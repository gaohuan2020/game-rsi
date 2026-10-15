'use strict';
const $ = id => document.getElementById(id);
const terminal = new Set(['completed', 'stopped', 'failed', 'interrupted']);
const names = {starting:'正在启动',switching:'正在切换游戏',stopping:'正在停止训练与自进化',loading:'加载冠军模型',loading_candidate:'加载候选模型',collecting:'模型游玩 / 教师标注',training:'更新本地模型权重',saving:'保存与校验 Checkpoint',evaluating:'同种子游戏评估',completed:'本轮实验已完成',failed:'任务失败',stopped:'任务已停止',interrupted:'任务被中断',playing:'实时游戏游玩'};
let latest = null, archived = null, tab = 'score', plotPoints = [], lastChart = '', lastLog = '', toastTimer, playPending=false, switchTimer, formInitialized=false;
let lastComparisons = '';
let lossSeries = 'both', chartSegments = [];
let trainingGame = null;
const trainingCache = new Map(), trainingRequests = new Map(), expandedComparisons = new Set();
const comparisonHistory = new Map(), comparisonRequests = new Map();
const fmt = (v, n=2) => Number.isFinite(v) ? v.toFixed(n) : '—';
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const active = job => job && !terminal.has(job.status);

function toast(message, error=false) {
  $('toast').textContent = message;
  $('toast').className = error ? 'error' : '';
  $('toast').hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {$('toast').hidden = true;}, error ? 14000 : 5000);
}

const gameGuides = {
  snake: ['吃到食物让蛇变长，同时避开墙壁和自身。', '学习寻找食物的安全路线，为蛇身增长保留移动空间。'],
  '2048': ['移动棋盘，合并相同数字的方块，争取更高得分。', '学习保持大数字的位置、留出空格，并减少无效移动。'],
  tetris: ['移动和旋转方块，填满整行即可消除；出生位置被堵住时失败。', '学习选择落点、减少空洞、控制堆叠高度，并创造连续消行机会。'],
  pacman: ['收集豆子和能量豆，避开幽灵追击。', '学习规划收集路线，判断幽灵威胁并利用能量豆脱险。'],
  racer: ['沿赛道向前行驶，控制方向和速度，避开交通车辆。', '学习提前判断弯道，在行驶距离和碰撞风险之间选择合适速度。'],
  invaders: ['移动飞船、躲避子弹并射击外星敌人。', '学习瞄准敌人、寻找安全位置，并在进攻时避开弹道。'],
  sokoban: ['将箱子推到目标位置，全部归位即可过关。', '学习规划推箱顺序，避免将箱子推入无法恢复的死角。'],
  mario: ['向前探索关卡，跳过障碍并躲避敌人。', '学习选择起跳时机、控制移动节奏，提高向前推进的距离。'],
  flappy: ['控制小鸟飞行高度，穿过水管之间的空隙。', '学习根据高度和下落速度调整振翅时机，增加通过的水管数。'],
  breakout: ['移动挡板接住小球，反弹小球击碎砖块。', '学习预测小球落点、调整挡板位置，延长回合并击碎更多砖块。']
};

function renderGameGuide(game) {
  const [goal, learning] = gameGuides[game] || gameGuides.snake;
  $('guide-game').textContent = gameName(game);
  $('guide-goal').textContent = t(goal);
  $('guide-learning').textContent = t(learning);
}

async function api(path, body) {
  const response = await fetch(path, body ? {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)} : {});
  const value = await response.json();
  if (!response.ok) {
    const detail = value.detail || t('请求失败');
    throw new Error(Array.isArray(detail) ? detail.map(x => t(x.msg)).join('; ') : t(detail));
  }
  return value;
}

function config() {
  const form = $('config-form');
  if (!form.reportValidity()) return null;
  const data = Object.fromEntries(new FormData(form));
  for (const key of ['rounds','train_steps','collect_steps','eval_episodes','batch_size','learning_rate','epsilon','teacher_mix','min_gain','max_episode_steps','seed','trajectory_candidates','trajectory_keep']) data[key] = Number(data[key]);
  data.tune_tower = form.elements.tune_tower.checked;
  data.require_ci = form.elements.require_ci.checked;
  data.actor = $('actor').value;
  localStorage.setItem('game-rsi-config', JSON.stringify(data));
  return data;
}

async function start(kind) {
  const cfg = config();
  if (!cfg) return;
  const button = $(kind === 'evolve' ? 'evolve-button' : 'play-button');
  if(kind === 'play') playPending=true;
  button.disabled = true;
  try {
    const job = await api('/api/'+kind, cfg);
    if (kind === 'evolve') {archived = null; $('history-select').value = 'latest';}
    toast(t(kind === 'evolve' ? '已启动自进化实验，同时运行真实游戏游玩。' : job.stopped_evolution_ids?.length ? '已请求停止上一游戏的训练与 RSI 自进化，正在切换游玩。' : job.switched ? '正在切换游戏，复用已加载的模型和浏览器。' : '已启动游戏进程，正在准备环境。'));
    playPending=false;
    render(await api('/api/state'));
  } catch (error) {playPending=false; toast(error.message, true); button.disabled = false;}
}

function jobGame(job) {return job?.game || job?.config?.game;}

function loadTraining(snapshot, job) {
  const current = snapshot.evolution;
  const signature = JSON.stringify(job);
  if (current?.job.id === job.id) {
    trainingCache.set(job.id,{signature,details:current});
    return current;
  }
  const cached = trainingCache.get(job.id);
  if (cached?.signature !== signature && !trainingRequests.has(job.id)) {
    const request = api('/api/jobs/'+encodeURIComponent(job.id)).then(details=>{
      trainingCache.set(job.id,{signature,details});
      return true;
    }).catch(error=>{
      if ($('game').value === jobGame(job)) toast(error.message,true);
      return false;
    }).finally(()=>{
      trainingRequests.delete(job.id);
    });
    trainingRequests.set(job.id,request);
    // Re-render both the training panel and score table for the currently selected game.
    request.then(updated=>{if (updated && latest) render(latest);});
  }
  return cached?.details || null;
}

function trainingForGame(snapshot, game) {
  if (archived && jobGame(archived.job) === game) return archived;
  const job = snapshot.jobs.find(j=>j.kind === 'evolve' && jobGame(j) === game)
    || (jobGame(snapshot.evolution?.job) === game ? snapshot.evolution.job : null);
  return job ? loadTraining(snapshot,job) : null;
}

function render(snapshot) {
  latest = snapshot;
  if(!formInitialized) {
    formInitialized=true;
    const current=active(snapshot.play?.job) ? snapshot.play.job.config : null;
    if(current) {
      for(const [key,value] of Object.entries(current)) {
        // Gameplay retains its original recipe; new training uses the form's defaults/preferences.
        if (key === 'train_steps' || key === 'collect_steps') continue;
        const field=$('config-form').elements[key];
        if(field) {if(field.type==='checkbox') field.checked=!!value; else field.value=value;}
      }
      $('actor').value=current.actor;
    }
  }
  const game = $('game').value;
  renderGameGuide(game);
  if (archived && !snapshot.jobs.some(j=>j.id === archived.job.id) && snapshot.evolution?.job.id !== archived.job.id) {
    archived=null;
    $('history-select').value='latest';
  }
  if (trainingGame !== game) {
    trainingGame = game;
    archived = null;
    $('history-select').value = 'latest';
    delete $('history-select').dataset.ids;
    lastChart=lastLog=lastComparisons='';
    $('chart-tooltip').hidden=true;
  }
  const train = trainingForGame(snapshot,game);
  const job = train?.job;
  const running = active(snapshot.evolution?.job);
  $('play-button').disabled = playPending;
  $('play-button').textContent = t(active(snapshot.play?.job) ? '▷ 切换游玩' : '▷ 启动游玩');
  $('evolve-button').disabled = !!running;
  $('stop-evolve').disabled = !running;
  $('stop-play').disabled = !active(snapshot.play?.job);
  $('run-state').textContent = t(job?.status === 'stopping' ? '正在停止训练与自进化' : active(job) ? '实验运行中' : job?.status === 'failed' ? '实验需要处理' : job ? '实验已结束' : '等待实验');
  $('run-led').className = 'status-led' + (active(job) ? ' active' : '');
  const play = jobGame(snapshot.play?.job) === game ? snapshot.play : null;
  $('game-status').textContent = t(names[play?.job.phase] || '待机');
  const rawFrame = $('frame-source').value === 'evolution' ? train?.frame : play?.frame || train?.frame;
  const frame = rawFrame && (!rawFrame.game || rawFrame.game === game) ? rawFrame : null;
  renderFrame(frame, $('frame-source').value === 'live' && !!frame && frame === play?.frame);
  const champ = (jobGame(play?.job.champion) === game ? play.job.champion : null)
    || snapshot.champions?.find(c=>c.game === game && c.backend === $('backend').value)
    || job?.champion;
  $('champion-version').textContent = champ?.version === 'base' || !champ ? 'base' : champ.version.split('/').pop();
  $('champion-model').textContent = champ?.checkpoint || $('model').value;
  $('champion-model').title = champ?.checkpoint || $('model').value;
  const phase = job?.phase || 'idle';
  $('phase-title').textContent = t(names[phase] || '等待启动实验');
  $('round-label').textContent = job ? t('ROUND {round} / {total}',{round:job.round || 0,total:job.config.rounds}) : 'ROUND —';
  $('phase-progress').textContent = Math.round((job?.progress || 0)*100)+'%';
  $('progress-fill').style.width = Math.round((job?.progress || 0)*100)+'%';
  $('phase-led').className = 'status-led' + (active(job) ? ' active' : job?.status === 'failed' ? ' error' : '');
  const note = job?.error || (phase === 'collecting' ? t('已采集 {count} / {target} 帧；动作由模型或教师执行，标签由游戏教师提供。',{count:job.collected || 0,target:job.collect_target})
    : phase === 'training' ? t('Step {step} / {total} · 使用真实轨迹更新 {backend} 权重。',{step:job.train_step || 0,total:job.train_steps || job.config.train_steps,backend:job.backend})
    : phase === 'evaluating' ? t('{model}模型 · 已完成 {count} / {total} 局。',{model:t(job.evaluation_model === 'candidate' ? '候选' : '冠军'),count:job.eval_episode || 0,total:job.eval_target || job.config.eval_episodes})
    : phase === 'stopping' ? t('正在停止当前训练与 RSI 自进化；当前步骤完成后退出。')
    : phase === 'saving' ? t('写入实际模型权重，并校验重新加载后的决策是否一致。')
    : phase === 'completed' ? t('全部轮次已完成。仅通过评估门槛的候选会成为冠军。')
    : phase === 'loading' || phase === 'loading_candidate' ? t('首次运行可能需要下载模型，完整进度可在原始日志中查看。')
    : t(job ? '每轮先预注册实验，再采样、更新权重和评估候选模型。' : '此游戏暂无训练记录，开始自进化后显示训练过程。'));
  $('phase-note').textContent = note;
  for (const node of $('pipeline').querySelectorAll('[data-phase]')) {
    const match = node.dataset.phase === phase || (phase === 'saving' && node.dataset.phase === 'training') || (phase === 'completed' && node.dataset.phase === 'promoted' && job?.last_evaluation?.passed);
    node.classList.toggle('active', match);
  }
  const metrics = (train?.events || []).filter(e => e.kind === 'metric');
  const losses = metrics.filter(e => Number.isFinite(e.loss));
  const validations = metrics.filter(e => Number.isFinite(e.agreement));
  const metric = losses.at(-1);
  $('loss-metric').textContent = fmt(metric?.loss, 4);
  $('loss-breakdown').textContent = t('KL {kl} · 标签熵 {entropy}',{kl:fmt(metric?.kl,6),entropy:fmt(metric?.target_entropy,4)});
  $('train-step').textContent = metric ? t('Round {round} · Step {step} / {total}',{round:metric.round,step:metric.step,total:job.config.train_steps}) : t('尚未开始权重更新');
  $('sample-count').textContent = job?.collected?.toLocaleString() || '—';
  $('agreement').textContent = validations.length ? fmt(validations.at(-1).agreement*100, 1)+'%' : '—';
  $('gpu-memory').textContent = metric ? fmt(metric.gpu_gb, 2)+' GB' : '—';
  const evaluation = (train?.events || []).filter(e => e.kind === 'evaluation').at(-1);
  renderEvaluation(evaluation);
  renderChart(metrics, (train?.events || []).filter(e => e.kind === 'evaluation'), job);
  renderRoundComparisons(snapshot);
  renderLog(train || play);
  const select = $('history-select'), selected = select.value;
  const options = snapshot.jobs.filter(j => j.kind === 'evolve' && jobGame(j) === game);
  if (select.dataset.ids !== options.map(j => j.id).join(',')) {
    select.innerHTML = `<option value="latest">${t('当前实验')}</option>`+options.map(j => {
      const label=Number.isFinite(j.created_at) ? new Date(j.created_at*1000).toLocaleString(currentLanguage,{month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',hour12:false}) : j.id;
      return `<option value="${esc(j.id)}" title="${esc(j.id)}">${esc(label)}</option>`;
    }).join('');
    select.value = selected;
    if (!select.value) select.value = 'latest';
    select.dataset.ids = options.map(j => j.id).join(',');
  }
  select.title=select.value === 'latest' ? t('当前实验') : select.value;
}

function renderFrame(frame, live=false) {
  $('game-frame').hidden = !frame;
  $('empty-game').hidden = !!frame;
  if (!frame) {
    $('live-score').textContent='—';
    $('live-episode').textContent=t('等待真实游戏画面');
    $('screen-source').textContent='NO SIGNAL';
    $('frame-time').textContent=t('等待首帧');
    $('latency').textContent='— ms / decision';
    $('input-badge').textContent=t('输入模式：等待模型加载');
    $('action-name').textContent='ACTION —';
    $('action-bars').innerHTML=`<p class="muted">${t('等待模型返回动作概率')}</p>`;
    return;
  }
  const source = live && frame.display_sequence ? '/api/play/stream' : frame.image;
  if(source && $('game-frame').getAttribute('src') !== source) $('game-frame').src = source;
  $('live-score').textContent = String(frame.score);
  $('live-episode').textContent = t('Episode {episode} · Step {step} · seed {seed}',{episode:(frame.episode || 0)+1,step:frame.step || 0,seed:frame.seed});
  $('screen-source').textContent = `${frame.game?.toUpperCase() || frame.source?.toUpperCase() || 'LIVE'} / ${frame.actor?.toUpperCase() || 'MODEL'}`;
  const fps = live && Number.isFinite(frame.display_fps) && active(latest?.play?.job) ? `${frame.display_fps} FPS · ` : '';
  $('frame-time').textContent = fps + new Date(frame.updated_at*1000).toLocaleTimeString(currentLanguage);
  $('latency').textContent = fmt(frame.latency_ms, 0)+' ms / decision';
  $('input-badge').textContent = t('输入模式：')+t(frame.input_mode === 'pixels' ? '游戏截图' : frame.actor === 'teacher' ? 'Teacher 内部状态' : frame.actor === 'random' ? 'Random 随机动作' : '结构化游戏状态');
  $('action-name').textContent = 'ACTION '+(frame.action || '—').toUpperCase();
  $('action-bars').innerHTML = frame.actions.map((name,i) => `<div class="action-row ${name === frame.action ? 'chosen' : ''}"><span>${esc(name)}</span><div class="action-track"><div class="action-fill" style="width:${Math.max(0, Math.min(100, (frame.probabilities?.[i] || 0)*100))}%"></div></div><span>${Math.round((frame.probabilities?.[i] || 0)*100)}%</span></div>`).join('');
}

function renderEvaluation(ev) {
  $('gain-metric').textContent = ev ? (ev.relative_gain >= 0 ? '+' : '')+fmt(ev.relative_gain*100,1)+'%' : '—';
  $('gain-metric').className = ev ? ev.delta >= 0 ? 'positive' : 'negative' : '';
  $('gain-note').textContent = ev ? t('{pairs} 局配对 · {result}',{pairs:ev.pairs,result:t(ev.passed ? '候选已晋升' : '保留原冠军')}) : t('评估后显示真实结果');
  $('gate-status').textContent = t(ev ? ev.passed ? '通过 / 已晋升' : '未通过 / 已保留记录' : '等待评估');
  const checks = ev ? Object.entries(ev.checks).filter(([k,v])=>!v && (!ev.required_checks || ev.required_checks.includes(k))).map(([k])=>t({score_gain:'得分增益未达门槛',confidence_interval:'置信区间未证明正增益',no_large_regression:'存在较大的单局退步'}[k])).join('; ') : '';
  const positive=ev?.promotion_rule === 'positive_gain' || (!ev && $('config-form').elements.promotion_rule.value === 'positive_gain');
  const rule=t(positive ? '平均配对得分提高即晋升；置信区间和单局退步仅供参考。' : '超过预设得分增益、通过置信区间和退步检查后，才自动切换冠军。');
  $('gate-description').textContent = ev ? rule+' '+t('训练前 {before} → 训练后 {after}；增益 CI {ci}。{checks}',{before:fmt(ev.baseline),after:fmt(ev.candidate),ci:`[${fmt(ev.ci95[0])}, ${fmt(ev.ci95[1])}]`,checks:checks || t('全部门槛通过，游戏将在下一局加载新的冠军。')}) : rule;
}

function gameName(game) {
  const option = [...$('game').options].find(o=>o.value === game);
  return option ? option.textContent.split(' · ').at(-1) : game;
}

function loadComparisonHistory(snapshot, game) {
  const jobs=snapshot.jobs.filter(j=>j.kind === 'evolve' && jobGame(j) === game);
  const signature=JSON.stringify(jobs.map(j=>[j.id,j.status,j.phase,j.round,j.last_evaluation]));
  const cached=comparisonHistory.get(game);
  if (cached?.signature !== signature && !comparisonRequests.has(game) && Date.now() >= (cached?.retryAfter || 0)) {
    const request=api('/api/games/'+encodeURIComponent(game)+'/comparisons').then(history=>{
      comparisonHistory.set(game,{signature,runs:history.runs});
    }).catch(error=>{
      comparisonHistory.set(game,{...cached,signature:null,retryAfter:Date.now()+5000,error:error.message});
      if ($('game').value === game) toast(error.message,true);
    }).finally(()=>{
      comparisonRequests.delete(game);
      if (latest) {renderRoundComparisons(latest);if (tab === 'score') renderChart([],[],null);}
    });
    comparisonRequests.set(game,request);
  }
  return cached;
}

function comparisonData(snapshot) {
  const game=$('game').value;
  const history=loadComparisonHistory(snapshot,game);
  const runs=new Map((history?.runs || []).filter(r=>jobGame(r.job) === game).map(r=>[r.job.id,r]));
  {
    for (const job of snapshot.jobs.filter(j=>j.kind === 'evolve' && jobGame(j) === game)) {
      runs.set(job.id,{job,evaluations:runs.get(job.id)?.evaluations || []});
    }
    if (jobGame(snapshot.evolution?.job) === game) {
      const current=snapshot.evolution;
      const evaluations=new Map((runs.get(current.job.id)?.evaluations || []).map(e=>[e.round,e]));
      for (const e of current.events || []) if (e.kind === 'evaluation') evaluations.set(e.round,e);
      runs.set(current.job.id,{job:current.job,evaluations:[...evaluations.values()]});
    }
  }
  const groups=[...runs.values()].sort((a,b)=>(a.job.created_at || 0)-(b.job.created_at || 0) || a.job.id.localeCompare(b.job.id))
    .map((run,index)=>({job:run.job,runIndex:index+1,loading:false,evaluations:run.evaluations.filter(e=>Number.isFinite(e.baseline) && Number.isFinite(e.candidate)).sort((a,b)=>a.round-b.round)}));
  const rows=groups.flatMap((group,index)=>{
    const entries=group.evaluations.map(ev=>({...group,ev,round:ev.round}));
    const round=Math.max(1,group.job.round || 0);
    if ((index === groups.length-1 || active(group.job)) && !entries.some(e=>e.round === round)) entries.push({...group,ev:null,round});
    return entries.sort((a,b)=>a.round-b.round);
  });
  return {game,history,groups,rows};
}

function renderRoundComparisons(snapshot) {
  const {game,history,groups,rows}=comparisonData(snapshot);
  const key=JSON.stringify([currentLanguage,game,!!history?.runs,history?.error,groups.map(({job,evaluations})=>[job.id,job.status,job.phase,job.round,evaluations])]);
  if (key === lastComparisons) return;
  lastComparisons=key;
  $('comparison-game-label').textContent=gameName(game);
  if (!history?.runs || !groups.length) {
    const message=!history?.runs ? history?.error ? '读取训练历史失败，请稍后重试。' : '正在读取训练历史…' : '此游戏暂无训练记录，开始自进化后显示训练过程。';
    $('round-comparisons').innerHTML=`<tr><td colspan="8" class="comparison-empty">${t(message)}</td></tr>`;
    return;
  }
  $('round-comparisons').innerHTML=rows.map(({job,ev,round,loading,runIndex},index)=>{
    const rowKey=job.id+':'+round, detailsId='comparison-'+job.id+'-'+round;
    const delta=ev ? Number.isFinite(ev.delta) ? ev.delta : ev.candidate-ev.baseline : null;
    const relative=ev ? Number.isFinite(ev.relative_gain) ? ev.relative_gain : delta/Math.max(Math.abs(ev.baseline),1) : null;
    const sign=delta > 0 ? '+' : '';
    const tone=delta > 0 ? 'positive' : delta < 0 ? 'negative' : 'neutral';
    const ci=Array.isArray(ev?.ci95) ? `[${fmt(ev.ci95[0])}, ${fmt(ev.ci95[1])}]` : '—';
    const ciNote=ev?.ci95 ? t(ev.ci95[0] > 0 ? '增益下界高于 0' : ev.ci95[1] < 0 ? '增益区间低于 0' : '增益区间包含 0') : '';
    const before=Array.isArray(ev?.baseline_scores) ? ev.baseline_scores : [];
    const after=Array.isArray(ev?.candidate_scores) ? ev.candidate_scores : [];
    const pairs=Math.min(before.length,after.length);
    const open=expandedComparisons.has(rowKey);
    const pending=loading ? t('读取中') : t(active(job) ? {training:'训练中',collecting:'采样中',evaluating:'评估中',saving:'保存中'}[job.phase] || '等待评估' : job.status === 'completed' ? '尚未评估' : names[job.phase] || '尚未评估');
    const paired=pairs ? `<table class="paired-score-table"><thead><tr><th>${t('Seed / 局')}</th><th>${t('训练前')}</th><th>${t('训练后')}</th><th>${t('增益')}</th></tr></thead><tbody>${before.slice(0,pairs).map((score,i)=>`<tr><td>${esc(ev.seeds?.[i] ?? i+1)}</td><td>${fmt(score)}</td><td>${fmt(after[i])}</td><td class="${after[i]-score > 0 ? 'positive' : after[i]-score < 0 ? 'negative' : ''}">${after[i]-score > 0 ? '+' : ''}${fmt(after[i]-score)}</td></tr>`).join('')}</tbody></table>` : `<p class="comparison-empty-pairs">${t('此轮未记录逐局成绩。')}</p>`;
    const date=Number.isFinite(job.created_at) ? new Date(job.created_at*1000).toLocaleString(currentLanguage,{month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',hour12:false}) : job.id;
    return `<tr class="round-comparison" data-game="${esc(jobGame(job))}" data-run-id="${esc(job.id)}" data-round="${esc(round)}">
      <th scope="row" class="comparison-round" title="${esc(t('累计第 {round} 轮',{round:index+1}))}">R${index+1}</th>
      <td class="comparison-run" title="${esc(job.id)}"><strong>${t('第 {run} 次 · R{round}',{run:runIndex,round})}</strong><small>${esc(date)}</small></td>
      <td class="comparison-score numeric"><strong>${fmt(ev?.baseline)}</strong></td>
      <td class="comparison-score candidate numeric"><strong>${fmt(ev?.candidate)}</strong></td>
      <td class="comparison-gain numeric ${tone}"><strong>${ev ? sign+fmt(relative*100,1)+'%' : '—'}</strong><small>${ev ? t('得分 {delta}',{delta:sign+fmt(delta)}) : ''}</small></td>
      <td class="comparison-ci"><strong>${ci}</strong><small>${ciNote}</small></td>
      <td class="comparison-result"><span class="status-tag ${ev?.passed ? 'promoted' : ev ? 'rejected' : 'pending'}">${ev?.passed ? '✓ ' : ''}${ev ? t(ev.passed ? '已晋升' : '未晋升') : pending}</span></td>
      <td class="comparison-detail-cell"><button type="button" class="round-details-toggle" data-round-key="${esc(rowKey)}" aria-expanded="${open}" aria-controls="${esc(detailsId)}" aria-label="${esc(t('{game} · 第 {round} 轮明细',{game:gameName(jobGame(job)),round}))}" ${ev ? '' : 'disabled'}>${t('明细')}<span class="comparison-chevron" aria-hidden="true"></span></button></td>
    </tr><tr id="${esc(detailsId)}" class="comparison-details" ${open ? '' : 'hidden'}><td colspan="8"><div class="comparison-expanded"><div class="comparison-expanded-heading"><strong>${esc(gameName(jobGame(job)))} · ${esc(t('第 {round} 轮',{round}))}</strong><span>${t('逐局配对得分 · {pairs} 局',{pairs})}</span></div><div class="table-scroll">${paired}</div><div class="comparison-checkpoint"><span>${t('实验')}</span><code>${esc(job.id)}</code>${ev?.checkpoint ? `<span>Checkpoint</span><code>${esc(ev.checkpoint)}</code>` : ''}</div></div></td></tr>`;
  }).join('');
}

function renderChart(metrics, evaluations, job) {
  const historical=tab === 'score' && latest ? comparisonData(latest).rows : [];
  const scores=historical.length ? historical.flatMap(({ev,job,runIndex,round},index)=>ev ? [{
    ...ev, round:index+1, runId:job.id,
    label:t('累计第 {round} 轮',{round:index+1})+' · '+t('第 {run} 次 · R{round}',{run:runIndex,round})
  }] : []) : evaluations;
  const key=JSON.stringify([tab,lossSeries,currentLanguage,job?.id,job?.config.train_steps,metrics,scores]);
  if (key === lastChart) return;
  lastChart=key;
  $('loss-chart-controls').hidden=tab !== 'loss';
  $('loss-chart-caption').hidden=false;
  $('loss-chart-caption').textContent=t(tab === 'loss' ? '累计训练步数 · 纵轴自动缩放 · 验证仅绘制记录点' : '累计轮次 · 按实验时间排序 · 保留未晋升结果');
  const validation=metrics.filter(e=>Number.isFinite(e.val_loss)).at(-1);
  $('validation-summary').textContent=validation ? t('验证 {loss} · R{round} Step {step}',{loss:fmt(validation.val_loss,5),round:validation.round,step:validation.step}) : t('验证 —');
  const stepsPerRound=job?.config.train_steps || job?.train_steps || Math.max(1,...metrics.map(e=>e.step || 0));
  function lossPoints(field) {
    const points=new Map();
    for (const e of metrics) if (Number.isFinite(e[field]) && Number.isInteger(e.round) && e.round >= 1 && Number.isInteger(e.step) && e.step >= 0) {
      points.set(e.round+':'+e.step,{x:(e.round-1)*stepsPerRound+e.step,y:e[field],round:e.round,step:e.step,label:t('R{round} step {step}',{round:e.round,step:e.step})});
    }
    return [...points.values()].sort((a,b)=>a.round-b.round || a.step-b.step);
  }
  const datasets=tab === 'loss' ? [
    {id:'train',name:t('Train'),color:'#92e7c6',points:lossPoints('loss')},
    {id:'validation',name:t('Validation'),color:'#eeb875',points:lossPoints('val_loss')}
  ].filter(d=>lossSeries === 'both' || lossSeries === d.id) : [
    {id:'before',name:t('训练前'),color:'#748d8a',points:scores.map(e=>({x:e.round,y:e.baseline,round:e.round,runId:e.runId,label:e.label || t('R{round} 训练前',{round:e.round})}))},
    {id:'after',name:t('训练后'),color:'#92e7c6',points:scores.map(e=>({x:e.round,y:e.candidate,round:e.round,runId:e.runId,label:e.label || t('R{round} 训练后',{round:e.round})}))}
  ];
  $('chart-legend').innerHTML=datasets.map(d=>`<span style="color:${d.color}">● ${esc(d.name)}</span>`).join('　');
  const all=datasets.flatMap(d=>d.points);
  $('empty-chart').hidden=all.length > 0;
  $('chart-tooltip').hidden=true;
  $('chart').setAttribute('aria-label',t(tab === 'loss' ? lossSeries === 'validation' ? '实际验证损失随训练步数的变化' : '真实训练和验证损失曲线' : '各轮训练前后模型的真实评估得分'));
  plotPoints=[]; chartSegments=[];
  if (!all.length) {$('chart').innerHTML='';return;}
  const maxX=Math.max(1,...all.map(p=>p.x));
  const dataMin=Math.min(...all.map(p=>p.y)), dataMax=Math.max(...all.map(p=>p.y));
  const padding=Math.max((dataMax-dataMin)*.12,Math.abs(dataMax)*.001,1e-6);
  const minY=tab === 'loss' ? Math.max(0,dataMin-padding) : Math.min(0,dataMin);
  const maxY=tab === 'loss' ? dataMax+padding : Math.max(1,dataMax)*1.12;
  const decimals=tab === 'loss' ? Math.min(8,Math.max(2,1-Math.floor(Math.log10((maxY-minY)/3)))) : 0;
  const X=x=>64+x/maxX*475, Y=y=>175-(y-minY)/(maxY-minY)*147;
  let svg='';
  for (let i=0;i<4;i++) {
    const y=minY+(maxY-minY)*i/3;
    svg+=`<line x1="64" y1="${Y(y)}" x2="539" y2="${Y(y)}" stroke="#2a3638" stroke-dasharray="3 5"/><text x="54" y="${Y(y)+3}" fill="#789196" text-anchor="end" font-size="9" font-family="monospace">${fmt(y,decimals)}</text>`;
  }
  for (let i=0;i<=4;i++) svg+=`<text x="${X(maxX*i/4)}" y="198" fill="#789196" text-anchor="middle" font-size="9" font-family="monospace">${fmt(maxX*i/4,maxX%4 ? 1 : 0)}</text>`;
  for (const d of datasets) {
    // Each round has its own replay/validation set. Never connect its endpoints to another round.
    const rounds=tab === 'loss' ? [...new Set(d.points.map(p=>p.round))] : [null];
    for (const round of rounds) {
      const points=d.points.filter(p=>round === null || p.round === round).map(p=>({...p,name:d.name,series:d.id,sx:X(p.x),sy:Y(p.y)}));
      if (!points.length) continue;
      const segmentIndex=chartSegments.length;
      chartSegments.push({points});
      if (points.length > 1) svg+=`<polyline data-chart-segment="${segmentIndex}" data-series="${d.id}" points="${points.map(p=>`${p.sx},${p.sy}`).join(' ')}" fill="none" stroke="${d.color}" stroke-width="${d.id === 'validation' ? 2 : 1.5}"/>`;
      for (const p of points) {
        const pointIndex=plotPoints.length;
        plotPoints.push(p);
        if (d.id === 'validation' || points.length < 30) svg+=`<circle data-chart-point="${pointIndex}" data-series="${d.id}" cx="${p.sx}" cy="${p.sy}" r="${d.id === 'validation' ? 3 : 2.5}" fill="${d.color}"/>`;
      }
    }
  }
  $('chart').innerHTML=svg;
}

function nearestChartPoint(x,y,target) {
  const pointTarget=target?.closest('[data-chart-point]');
  if (pointTarget) return plotPoints[Number(pointTarget.dataset.chartPoint)];
  const segmentTarget=target?.closest('[data-chart-segment]');
  let selected=segmentTarget ? chartSegments[Number(segmentTarget.dataset.chartSegment)] : null;
  if (!selected) {
    let closest=Infinity;
    for (const segment of chartSegments) {
      const points=segment.points;
      for (let i=0;i<Math.max(1,points.length-1);i++) {
        const a=points[i], b=points[i+1] || a;
        const dx=b.sx-a.sx, dy=b.sy-a.sy, length=dx*dx+dy*dy;
        const ratio=length ? Math.max(0,Math.min(1,((x-a.sx)*dx+(y-a.sy)*dy)/length)) : 0;
        const distance=(x-a.sx-ratio*dx)**2+(y-a.sy-ratio*dy)**2;
        if (distance < closest) {closest=distance;selected=segment;}
      }
    }
  }
  // The tooltip always reports a logged observation, even between validation markers.
  return selected?.points.reduce((best,p)=>Math.abs(p.sx-x)<Math.abs(best.sx-x) ? p : best);
}

function renderLog(details) {
  const events = details?.events || [];
  const key = JSON.stringify([details?.job.id,events.slice(-40)]);
  $('raw-log').textContent = details?.logs || t('尚无日志');
  if (key === lastLog) return;
  lastLog = key;
  const text = e => {
    if(e.kind === 'metric') return `${t('R{round} step {step}',{round:e.round,step:e.step})} ${Number.isFinite(e.loss) ? 'loss='+fmt(e.loss,4) : ''} ${Number.isFinite(e.kl) ? 'KL='+fmt(e.kl,6) : ''} ${Number.isFinite(e.val_loss) ? 'val='+fmt(e.val_loss,4)+' '+t('教师一致率')+'='+fmt(e.agreement*100,1)+'%' : ''}`;
    if(e.kind === 'phase') return t(names[e.phase] || e.phase);
    if(e.kind === 'episode') return t('{source} · seed {seed} · score {score}',{source:e.source,seed:e.seed,score:e.score});
    if(e.kind === 'evaluation') return `R${e.round} ${t(e.passed ? '已晋升' : '未晋升')} · delta ${fmt(e.delta)} · CI [${e.ci95.map(v=>fmt(v)).join(', ')}]`;
    if(e.kind === 'experiment') return t('注册 R{round} · {hypothesis}',{round:e.round,hypothesis:e.hypothesis});
    if(e.kind === 'collected') return t('{samples} 帧 / {episodes} episodes',{samples:e.samples,episodes:e.episodes});
    if(e.kind === 'data_selection') return t('训练 {train} 条 · 验证 {dev} 条 · 纠正样本 {corrections} 条',{train:e.train_samples,dev:e.dev_samples,corrections:e.correction_samples || 0});
    if(e.kind === 'trained') return t('Checkpoint 已保存 · reload parity={parity}',{parity:e.reload_parity});
    if(e.kind === 'training_setup') return t('{train} train / {dev} dev · {parameters} 可训练参数',{train:e.train_samples,dev:e.dev_samples,parameters:e.trainable_parameters.toLocaleString(currentLanguage)});
    return t(e.message || e.kind);
  };
  $('event-log').innerHTML = events.length ? events.slice(-40).map(e => `<div class="event-line"><time>${new Date(e.time*1000).toLocaleTimeString(currentLanguage)}</time><span>${esc(e.kind.toUpperCase())}</span><span>${esc(text(e))}</span></div>`).join('') : `<p class="muted">${t('等待任务启动。所有指标来自本地训练与游戏进程。')}</p>`;
  $('event-log').scrollTop = $('event-log').scrollHeight;
}

$('config-form').addEventListener('submit', e => {e.preventDefault(); start('evolve');});
$('play-button').addEventListener('click', () => start('play'));
function switchLiveGame() {
  if (latest) render(latest);
  if(!active(latest?.play?.job) && !active(latest?.evolution?.job)) return;
  clearTimeout(switchTimer);
  switchTimer=setTimeout(() => start('play'),150);
}
$('game').addEventListener('change', switchLiveGame);
$('actor').addEventListener('change', switchLiveGame);
$('fullscreen-button').addEventListener('click', async () => {
  try {
    if(document.fullscreenElement) await document.exitFullscreen();
    else await document.querySelector('.game-panel').requestFullscreen();
  } catch(error) {toast(t('当前浏览器未允许全屏显示。'),true);}
});
document.addEventListener('fullscreenchange', () => {$('fullscreen-button').textContent=t(document.fullscreenElement ? '退出全屏' : '⛶ 全屏');});
for (const kind of ['play','evolve']) $('stop-'+kind).addEventListener('click', async () => {
  const job = kind === 'play' ? latest?.play?.job : latest?.evolution?.job;
  if (!job) return;
  try {await api(`/api/jobs/${job.id}/stop`, {}); toast(t('已请求停止，进程会在当前步骤完成后退出。'));} catch(error) {toast(error.message,true);}
});
$('backend').addEventListener('change', () => {
  const pixel = $('backend').value === 'playjev';
  $('model').value = pixel ? 'OmniJev/PlayJev-0.8B' : 'shgao/rsi-jev-v1.0-qwen3.5-0.8b';
  $('input_mode').value = pixel ? 'pixels' : 'auto';
  $('config-form').elements.learning_rate.value = pixel ? '0.00001' : '0.0001';
  switchLiveGame();
});
$('frame-source').addEventListener('change', () => {if (latest) render(latest);});
$('config-form').elements.promotion_rule.addEventListener('change', () => {if (latest) render(latest);});
$('round-comparisons').addEventListener('click',event=>{
  const button=event.target.closest('.round-details-toggle');
  if (!button || button.disabled) return;
  const open=button.getAttribute('aria-expanded') !== 'true';
  button.setAttribute('aria-expanded',String(open));
  $(button.getAttribute('aria-controls')).hidden=!open;
  if (open) expandedComparisons.add(button.dataset.roundKey);
  else expandedComparisons.delete(button.dataset.roundKey);
});
$('history-select').addEventListener('change', async () => {
  const id=$('history-select').value, game=$('game').value;
  try {
    const details=id === 'latest' ? null : await api('/api/jobs/'+encodeURIComponent(id));
    if ($('game').value !== game || $('history-select').value !== id) return;
    archived=details; lastChart=lastLog=lastComparisons='';
    if(latest) render(latest);
  } catch(error){if ($('game').value === game) toast(error.message,true);}
});
for (const [id, value] of [['loss-tab','loss'],['score-tab','score']]) $(id).addEventListener('click', () => {
  tab=value; $('loss-tab').classList.toggle('selected',tab==='loss'); $('score-tab').classList.toggle('selected',tab==='score');
  $('loss-tab').setAttribute('aria-selected',String(tab==='loss')); $('score-tab').setAttribute('aria-selected',String(tab==='score'));
  if(latest) render(latest);
});
$('log-toggle').addEventListener('click', () => {$('raw-log').hidden = !$('raw-log').hidden; $('log-toggle').textContent = t($('raw-log').hidden ? '查看原始日志 ↗' : '收起原始日志 ↑');});
$('loss-chart-controls').addEventListener('click',event=>{
  const button=event.target.closest('[data-loss-series]');
  if (!button) return;
  lossSeries=button.dataset.lossSeries;
  for (const item of $('loss-chart-controls').querySelectorAll('[data-loss-series]')) item.setAttribute('aria-pressed',String(item === button));
  lastChart='';
  if(latest) render(latest);
});
$('chart').addEventListener('pointermove',event=>{
  if(!plotPoints.length) return;
  const matrix=$('chart').getScreenCTM();
  if (!matrix) return;
  const point=$('chart').createSVGPoint(); point.x=event.clientX; point.y=event.clientY;
  const local=point.matrixTransform(matrix.inverse());
  const p=nearestChartPoint(local.x,local.y,event.target);
  if (!p) return;
  const tooltip=$('chart-tooltip'); tooltip.hidden=false; tooltip.textContent=`${p.label} · ${p.name}: ${fmt(p.y,tab === 'loss' ? 6 : 2)}`;
  const box=$('chart').getBoundingClientRect();
  tooltip.style.left=Math.max(0,Math.min(box.width-tooltip.offsetWidth,event.clientX-box.left+10))+'px';
  tooltip.style.top='12px';
});
$('chart').addEventListener('pointerleave', () => {$('chart-tooltip').hidden=true;});
setInterval(() => {$('clock').textContent=new Date().toLocaleTimeString(currentLanguage);},1000);
try {
  const saved=JSON.parse(localStorage.getItem('game-rsi-config') || 'null');
  if (localStorage.getItem('game-rsi-default-collect-steps') !== '1000') {
    if (saved) {saved.collect_steps=1000;localStorage.setItem('game-rsi-config',JSON.stringify(saved));}
    localStorage.setItem('game-rsi-default-collect-steps','1000');
  }
  if (localStorage.getItem('game-rsi-default-train-steps') !== '70') {
    if (saved?.train_steps === 80) {saved.train_steps=70;localStorage.setItem('game-rsi-config',JSON.stringify(saved));}
    localStorage.setItem('game-rsi-default-train-steps','70');
  }
  if(saved) for(const [k,v] of Object.entries(saved)) {
    const field=$('config-form').elements[k];
    if(field) {if(field.type==='checkbox') field.checked=!!v; else field.value=v;}
  }
} catch (_) {}
const stream=new EventSource('/api/events');
stream.addEventListener('snapshot', e => {
  $('connection-led').className='status-led connected'; $('connection-label').textContent=t('本地服务已连接');
  try {render(JSON.parse(e.data));} catch(error) {console.error(error);}
});
stream.onerror=() => {$('connection-led').className='status-led error'; $('connection-label').textContent=t('连接中断，正在重连');};
api('/api/state').then(render).catch(error=>toast(error.message,true));

document.addEventListener('languagechange',()=>{
  lastChart=lastLog=lastComparisons='';
  delete $('history-select').dataset.ids;
  $('connection-label').textContent=t($('connection-led').classList.contains('connected') ? '本地服务已连接' : '连接中断，正在重连');
  $('fullscreen-button').textContent=t(document.fullscreenElement ? '退出全屏' : '⛶ 全屏');
  $('log-toggle').textContent=t($('raw-log').hidden ? '查看原始日志 ↗' : '收起原始日志 ↑');
  if(latest) render(latest);
});
