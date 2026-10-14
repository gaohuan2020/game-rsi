'use strict';
const $ = id => document.getElementById(id);
const terminal = new Set(['completed', 'stopped', 'failed', 'interrupted']);
const names = {starting:'正在启动',switching:'正在切换游戏',stopping:'正在停止训练与自进化',loading:'加载冠军模型',loading_candidate:'加载候选模型',collecting:'模型游玩 / 教师标注',training:'更新本地模型权重',saving:'保存与校验 Checkpoint',evaluating:'同种子游戏评估',completed:'本轮实验已完成',failed:'任务失败',stopped:'任务已停止',interrupted:'任务被中断',playing:'实时游戏游玩'};
let latest = null, archived = null, tab = 'loss', plotPoints = [], lastChart = '', lastLog = '', toastTimer, playPending=false, switchTimer, formInitialized=false;
let lastComparisons = '';
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
  for (const key of ['rounds','train_steps','collect_steps','eval_episodes','batch_size','learning_rate','epsilon','teacher_mix','min_gain','max_episode_steps','seed']) data[key] = Number(data[key]);
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

function render(snapshot) {
  latest = snapshot;
  if(!formInitialized) {
    formInitialized=true;
    const current=active(snapshot.play?.job) ? snapshot.play.job.config : null;
    if(current) {
      for(const [key,value] of Object.entries(current)) {
        const field=$('config-form').elements[key];
        if(field) {if(field.type==='checkbox') field.checked=!!value; else field.value=value;}
      }
      $('actor').value=current.actor;
    }
  }
  const train = archived || snapshot.evolution;
  const job = train?.job;
  const running = active(snapshot.evolution?.job);
  $('play-button').disabled = playPending;
  $('play-button').textContent = t(active(snapshot.play?.job) ? '▷ 切换游玩' : '▷ 启动游玩');
  $('evolve-button').disabled = !!running;
  $('stop-evolve').disabled = !running;
  $('stop-play').disabled = !active(snapshot.play?.job);
  $('run-state').textContent = t(snapshot.evolution?.job?.status === 'stopping' ? '正在停止训练与自进化' : running ? '实验运行中' : snapshot.evolution?.job?.status === 'failed' ? '实验需要处理' : snapshot.evolution ? '实验已结束' : '等待实验');
  $('run-led').className = 'status-led' + (running ? ' active' : '');
  $('game-status').textContent = t(names[snapshot.play?.job?.phase] || '待机');
  const frame = $('frame-source').value === 'evolution' ? train?.frame : snapshot.play?.frame || train?.frame;
  renderFrame(frame, $('frame-source').value === 'live' && !!snapshot.play?.frame);
  const champ = snapshot.play?.job?.champion || snapshot.evolution?.job?.champion;
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
    : t('每轮先预注册实验，再采样、更新权重和评估候选模型。'));
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
  renderExperiments((train?.events || []).filter(e => e.kind === 'evaluation'));
  renderRoundComparisons((train?.events || []).filter(e => e.kind === 'evaluation'), job);
  renderLog(train || snapshot.play);
  const select = $('history-select'), selected = select.value;
  const options = snapshot.jobs.filter(j => j.kind === 'evolve');
  if (select.dataset.ids !== options.map(j => j.id).join(',')) {
    select.innerHTML = `<option value="latest">${t('当前实验')}</option>`+options.map(j => `<option value="${esc(j.id)}">${esc(j.id)} · ${esc(j.game)}</option>`).join('');
    select.value = selected;
    if (!select.value) select.value = 'latest';
    select.dataset.ids = options.map(j => j.id).join(',');
  }
}

function renderFrame(frame, live=false) {
  $('game-frame').hidden = !frame;
  $('empty-game').hidden = !!frame;
  if (!frame) return;
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
  const checks = ev ? Object.entries(ev.checks).filter(([k,v])=>!v).map(([k])=>t({score_gain:'得分增益未达门槛',confidence_interval:'置信区间未证明正增益',no_large_regression:'存在较大的单局退步'}[k])).join('; ') : '';
  $('gate-description').textContent = ev ? t('冠军 {before} → 候选 {after}；增益 CI {ci}。{checks}',{before:fmt(ev.baseline),after:fmt(ev.candidate),ci:`[${fmt(ev.ci95[0])}, ${fmt(ev.ci95[1])}]`,checks:checks || t('全部门槛通过，游戏将在下一局加载新的冠军。')}) : t('超过预设得分增益、通过置信区间和退步检查后，才自动切换冠军。');
}

function renderExperiments(rows) {
  $('experiments').innerHTML = rows.length ? rows.map(e=>`<tr><td>R${esc(e.round)}<span class="status-tag ${e.passed ? '' : 'rejected'}">${t(e.passed ? '晋升' : '未晋升')}</span></td><td>${fmt(e.baseline)}</td><td>${fmt(e.candidate)}</td><td class="${e.delta >= 0 ? 'positive' : 'negative'}">${e.delta > 0 ? '+' : ''}${fmt(e.delta)}</td><td>[${fmt(e.ci95[0])}, ${fmt(e.ci95[1])}]</td><td title="${esc(e.checkpoint)}">${esc((e.checkpoint || '').split('/').slice(-3).join('/'))}</td></tr>`).join('') : `<tr><td colspan="6" class="empty-table">${t('每一个成功或失败的实验，都在这里留下记录。')}</td></tr>`;
}

function renderRoundComparisons(rows, job) {
  const key = JSON.stringify([currentLanguage,job?.id,rows]);
  if (key === lastComparisons) return;
  lastComparisons = key;
  const results = rows.filter(e=>Number.isFinite(e.baseline) && Number.isFinite(e.candidate)).sort((a,b)=>a.round-b.round);
  $('comparison-context').textContent = job ? t('{game} · {rounds} 轮已评估',{game:job.game || job.config.game,rounds:results.length}) : t('完成每轮评估后更新');
  if (!results.length) {
    $('round-comparisons').innerHTML = `<p class="comparison-empty">${t('每轮评估完成后，显示冠军与候选的得分对比、配对增益和逐局成绩。')}</p>`;
    return;
  }
  const scale = Math.max(1,...results.flatMap(e=>[Math.abs(e.baseline),Math.abs(e.candidate)]));
  $('round-comparisons').innerHTML = results.map(e=>{
    const delta = Number.isFinite(e.delta) ? e.delta : e.candidate-e.baseline;
    const relative = Number.isFinite(e.relative_gain) ? e.relative_gain : delta/Math.max(Math.abs(e.baseline),1);
    const sign = delta > 0 ? '+' : '';
    const ci = Array.isArray(e.ci95) ? `[${fmt(e.ci95[0])}, ${fmt(e.ci95[1])}]` : '—';
    const before = Array.isArray(e.baseline_scores) ? e.baseline_scores : [];
    const after = Array.isArray(e.candidate_scores) ? e.candidate_scores : [];
    const pairs = Math.min(before.length,after.length);
    const paired = pairs ? `<details class="comparison-details"><summary>${t('逐局配对得分 · {pairs} 局',{pairs})}</summary><div class="table-scroll"><table><thead><tr><th>${t('Seed / 局')}</th><th>${t('冠军')}</th><th>${t('候选')}</th><th>${t('增益')}</th></tr></thead><tbody>${before.slice(0,pairs).map((score,i)=>`<tr><td>${esc(e.seeds?.[i] ?? i+1)}</td><td>${fmt(score)}</td><td>${fmt(after[i])}</td><td class="${after[i]-score >= 0 ? 'positive' : 'negative'}">${after[i]-score > 0 ? '+' : ''}${fmt(after[i]-score)}</td></tr>`).join('')}</tbody></table></div></details>` : '';
    return `<article class="round-comparison" aria-label="${esc(t('第 {round} 轮成绩对比',{round:e.round}))}"><div class="comparison-title"><strong>${esc(t('第 {round} 轮',{round:e.round}))}</strong><span class="status-tag ${e.passed ? '' : 'rejected'}">${t(e.passed ? '已晋升' : '未晋升')}</span></div><div class="comparison-score"><span>${t('冠军平均分')}</span><div class="comparison-track"><div style="width:${Math.abs(e.baseline)/scale*100}%"></div></div><strong>${fmt(e.baseline)}</strong></div><div class="comparison-score candidate"><span>${t('候选平均分')}</span><div class="comparison-track"><div style="width:${Math.abs(e.candidate)/scale*100}%"></div></div><strong>${fmt(e.candidate)}</strong></div><div class="comparison-gain ${delta >= 0 ? 'positive' : 'negative'}">${t('配对增益 {delta} · {percent}%',{delta:sign+fmt(delta),percent:sign+fmt(relative*100,1)})}</div><p class="comparison-ci">${t('增益 95% CI {ci}',{ci})}</p>${paired}</article>`;
  }).join('');
}

function renderChart(metrics, evaluations, job) {
  const key = tab+JSON.stringify([metrics,evaluations]);
  if (key === lastChart) return;
  lastChart = key;
  const datasets = tab === 'loss' ? [
    {name:t('Train'),color:'#92e7c6',points:metrics.filter(e => Number.isFinite(e.loss)).map(e => ({x:(e.round-1)*(job?.config.train_steps || 80)+e.step,y:e.loss,label:t('R{round} step {step}',{round:e.round,step:e.step})}))},
    {name:t('Validation'),color:'#eeb875',points:metrics.filter(e => Number.isFinite(e.val_loss)).map(e => ({x:(e.round-1)*(job?.config.train_steps || 80)+e.step,y:e.val_loss,label:t('R{round} step {step}',{round:e.round,step:e.step})}))}
  ] : [
    {name:t('Champion'),color:'#748d8a',points:evaluations.map(e => ({x:e.round,y:e.baseline,label:t('R{round} 冠军',{round:e.round})}))},
    {name:t('Candidate'),color:'#92e7c6',points:evaluations.map(e => ({x:e.round,y:e.candidate,label:t('R{round} 候选',{round:e.round})}))}
  ];
  $('chart-legend').innerHTML = tab === 'loss' ? `● ${t('Train')}　<span>● ${t('Validation')}</span>` : `<span style="color:#748d8a">● ${t('Champion')}</span>　● ${t('Candidate')}`;
  const all = datasets.flatMap(d => d.points);
  $('empty-chart').hidden = all.length > 0;
  $('chart').setAttribute('aria-label', t(tab === 'loss' ? '真实训练和验证损失曲线' : '各轮冠军与候选模型的真实评估得分'));
  const maxX = Math.max(1,...all.map(p => p.x));
  const minY = Math.min(...all.map(p => p.y),0);
  const maxY = Math.max(1,...all.map(p => p.y))*1.12;
  const X = x => 44+x/maxX*495, Y = y => 175-(y-minY)/(maxY-minY)*147;
  let svg = '';
  for (let i=0;i<4;i++) {
    const y = minY+(maxY-minY)*i/3;
    svg += `<line x1="44" y1="${Y(y)}" x2="539" y2="${Y(y)}" stroke="#2a3638" stroke-dasharray="3 5"/><text x="34" y="${Y(y)+3}" fill="#647c82" text-anchor="end" font-size="9" font-family="monospace">${fmt(y,tab === 'loss' ? 2 : 0)}</text>`;
  }
  for (let i=0;i<=4;i++) svg += `<text x="${X(maxX*i/4)}" y="198" fill="#647c82" text-anchor="middle" font-size="9" font-family="monospace">${fmt(maxX*i/4,0)}</text>`;
  plotPoints = [];
  for (const d of datasets) {
    if (!d.points.length) continue;
    svg += `<polyline points="${d.points.map(p=>`${X(p.x)},${Y(p.y)}`).join(' ')}" fill="none" stroke="${d.color}" stroke-width="1.8"/>`;
    for (const p of d.points) {
      plotPoints.push({...p,name:d.name,sx:X(p.x),sy:Y(p.y)});
      if (d.points.length<30) svg += `<circle cx="${X(p.x)}" cy="${Y(p.y)}" r="2.5" fill="${d.color}"/>`;
    }
  }
  $('chart').innerHTML = svg;
}

function renderLog(details) {
  const events = details?.events || [];
  const key = JSON.stringify(events.slice(-40));
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
$('history-select').addEventListener('change', async () => {
  try {archived = $('history-select').value === 'latest' ? null : await api('/api/jobs/'+$('history-select').value); lastChart=''; lastLog=''; if(latest) render(latest);} catch(error){toast(error.message,true);}
});
for (const [id, value] of [['loss-tab','loss'],['score-tab','score']]) $(id).addEventListener('click', () => {
  tab=value; $('loss-tab').classList.toggle('selected',tab==='loss'); $('score-tab').classList.toggle('selected',tab==='score');
  $('loss-tab').setAttribute('aria-selected',String(tab==='loss')); $('score-tab').setAttribute('aria-selected',String(tab==='score'));
  if(latest) render(latest);
});
$('log-toggle').addEventListener('click', () => {$('raw-log').hidden = !$('raw-log').hidden; $('log-toggle').textContent = t($('raw-log').hidden ? '查看原始日志 ↗' : '收起原始日志 ↑');});
$('chart').addEventListener('pointermove', e => {
  if(!plotPoints.length) return;
  const box=$('chart').getBoundingClientRect(); const x=(e.clientX-box.left)*560/box.width;
  const p=plotPoints.reduce((best,v) => Math.abs(v.sx-x)<Math.abs(best.sx-x)?v:best, plotPoints[0]);
  const tooltip=$('chart-tooltip'); tooltip.hidden=false; tooltip.textContent=`${p.label} · ${p.name}: ${fmt(p.y,4)}`;
  tooltip.style.left=Math.min(box.width-195,Math.max(0,e.clientX-box.left+10))+'px'; tooltip.style.top='12px';
});
$('chart').addEventListener('pointerleave', () => {$('chart-tooltip').hidden=true;});
setInterval(() => {$('clock').textContent=new Date().toLocaleTimeString(currentLanguage);},1000);
try {
  const saved=JSON.parse(localStorage.getItem('game-rsi-config') || 'null');
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
