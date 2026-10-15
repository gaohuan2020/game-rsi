"""Verify bilingual UI, paired round scores, and game-switch requests without starting GPU tasks."""
import asyncio
import json
from pathlib import Path

from playwright.async_api import async_playwright

from game_rsi.config import RunConfig


async def main():
    root = Path(__file__).resolve().parents[1]
    cfg = RunConfig(rounds=2).model_dump()
    cfg['collect_steps'] = 512  # An existing run must not overwrite the new-task budget.
    def result(round_index, before, after, passed):
        delta = after-before
        return {'kind':'evaluation','time':round_index,'round':round_index,
                'baseline':before,'candidate':after,'delta':delta,'relative_gain':delta/max(abs(before),1),
                'ci95':[delta,delta],'passed':passed,'pairs':2,'baseline_scores':[before,before],
                'candidate_scores':[after,after],'seeds':[90001088,90001089],
                'checks':{'score_gain':passed,'confidence_interval':passed,'no_large_regression':passed},
                'checkpoint':f'checkpoints/round-{round_index}'}
    events = [result(1,1.,26.5,True),result(2,26.5,1.,False)]
    state = {'jobs':[], 'play':{'job':{'id':'play-fixture','kind':'play','status':'running','phase':'playing','config':cfg},'frame':None,'events':[]},
             'evolution':{'job':{'id':'evolution-fixture','kind':'evolve','game':'snake','status':'completed','phase':'completed','config':cfg,'round':2},'events':events,'frame':None}}
    errors, mutations = [], []
    async def route_api(route):
        request = route.request
        if request.method == 'POST':
            data = request.post_data_json
            mutations.append({'url':request.url,'data':data})
            assert request.url.endswith('/api/play'), request.url
            state['evolution']['job'].update(status='stopped',phase='stopped')
            state['play']={'job':{'id':'new-play','kind':'play','status':'running','phase':'loading','config':data},'frame':None,'events':[]}
            response={'id':'new-play','switched':True,'stopped_evolution_ids':['evolution-fixture']}
            await route.fulfill(json=response)
        elif '/api/games/' in request.url and request.url.endswith('/comparisons'):
            game=request.url.split('/')[-2]
            d=state.get('evolution')
            runs=[{'job':d['job'],'evaluations':[e for e in d['events'] if e['kind']=='evaluation']}] if d and d['job']['game']==game else []
            await route.fulfill(json={'game':game,'runs':runs})
        elif request.url.endswith('/api/events'):
            await route.fulfill(content_type='text/event-stream',body='event: snapshot\ndata: '+json.dumps(state)+'\n\n')
        elif request.url.endswith('/api/state'):
            await route.fulfill(json=state)
        else:
            await route.continue_()
    async with async_playwright() as p:
        browser=await p.chromium.launch(headless=True,args=['--no-sandbox'])
        page=await browser.new_page(viewport={'width':1512,'height':1100})
        page.on('pageerror',lambda e:errors.append(str(e)))
        await page.route('**/api/**',route_api)
        await page.goto('http://127.0.0.1:8092',wait_until='domcontentloaded')
        await page.wait_for_selector('.round-comparison')
        assert await page.locator('.round-comparison').count()==2
        assert await page.locator('#score-tab').get_attribute('aria-selected')=='true'
        assert await page.locator('#chart polyline').count()==2
        positions = {}
        for name in ('game-panel','decision-panel','experiment-panel','log-panel',
                     'train-panel','guide-panel','config-panel'):
            positions[name] = await page.locator('.'+name).bounding_box()
        assert positions['game-panel']['x'] < positions['train-panel']['x']
        assert positions['game-panel']['y'] == positions['train-panel']['y']
        for column in (('game-panel','decision-panel','experiment-panel','log-panel'),
                       ('train-panel','guide-panel','config-panel')):
            for before, after in zip(column,column[1:]):
                assert positions[before]['y']+positions[before]['height'] <= positions[after]['y']
        assert await page.evaluate("config().game")=='snake', 'Game selector must remain part of the configuration form'
        assert await page.evaluate("config().max_episode_steps")==1000
        assert await page.evaluate("config().collect_steps")==1000
        assert await page.evaluate("config().data_strategy")=='trajectory'
        assert await page.evaluate("config().promotion_rule")=='positive_gain'
        screenshot=root/'artifacts/qa/compact-layout.png'
        screenshot.parent.mkdir(parents=True,exist_ok=True)
        await page.screenshot(path=str(screenshot),full_page=True)
        assert '26.50' in await page.locator('.round-comparison').nth(0).inner_text()
        assert '+25.50' in await page.locator('.comparison-gain').nth(0).inner_text()
        assert '-25.50' in await page.locator('.comparison-gain').nth(1).inner_text()
        await page.locator('.round-details-toggle').first.click()
        assert await page.locator('.comparison-details').first.locator('tbody tr').count()==2
        assert '90001088' in await page.locator('.comparison-details').first.inner_text()
        await page.select_option('#language-select','en')
        assert await page.locator('html').get_attribute('lang')=='en'
        assert await page.locator('.experiment-panel h2').inner_text()=='Scores by evolution round'
        assert 'Before training' in await page.locator('.comparison-table > thead').inner_text()
        visible_chinese=await page.evaluate("/[\\u4e00-\\u9fff]/.test(document.body.innerText.replace('中文',''))")
        assert not visible_chinese, await page.locator('body').inner_text()
        for width in (1512,390):
            await page.set_viewport_size({'width':width,'height':1100 if width>700 else 844})
            assert not await page.evaluate('document.documentElement.scrollWidth>innerWidth'), width
        await page.reload(wait_until='domcontentloaded')
        await page.wait_for_selector('.round-comparison')
        assert await page.locator('#language-select').input_value()=='en'
        assert await page.locator('[name="collect_steps"]').input_value()=='1000'
        assert await page.locator('#run-config').is_visible()
        assert 'After training' in await page.locator('.comparison-table > thead').inner_text()
        await page.select_option('#language-select','zh-CN')
        assert await page.locator('.experiment-panel h2').inner_text()=='每轮进化成绩对比'
        assert '训练后平均分' in await page.locator('.comparison-table > thead').inner_text()
        assert not mutations, 'Changing language must not start or stop any task'
        gate = await page.evaluate("""() => {renderEvaluation({baseline:10,candidate:11,delta:1,relative_gain:.1,
          pairs:8,passed:true,ci95:[-2,4],promotion_rule:'positive_gain',required_checks:['score_gain'],
          checks:{score_gain:true,confidence_interval:false,no_large_regression:false}});
          return {status:document.getElementById('gate-status').textContent,
                  description:document.getElementById('gate-description').textContent};}""")
        assert gate['status']=='通过 / 已晋升'
        assert '单局退步仅供参考' in gate['description']
        assert '存在较大的单局退步' not in gate['description']
        state['play']=None
        state['evolution']['job'].update(status='running',phase='training')
        await page.evaluate('snapshot=>render(snapshot)',state)
        await page.select_option('#language-select','en')
        await page.select_option('#game','mario')
        await page.wait_for_function("document.getElementById('toast').textContent.includes('Stopping the previous')")
        assert len(mutations)==1 and mutations[0]['data']['game']=='mario'
        assert await page.locator('#run-state').inner_text()=='Waiting for an experiment'
        await browser.close()
    report={'page_errors':errors,'languages':['zh-CN','en'],'round_comparison':True,'paired_scores':True,
            'language_persistence':True,'mobile_overflow':False,'game_switch_without_live_worker':True,
            'two_column_layout':True,'external_game_selector':True,'default_episode_steps':1000}
    output=root/'artifacts/qa/ui-features.json';output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report));assert not errors


if __name__=='__main__':
    asyncio.run(main())
