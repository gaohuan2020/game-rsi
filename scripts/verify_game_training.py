"""Check game-scoped training views and late history responses without GPU mutations."""
import asyncio
import copy
import json
from pathlib import Path

from playwright.async_api import async_playwright

from game_rsi.config import RunConfig


def experiment(task_id, game, loss, gain):
    cfg = RunConfig(game=game).model_dump()
    job = {"id": task_id, "kind": "evolve", "game": game, "config": cfg,
           "status": "completed", "phase": "completed", "round": 1,
           "collected": 512, "updated_at": 1, "progress": 1,
           "created_at": 1 if task_id.endswith("older") else 2}
    metrics = {"kind": "metric", "time": 1, "round": 1, "step": cfg["train_steps"],
               "loss": loss, "val_loss": loss, "kl": .01, "target_entropy": .3,
               "agreement": .9, "gpu_gb": 3}
    evaluation = {"kind": "evaluation", "time": 2, "round": 1,
                  "baseline": 10, "candidate": 10 + gain, "delta": gain,
                  "relative_gain": gain / 10, "ci95": [gain - 1, gain + 1],
                  "passed": gain > 1, "pairs": 2, "checks": {},
                  "baseline_scores": [10, 10], "candidate_scores": [10 + gain, 10 + gain]}
    return {"job": job, "events": [metrics, evaluation], "frame": None, "logs": task_id}


async def main():
    mario = experiment("mario-latest", "mario", 1.6, 3)
    snake = experiment("snake-latest", "snake", .325, 5)
    older = experiment("snake-older", "snake", .77, 2)
    details = {d["job"]["id"]: d for d in [mario, snake, older]}
    state = {"jobs": [d["job"] for d in [mario, snake, older]], "evolution": mario,
             "play": {"job": {"id": "play", "kind": "play", "status": "running",
                              "phase": "playing", "config": RunConfig(game="mario").model_dump()},
                      "events": [], "frame": None}}
    late_response = asyncio.Event()
    delay_history = False
    requests, mutations, errors = [], [], []

    async def route_api(route):
        nonlocal delay_history
        request = route.request
        path = request.url.split("8092", 1)[1]
        if request.method == "POST":
            mutations.append(path)
            assert path == "/api/play"
            state["play"]["job"]["config"] = request.post_data_json
            await route.fulfill(json={"id": "play", "switched": True})
        elif path == "/api/state":
            await route.fulfill(json=state)
        elif path.startswith("/api/games/") and path.endswith("/comparisons"):
            game=path.split("/")[-2]
            runs=[{"job":d["job"],"evaluations":[e for e in d["events"] if e["kind"]=="evaluation"]} for d in details.values() if d["job"]["game"]==game]
            await route.fulfill(json={"game":game,"runs":runs})
        elif path == "/api/events":
            await route.fulfill(content_type="text/event-stream",
                                body="event: snapshot\ndata: " + json.dumps(state) + "\n\n")
        elif path.startswith("/api/jobs/"):
            task_id = path.rsplit("/", 1)[-1]
            requests.append(task_id)
            response = copy.deepcopy(details[task_id])
            if delay_history and task_id == "snake-latest":
                await late_response.wait()
            await route.fulfill(json=response)
        else:
            await route.continue_()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])
        page = await browser.new_page(viewport={"width": 1512, "height": 1100})
        page.on("pageerror", lambda error: errors.append(str(error)))
        await page.route("**/api/**", route_api)
        await page.goto("http://127.0.0.1:8092", wait_until="domcontentloaded")
        await page.wait_for_function("document.getElementById('loss-metric').textContent === '1.6000'")
        await page.evaluate("stream.close()")
        await page.select_option("#game", "snake")
        await page.wait_for_function("document.getElementById('loss-metric').textContent === '0.3250'")
        await page.wait_for_function("document.querySelectorAll('.round-comparison[data-game=snake]').length===2")
        assert await page.locator(".round-comparison[data-game=mario]").count() == 0
        assert await page.locator("#history-select option").count() == 3
        assert "mario" not in await page.locator("#history-select").inner_text()
        assert await page.locator("#chart circle").count() == 2
        await page.select_option("#history-select", "snake-older")
        await page.wait_for_function("document.getElementById('loss-metric').textContent === '0.7700'")
        await page.select_option("#game", "sokoban")
        for field in ["loss-metric", "gain-metric", "sample-count", "agreement", "gpu-memory"]:
            assert await page.locator("#" + field).inner_text() == "—", field
        assert await page.locator("#chart polyline").count() == 0
        assert await page.locator(".round-comparison").count() == 0
        assert await page.locator("#history-select option").count() == 1
        assert await page.locator("#history-select").input_value() == "latest"
        assert await page.locator("#phase-progress").inner_text() == "0%"
        assert await page.locator("#gate-status").inner_text() == "等待评估"
        assert "暂无训练记录" in await page.locator("#phase-note").inner_text()
        assert "mario-latest" not in await page.locator("#raw-log").inner_text()
        assert "snake-older" not in await page.locator("#raw-log").inner_text()
        await page.select_option("#game", "snake")
        await page.wait_for_function("document.getElementById('loss-metric').textContent === '0.3250'")
        assert requests.count("snake-latest") == 1, "Completed history should be cached"
        # A new snapshot changes the selected game's latest experiment. Hold its response
        # while switching away, then ensure it cannot restore that game's stale metrics.
        delay_history = True
        snake["job"]["updated_at"] = 2
        snake["events"][0]["loss"] = .25
        await page.evaluate("snapshot=>render(snapshot)", state)
        await page.wait_for_function("trainingRequests.has('snake-latest')")
        await page.select_option("#game", "mario")
        late_response.set()
        await page.wait_for_function("!trainingRequests.has('snake-latest')")
        assert await page.locator("#loss-metric").inner_text() == "1.6000"
        assert await page.locator(".round-comparison[data-game=mario]").count() == 1
        assert await page.locator(".round-comparison[data-game=snake]").count() == 0
        await page.select_option("#game", "snake")
        await page.wait_for_function("document.getElementById('loss-metric').textContent === '0.2500'")
        await page.select_option("#language-select", "en")
        await page.select_option("#game", "sokoban")
        assert "No training history" in await page.locator("#phase-note").inner_text()
        assert not await page.evaluate("/[\\u4e00-\\u9fff]/.test(document.body.innerText.replace('中文',''))")
        # Append another experiment without replacing the game's prior scored rounds.
        third=experiment("snake-third","snake",.21,7)
        third["job"].update(created_at=3,status="running",phase="collecting",round=0)
        saved_events=third["events"]
        third["events"]=[]
        details["snake-third"]=third
        state["jobs"].insert(0,third["job"])
        state["evolution"]=third
        state["play"]=None
        await page.evaluate("snapshot=>{document.getElementById('game').value='snake';render(snapshot)}",state)
        await page.wait_for_function("document.querySelectorAll('.round-comparison').length===3")
        assert await page.locator('.round-comparison[data-run-id="snake-older"]').count()==1
        pending=page.locator('.round-comparison[data-run-id="snake-third"]')
        assert await pending.locator('.comparison-score').first.inner_text()=="—"
        third["job"].update(status="completed",phase="completed",round=2)
        third["job"]["config"]["rounds"]=2
        second=copy.deepcopy(saved_events[-1])
        second.update(round=2,time=3,baseline=17,candidate=20,delta=3,relative_gain=3/17,
                      ci95=[2,4],baseline_scores=[17,17],candidate_scores=[20,20])
        third["events"]=saved_events+[second]
        third["job"]["last_evaluation"]=second
        for _ in range(3):
            await page.evaluate("snapshot=>render(snapshot)",state)
        await page.wait_for_function("document.querySelectorAll('.round-comparison').length===4")
        assert await page.locator('.round-comparison[data-run-id="snake-third"]').count()==2
        assert await page.locator('.comparison-round').all_inner_texts()==["R1","R2","R3","R4"]
        assert "Run 3 · R2" in await page.locator('.round-comparison').last.inner_text()
        assert "20.00" in await page.locator('.round-comparison').last.inner_text()
        await page.locator('.round-details-toggle').first.click()
        assert "snake-older" in await page.locator('.comparison-details').first.inner_text()
        # Removed sidebar controls cannot leave an old persisted hidden state behind.
        await page.evaluate("localStorage.setItem('game-rsi-sidebar-hidden','true')")
        await page.reload(wait_until="domcontentloaded")
        await page.wait_for_function("latest!==null")
        await page.select_option("#game","snake")
        await page.wait_for_function("document.querySelectorAll('.round-comparison').length===4")
        assert await page.locator('#sidebar').is_visible()
        assert await page.locator('#sidebar-toggle').count()==0
        assert await page.locator('.round-comparison[data-game=mario]').count()==0
        await browser.close()
    report = {"page_errors": errors, "game_history_restored": True, "empty_game_cleared": True,
              "history_filtered": True, "late_response_ignored": True, "cached_history_updated": True,
              "cumulative_history": True, "appended_rounds": 4, "history_after_reload": True, "sidebar_fixed": True}
    output = Path(__file__).resolve().parents[1] / "artifacts/qa/game-training.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    assert not errors


if __name__ == "__main__":
    asyncio.run(main())
