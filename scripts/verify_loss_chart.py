"""Verify loss chart coordinates, validation hover, and round boundaries using logged fixtures."""
import asyncio
import json
from pathlib import Path

from playwright.async_api import async_playwright

from game_rsi.config import RunConfig


async def main():
    cfg = RunConfig(train_steps=70, rounds=2).model_dump()
    metrics = []
    for round_index in (1, 2):
        for step in range(71):
            row = {"kind": "metric", "time": round_index * 100 + step,
                   "round": round_index, "step": step}
            if step:
                row.update(loss=.7 + (step % 13) / 10, gpu_gb=3)
            if step in (0, 17, 34, 51, 68, 70):
                row.update(val_loss=1.324 - .008 * (round_index - 1) - .01 * step / 70,
                           agreement=.6)
            metrics.append(row)
    job = {"id": "loss-fixture", "kind": "evolve", "game": "snake", "config": cfg,
           "status": "completed", "phase": "completed", "round": 2}
    state = {"jobs": [job], "play": None,
             "evolution": {"job": job, "events": metrics, "frame": None}}
    errors, mutations = [], []

    async def route_api(route):
        if route.request.method == "POST":
            mutations.append(route.request.url)
            await route.fulfill(status=400, json={"detail": "No mutations allowed"})
        elif "/api/games/" in route.request.url and route.request.url.endswith("/comparisons"):
            await route.fulfill(json={"game":"snake","runs":[{"job":job,"evaluations":[]}]})
        elif route.request.url.endswith("/api/events"):
            await route.fulfill(content_type="text/event-stream",
                                body="event: snapshot\ndata: " + json.dumps(state) + "\n\n")
        elif route.request.url.endswith("/api/state"):
            await route.fulfill(json=state)
        else:
            await route.continue_()

    output = Path(__file__).resolve().parents[1] / "artifacts/qa"
    output.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])
        page = await browser.new_page(viewport={"width": 1512, "height": 1100})
        page.on("pageerror", lambda error: errors.append(str(error)))
        await page.route("**/api/**", route_api)
        await page.goto("http://127.0.0.1:8092", wait_until="domcontentloaded")
        await page.locator("#loss-tab").click()
        await page.wait_for_selector("#chart polyline")
        await page.evaluate("stream.close()")
        expected = [{"x": (e["round"] - 1) * 70 + e["step"], "y": e["val_loss"],
                     "round": e["round"], "step": e["step"]} for e in metrics if "val_loss" in e]
        actual = await page.evaluate("plotPoints.filter(p=>p.series==='validation').map(({x,y,round,step})=>({x,y,round,step}))")
        assert actual == expected, "Chart observations must exactly match recorded steps and values"
        assert await page.locator("#chart polyline").count() == 4, "Separate lines per round and series"
        assert await page.locator("#chart circle[data-series=validation]").count() == 12
        assert await page.evaluate("chartSegments.every(s=>new Set(s.points.map(p=>p.round)).size===1)")
        assert actual[5]["x"] == actual[6]["x"] == 70
        assert actual[-1]["x"] == 140
        screen = await page.evaluate("""() => {
          const p=plotPoints.find(p=>p.series==='validation'&&p.round===1&&p.step===34);
          const point=document.getElementById('chart').createSVGPoint(); point.x=p.sx; point.y=p.sy;
          const screen=point.matrixTransform(document.getElementById('chart').getScreenCTM());
          return {x:screen.x,y:screen.y,value:p.y.toFixed(6)};
        }""")
        await page.mouse.move(screen["x"], screen["y"])
        await page.wait_for_function("document.getElementById('chart-tooltip').textContent.includes('验证')")
        assert screen["value"] in await page.locator("#chart-tooltip").inner_text()
        # Sparse validation lines select an actual logged point even between observations.
        assert await page.evaluate("""() => {
          const segment=chartSegments.find(s=>s.points[0].series==='validation');
          const [a,b]=segment.points; const p=nearestChartPoint((a.sx+b.sx)/2,(a.sy+b.sy)/2,null);
          return p.series==='validation'&&[0,17].includes(p.step);
        }""")
        before_span = await page.evaluate("Math.abs(plotPoints.find(p=>p.series==='validation'&&p.round===1&&p.step===0).sy-plotPoints.find(p=>p.series==='validation'&&p.round===1&&p.step===70).sy)")
        await page.locator('[data-loss-series="validation"]').click()
        assert await page.locator("#chart polyline").count() == 2
        assert await page.evaluate("plotPoints.every(p=>p.series==='validation')")
        after_span = await page.evaluate("Math.abs(plotPoints.find(p=>p.series==='validation'&&p.round===1&&p.step===0).sy-plotPoints.find(p=>p.series==='validation'&&p.round===1&&p.step===70).sy)")
        assert after_span > 50 and after_span > before_span * 10
        assert "1.30600" in await page.locator("#validation-summary").inner_text()
        await page.locator(".train-panel").screenshot(path=str(output / "validation-chart-fixture.png"))
        await page.locator("#score-tab").click()
        assert not await page.locator("#loss-chart-controls").is_visible()
        await page.locator("#loss-tab").click()
        assert await page.locator('[data-loss-series="validation"]').get_attribute("aria-pressed") == "true"
        await page.select_option("#language-select", "en")
        assert "Validation" in await page.locator("#chart-legend").inner_text()
        assert "Step 70" in await page.locator("#validation-summary").inner_text()
        for width in (320, 390, 1512):
            await page.set_viewport_size({"width": width, "height": 1100})
            assert not await page.evaluate("document.documentElement.scrollWidth>innerWidth"), width
        await page.evaluate("renderChart([],[],null)")
        assert await page.locator("#chart").inner_html() == ""
        assert await page.locator("#empty-chart").is_visible()
        await browser.close()
    report = {"page_errors": errors, "recorded_validation_points": len(expected),
              "configured_steps": 70, "two_round_final_step": 140, "validation_hover": True,
              "no_cross_round_lines": True, "validation_zoom": True,
              "mutations": mutations, "mobile_overflow": False}
    (output / "loss-chart.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    assert not errors and not mutations


if __name__ == "__main__":
    asyncio.run(main())
