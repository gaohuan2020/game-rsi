"""Browser QA against a running local server; screenshots stay in artifacts/qa."""
import asyncio
import json
from pathlib import Path

from playwright.async_api import async_playwright


async def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "artifacts" / "qa"
    output.mkdir(parents=True, exist_ok=True)
    errors = []
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True, args=["--no-sandbox"])
        page = await browser.new_page(viewport={"width": 1512, "height": 1100}, device_scale_factor=1)
        page.on("pageerror", lambda error: errors.append(str(error)))
        await page.goto("http://127.0.0.1:8092", wait_until="domcontentloaded")
        await page.wait_for_function("document.getElementById('connection-label').textContent === '本地服务已连接'")
        await page.screenshot(path=str(output / "desktop.png"), full_page=True)
        await page.locator("#score-tab").click()
        assert await page.locator("#score-tab").get_attribute("aria-selected") == "true"
        await page.locator("#loss-tab").click()
        await page.locator("#log-toggle").click()
        assert await page.locator("#raw-log").is_visible()
        await page.set_viewport_size({"width": 390, "height": 844})
        await page.screenshot(path=str(output / "mobile.png"), full_page=True)
        overflow = await page.evaluate("document.documentElement.scrollWidth > innerWidth")
        await browser.close()
    report = {"page_errors": errors, "mobile_horizontal_overflow": overflow,
              "desktop": str(output / "desktop.png"), "mobile": str(output / "mobile.png")}
    (output / "browser-report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    assert not errors and not overflow


if __name__ == "__main__":
    asyncio.run(main())
