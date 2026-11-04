"""用 Playwright 抓 Kaggle 比賽公開頁面（不需登入）的文字，存到 kaggle_pages/。"""
from pathlib import Path
from playwright.sync_api import sync_playwright

COMP = "https://www.kaggle.com/competitions/filament-segmentation-2026"
TABS = {"overview": "/overview", "data": "/data", "rules": "/rules"}
OUT = Path(__file__).parent / "kaggle_pages"
OUT.mkdir(exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1400, "height": 1000})
    for name, suffix in TABS.items():
        url = COMP + suffix
        page.goto(url, wait_until="networkidle", timeout=90_000)
        page.wait_for_timeout(3000)
        text = page.locator("main").inner_text() if page.locator("main").count() else page.inner_text("body")
        (OUT / f"{name}.txt").write_text(f"URL: {url}\n\n{text}", encoding="utf-8")
        page.screenshot(path=str(OUT / f"{name}.png"), full_page=True)
        print(f"{name}: {len(text)} chars")
    browser.close()
