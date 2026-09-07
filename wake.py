"""Wake the Streamlit demo and verify it actually renders.

Streamlit Community Cloud's sleep page answers HTTP 200, so a plain request
reports success while the app is down. This drives a real browser instead: it
clicks the wake button when one is present and waits for the app container to
render. Holding the page open afterwards keeps the websocket session alive,
which is what Streamlit counts as activity - an HTTP request alone does not
reset the sleep timer.
"""

import sys

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

APP_URL = "https://banking-support-agent.streamlit.app"

WAKE_BUTTON = "Yes, get this app back up!"
APP_CONTAINER = '[data-testid="stAppViewContainer"]'

LOAD_TIMEOUT_MS = 60_000
RENDER_TIMEOUT_MS = 180_000
HOLD_SECONDS = 15


def main() -> int:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            page.goto(APP_URL, wait_until="domcontentloaded", timeout=LOAD_TIMEOUT_MS)

            wake = page.get_by_role("button", name=WAKE_BUTTON)
            try:
                wake.wait_for(state="visible", timeout=10_000)
                print("App was asleep - clicking the wake button.")
                wake.click()
            except PlaywrightTimeoutError:
                pass  # already awake

            try:
                page.wait_for_selector(APP_CONTAINER, timeout=RENDER_TIMEOUT_MS)
            except PlaywrightTimeoutError:
                page.screenshot(path="failure.png", full_page=True)
                print(
                    f"App did not render within {RENDER_TIMEOUT_MS // 1000}s - "
                    "see failure.png."
                )
                return 1

            # Streamlit treats an open websocket session as traffic, so linger
            # here rather than exiting the moment the container appears.
            page.wait_for_timeout(HOLD_SECONDS * 1000)
            page.screenshot(path="healthy.png", full_page=True)
            print("App is UP")
            return 0
        finally:
            browser.close()


if __name__ == "__main__":
    sys.exit(main())
