import os


def main():
    import re
    from playwright.sync_api import Playwright, sync_playwright, expect

    def run(playwright: Playwright) -> None:
        browser = playwright.chromium.launch(headless=False)
        context = browser.new_context(color_scheme="dark", storage_state="auth.json")
        page = context.new_page()
        page.goto("https://wochirou.com/")
        page.get_by_role("button").nth(3).click()
        page.get_by_role("link", name="首页").click()
        page.get_by_role("link", name="首页").click()
        page.get_by_role("link", name="聊天").click()
        page.get_by_role("link", name="价格").click()
        page.get_by_role("button").filter(has_text=re.compile(r"^$")).nth(3).click()
        page.get_by_role("button", name="设置").click()
        page.get_by_role("button", name="仪表盘").click()
        page.locator("div:nth-child(25) > div > .MuiChip-root > .MuiChip-label > .MuiStack-root > .MuiTypography-root").click()
        page.get_by_text("20未签到").click()

        # ---------------------
        context.storage_state(path="auth.json")
        context.close()
        browser.close()

    with sync_playwright() as playwright:
        run(playwright)

    pass

if __name__ == '__main__':
    """
    https://wochirou.com
    https://demo.voapi.top/
    https://goapi.gptnb.ai/
    https://api.aigc369.com/
    https://api.mjdjourney.cn/
    """
    # main()
    os.system("playwright codegen --load-storage=auth.json --color-scheme=dark https://wochirou.com --save-storage=auth.json")
