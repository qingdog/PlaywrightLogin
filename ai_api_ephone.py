import os
import re

from dotenv import load_dotenv
from playwright.sync_api import Playwright, sync_playwright, expect

from find_chrome_util import find_chrome_util


def run(playwright: Playwright) -> None:
    browser = playwright.chromium.launch(headless=True, executable_path=find_chrome_util())
    context = browser.new_context(color_scheme="dark", storage_state="auth.json")
    context.set_default_timeout(10000)  # 设置默认10s
    page = context.new_page()
    page.goto("https://api.ephone.chat/login")

    load_dotenv()
    page.get_by_role("textbox", name="用户名/邮箱").click()
    page.get_by_role("textbox", name="用户名/邮箱").fill(os.getenv("api_ephone_name"))
    page.get_by_role("textbox", name="用户名/邮箱").press("Tab")
    page.get_by_role("textbox", name="密码").fill(os.getenv("api_ephone_pass"))
    page.get_by_role("textbox", name="密码").press("Enter")
    page.get_by_role("button", name="登录").click()

    # page.get_by_role("button", name="今日不再提醒").click()
    if page.get_by_role("button", name="确定").is_visible():
        page.get_by_role("button", name="确定").click()
    page.get_by_role("link", name="工作台").click()
    page.get_by_role("link", name=" 工作台").click()
    page.get_by_text("签到日历").click()
    print(page.get_by_text("👋 你好，17597658361759765836 7694当前余额").text_content())
    page.get_by_role("button", name=" 去签到").click()
    page.get_by_text("签到成功").click()
    page.get_by_text("签到成功").text_content()
    expect(page.get_by_label("success type")).to_contain_text("签到成功")

    # ---------------------
    context.storage_state(path="auth.json")
    context.close()
    browser.close()


def main():
    if not os.path.exists("auth.json"):
        with open("auth.json", "w", encoding="utf-8") as file: file.write("{}")  # 写入空 JSON 对象
    with sync_playwright() as playwright: run(playwright)


if __name__ == '__main__':
    main()
    # os.system("playwright codegen --load-storage=auth.json --color-scheme=dark https://api.ephone.chat/ --save-storage=auth.json")
