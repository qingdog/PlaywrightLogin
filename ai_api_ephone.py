import logging
import os
import platform

from dotenv import load_dotenv
from playwright.sync_api import Playwright, sync_playwright, expect, TimeoutError

from find_chrome_util import find_chrome_util


def run(playwright: Playwright) -> None:
    browser = playwright.chromium.launch(headless=platform.system() != "Windows", executable_path=find_chrome_util())
    # context = browser.new_context(color_scheme="dark", storage_state="auth.json")
    context = browser.new_context(color_scheme="dark")
    context.set_default_timeout(10000)  # 设置默认10s
    page = context.new_page()
    page.goto("https://api.ephone.chat/login")
    # 等待网络请求闲置至少 500ms，即没有新的请求发送或进行中。等待 JavaScript 运行完毕、数据加载完成
    try: page.wait_for_load_state(state="networkidle", timeout=5000)  # 5s后超时
    except Exception as e: logging.error(e, exc_info=True)

    # page.get_by_role("button", name="今日不再提醒").click()
    # page.get_by_role("button", name="确定").click()
    if page.get_by_role("button", name="确定").is_visible():
        page.get_by_role("button", name="确定").click()

    load_dotenv()
    page.get_by_role("textbox", name="用户名/邮箱").click()
    page.get_by_role("textbox", name="用户名/邮箱").fill(os.getenv("api_ephone_name"))
    page.get_by_role("textbox", name="用户名/邮箱").press("Tab")
    page.get_by_role("textbox", name="密码").fill(os.getenv("api_ephone_pass"))
    page.get_by_role("button", name="登录").click()

    if page.get_by_role("button", name="确定").is_visible():
        page.get_by_role("button", name="确定").click()

    page.get_by_role("link", name="工作台").click()
    page.get_by_role("link", name=" 工作台").click()
    page.get_by_text("签到日历").click()
    logging.getLogger().setLevel(logging.INFO)
    logging.info(page.get_by_text("👋 你好，17597658361759765836 7694当前余额").text_content())
    
    
    check_button = False
    try:
        try:
            page.get_by_role("button", name=" 去签到").click()
        except TimeoutError as e:
            check_button = True
            raise e
        try: page.wait_for_load_state(state="load", timeout=1000)  # 最长只等待1s，不管是否load完成，就进行下一步
        except Exception as e: logging.error(e, exc_info=True)

        alert_success_locator = page.locator('div[role="alert"][aria-label="success type"]')
        # 断言为签到成功
        expect(alert_success_locator.last).to_contain_text("签到成功")

        # 疑似在断言时间里，产生了等待1s的行为（导致前面打印失败了，但断言成功了）。这里进行重新打印
        logging.info(f"all_outer_text: {alert_success_locator.evaluate_all("elements => elements.map(e => e.outerText)")}")
    except Exception as e:
        if check_button:
            logging.warning("未找到签到按钮，疑似已经签到······")
            logging.warning(e, exc_info=True)
        else:
            logging.error(e, exc_info=True)
    finally:
        try: page.wait_for_load_state(state="networkidle", timeout=1000)  # 1s后超时
        except Exception as e: logging.error(e, exc_info=True)
        logging.info(page.get_by_text("👋 你好，17597658361759765836 7694当前余额").text_content())

    # ---------------------
    # context.storage_state(path="auth.json") # 不保存状态
    context.close()
    browser.close()


def main():
    if not os.path.exists("auth.json"):
        with open("auth.json", "w", encoding="utf-8") as file: file.write("{}")  # 写入空 JSON 对象
    with sync_playwright() as playwright: run(playwright)


if __name__ == '__main__':
    main()
    # os.system("playwright codegen --load-storage=auth.json --color-scheme=dark https://api.ephone.chat/ --save-storage=auth.json")
    # os.system("playwright codegen --color-scheme=dark https://api.ephone.chat/ ")
