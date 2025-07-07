import logging
import os
import platform

from dotenv import load_dotenv
from playwright.sync_api import Playwright, sync_playwright, expect, TimeoutError

from find_chrome_util import find_chrome_util


def run(playwright: Playwright) -> None:
    browser = playwright.chromium.launch(headless=platform.system() != "Windows", executable_path=find_chrome_util())
    #context = browser.new_context(color_scheme="dark", storage_state="auth.json")
    context = browser.new_context(color_scheme="dark", viewport={"width": 1020, "height": 880})
    context.set_default_timeout(30000)  # 设置默认10s
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
    
    # 2025修复校验
    import slide_validate
    js = """
        const elements = document.querySelectorAll('span'); // 查找所有元素
        elements.forEach(element => {
          if (element.textContent.trim() === "请完成人机验证后继续") {
            element.click(); // 点击该元素
          }
        });"""
    slide_validate.validate(page,page_url=None, page_evaluate=js, background_css="img.gocaptcha-module_picture__LRwbY", slider_css="div.index-module_tile__8pkQD img", background_size=(300, 220), slider_down_css_xpath="div.gocaptcha-module_dragBlock__bFlwx", distance_correction=-11)
    
    page.get_by_role("button", name="登录").click()
    

    if page.get_by_role("button", name="确定").is_visible():
        page.get_by_role("button", name="确定").click()

    page.get_by_role("link", name="工作台").click()
    page.get_by_role("link", name=" 工作台").click()
    page.get_by_text("签到日历").click()
    logging.getLogger().setLevel(logging.INFO)
    
    logging.info(page.query_selector('div.semi-spin-children div.mb-4').text_content())
    
    
    check_button = False
    try:
        #page.set_viewport_size({'width': 1920, 'height': 1040})  # 设置合适的视口大小
        # 获取“签到日历”元素  
        calendar_element = page.get_by_text("签到日历")  
        if calendar_element:  # 确保元素存在并可见  
            bounding_box = calendar_element.bounding_box()  # 获取元素的坐标  
            if bounding_box:  # 计算中间坐标  
                mid_x = bounding_box['x'] + bounding_box['width'] // 2  
                mid_y = bounding_box['y'] + bounding_box['height'] // 2
                page.wait_for_timeout(5 * 1000)
                page.mouse.click(mid_x, mid_y)  # 使用鼠标点击中间坐标
                page.wait_for_timeout(2 * 1000)
                page.mouse.click(mid_x, mid_y)  # 使用鼠标点击中间坐标
                print('已在“签到日历”上点击。')  
            else:  
                print('无法获取元素的边界框。')  
        else:  
            print('未找到“签到日历”元素。')
        
        #page.get_by_text("签到日历").click()
        try:
            page.get_by_role("button", name=" 去签到").click()
            js = "document.title"
            slide_validate.validate(page,page_url=None, page_evaluate=js, background_css="img.gocaptcha-module_picture__LRwbY", slider_css="div.index-module_tile__8pkQD img", background_size=(300, 220), slider_down_css_xpath="div.gocaptcha-module_dragBlock__bFlwx", distance_correction=-11)
        except TimeoutError as e:
            check_button = True
            raise e
            
        # 断言为成功
        try: expect(page.locator("span.semi-toast-content-text").last).to_contain_text("验证成功")
        except Exception as e: 
            logging.error(e, exc_info=True)
        logging.info(f"validate: {page.locator('span.semi-toast-content-text').evaluate_all('elements => elements.map(e => e.outerText)')}")
        
        try: page.wait_for_load_state(state="load", timeout=1000)  # 最长只等待1s，不管是否load完成，就进行下一步
        except Exception as e: logging.error(e, exc_info=True)
        
        
        page.get_by_role("button", name=" 去签到").click()
        alert_success_locator = page.locator('div[role="alert"][aria-label="success type"]')
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
        try: 
            page.wait_for_load_state(state="networkidle", timeout=3000)  # 3s后超时
            page.reload(wait_until="networkidle" , timeout=3000) # 先等待网络空闲，再执行reload刷新页面，再等3s之内网络空闲
        except Exception as e: logging.error(e, exc_info=True)
        logging.info(page.query_selector('div.semi-spin-children div.mb-4').text_content())

    page.wait_for_timeout(20 * 1000)
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
