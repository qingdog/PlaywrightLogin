import logging
import os
import platform

from dotenv import load_dotenv
from playwright.sync_api import Playwright, sync_playwright, expect, TimeoutError

from find_chrome_util import find_chrome_util


def run(playwright: Playwright) -> None:
    #browser = playwright.chromium.launch(headless=platform.system() != "Windows", executable_path=find_chrome_util(), args=["--lang=zh-CN"])
    browser = playwright.chromium.launch(headless=True, executable_path=find_chrome_util(), args=["--lang=en-US"])
    #context = browser.new_context(color_scheme="dark", storage_state="auth.json")
    context = browser.new_context(color_scheme="dark", viewport={"width": 1920, "height": 1080}) # 为了确定UI整体布局位置
    context.set_default_timeout(30000)  # 设置默认10s
    page = context.new_page()
    page.goto("https://api.ephone.chat/login")
    # 等待网络请求闲置至少 500ms，即没有新的请求发送或进行中。等待 JavaScript 运行完毕、数据加载完成
    try: page.wait_for_load_state(state="networkidle", timeout=11000)  # 5s后超时
    except Exception as e: logging.error(e, exc_info=True)
    
    try:
        page.get_by_role("button", name="Switch language").click()
        page.get_by_role("menuitem", name="简体中文").click()
    except Exception as e: 
        logging.error(e, exc_info=True)

    # page.get_by_role("button", name="今日不再提醒").click()
    # page.get_by_role("button", name="确定").click()
    if page.get_by_role("button", name="确定").is_visible():
        page.get_by_role("button", name="确定").click()

    load_dotenv()
    #2026renew
    '''page.get_by_role("textbox", name="用户名/邮箱").click()
    page.get_by_role("textbox", name="用户名/邮箱").fill(os.getenv("api_ephone_name"))
    page.get_by_role("textbox", name="用户名/邮箱").press("Tab")
    page.get_by_role("textbox", name="密码").fill(os.getenv("api_ephone_pass"))'''
    
    
    '''try: page.get_by_role("button", name="密码登录").click()
    except Exception as e: 
        logging.error(e, exc_info=True)
        
        print("\n包含关键字的完整片段：")
        print("---------------------------------------------------------------------------------------------")
        button_texts = page.locator("button").all_inner_texts()
        
        print(button_texts)
        ###
        # 匹配包含 aaa 和 bbb 的完整片段
        #pattern_with_keys = r"aaa.*?bbb"
        
        pattern_with_keys = r"加载数据.*?便捷导入"
        matches_full = re.findall(pattern_with_keys, body, flags=re.DOTALL)
        
        for m in matches_full:
            import re
            print(re.sub(f"\n(\n)+|\r\n(\r\n)+", "", m))'''
    
    page.get_by_role("textbox", name="用户名或邮箱").click()
    page.get_by_role("textbox", name="用户名或邮箱").fill(os.getenv("api_ephone_name"))
    page.get_by_role("textbox", name="密码").click()
    page.get_by_role("textbox", name="密码").fill(os.getenv("api_ephone_pass"))
    page.get_by_role("button", name="请完成验证").click()
    
    # 2025修复校验
    import slide_validate
    js = """
        /*const elements = document.querySelectorAll('span'); // 查找所有元素
        elements.forEach(element => {
          if (element.textContent.trim() === "请完成人机验证后继续") {
            element.click(); // 点击该元素
          }
        });*/
        """
    slide_validate.validate(page,page_url=None, page_evaluate=js, background_css="img.gocaptcha-module_picture__LRwbY", slider_css="div.index-module_tile__8pkQD img", background_size=(300, 220), slider_down_css_xpath="div.gocaptcha-module_dragBlock__bFlwx", distance_correction=0)#-11
    
    #page.wait_for_timeout(2*1000)
    #page.get_by_role("button", name="登录").click()
    page.get_by_role("button", name="登录", exact=True).click()
    
    page.wait_for_timeout(5 * 1000)
    
    try:
        page.get_by_role("button", name="Close dialog").click()
    except Exception as e:
        print(e)
    '''try:
        page.get_by_role("button", name="Close dialog").click()
    except Exception as e:
        print(e)'''
    
    page.wait_for_timeout(2 * 1000)
    
    #20260420page.get_by_role("tab", name="签到日历").click()


    '''if page.get_by_role("button", name="确定").is_visible():
        page.get_by_role("button", name="确定").click()

    page.get_by_role("link", name="工作台").click()
    page.get_by_role("link", name=" 工作台").click()
    page.get_by_text("签到日历").click()'''
    logging.getLogger().setLevel(logging.INFO)
    
    #logging.info(page.query_selector('div.semi-spin-children div.mb-4').text_content())
    
    
    check_button = False
    try:
        #page.set_viewport_size({'width': 1920, 'height': 1040})  # 设置合适的视口大小
        # 获取“签到日历”元素  
        #calendar_element = page.get_by_text("签到日历")
        import re
        
        '''calendar_element = page.get_by_role("main", name="Main content").get_by_role("button").filter(
            has_text=re.compile(r"今天，即.*")
        )
        
        if calendar_element:  # 确保元素存在并可见  
            bounding_box = calendar_element.bounding_box()  # 获取元素的坐标  
            if bounding_box:  # 计算中间坐标  
                mid_x = bounding_box['x'] + bounding_box['width'] // 2  
                mid_y = bounding_box['y'] + bounding_box['height'] // 2
                page.wait_for_timeout(5 * 1000)
                page.mouse.click(mid_x, mid_y)  # 使用鼠标点击中间坐标
                page.wait_for_timeout(2 * 1000)
                page.mouse.click(mid_x, mid_y)  # 使用鼠标点击中间坐标
                print('已在日历点击。')  
            else:  
                print('无法获取元素的边界框。')  
        else:  
            print('未找到日历元素。')'''
        # 
        
        # 1. 构建 Locator (此时并未真正查找元素，只是定义规则)
        # 建议给 locator 起个复数名字表示集合，或者直接用 target_btn
        '''target_btns = (
            page.get_by_role("main", name="Main content")
            .get_by_role("button")
            .filter(has_text=re.compile(r"今天，即.*"))
        )
        

        # 关键修正：先检查 count
        if target_btns.count() > 0:
            btn = target_btns.first()
            
            # 确保元素在视口中并获取边界框
            # wait_for_state('visible') 确保元素已渲染
            btn.wait_for(state="visible") 
            
            box = btn.bounding_box()
            
            if box:
                mid_x = box['x'] + box['width'] / 2
                mid_y = box['y'] + box['height'] / 2
                
                # 可选：先滚动到元素位置，防止坐标计算偏差
                btn.scroll_into_view_if_needed()
                
                page.mouse.click(mid_x, mid_y)
                print('已通过鼠标坐标点击。')
            else:
                print('无法获取边界框（元素可能不可见）。')
        else:
            print('未找到元素。')
        '''
        #btn = page.locator("td>div.relative>div>div>div>div>div>span").last().click()
        
        page.get_by_role("button", name="签到").click()
        page.wait_for_timeout(1 * 1000)
        btn = (
            page.locator("td div.relative span")  # 简化选择器，只定位到关键的 span
            .filter(has_text=re.compile(r"签到")) # 根据实际文本调整正则
            .last # 取最后一个
        )
        try:
            # wait_for 确保元素可见且可交互，避免点击失败
            btn.wait_for(state="visible", timeout=5000)
            btn.click()
            print("已成功点击目标日期")
            
            #
            slide_validate.validate(page,page_url=None, page_evaluate=js, background_css="img.gocaptcha-module_picture__LRwbY", slider_css="div.index-module_tile__8pkQD img", background_size=(300, 220), slider_down_css_xpath="div.gocaptcha-module_dragBlock__bFlwx", distance_correction=-12, track_add=-1)#-11
            
            results = page.locator("ol").all_inner_texts()
            print(results)
            
            try:
                # 断言疑似存在问题：时间太短来不及断言
                #expect(page.get_by_text("Verification successful")).to_be_visible() # 等待可见
                #expect(page.locator("ol")).to_contain_text("Verification successful") # 文本断言
                expect(page.locator("ol")).to_contain_text("成功") # 文本断言
            except Exception as e:
                if "成功" in results:
                    raise "成功！"
                print(f"第一次重试：{e}")
                btn.wait_for(state="visible", timeout=5000)
                btn.click()
                print("已成功点击目标日期")
                slide_validate.validate(page,page_url=None, page_evaluate=js, background_css="img.gocaptcha-module_picture__LRwbY", slider_css="div.index-module_tile__8pkQD img", background_size=(300, 220), slider_down_css_xpath="div.gocaptcha-module_dragBlock__bFlwx", distance_correction=-14, track_add=-1)#-11
                try:
                    expect(page.get_by_text("Verification successful")).to_be_visible() # 等待可见
                    expect(page.locator("ol")).to_contain_text("Verification successful") # 文本断言
                except Exception as e:
                    print(f"第二次重试：{e}")
                    btn.wait_for(state="visible", timeout=5000)
                    btn.click()
                    print("已成功点击目标日期")
                    slide_validate.validate(page,page_url=None, page_evaluate=js, background_css="img.gocaptcha-module_picture__LRwbY", slider_css="div.index-module_tile__8pkQD img", background_size=(300, 220), slider_down_css_xpath="div.gocaptcha-module_dragBlock__bFlwx", distance_correction=0)#-11
                    '''try:
                        expect(page.get_by_text("Verification successful")).to_be_visible() # 等待可见
                        expect(page.locator("ol")).to_contain_text("Verification successful") # 文本断言
                    except Exception as e:
                        slide_validate.validate(page,page_url=None, page_evaluate=js, background_css="img.gocaptcha-module_picture__LRwbY", slider_css="div.index-module_tile__8pkQD img", background_size=(300, 220), slider_down_css_xpath="div.gocaptcha-module_dragBlock__bFlwx", distance_correction=0)#-11'''
            print(page.locator("ol").all_inner_texts())
            
        except Exception as e:
            print(f"疑似点击失败..")
            logging.error(e, exc_info=True)
            # 调试：打印一下当前匹配到的元素文本，看看选对了没
            if btn.count() > 0:
                print(f"匹配到的元素文本: {btn.text_content()}")
                
        #print(r"page.locator('td div.relative span').all_inner_texts()==============================")
        #all_texts = page.locator("td div.relative span").all_inner_texts()
        #print(all_texts)
            
        '''str_arr_join = "".join(map(str, all_texts)) 
        if "签到" not in str_arr_join:
            print(f"成功！")
        else:
            logging.error("失败！！")'''
        
        
        #page.get_by_text("签到日历").click()
        '''try:
            page.get_by_role("button", name=" 去签到").click()
            js = "document.title"
            slide_validate.validate(page,page_url=None, page_evaluate=js, background_css="img.gocaptcha-module_picture__LRwbY", slider_css="div.index-module_tile__8pkQD img", background_size=(300, 220), slider_down_css_xpath="div.gocaptcha-module_dragBlock__bFlwx", distance_correction=-11)
        except TimeoutError as e:
            check_button = True
            raise e'''
            
        # 断言为成功
        '''try: expect(page.locator("span.semi-toast-content-text").last).to_contain_text("验证成功")
        except Exception as e: 
            logging.error(e, exc_info=True)
        logging.info(f"validate: {page.locator('span.semi-toast-content-text').evaluate_all('elements => elements.map(e => e.outerText)')}")
        
        try: page.wait_for_load_state(state="load", timeout=1000)  # 最长只等待1s，不管是否load完成，就进行下一步
        except Exception as e: logging.error(e, exc_info=True)
        
        
        #page.get_by_role("button", name=" 去签到").click()
        alert_success_locator = page.locator('div[role="alert"][aria-label="success type"]')
        expect(alert_success_locator.last).to_contain_text("签到成功")

        # 疑似在断言时间里，产生了等待1s的行为（导致前面打印失败了，但断言成功了）。这里进行重新打印
        logging.info(f"all_outer_text: {alert_success_locator.evaluate_all("elements => elements.map(e => e.outerText)")}")'''
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
        #logging.info(page.query_selector('div.semi-spin-children div.mb-4').text_content())

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
    # os.system("playwright codegen --lang=en-US --color-scheme=dark https://api.ephone.chat/ ")
