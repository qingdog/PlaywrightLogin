# pip install opencv-python playwright
import base64
import logging
import random

import cv2
import requests
from PIL import Image
from io import BytesIO
import numpy as np
from playwright.sync_api import sync_playwright, Page, ElementHandle
import psutil
import os


# 定义不同的边缘处理方法和匹配方法组合来测试
def adaptive_canny(image_path):
    """使用图像中值自适应调整 Canny 阈值（会匹配到得分更高的位置而不是滑块缺口）"""
    image = cv2.imread(image_path, 0)
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    median_val = np.median(blurred)
    lower = int(max(0, 0.66 * median_val))
    upper = int(min(255, 1.33 * median_val))
    return cv2.Canny(blurred, lower, upper)


def combined_gray_edge(image_path):
    """融合灰度图和边缘图

    边缘不够稳定，可以组合灰度图与边缘图来辅助匹配"""
    gray = cv2.imread(image_path, 0)
    edge = apply_canny(image_path)
    return cv2.addWeighted(gray, 0.5, edge, 0.5, 0)


def apply_canny(image_path):
    """
    对输入图像进行高斯模糊并应用 Canny 边缘检测。提取图像的边缘轮廓

    滑块验证码识别中非常关键，因为相比颜色、纹理，边缘特征更稳定，更容易用于模板匹配
    :param image_path: 输入的图像路径
    :return: 处理后的二值图像，仅保留边缘信息
    """
    image = cv2.imread(image_path, 0)  # 读取滑块图片并转换为灰度图
    # 对图像进行高斯模糊，降低噪声，避免边缘检测出现误检
    image = cv2.GaussianBlur(image, (3, 3), 0)
    # 使用 Canny 算法进行边缘检测
    # 第一个阈值 50：较小的边缘强度
    # 第二个阈值 150：较大的边缘强度
    # 两阈值间的边缘只有在与强边缘相连时才会被保留
    return cv2.Canny(image, 50, 150)
    # 尝试提高边缘检测的敏感度
    # return cv2.Canny(image, 30, 100)


def match_template(bg_path, hx_path):
    """
    模板匹配，查找滑块在背景图中的位置。
    :param bg_path: 背景图片路径
    :param hx_path: 滑块图片路径
    :return: 缺口的横向位置（int）
    """
    # 融合灰度图和边缘图 - TM_CCORR_NORMED（匹配度得分更高但是不符合滑块滑动）
    # res = cv2.matchTemplate(combined_gray_edge(hx_path), combined_gray_edge(bg_path), cv2.TM_CCORR_NORMED)
    # 对滑块图和背景图都进行边缘检测后进行模板匹配，返回匹配结果矩阵
    #res = cv2.matchTemplate(apply_canny(hx_path), apply_canny(bg_path), cv2.TM_CCOEFF_NORMED)
    methods = [
        ('cv2.TM_CCOEFF_NORMED', cv2.TM_CCOEFF_NORMED),
        ('cv2.TM_CCORR_NORMED', cv2.TM_CCORR_NORMED),
        ('cv2.TM_SQDIFF_NORMED', cv2.TM_SQDIFF_NORMED)
    ]

    results = [] # 多个模板匹配结果集合 为置信度范围是[0,1]
    for method_name, method in methods:
        # 使用边缘进行匹配
        result = cv2.matchTemplate(apply_canny(hx_path), apply_canny(bg_path), method)

        if method == cv2.TM_SQDIFF_NORMED:
            # 对于SQDIFF方法，值越小表示匹配度越高
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            confidence = 1 - min_val  # 转换为置信度
            match_loc = min_loc
            print(f"使用方法: {method_name} 最小值: {min_val:.4f}, 置信度: {confidence:.4f}, 最佳匹配位置: {min_loc}")
        else:
            # 对于其他方法，值越大表示匹配度越高（获取匹配度最高的位置）
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            confidence = max_val
            match_loc = max_loc
            print(f"使用方法: {method_name} 最大值: {max_val:.4f}, 置信度: {confidence:.4f}, 最佳匹配位置: {max_loc}")
            
        results.append((method_name, match_loc, confidence, result))
        
    # 找到置信度最高的结果
    best_result = max(results, key=lambda x: x[2])
    print(f"\n最佳匹配结果: 方法: {best_result[0]} 位置: {best_result[1]} 置信度: {best_result[2]:.4f}")

    # 获取匹配位置的左上角坐标 返回x即可
    x, y = best_result[1]

    import platform
    os_name = platform.system()
    if os_name == "Windows":
        # 显示灰度化、高斯模糊、降噪、提取边缘轮廓 后的匹配位置
        show_draw_notch_box(hx_path, bg_path, max_val=best_result[2], x=x, y=y)
    return x  # 返回横坐标即可


def show_draw_notch_box(hx_path, bg_path, max_val, x, y):
    """红色矩形框标记出匹配位置并弹出处理后的图像窗口"""
    slider_img = cv2.imread(hx_path, 0)  # 读取滑块图片并转换为灰度图
    # 获取滑块图像的宽高，用于绘制矩形框
    w, h = slider_img.shape[::-1]

    # 在彩色背景图上绘制红色矩形框标记出匹配位置
    bg_img_color = cv2.imread(bg_path)  # 读取背景图片的彩色图，用于在识别结果上画出缺口位置
    cv2.rectangle(bg_img_color, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    # 显示
    cv2.imshow(f'cv2-({x},{y})-{max_val}', bg_img_color)
    # 等待1秒给用户查看图片
    cv2.waitKey(1000)
    # 销毁
    cv2.destroyAllWindows()


def get_track_list(distance):
    """
    生成模拟人类滑动行为的轨迹：加速 + 减速 + 收尾修正，保证总和 ≈ distance
    :param distance: 总滑动距离
    :return: 轨迹列表，每步滑动距离
    """
    tracks = []
    current = 0
    mid = distance * 0.6  # 加速到 60% 位置后开始减速

    t = 0.3  # 时间间隔
    v = 6  # 初速度

    while current < distance:
        if current < mid:
            a = random.uniform(2, 4)  # 加速阶段
        else:
            a = -random.uniform(3, 5)  # 减速阶段

        v0 = v
        v = v0 + a * t
        move = v0 * t + 0.5 * a * t * t
        move = round(move)

        if move == 0:
            continue

        current += move
        if current > distance:
            break

        tracks.append(move)

    # 补上差值，使总和正好为 distance
    diff = distance - sum(tracks)
    if diff != 0:
        tracks.append(diff)

    return tracks


def simulate_slider(page: Page, distance, slider_down_css_xpath, is_bezier=False):
    """simulate_slider
    控制滑块模拟移动。
    :param page: Playwright 页面对象
    :param distance: 移动距离（像素）
    :param slider_down_css_xpath: 滑块元素的XPath或者css
    :param is_bezier: 贝塞尔曲线实现模拟人手动滑动的轨迹
    """
    # 等待滑块出现
    slider = page.wait_for_selector(slider_down_css_xpath, strict=True)
    box = slider.bounding_box()
    start_x = box["x"] + box["width"] / 2
    start_y = box["y"] + box["height"] / 2

    page.mouse.move(start_x, start_y)
    page.mouse.down()

    if is_bezier:
        from cBezier import bezierTrajectory  # 替换为你的模块名
        bt = bezierTrajectory()
        # 生成轨迹
        result = bt.trackArray(
            start=[0, 0],
            end=[distance, 0],  # Y 为 0 起止，轨迹内部自动加入偏移
            numberList=60,  # 轨迹点数量30~80
            le=4,  # 贝塞尔曲线阶数，控制曲率复杂度    2~4
            deviation=4,   # 上下波动范围（Y轴幅度）3~10
            bias=0.4,  # 控制波动中线位置
            type=2,  # 速度类型：2 = 先快后慢（自然人滑动特性）
            cbb=0  # ❗️关闭终点回摆 # yhh 终点回摆幅度（建议 <10）
        )
        points = result["trackArray"]
        print("轨迹点：", points)
        # 实际滑动（加上起始位置）
        for point in points:
            x = start_x + point[0]
            y = start_y + point[1]
            page.mouse.move(x, y, steps=1)
    else:
        tracks = get_track_list(distance)
        # 遍历轨迹列表，逐步移动鼠标，实现“人类风格”的滑动
        print(tracks)
        for move_x in tracks:
            start_x += move_x
            if move_x >= 10:
                page.mouse.move(start_x, start_y, steps=2)  # 大步才平滑
            else:
                page.mouse.move(start_x, start_y, steps=1)  # 小步快速滑

    page.mouse.up()


def test_simulate_slider():
    """测试滑动"""
    def check_port(port):
        """检查端口是否被占用"""
        for conn in psutil.net_connections(kind='inet'):
            if conn.laddr.port == port:
                return True
        return False
    def start_remote_chrome_port(chrome_path=r"C:/Program Files/Google/Chrome/Application/chrome.exe", port=9527):
        """初始化浏览器连接"""
        if not check_port(port):
            print(f"端口 {port} 没有被占用，启动 Chrome 浏览器...")
            os.system(
                rf'start "" "{chrome_path}" --remote-debugging-port={port} '  # 避免start遇到空格就结束
                r'--user-data-dir="D:\\selenium"&')
        else:
            print(f"端口 {port} 已经被占用，Chrome 浏览器已启动。")
        return port
        
    chrome_port = start_remote_chrome_port()
    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp(f"http://127.0.0.1:{chrome_port}")
        # browser = p.chromium.launch(executable_path=chrome_executable_path, headless=False)
        context = browser.contexts[0]
        page = context.pages[0]
        # 打印 navigator.webdriver 的值
        print(f"navigator.webdriver: {page.evaluate('navigator.webdriver')}")

        #page.goto("http://localhost:9527/json/version")
        #page.goto("https://bot.sannysoft.com/")
        page.wait_for_timeout(5 * 1000)
        simulate_slider(page, 223-12, slider_down_css_xpath="div.gocaptcha-module_dragBlock__bFlwx", is_bezier=False)
                
        # 执行操作，例如获取页面标题
        print(page.title())
        # 延时，便于观察结果
        page.wait_for_timeout(10 * 1000)
        browser.close()


def page_open(page: Page, page_url, wait_until="domcontentloaded", page_evaluate="true;"):
    # page.goto("https://dun.163.com/trial/jigsaw", wait_until="onload")
    if page_url:
        page.goto(page_url, wait_until=wait_until)
        page.wait_for_function("document.readyState === 'complete'")
        # 等待2s后再执行
        page.wait_for_timeout(2 * 1000)
    page.evaluate(f"{page_evaluate}")

def crop_top_bottom_transparency(image, cropped_img_name="cropped_image.png", top=None, bottom=None):  
    """裁剪图片上下透明部分，保留左右宽度不变"""
    img = image.convert("RGBA")
    if not top or not bottom:
        data = np.array(img)
        # 获取所有非透明像素的行索引（alpha > 0）
        rows_with_content = np.where(data[:, :, 3] > 0)[0]

        if len(rows_with_content) == 0:
            return image  # 完全透明，返回原图

        # 计算上下边界
        top = rows_with_content.min()
        bottom = rows_with_content.max()

    # 裁剪图片（左=0, 上=top, 右=原宽度, 下=bottom+1）
    cropped_img = image.crop((0, top, img.width, bottom + 1))
    ###
    cropped_img.save(cropped_img_name)  # 保存裁剪后的图片
    print(f"裁剪后尺寸: {cropped_img.size}")  # (宽度, 新高度)

    return cropped_img, top, bottom


def download_src(page: Page, background_css, slider_css, background_size=None, slider_size=None, is_crop=0):
    # 替换为你的滑块图片路径和背景图路径
    background_filename = 'background.png'
    slider_filename = 'slider.png'
    # 等待图片加载
    page.wait_for_selector(background_css)
    
    # 获取 滑块Base64 图片数据
    img_src = page.get_attribute(slider_css, "src")
    #print(f"图标滑块: {img_src}")
    top, bottom = -1, -1
    if img_src.startswith('http'):
        response = requests.get(img_src)
        # 检查请求是否成功
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            # 调整大小
            if slider_size: img = img.resize(slider_size)
            # 保存图像
            img.save(slider_filename)
            if is_crop == 1:
                cropped_img, top, bottom = crop_top_bottom_transparency(img, slider_filename)
    #elif img_src.startswith('data:'):
    else:
        base64_str = img_src.split(',')[1]
        image_data = base64.b64decode(base64_str)
        img = Image.open(BytesIO(image_data))
        if slider_size: img = img.resize(slider_size)
        img.save(slider_filename)
        # 测试001
        if is_crop == 1:
            cropped_img, top, bottom = crop_top_bottom_transparency(img, slider_filename)
    '''
    # 针对滑块的高度在html标签的样式中，而不是图片为透明的滑块
    element = page.query_selector('div.index-module_tile__8pkQD[style*="top:"]') # 使用选择器定位元素（根据类名和样式）
    if element:
        # 获取元素的style属性中的top值
        style = element.get_attribute("style")
        top = style.split("top:")[1].split("px")[0].strip()
        print(f"元素的top值为: {top}px")
        bottom = top + Image.open("example.jpg").height + 10
    '''
        
    # 获取背景图片的 src 属性
    img_src = page.get_attribute(background_css, "src")
    #print(f"背景图片: {img_src}")
    if img_src.startswith('http'):
        response = requests.get(img_src)
        # 检查请求是否成功
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            # 调整大小
            if background_size: img = img.resize(background_size)
            # 保存图像
            img.save(background_filename)
            if is_crop == 1:
                cropped_img, top, bottom = crop_top_bottom_transparency(img, background_filename, top, bottom)
    else:
        # 提取 Base64 数据（去掉前缀部分）
        base64_str = img_src.split(',')[1]
        image_data = base64.b64decode(base64_str)
        img = Image.open(BytesIO(image_data))
        if background_size: img = img.resize(background_size)
        img.save(background_filename)
        if is_crop == 1:
            cropped_img, top, bottom = crop_top_bottom_transparency(img, background_filename, top, bottom)

    return background_filename, slider_filename

def main(background_css, slider_css, page_url=None, page_open_func=page_open, page_evaluate="document.baseURI", 
         download_src_func=download_src, background_size=None, slider_size=None, slider_down_css_xpath=None, distance_correction=0):
    if not slider_down_css_xpath: slider_down_css_xpath = slider_css
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(
            viewport={"width": 1600, "height": 900},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
            locale="zh-CN"
        )
        context.route('**/*', lambda route: route.continue_())
        page = context.new_page()
        
        validate(page, background_css, slider_css, page_url, page_open_func, page_evaluate, 
         download_src_func, background_size, slider_size, slider_down_css_xpath, distance_correction)

        #page.wait_for_timeout(30 * 1000)

def validate(page, background_css, slider_css, page_url=None, page_open_func=page_open, page_evaluate="document.baseURI", 
         download_src_func=download_src, background_size=None, slider_size=None, slider_down_css_xpath=None, distance_correction=0, 
         validate_success_css=None):
    '''
    :param background_css:背景图片css或xpath
    :param slider_css:滑块图片css或xpath
    :param page_open_func:可保持默认或自定义打开页面函数并执行自定义的js
    :param page_evaluate:自定义的js
    :param download_src_func:下载两张图片的函数，默认只支持base64和url
    :param background_size:用于缩放图片确定距离比例
    :param slider_size:用于缩放图片确定距离比例
    :param slider_down_css_xpath:滑块图片下方滑动块的css（滑块图片不能直接滑动需要使用这个）
    :param distance_correction:距离_校正（识别出缺口距离后会加上这个）
    :param validate_success_css:验证成功css（用于重试）
    :return:None
    '''
    # 打开页面
    page_open_func(page, page_url, page_evaluate=page_evaluate)
    # 下载图片
    background_filename, slider_filename = download_src_func(page, background_css, slider_css, background_size, slider_size)
    # 识别滑块缺口位置
    distance_notch = match_template(background_filename, slider_filename)
    # 控制滑块模拟移动
    #simulate_slider(page, distance_notch, slider_down_css_xpath='div.yidun_slide_indicator')
    # 增加距离偏移（可选）
    if "dun.163.com" in page.url:
        difference_arr = [4, 8, 12]
        print("距离增加12像素")
        distance_notch += 12
    elif "api.ephone.chat" in page.url:
        #distance_notch += -12
        pass
    simulate_slider(page, distance_notch+distance_correction, slider_down_css_xpath=slider_down_css_xpath)
    

    for i in ([1,2,3,4,5]):
        if validate_success_css:
            page.locator(validate_success_css).wait_for(state="visible", timeout=10*1000)
            print("validate_success_css验证通过---执行完毕！")
            break
        else:
            from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
            try:
                # 假设验证接口50ms，验证弹窗收起动画1s，验证成功提示3s。 1.1s < wait_for_timeout < 2.9s
                page.wait_for_timeout(1.7 * 1000)
                # 重新获取 Base64 图片数据，如果获取到了则说明验证码失败了
                page.locator(background_css).wait_for(timeout=200)
                print("验证不通过---不一定准确，受网络响应时间影响")
                
                # 验证失败了，再执行一次滑动逻辑...（可能网站验证滑动存在一定像素距离随机增减，这里使用重复校验试图通过）
                background_filename, slider_filename = download_src_func(page, background_css, slider_css, background_size, slider_size)
                distance_notch = match_template(background_filename, slider_filename)
                simulate_slider(page, distance_notch+distance_correction, slider_down_css_xpath=slider_down_css_xpath)
            except PlaywrightTimeoutError as e:
                print("验证通过---执行完毕！")
                break
    

if __name__ == "__main__":
    ''''''
    js = """
        // 查找文本内容为 "弹出式" 的元素并点击它
        const elements = document.querySelectorAll('ul li.tcapt-tabs__tab'); // 查找所有元素
        elements.forEach(element => {
          if (element.textContent.trim() === "弹出式") {
            element.click(); // 点击该元素
          }
        });
        setTimeout(() => {
        document.querySelectorAll('div.u-fitem-capt button.tcapt-bind_btn--login')[0].click();}, 200);
    """
    #main(page_url="https://dun.163.com/trial/jigsaw", background_css="img.yidun_bg-img", slider_css="img.yidun_jigsaw", page_evaluate=js, background_size=(320,160), slider_size=[61, 160])
    
    ''''''
    js = """
        document.querySelector('[placeholder="请输入用户名"]').value="admin";
        const elements = document.querySelectorAll('span'); // 查找所有元素
        elements.forEach(element => {
          if (element.textContent.trim() === "登录") {
            element.click(); // 点击该元素
          }
        });
    """
    main(page_url="http://192.168.50.227/login?redirect=/index", page_evaluate=js, background_css="div.verify-img-panel img", slider_css="div.verify-sub-block img", background_size=(400, 200), slider_size=(60, 200))
    
    ''''''
    js = """
        const elements2 = document.querySelectorAll('[class="semi-button-content"]');
        elements2.forEach(element => {
          if (element.textContent.trim() === "确定") {
            element.click(); // 点击该元素
          }
        });
        const elements = document.querySelectorAll('span'); // 查找所有元素
        elements.forEach(element => {
          if (element.textContent.trim() === "请完成人机验证后继续") {
            element.click(); // 点击该元素
          }
        });
    """
    #main(page_url="https://api.ephone.chat/login?expired=true", page_evaluate=js, background_css="img.gocaptcha-module_picture__LRwbY", slider_css="div.index-module_tile__8pkQD img", background_size=(300, 220), slider_down_css_xpath="div.gocaptcha-module_dragBlock__bFlwx", distance_correction=-12)
    
    #test_simulate_slider()
