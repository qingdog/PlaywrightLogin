import os
import platform
import re
import subprocess

from dotenv import load_dotenv


def get_win_browser_path(path="C:\\", file_list=None, program_name=r"(?i)chrome\.exe$", program_dir=r"(?i)Program Files|AppData", depth_path=r".:\\Users\\[^\\]+?$"):
    """使用windows软件的默认安装目录查找exe文件（Program Files/Users）

    从path查找program_name。从path->Users中查找第二层目录中的AppData里的program_name"""
    if file_list is None:
        file_list = []
    try:
        listdir = os.listdir(path)
    except Exception:
        return file_list  # 如果访问目录出错，返回当前文件列表

    for filename in listdir:
        file_path = os.path.join(path, filename)
        if re.search(depth_path, file_path):
            try:
                depth_listdir = os.listdir(file_path)
            except Exception:
                continue  # 如果访问深层目录出错，继续循环

            for depth_filename in depth_listdir:
                depth_filepath = os.path.join(file_path, depth_filename)
                # 查找 depth_path 目录下的文件夹是否包含 program_dir
                if os.path.isdir(depth_filepath) and re.search(program_dir, depth_filepath):
                    get_win_browser_path(depth_filepath, file_list, program_name)

        elif os.path.isdir(file_path) and re.search(program_dir, file_path):
            get_win_browser_path(file_path, file_list, program_name)
        elif os.path.isfile(file_path):
            if re.search(program_name, filename):
                file_list.append(file_path)  # 添加找到的文件路径
    return file_list


def find_win_exe(program_name=r"(?i)chrome\.exe$", default_exe_dirs=None):
    exe_file_paths = []
    if default_exe_dirs is None:
        default_exe_dirs = ["C:\\Program Files", "C:\\Program Files (x86)",
                    "D:\\Program Files", "D:\\Program Files (x86)", "D:\\Users"]
    for exe_dir in default_exe_dirs:
        found_files = get_win_browser_path(path=exe_dir, program_name=program_name)
        exe_file_paths.extend(found_files)  # 收集找到的文件
    return exe_file_paths


def find_chrome_util(exe_dirs=None) -> str | None:
    os_name = platform.system()

    def find_chrome_path(exe_dir=None):
        """查找 Chrome 浏览器路径"""
        def find_win_chrome_path():
            """查找 Windows 系统下的 Chrome 路径"""
            default_paths = [r"C:\Program Files\Google\Chrome\Application\chrome.exe", r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"]
            # 检查默认路径 chrome://version/
            for path in default_paths:
                if os.path.exists(path): return path
            # 加载环境变量
            load_dotenv()
            chrome_path = os.getenv("CHROME_PATH")
            if chrome_path: return chrome_path
            # 查找指定目录下的 Chrome
            return find_win_exe(program_name=r"(?i)chrome\.exe$", default_exe_dirs=exe_dir)

        def find_linux_chrome_path():
            """查找 Linux 系统下的 Chrome 路径"""
            # https://github.com/actions/runner-images/blob/ubuntu22/20241006.1/images/ubuntu/Ubuntu2204-Readme.md
            common_paths = ['/usr/bin/google-chrome', '/usr/bin/chromium-browser', '/opt/google/chrome/chrome', '/opt/google/chrome/google-chrome']
            for path in common_paths:
                if os.path.exists(path): return path
            # 使用 which 命令查找
            try:
                return subprocess.check_output(['which', 'google-chrome']).decode().strip()
            except subprocess.CalledProcessError:
                return None

        if os_name == "Windows":
            return find_win_chrome_path()
        else:
            return find_linux_chrome_path()

    chrome_executable_path = find_chrome_path(exe_dir=exe_dirs)
    # Chromium 内核浏览器的启动配置，控制浏览器行为和初始状态：https://peter.sh/experiments/chromium-command-line-switches/
    chrome_startup_args = []  # ['--window-size=1800,900']  # 最大化启动--start-maximized
    is_headless = os_name != "Windows"  # 非windows则以无头模式启动
    # chrome_startup_args.append("--disable-infobars")  # 禁用信息栏
    if is_headless:
        # chrome_startup_args.append("--disable-gpu")  # 兼容谷歌文档
        # Disables the sandbox for all process types that are normally sandboxed. Meant to be used as a browser-level switch for testing purposes only. ↪
        chrome_startup_args.append("--no-sandbox")  # 禁用沙箱进程
        chrome_startup_args.append("--disable-setuid-sandbox") # Disable the setuid sandbox (Linux only).
    return chrome_executable_path

if __name__ == '__main__':
    print(find_chrome_util())