## 天翼云签到(云函数版)

### 步骤

1. 选择 `python3.7`, 改执行方法为 `index.main`

2. ctrl + ` 进入终端

3. 输入 `pip3 install -r ./src/requirements.txt -t ./src` 安装依赖 (腾讯云函数)

华为云函数需要在底下创建依赖

---
华为云函数：函数工作流 FunctionGraph - 控制台
* https://console.huaweicloud.com/functiongraph/?region=ap-southeast-1#/serverless/dashboard
* 函数执行入口 index.handler
* 执行超时时间（秒）180
* 定时触发器 (TIMER)(共1个) Cron表达式 0 0 9 * * *

```python
def main(*_):
    configs = config.get("multi")
    push_together = config.get("push")

    messages = []

    for conf in configs:
        obj = ecloud(**conf)

        res = obj.start()

        push = conf.get("push")

        if push is None:
            if push_together is not None:
                messages.append(res)
        else:
            push_message(res, push)

    if len(messages) != 0 and push_together is not None:
        m = []
        for msg in messages:
            m.extend(msg["message"])

        msg = {"title": messages[0]["title"], "message": m}
        push_message(msg, push_together)


if __name__ == "__main__":
    main()

# -*- coding:utf-8 -*-
import json
def handler (event, context):
    main()
    return {
        "statusCode": 200,
        "isBase64Encoded": False,
        "body": json.dumps(event),
        "headers": {
            "Content-Type": "application/json"
        }
    }
```
* 环境变量
USERNAME189
PASSWORD189

* 依赖
urllib3-2.2.3-py3-none-any.whl
requests-2.32.3-py3-none-any.whl
python_dotenv-1.0.1-py3-none-any.whl
rsa-4.9-py3-none-any.whl

