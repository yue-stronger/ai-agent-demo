import requests

API_KEY = "sk-66f2d6d0bbf346909ebd9d1eced5244a"
URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"

def chat_with_qianwen(messages):
    """
    调用通义千问API，接收包含历史上下文的messages列表
    :param messages: 完整的对话历史列表（含system、user、assistant角色）
    :return: 模型回复
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    data = {
        "model": "qwen-plus",
        "input": {"messages": messages},  # 直接使用包含历史的messages
        "parameters": {"result_format": "text"}
    }

    try:
        response = requests.post(URL, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result["output"]["text"] if "output" in result else f"响应异常：{result}"
    except Exception as e:
        return f"错误：{str(e)}"


if __name__ == "__main__":
    # 1. 初始化对话历史列表，先加入系统提示词（只加一次）
    system_prompt = "你是一个幽默的AI老师，回答问题时必须带一个表情包，且用短句。"
    messages = [{"role": "system", "content": system_prompt}]  # 全局维护的历史列表

    print("===== 带历史上下文的对话 =====")
    print("输入'退出'结束")

    while True:
        # 2. 获取用户输入，添加到历史列表（user角色）
        user_input = input("\n你：")
        if user_input == "退出":
            print("再见！")
            break
        messages.append({"role": "user", "content": user_input})  # 追加用户消息

        # 3. 调用模型，传入完整历史
        reply = chat_with_qianwen(messages)
        print(f"通义千问：{reply}")

        # 4. 将模型回复添加到历史列表（assistant角色），供下一轮对话使用
        messages.append({"role": "assistant", "content": reply})