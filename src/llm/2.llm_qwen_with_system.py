import requests

API_KEY = "sk-66f2d6d0bbf346909ebd9d1eced5244a"
URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"


def chat_with_qwen(user_prompt, system_prompt):
    """
    调用通义千问API，支持系统提示词和用户提示词
    :param prompt: 用户输入的问题（用户提示词）
    :param system_prompt: 系统提示词（定义模型行为）
    :return: 模型回复
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    # 构建消息列表（先加系统提示词，再加用户提示词）
    # 添加系统提示词（role="system"）
    # 添加用户提示词（role="user"）
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    data = {
        "model": "qwen-plus",
        "input": {"messages": messages},
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
    # 定义系统提示词（可根据需求修改）
    system_prompt = "你是一个幽默的AI老师，回答问题时必须带一个表情包，且用短句。"

    print("===== 带系统提示词的对话 =====")
    print("输入'退出'结束")

    while True:
        user_input = input("\n你：")
        if user_input == "退出":
            print("再见！")
            break
        # 调用时传入用户提示和系统提示
        reply = chat_with_qwen(user_input, system_prompt)
        print(f"通义千问：{reply}")
