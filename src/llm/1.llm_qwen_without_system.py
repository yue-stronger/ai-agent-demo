import requests

API_KEY = "sk-66f2d6d0bbf346909ebd9d1eced5244a"
URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"


def chat_with_qianwen(prompt):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    messages = [{"role": "user", "content": prompt}]
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

    print("输入'退出'结束")

    while True:
        user_input = input("\n你：")
        if user_input == "退出":
            print("再见！")
            break

        reply = chat_with_qianwen(user_input)
        print(f"通义千问：{reply}")