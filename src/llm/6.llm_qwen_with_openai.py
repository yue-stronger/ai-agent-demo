from openai import OpenAI

API_KEY = "sk-66f2d6d0bbf346909ebd9d1eced5244a"
URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

if __name__ == "__main__":
    # 1. 配置客户端：指定通义千问的 OpenAI 兼容接口和 API-KEY
    client = OpenAI(
        api_key=API_KEY,  # 替换为你的 API-KEY
        base_url=URL  # 通义千问的 OpenAI 兼容接口地址
    )

    # 2. 调用聊天接口（语法与 OpenAI 完全一致）
    response = client.chat.completions.create(
        model="qwen-plus",  # 通义千问模型名（如 qwen-plus、qwen-turbo）
        messages=[
            {"role": "system", "content": "你是一个助手，用简洁的语言回答问题。"},
            {"role": "user", "content": "做一个自我介绍"}
        ],
        temperature=0.7  # 支持设置 temperature 等参数
    )

    # 3. 提取回复
    print("大模型回复：")
    print(response.choices[0].message.content)

