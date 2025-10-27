import requests

API_KEY = "sk-66f2d6d0bbf346909ebd9d1eced5244a"
URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"


def chat_with_qwen(user_prompt, system_prompt, temperature, top_k, top_p):
    """
    调用通义千问API，支持设置temperature、top-k、top-p参数
    :param user_prompt: 用户提示词
    :param system_prompt: 系统提示词
    :param temperature: 随机性（0~1）
    :param top_k: 候选词数量（1~100）
    :param top_p: 累积概率（0~1）
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    data = {
        "model": "qwen-plus",
        "input": {"messages": messages},
        # 核心：在parameters中添加控制参数
        "parameters": {
            "result_format": "text",
            "temperature": temperature,  # 设置随机性
            "top_k": top_k,  # 设置候选词数量
            "top_p": top_p  # 设置累积概率
        }
    }

    try:
        response = requests.post(URL, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result["output"]["text"] if "output" in result else f"响应异常：{result}"
    except Exception as e:
        return f"错误：{str(e)}"


if __name__ == "__main__":
    # 示例：设置不同参数（根据需求调整）
    system_prompt = "你是一个创意写作助手，生成内容要多样有趣。"
    # 高随机性（适合创意生成）：temperature=1.8, top_k=100, top_p=0.9
    # 高确定性（适合事实回答）：temperature=0.1, top_k=10, top_p=0.5
    reply = chat_with_qwen(
        user_prompt="以'秋天'为主题写一句诗",
        system_prompt=system_prompt,
        temperature=1.8,  # 高随机性
        top_k=100,
        top_p=0.9
    )
    # 通义千问默认值
    # temperature：0.8（控制随机性，默认偏中等灵活）
    # top - k：50（默认从概率前50的候选词中选择）
    # top - p：0.8（默认累积概率达到80 % 的候选词集合）

    print(f"通义千问：{reply}")