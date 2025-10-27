import requests
import time

API_KEY = "sk-66f2d6d0bbf346909ebd9d1eced5244a"
URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"


def call_qwen(prompt, temp, top_k, top_p):
    """封装调用函数，接收参数并返回结果"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    data = {
        "model": "qwen-plus",
        "input": {"messages": [{"role": "user", "content": prompt}]},
        "parameters": {
            "result_format": "text",
            "temperature": temp,
            "top_k": top_k,
            "top_p": top_p
        }
    }
    try:
        response = requests.post(URL, headers=headers, json=data)
        return response.json()["output"]["text"]
    except Exception as e:
        return f"错误：{str(e)}"


if __name__ == "__main__":
    # 测试问题（生成类，适合观察多样性）
    test_prompt = "用‘月亮’造3个不同的句子"

    # 定义3组参数：默认值、高随机性、高确定性
    # 默认参数 多次结果有一定差异，但不会太离谱（平衡灵活性和稳定性）。
    # 高随机性参数 多次结果差异很大（句子结构、用词可能完全不同），甚至会出现更有创意的表达。
    # 高确定性参数 多次结果非常相似（几乎是同一类句子，用词重复率高），稳定性强但缺乏多样性。
    param_groups = {
        "默认参数": {"temp": 0.8, "top_k": 50, "top_p": 0.7},
        "高随机性（创意）": {"temp": 1.8, "top_k": 100, "top_p": 0.95},
        "高确定性（稳定）": {"temp": 0.1, "top_k": 10, "top_p": 0.5}
    }


    # 每组参数测试3次（观察多次结果的差异）
    for name, params in param_groups.items():
        print(f"\n===== {name} 测试结果 =====")
        for i in range(3):
            print(f"\n第{i + 1}次：")
            result = call_qwen(
                test_prompt,
                temp=params["temp"],
                top_k=params["top_k"],
                top_p=params["top_p"]
            )
            print(result)
            time.sleep(1)  # 避免请求过于频繁