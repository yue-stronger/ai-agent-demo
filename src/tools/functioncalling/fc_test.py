import requests
import json

# 1. 配置通义千问API
API_KEY = "sk-66f2d6d0bbf346909ebd9d1eced5244a"
URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"


# 2. 定义可用的工具函数（实际场景中可对接真实API）
def get_weather(city: str) -> str:
    """查询指定城市的天气（模拟）"""
    # 模拟天气数据，实际应调用天气API
    weather_data = {
        "北京": "晴，25℃，微风",
        "上海": "多云，28℃，东风3级",
        "广州": "小雨，30℃，南风2级"
    }
    return f"{city}当前天气：{weather_data.get(city, '暂未获取到数据')}"


# 3. 工具映射表：让代码能根据函数名找到对应的工具（关键）
TOOL_MAP = {
    "get_weather": get_weather
}


# 4. 调用大模型的函数
def call_model(messages):
    """发送消息给通义千问，返回模型响应"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    data = {
        "model": "qwen-plus",
        "input": {"messages": messages},
        "parameters": {"result_format": "text"}
    }
    response = requests.post(URL, headers=headers, json=data)
    response.raise_for_status()
    return response.json()["output"]["text"]


# 5. 解析模型返回的工具调用指令
def parse_function_call(response_text):
    """
    从模型回复中提取工具调用信息
    假设模型用<FunctionCall>和</FunctionCall>包裹JSON格式的调用指令
    例如：<FunctionCall>{"name":"get_weather","parameters":{"city":"北京"}}</FunctionCall>
    """
    start_tag = "<FunctionCall>"
    end_tag = "</FunctionCall>"

    if start_tag in response_text and end_tag in response_text:
        # 提取标签内的JSON字符串
        call_str = response_text.split(start_tag)[1].split(end_tag)[0].strip()
        try:
            return json.loads(call_str)  # 返回解析后的字典：{"name": "...", "parameters": {...}}
        except json.JSONDecodeError:
            return None  # 解析失败
    return None  # 没有工具调用


# 6. 主对话逻辑（核心：处理工具调用循环）
def main():
    # 系统提示词：告诉模型有哪些工具及调用格式（关键配置）
    system_prompt = """
    你可以调用以下工具解决问题：
    1. 函数名：get_weather，参数：city（城市名，字符串），功能：查询指定城市的天气。

    若需要调用工具，请用<FunctionCall>和</FunctionCall>包裹JSON格式的调用指令，例如：
    <FunctionCall>{"name":"get_weather","parameters":{"city":"北京"}}</FunctionCall>

    若不需要调用工具，直接回答用户问题即可。
    """

    # 初始化对话历史（包含系统提示）
    messages = [{"role": "system", "content": system_prompt}]

    print("===== 支持工具调用的对话（输入'退出'结束） =====")
    while True:
        # 用户输入
        user_input = input("\n你：")
        if user_input == "退出":
            print("再见！")
            break
        messages.append({"role": "user", "content": user_input})

        # 第一次调用模型：判断是否需要调用工具
        model_response = call_model(messages)
        print(f"模型初步响应：{model_response}")

        # 检查是否有工具调用指令
        function_call = parse_function_call(model_response)
        if not function_call:
            # 不需要调用工具，直接记录并展示结果
            messages.append({"role": "assistant", "content": model_response})
            print(f"最终回答：{model_response}")
            continue

        # 需要调用工具：解析函数名和参数
        func_name = function_call.get("name")
        func_params = function_call.get("parameters", {})

        # 执行工具函数
        if func_name in TOOL_MAP:
            try:
                # 调用工具（**func_params 解包参数）
                tool_result = TOOL_MAP[func_name](**func_params)
                print(f"工具返回结果：{tool_result}")

                # 将工具结果加入对话历史，再次调用模型生成最终回答
                messages.append({"role": "assistant", "content": model_response})  # 记录模型的调用指令
                messages.append({
                    "role": "assistant",
                    "content": f"工具调用返回结果：{tool_result}"  # 明确告诉模型这是工具结果
                })

                # 第二次调用模型：基于工具结果生成最终回答
                final_response = call_model(messages)
                print(f"最终回答：{final_response}")
                messages.append({"role": "assistant", "content": final_response})

            except Exception as e:
                error_msg = f"工具调用失败：{str(e)}"
                print(error_msg)
                messages.append({"role": "assistant", "content": error_msg})
        else:
            error_msg = f"未知工具：{func_name}"
            print(error_msg)
            messages.append({"role": "assistant", "content": error_msg})


if __name__ == "__main__":
    main()