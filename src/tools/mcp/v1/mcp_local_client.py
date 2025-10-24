from typing import Any

import requests
import json
import dashscope  # 通义千问SDK
from dashscope import Generation

# 配置通义千问API密钥
dashscope.api_key = "sk-66f2d6d0bbf346909ebd9d1eced5244a"

# MCP服务地址（与本地MCP服务对应）
MCP_SERVICE_URL = "http://localhost:18000/mcp/call"


def llm_decide_tool(user_query: str) -> tuple[None, None, str] | tuple[Any, Any, None]:
    """
    调用通义千问Plus模型，判断是否需要调用工具及参数
    返回：(工具名, 参数, 直接回答内容)
    """
    # 告诉模型可用的工具及MCP调用格式
    prompt = f"""
    你需要处理用户问题，并决定是否调用工具。
    可用工具：
    1. 工具名：get_weather，功能：查询城市天气，参数：{{"city": "城市名"}}
    若需要调用工具，返回JSON格式：{{"tool_name": "工具名", "parameters": {{...}}}}
    若无需调用工具，直接返回自然语言回答。

    用户问题：{user_query}
    """

    # 调用通义千问Plus模型
    response = Generation.call(
        model="qwen-plus",  # 通义千问Plus模型
        prompt=prompt,
        result_format="text"  # 返回文本格式
    )

    # 解析模型返回结果
    if response.status_code == 200:
        content = response.output.text.strip()
        # 判断是否返回工具调用指令（JSON格式）
        if content.startswith("{") and "tool_name" in content:
            try:
                tool_info = json.loads(content)
                return tool_info.get("tool_name"), tool_info.get("parameters", {}), None
            except json.JSONDecodeError:
                # 格式错误，视为直接回答
                return None, None, content
        else:
            # 直接回答
            return None, None, content
    else:
        return None, None, f"模型调用失败：{response.message}"


def call_mcp_service(tool_name: str, parameters: dict) -> dict:
    """调用MCP服务，发送工具调用请求"""
    try:
        response = requests.post(
            url=MCP_SERVICE_URL,
            json={"tool_name": tool_name, "parameters": parameters},
            timeout=5
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"success": False, "error": f"MCP服务调用失败：{str(e)}"}


def main():
    print("===== 通义千问+MCP工具调用Demo =====")
    user_query = input("请输入问题：")

    # 1. 调用通义千问，决定是否调用工具
    tool_name, params, direct_answer = llm_decide_tool(user_query)

    if direct_answer:
        # 无需调用工具，直接返回模型回答
        print(f"通义千问回答：{direct_answer}")
        return

    # 2. 通过MCP服务调用工具
    print(f"通义千问决定调用工具：{tool_name}，参数：{params}")
    mcp_result = call_mcp_service(tool_name, params)

    # 3. 结合工具结果生成最终回答
    if mcp_result["success"]:
        print(f"通义千问回答：根据查询，{mcp_result['data']}")
    else:
        print(f"通义千问回答：查询失败，原因：{mcp_result['error']}")


if __name__ == "__main__":
    main()