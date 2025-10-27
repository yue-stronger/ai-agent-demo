import json

import dashscope
import requests
from dashscope import Generation


class MCPClient:
    def __init__(self, server_url):
        self.server_url = server_url
        self.request_id = 0  # 用于生成唯一请求ID

    def _get_next_id(self):
        """生成递增的请求ID"""
        self.request_id += 1
        return self.request_id

    def list_tools(self):
        """拉取服务器注册的工具列表"""
        payload = {
            "jsonrpc": "2.0",
            "id": self._get_next_id(),
            "method": "list_tools",
            "params": {}
        }
        response = requests.post(
            self.server_url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=10
        )
        response.raise_for_status()
        return response.json()

    def call_tool(self, tool_name, arguments):
        """调用指定工具"""
        payload = {
            "jsonrpc": "2.0",
            "id": self._get_next_id(),
            "method": "call_tool",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
        response = requests.post(
            self.server_url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=10
        )
        response.raise_for_status()
        return response.json()

# 初始化
dashscope.api_key = "sk-66f2d6d0bbf346909ebd9d1eced5244a"  # 从环境变量获取通义千问API_KEY
mcp_client = MCPClient("http://localhost:18001/mcp")  # 连接MCP服务器
chat_history = []  # 维护多轮对话历史


def format_tools_for_qwen(tools):
    """将MCP工具列表转换为通义千问要求的标准格式（含type和function字段）"""
    qwen_tools = []
    for tool in tools:
        # 严格按通义千问格式构造工具：type=function + function嵌套核心信息
        qwen_tool = {
            "type": "function",  # 必须包含，固定值为"function"
            "function": {
                "name": tool["name"],  # 工具名（来自MCP）
                "description": tool["description"],  # 工具描述（来自MCP）
                "parameters": {  # 参数结构（来自MCP的inputSchema）
                    "type": "object",
                    "properties": tool["inputSchema"]["properties"],
                    "required": tool["inputSchema"]["required"]
                }
            }
        }
        qwen_tools.append(qwen_tool)
    return qwen_tools


def run_multi_turn_chat():
    """多轮对话主逻辑：用户输入→模型处理→工具调用（如需）→返回结果"""
    # 1. 拉取MCP工具列表，告知通义千问
    mcp_tools = mcp_client.list_tools()["result"]["tools"]
    qwen_tools = format_tools_for_qwen(mcp_tools)
    print("已加载工具：", [t["name"] for t in mcp_tools])

    while True:
        user_input = input("\n你：")
        if user_input.lower() in ["exit", "退出"]:
            print("对话结束")
            break
        chat_history.append({"role": "user", "content": user_input})

        # 调用通义千问
        try:
            response = Generation.call(
                model="qwen-plus",  # 确保模型名称正确
                messages=chat_history,
                tools=qwen_tools,
                result_format="message"
            )
            print("API原始响应：", response)  # 打印完整响应，方便排查

            # 检查HTTP状态码
            if response.status_code != 200:
                # 提取错误信息（适配dashscope的响应结构）
                error_msg = ""
                if hasattr(response, 'output') and response.output:
                    error_msg = response.output.get('error', {}).get('message', '未知错误')
                elif hasattr(response, 'error'):
                    error_msg = str(response.error)
                else:
                    error_msg = "未获取到具体错误信息"

                print(f"API调用失败（状态码{response.status_code}）：{error_msg}")
                chat_history.pop()  # 回滚用户输入，避免影响后续对话
                continue

            # 检查output是否存在
            if not hasattr(response, 'output') or not response.output:
                print("API未返回有效结果，可能是参数错误或模型不可用")
                chat_history.pop()
                continue

            # 解析模型消息
            model_msg = response.output.choices[0].message
            chat_history.append(model_msg)

        except Exception as e:
            print(f"调用通义千问时发生异常：{str(e)}")
            chat_history.pop()  # 回滚用户输入
            continue

        # 5. 判断是否需要调用工具
        if "tool_calls" in model_msg:
            print("模型：正在调用工具...")
            # 执行所有工具调用
            tool_results = []
            for tool_call in model_msg["tool_calls"]:
                # 修复：从function字段中获取name和arguments
                tool_name = tool_call["function"]["name"]
                tool_args = json.loads(tool_call["function"]["arguments"])
                print(f"调用工具：{tool_name}，参数：{tool_args}")

                # 调用MCP工具
                try:
                    tool_response = mcp_client.call_tool(tool_name, tool_args)
                    result_content = tool_response["result"]["content"][0]["text"]
                    tool_results.append({
                        "role": "tool",
                        "content": result_content,
                        "tool_call_id": tool_call["id"]  # 关联工具调用ID
                    })
                except Exception as e:
                    tool_results.append({
                        "role": "tool",
                        "content": f"工具调用失败：{str(e)}",
                        "tool_call_id": tool_call["id"]
                    })

            # 6. 将工具结果回传给模型，让模型整理回答
            chat_history.extend(tool_results)
            final_response = Generation.call(
                model="qwen-plus",
                messages=chat_history,
                result_format="message"
            )
            final_msg = final_response.output.choices[0].message
            chat_history.append(final_msg)
            print(f"模型：{final_msg['content']}")

        else:
            # 无需调用工具，直接返回回答
            print(f"模型：{model_msg['content']}")


if __name__ == "__main__":
    print("开始对话（输入exit退出）...")
    run_multi_turn_chat()



