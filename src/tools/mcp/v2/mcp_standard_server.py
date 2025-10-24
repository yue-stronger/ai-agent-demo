from flask import Flask, request, jsonify
from jsonschema import validate, ValidationError

app = Flask(__name__)

# --------------------------
# 1. 工具定义（MCP工具注册）
# --------------------------
# 工具1：查询天气
weather_tool = {
    "name": "get_weather",
    "description": "查询指定城市的实时天气",
    "inputSchema": {  # 参数校验规则（JSON Schema）
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "城市名称（如北京、上海）"}
        },
        "required": ["city"]  # 必选参数
    }
}

# 工具2：两数相加
sum_tool = {
    "name": "calculate_sum",
    "description": "计算两个整数的和",
    "inputSchema": {
        "type": "object",
        "properties": {
            "a": {"type": "integer", "description": "第一个整数"},
            "b": {"type": "integer", "description": "第二个整数"}
        },
        "required": ["a", "b"]
    }
}

# 工具列表（服务器注册的所有工具）
registered_tools = [weather_tool, sum_tool]


# --------------------------
# 2. 工具执行逻辑
# --------------------------
def execute_get_weather(params):
    """执行天气查询工具"""
    city = params["city"]
    return {"type": "text", "text": f"{city}当前天气：晴，25℃"}

def execute_calculate_sum(params):
    """执行加法计算工具"""
    a = params["a"]
    b = params["b"]
    return {"type": "text", "text": f"{a} + {b} = {a + b}"}


# --------------------------
# 3. MCP协议接口（JSON-RPC 2.0）
# --------------------------
@app.route("/mcp", methods=["POST"])
def mcp_handler():
    """处理客户端的MCP请求（拉取工具列表/调用工具）"""
    try:
        request_data = request.get_json()
        # 验证JSON-RPC基础格式
        if not all(k in request_data for k in ["jsonrpc", "id", "method"]):
            return jsonify({
                "jsonrpc": "2.0",
                "id": request_data.get("id"),
                "error": {"code": -32600, "message": "无效的请求格式，缺少必要字段"}
            }), 400

        # 1. 拉取工具列表（method: list_tools）
        if request_data["method"] == "list_tools":
            return jsonify({
                "jsonrpc": "2.0",
                "id": request_data["id"],
                "result": {"tools": registered_tools}  # 返回注册的工具列表
            })

        # 2. 调用工具（method: call_tool）
        elif request_data["method"] == "call_tool":
            params = request_data.get("params", {})
            tool_name = params.get("name")
            tool_args = params.get("arguments", {})

            # 校验工具是否存在
            tool = next((t for t in registered_tools if t["name"] == tool_name), None)
            if not tool:
                return jsonify({
                    "jsonrpc": "2.0",
                    "id": request_data["id"],
                    "error": {"code": -32601, "message": f"工具不存在：{tool_name}"}
                }), 404

            # 校验参数是否符合inputSchema
            try:
                validate(instance=tool_args, schema=tool["inputSchema"])
            except ValidationError as e:
                return jsonify({
                    "jsonrpc": "2.0",
                    "id": request_data["id"],
                    "error": {
                        "code": -32602,
                        "message": f"参数错误：{str(e)}"
                    }
                }), 400

            # 执行工具
            if tool_name == "get_weather":
                result = execute_get_weather(tool_args)
            elif tool_name == "calculate_sum":
                result = execute_calculate_sum(tool_args)
            else:
                result = {"type": "text", "text": "未知工具"}

            # 返回工具执行结果
            return jsonify({
                "jsonrpc": "2.0",
                "id": request_data["id"],
                "result": {
                    "isError": False,
                    "content": [result]
                }
            })

        # 未知方法
        else:
            return jsonify({
                "jsonrpc": "2.0",
                "id": request_data["id"],
                "error": {"code": -32601, "message": f"未知方法：{request_data['method']}"}
            }), 404

    except Exception as e:
        return jsonify({
            "jsonrpc": "2.0",
            "id": request_data.get("id"),
            "error": {"code": -32000, "message": f"服务器错误：{str(e)}"}
        }), 500


if __name__ == "__main__":
    # 启动服务器，监听本地5000端口
    app.run(host="0.0.0.0", port=18001, debug=True)