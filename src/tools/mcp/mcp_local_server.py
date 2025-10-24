from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

# 初始化FastAPI应用（MCP服务）
app = FastAPI(title="本地MCP服务（大模型工具调用协议）")


# --------------------------
# 定义MCP协议数据结构（请求/响应格式）
# --------------------------
class MCPRequest(BaseModel):
    """大模型发送给MCP的工具调用请求"""
    tool_name: str  # 工具名称
    parameters: Dict[str, Any]  # 工具参数（键值对）


class MCPResponse(BaseModel):
    """MCP返回给大模型的响应"""
    success: bool
    data: Any = None  # 工具执行结果
    error: str = ""  # 错误信息（若失败）


# --------------------------
# 示例工具：天气查询（在MCP服务中实现）
# --------------------------
class WeatherTool:
    @staticmethod
    def run(city: str) -> str:
        """查询天气的具体实现（模拟）"""
        if not city:
            return "错误：缺少城市参数"
        # 模拟天气数据
        weather_data = {
            "北京": "晴，25℃，北风2级",
            "上海": "多云，27℃，东风3级",
            "广州": "雷阵雨，29℃，南风1级"
        }
        return weather_data.get(city, f"未查询到{city}的天气数据")


# 注册工具列表（MCP服务管理的工具）
TOOLS = {
    "get_weather": WeatherTool  # 工具名 -> 工具类映射
}


# --------------------------
# MCP核心接口：处理工具调用请求
# --------------------------
@app.post("/mcp/call", response_model=MCPResponse)
def mcp_tool_call(request: MCPRequest) -> MCPResponse:
    """
    MCP服务的工具调用接口
    接收大模型的调用请求，执行对应工具，返回结果
    """
    # 1. 检查工具是否存在
    if request.tool_name not in TOOLS:
        return MCPResponse(
            success=False,
            error=f"工具不存在：{request.tool_name}，可用工具：{list(TOOLS.keys())}"
        )

    # 2. 调用工具（传递参数）
    try:
        tool_class = TOOLS[request.tool_name]
        result = tool_class.run(**request.parameters)  # 执行工具
        return MCPResponse(success=True, data=result)
    except Exception as e:
        return MCPResponse(success=False, error=f"工具执行失败：{str(e)}")


# 启动服务：python mcp_server.py
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=18000)  # MCP服务地址：http://localhost:8000