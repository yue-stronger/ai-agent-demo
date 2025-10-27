import time

import requests
import yfinance as yf  # 股票API库
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_classic.agents import initialize_agent, AgentType
from langchain_community.chat_models import ChatTongyi
from yfinance.exceptions import YFRateLimitError

# 加载API密钥（即使没用到.env，也不影响）
load_dotenv()
DASHSCOPE_API_KEY = "sk-66f2d6d0bbf346909ebd9d1eced5244a"  # 通义千问密钥


# --------------------------
# 1. 定义工具（修复get_weather的docstring和返回格式）
# --------------------------

@tool
def get_stock_price(symbol: str) -> str:
    """查询股票实时价格（参数：股票代码，如APPLE对应AAPL、谷歌对应GOOGL）"""
    try:
        time.sleep(1)  # 避免Yahoo Finance限流
        stock = yf.Ticker(symbol)
        hist = stock.history(period="1d")
        if hist.empty:
            hist = stock.history(period="5d")
        if hist.empty:
            return f"未查询到{symbol}的股价数据（请确认股票代码是否正确）"
        price = hist['Close'].iloc[-1]  # 最新收盘价
        return f"{symbol}实时股价：{price:.2f}美元"
    except YFRateLimitError:
        return "查询过于频繁，请5分钟后再试（Yahoo Finance限流）"
    except Exception as e:
        return f"股价查询失败：{str(e)}（可能是股票代码错误或网络问题）"


@tool  # 现在函数有docstring，装饰器不会报错
def get_weather(city: str) -> str:
    """查询指定城市的实时天气（参数：城市名称，如北京、上海、New York）"""
    try:
        url = "https://mcp-weather.vercel.app/mcp"
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "get_current_weather",
                "arguments": {"city": city}
            }
        }
        response = requests.post(url, json=payload, timeout=10)  # 加超时避免卡住
        response.raise_for_status()  # 若HTTP状态码非200，抛异常
        result = response.json()

        # 解析JSON结果，转为可读字符串（避免返回字典）
        if "result" in result and "content" in result["result"]:
            weather_info = result["result"]["content"][0]
            # 若weather_info是字典，提取关键信息；否则直接返回
            if isinstance(weather_info, dict):
                return f"{city}实时天气：{weather_info.get('weather', '未知')}，温度{weather_info.get('temperature', '未知')}，湿度{weather_info.get('humidity', '未知')}"
            else:
                return f"{city}实时天气：{weather_info}"
        else:
            return f"查询{city}天气失败：未获取到有效数据"
    except requests.exceptions.RequestException as e:
        return f"查询{city}天气失败：网络错误（{str(e)}）"
    except Exception as e:
        return f"查询{city}天气失败：{str(e)}"


# 工具列表（现在两个工具都合规）
tools = [get_weather, get_stock_price]

# --------------------------
# 2. 初始化大模型（不变）
# --------------------------
llm = ChatTongyi(model_name="qwen-plus", dashscope_api_key=DASHSCOPE_API_KEY)  # 复用密钥变量

# --------------------------
# 3. 创建Agent（不变）
# --------------------------
agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,  # 开启 verbose 可查看Agent思考过程
    handle_parsing_errors=True  # 处理解析错误
)

# --------------------------
# 4. 测试调用（不变）
# --------------------------
if __name__ == "__main__":
    print("===== 大模型+MCP工具调用 =====")
    user_query = input("请输入问题：")
    result = agent_executor.invoke({"input": user_query})["output"]
    print(f"回答：{result}")