import time

# import openweathermap_api  # 天气API库
import yfinance as yf  # 股票API库
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_classic.agents import initialize_agent, AgentType
from langchain_community.chat_models import ChatTongyi
from yfinance.exceptions import YFRateLimitError

# 加载API密钥
load_dotenv()
# OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")  # 从OpenWeatherMap获取
DASHSCOPE_API_KEY = "sk-66f2d6d0bbf346909ebd9d1eced5244a"  # 通义千问密钥


# --------------------------
# 1. 定义工具（符合LangChain的tool规范，即MCP工具元数据）
# --------------------------
# @tool
# def get_weather(city: str) -> str:
#     """查询指定城市的实时天气（参数：城市名）"""
#     owm = openweathermap_api.OpenWeatherMap(OPENWEATHER_API_KEY)
#     data = owm.get_current_weather(q=city, units="metric")  # 摄氏度
#     if data:
#         return f"{city}天气：{data['weather'][0]['description']}，温度：{data['main']['temp']}℃"
#     return f"未查询到{city}的天气数据"

@tool
def get_stock_price(symbol: str) -> str:
    """查询股票实时价格（参数：股票代码，如APPLE、GOOGL）"""
    try:
        # 增加延迟，避免触发限流
        time.sleep(1)  # 每次请求间隔1秒
        stock = yf.Ticker(symbol)
        # 获取最近1天数据，若为空则尝试获取最近5天
        hist = stock.history(period="1d")
        if hist.empty:
            hist = stock.history(period="5d")
        if hist.empty:
            return f"未查询到{symbol}的股价数据"
        price = hist['Close'].iloc[-1]  # 最新收盘价
        return f"{symbol}实时股价：{price:.2f}美元"
    except YFRateLimitError:
        return "查询过于频繁，请稍后再试（Yahoo Finance限流）"
    except Exception as e:
        return f"股价查询失败：{str(e)}"


# 工具列表（MCP服务管理的工具）
# tools = [get_weather, get_stock_price]
tools = [get_stock_price]

# --------------------------
# 2. 初始化大模型（通义千问）
# --------------------------
llm = ChatTongyi(model_name="qwen-plus", dashscope_api_key="sk-66f2d6d0bbf346909ebd9d1eced5244a")

# --------------------------
# 3. 创建Agent（大模型+MCP工具调用逻辑）
# --------------------------
agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

# --------------------------
# 4. 测试调用
# --------------------------
if __name__ == "__main__":
    print("===== 大模型+MCP工具调用 =====")
    user_query = input("请输入问题：")
    result = agent_executor.invoke({"input": user_query})["output"]
    print(f"回答：{result}")
