from langchain.tools import tool
from langchain_classic import hub
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_community.chat_models import ChatTongyi


# 1. 定义工具函数
@tool
def get_weather(city: str) -> str:
    """查询指定城市的实时天气"""
    return f"{city}当前天气：晴，25℃"


# 2. 拉起通义千问-plus模型，绑定可用函数
llm = ChatTongyi(model_name="qwen-plus", dashscope_api_key="sk-66f2d6d0bbf346909ebd9d1eced5244a")
tools = [get_weather]

# 3. 使用 ReAct Prompt（通义千问-plus 支持 ReAct 框架）
# https://smith.langchain.com/hub/hwchase17/react
prompt = hub.pull("hwchase17/react")

# 4. 组装 Agent
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 5. 极简多轮：手动维护历史，列表里放字符串即可
chat_history = []          # ← 全局历史

def chat(user_input: str) -> str:
    """单轮入口，返回助手回复并更新历史"""
    # 把历史拼成“人类: xxx\n助手: yyy”的字符串
    history_str = "\n".join(chat_history)
    # 构造 Agent 输入
    resp = agent_executor.invoke({
        "input": user_input,
        "chat_history": history_str          # ← 传给 prompt 的 {chat_history} 占位符
    })
    assistant_reply = resp["output"]
    # 更新历史
    chat_history.append(f"人类: {user_input}")
    chat_history.append(f"助手: {assistant_reply}")
    return assistant_reply

# 6. 多轮 REPL
if __name__ == "__main__":
    while True:
        try:
            q = input("你：")
            if q in {"退出"}:
                break
            print("AI：", chat(q))
        except KeyboardInterrupt:
            break
