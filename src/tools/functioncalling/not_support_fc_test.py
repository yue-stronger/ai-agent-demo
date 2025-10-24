import requests
import json  

def test_deepseek_api_no_function_calling():  
    # 1. 获取AK（从环境变量读取，避免硬编码）  
    deepseek_ak = "sk-1c9871969c0d4fbdbf3d64da23704529"

    # 2. 配置API参数（请根据Deepseek官方文档修改URL和模型名称）  
    api_url = "https://api.deepseek.com/v1/chat/completions"  # API端点（示例，以官方为准）  
    # model_name = "deepseek-7b-base"  # 不支持Function Calling的模型（需与官方一致）
    model_name = "deepseek-chat"

    # 3. 构造测试提示（与原逻辑一致，要求模型调用工具）  
    prompt = """  
    你可以调用工具`get_weather(city: str)`查询天气，调用时请返回JSON格式：{"function":"get_weather","params":{"city":"城市名"}}。  
    用户的问题：查询上海今天的天气。  
    """  

    # 4. 构造API请求体（符合Chat Completion格式）  
    request_data = {  
        "model": model_name,  
        "messages": [{"role": "user", "content": prompt}],  # 用户消息  
        "max_tokens": 100,  # 限制生成长度  
        "temperature": 0.7  # 中等随机性  
    }  

    # 5. 构造请求头（携带AK认证，字段名以官方文档为准）  
    headers = {  
        "Content-Type": "application/json",
        "X-API-Key": deepseek_ak
        # "Authorization":  f"Bearer {deepseek_ak}"  # 假设官方要求用X-API-Key传递AK，若不同请修改
        # 若官方要求Bearer格式："Authorization": f"Bearer {deepseek_ak}"  
    }  

    # 6. 发送API请求并处理结果  
    try:  
        response = requests.post(  
            url=api_url,  
            headers=headers,  
            data=json.dumps(request_data),  
            timeout=30  # 超时时间  
        )  
        response.raise_for_status()  # 抛出HTTP错误（如401认证失败）  
    except requests.exceptions.RequestException as e:
        print(f"API请求失败：{e}")
        # 关键：打印响应的详细文本（告诉你具体哪里错了）
        if 'response' in locals():  # 确保response变量存在
            print("错误详情：", response.text)
        return

        # 7. 解析并打印返回结果
    result = response.json()  
    if "choices" in result and len(result["choices"]) > 0:  
        model_response = result["choices"][0]["message"]["content"]  
        print("模型返回结果：")  
        print(model_response)  
    else:  
        print("API返回格式异常：")  
        print(result)  

if __name__ == "__main__":  
    test_deepseek_api_no_function_calling()  