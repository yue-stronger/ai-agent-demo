from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,  # 递归字符分块（最常用）
    CharacterTextSplitter,  # 简单字符分块
    TokenTextSplitter,  # 基于语义模型的Token分块
)

# 示例长文本（模拟一篇产品说明书）
long_text = """  
            产品X是2024年推出的智能手表，支持以下功能：  
            1. 健康监测：心率、血氧、睡眠质量检测，数据每5分钟更新一次。  
            2. 运动模式：跑步、游泳、骑行等15种运动模式，自动识别运动类型并生成报告。  
            3. 续航能力：普通模式下续航7天，省电模式下可达14天，支持无线快充（30分钟充满80%）。  
            
            注意事项：  
            - 游泳时仅支持50米防水，不建议深潜使用。  
            - 充电时需使用原装充电器，避免第三方设备导致损坏。  
            - 系统更新需保持电量在50%以上，更新过程中不可断电。  
            
            常见问题：  
            Q：手表无法连接手机怎么办？  
            A：1. 检查蓝牙是否开启；2. 重启手表和手机；3. 重新安装APP并配对。  
            Q：睡眠数据不准确？  
            A：确保手表佩戴紧密，且睡眠时间超过4小时才会生成完整报告。  
            """


def demo_text_splitting():
    # 策略1：递归字符分块（RecursiveCharacterTextSplitter，最推荐）
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,  # 每个块的字符数
        chunk_overlap=20,  # 块之间的重叠字符数（保持上下文连贯）
        separators=["\n\n", "\n", "。", "，", " "]  # 优先按分隔符拆分
    )
    recursive_chunks = recursive_splitter.split_text(long_text)
    print("===== 递归字符分块结果 =====")
    for i, chunk in enumerate(recursive_chunks):
        print(f"块{i + 1}（{len(chunk)}字符）：{chunk}\n")

    # 策略2：简单字符分块（CharacterTextSplitter）
    simple_splitter = CharacterTextSplitter(
        separator="\n",  # 仅按换行符拆分
        chunk_size=100,
        chunk_overlap=0
    )
    simple_chunks = simple_splitter.split_text(long_text)
    print("===== 简单字符分块结果 =====")
    for i, chunk in enumerate(simple_chunks):
        print(f"块{i + 1}（{len(chunk)}字符）：{chunk}\n")

    # 策略3：基于Token分块（TokenTextSplitter，按大模型Token计数）
    token_splitter = TokenTextSplitter(
        chunk_size=50,  # 每个块的Token数（约1Token=0.75英文词/0.5中文词）
        chunk_overlap=5
    )
    token_chunks = token_splitter.split_text(long_text)
    print("===== Token分块结果 =====")
    for i, chunk in enumerate(token_chunks):
        print(f"块{i + 1}：{chunk}\n")


if __name__ == "__main__":
    demo_text_splitting()
