from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# 1. 准备示例文本（分块用）
text = """  
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

if __name__ == "__main__":
    # 2. 文本分块（复用递归字符分块策略）
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        separators=["\n", "。", "，", " "]
    )
    chunks = text_splitter.split_text(text)
    print("===== 分块结果 =====")
    for i, chunk in enumerate(chunks):
        print(f"块{i + 1}：{chunk}\n")

    # 3. 初始化嵌入模型（选择一个即可，这里以开源模型为例）
    # 方案1：开源模型
    # 本地模型路径
    local_model_path = "../../local_models/all-MiniLM-L6-v2"
    embedding_model = HuggingFaceEmbeddings(
        model_name=local_model_path,  # 本地路径,  # 模型名称或本地路径
        model_kwargs={'device': 'cpu'},  # 强制用CPU（无GPU时）
        encode_kwargs={'normalize_embeddings': True}  # 归一化向量（推荐）
    )

    # 方案2：通义千问嵌入API（需阿里云密钥，768维向量）
    # import os
    # os.environ["DASHSCOPE_API_KEY"] = "你的通义千问密钥"
    # embedding_model = DashScopeEmbeddings(model="text-embedding-v1")

    # 4. 对分块后的文本进行向量化
    print("===== 向量化结果 =====")
    for i, chunk in enumerate(chunks):
        vector = embedding_model.embed_query(chunk)  # 生成向量
        # 取前10个元素，后面用...表示省略
        displayed_vector = vector[:10] + ['...']  # 前10个元素 + 省略号标记
        print(f"块{i + 1}向量（维度：{len(vector)}）：")
        print(displayed_vector)  # 直接打印包含前10个元素和省略号的列表
        print()  # 空行分隔，更清晰
