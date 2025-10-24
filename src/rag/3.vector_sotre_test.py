import uuid

from langchain_chroma import Chroma
from langchain_core.documents import Document  # 导入Document类（用于包装文本）
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 2. 示例文本
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
    # 3. 文本分块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=80,
        chunk_overlap=20,
        separators=["\n\n", "\n", "。", "，"]
    )
    chunks = text_splitter.split_text(text)  # chunks是字符串列表

    # 4. 初始化嵌入模型
    local_model_path = "../../local_models/bge-small-zh-v1"  # 你的本地模型路径
    embedding_model = HuggingFaceEmbeddings(
        model_name=local_model_path,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # 5. 初始化ChromaDB
    persist_directory = "../../store/chroma_db"
    db = Chroma(
        collection_name="product_docs",
        embedding_function=embedding_model,
        persist_directory=persist_directory,
    )

    # 6. 关键调整：将字符串列表转换为Document对象列表
    # 为每个chunk创建Document对象（page_content为文本，metadata可绑定元数据）
    documents = [
        Document(
            page_content=chunk,  # 文本内容
            metadata={"source": "product_manual_2024", "chunk_index": i}  # 元数据
        )
        for i, chunk in enumerate(chunks)
    ]

    # 生成唯一ID（每个Document对应一个ID）
    chunk_ids = [f"chunk_{uuid.uuid4().hex[:8]}" for _ in documents]

    # 7. 数据入库（用转换后的Document列表）
    db.add_documents(
        documents=documents,  # 传入Document对象列表（而非字符串列表）
        ids=chunk_ids  # 唯一ID
    )

    # 持久化
    # db.persist()

    # 8. 验证数据（修正查询方式）
    print("\n===== 验证：数据库中已存储的分块数量 =====")
    print(f"总存储分块数：{len(db.get(ids=chunk_ids)['ids'])}")  # 通过ID查询验证数量

    # 检索测试
    # query = "产品X的续航时间"
    query = "游泳"
    results = db.similarity_search(query, k=2)

    print("\n===== 检索测试：与“游泳”相关的分块 =====")
    for i, result in enumerate(results):
        print(f"相关分块{i + 1}：{result.page_content}")  # 现在result是Document对象，有page_content属性
        print(f"元数据：{result.metadata}\n")


    # # 9. 追加：使用similarity_search_with_score获取并打印相似度分数
    # print("\n===== 带相似度分数的检索测试：与“游泳”相关的分块 =====")
    # # similarity_search_with_score返回格式：[(Document对象, 相似度分数), (Document对象, 相似度分数), ...]
    # results_with_score = db.similarity_search_with_score(query="游泳", k=2)  # k=2和之前检索测试保持一致
    #
    # for i, (doc, score) in enumerate(results_with_score, 1):
    #     print(f"相关分块{i}：")
    #     print(f"  文本内容：{doc.page_content}")  # Document对象的文本内容
    #     print(f"  元数据：{doc.metadata}")  # Document对象的元数据
    #     print(f"  相似度分数：{score:.4f}")  # 打印相似度分数（保留4位小数，分数越高相关性越强）
    #     print()  # 空行分隔，提升可读性