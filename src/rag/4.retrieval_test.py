from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def retrieval():
    # 1. 初始化嵌入模型（需与存入数据库时使用的模型一致，否则向量不匹配）
    local_model_path = "../../local_models/bge-small-zh-v1"  # 本地模型路径
    embedding_model = HuggingFaceEmbeddings(
        model_name=local_model_path,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # 2. 加载已有的Chroma数据库（路径需与存入时一致）
    persist_directory = "../../store/chroma_db"
    db = Chroma(
        collection_name="product_docs",  # 必须与存入时的集合名称一致
        embedding_function=embedding_model,
        persist_directory=persist_directory,
    )

    print("===== 产品手册知识检索工具 =====")
    print("提示：输入问题查询产品信息，输入'exit'退出程序\n")

    while True:
        # 3. 接收用户控制台输入
        user_query = input("请输入你的问题：")

        # 退出条件
        if user_query.lower() == "exit":
            print("程序已退出，再见！")
            break

        # 4. 检索数据库（获取最相关的3个分块）
        results = db.similarity_search(
            query=user_query,
            k=3  # 返回Top3最相关的结果
        )

        # 5. 展示检索结果
        if not results:
            print("未找到相关信息，请尝试其他问题。\n")
            continue

        print("\n===== 检索到的相关信息 =====")
        for i, doc in enumerate(results, 1):
            print(f"【相关片段{i}】")
            print(f"内容：{doc.page_content}")
            print(f"来源：{doc.metadata['source']}（片段序号：{doc.metadata['chunk_index']}）\n")


if __name__ == "__main__":
    retrieval()
