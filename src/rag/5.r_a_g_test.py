import dashscope  # 通义千问SDK
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 1. 加载环境变量（存储API密钥，避免硬编码）
dashscope.api_key = "sk-66f2d6d0bbf346909ebd9d1eced5244a"  # 从.env文件获取通义千问API密钥


def build_rag_prompt(user_query: str, retrieved_docs: list) -> str:
    """
    Augmented：构建RAG提示词（整合用户问题+检索到的文档片段）
    :param user_query: 用户输入的问题
    :param retrieved_docs: 检索到的相关文档片段（Document对象列表）
    :return: 完整的RAG提示词
    """
    # 系统指令：约束大模型仅基于提供的文档回答，不编造信息
    system_prompt = """
    你是一个基于指定文档的问答助手，必须严格遵循以下规则：
    1. 所有回答必须基于【参考文档片段】中的内容，不得使用文档外的知识；
    2. 如果文档中没有相关信息，直接回复“未找到足够信息回答该问题”；
    3. 回答需简洁明了，分点说明（如果有多个要点），并标注参考的文档片段序号。
    """

    # 整理检索到的文档片段，格式化为“参考文档”
    reference_docs = ""
    for i, doc in enumerate(retrieved_docs, 1):
        reference_docs += f"""
    【参考文档片段{i}】
    内容：{doc.page_content}
    来源：{doc.metadata['source']}（片段序号：{doc.metadata['chunk_index']}）
        """

    # 完整提示词：系统指令 + 参考文档 + 用户问题
    full_prompt = f"""
    {system_prompt}
    {reference_docs}

    【用户问题】
    {user_query}

    【回答】
    """
    return full_prompt


def call_qwen_model(prompt: str) -> str:
    """
    Generation：调用通义千问大模型生成回答（适配新版dashscope SDK）
    :param prompt: 构建好的RAG提示词
    :return: 大模型生成的回答
    """
    try:
        # 调用通义千问轻量版（qwen-turbo）
        response = dashscope.Generation.call(
            model=dashscope.Generation.Models.qwen_turbo,
            prompt=prompt,
            temperature=1.8,
            top_p=0.9,
            max_tokens=1024
        )

        # 新版判断逻辑：直接用response.status_code == 200（200代表调用成功）
        if response.status_code == 200:
            # 提取回答内容（新版响应结构不变，仍从output["text"]获取）
            return response.output["text"].strip()
        else:
            # 错误信息处理：从response中提取详细错误（适配新版错误返回格式）
            error_msg = response.get("error", {}).get("message", "未知错误")
            return f"大模型调用失败：{error_msg}（错误码：{response.status_code}）"

    except Exception as e:
        # 捕获网络错误、超时等异常
        return f"调用异常：{str(e)}（可能是网络问题或SDK版本不兼容）"


def rag_qa_system():
    """完整RAG问答系统：Retrieval（检索）→ Augmented（提示词）→ Generation（生成）"""
    # 1. Retrieval：初始化嵌入模型+加载Chroma数据库（复用你的检索逻辑）
    local_model_path = "../../local_models/bge-small-zh-v1"  # 中文embedding模型
    embedding_model = HuggingFaceEmbeddings(
        model_name=local_model_path,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    persist_directory = "../../store/chroma_db"
    db = Chroma(
        collection_name="product_docs",
        embedding_function=embedding_model,
        persist_directory=persist_directory,
    )

    print("===== 产品X RAG问答系统 =====")
    print("提示：输入问题查询产品信息（如“游泳时能戴产品X吗？”），输入'退出'程序\n")

    while True:
        # 2. 接收用户问题
        user_query = input("请输入你的问题：")
        if user_query.lower() == "退出":
            print("程序已退出，再见！")
            break
        if not user_query.strip():
            print("请输入有效的问题，不能为空！\n")
            continue

        # 3. Retrieval：检索相关文档（Top3最相关片段）
        print("\n正在检索相关信息...")
        retrieved_docs = db.similarity_search(query=user_query, k=3)

        # 4. Augmented：构建RAG提示词
        print("正在整理参考信息...")
        rag_prompt = build_rag_prompt(user_query, retrieved_docs)

        # 5. Generation：调用通义千问生成回答
        print("正在生成回答...\n")
        answer = call_qwen_model(rag_prompt)

        # 6. 输出结果
        print("===== 回答结果 =====")
        print(answer)
        print("\n" + "-" * 50 + "\n")  # 分隔符，提升可读性


if __name__ == "__main__":
    rag_qa_system()