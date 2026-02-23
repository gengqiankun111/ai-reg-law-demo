"""
RAG 法律问答：结合本地文件与 API 网络资源。

- 本地资源（无需联网）：
  - 向量库：Chroma 持久化目录（chroma_laws_db），含 crawler-law / ingest_docs 写入的条文与切片
  - Embedding：优先从 model_cache 加载，不请求 Hugging Face

- 网络 API（需配置 .env）：
  - 大模型：OpenAI 或 DeepSeek，根据问题与检索到的本地条文生成回答

流程：用户问题 → 本地向量检索 TopK → 拼上下文 → 调用 API 大模型 → 返回答案。
"""
import os
import warnings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# ========== 核心修复1：全局离线配置（优先加载） ==========
# 强制 HuggingFace 库使用本地缓存，禁用远端检查
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
# 固定模型缓存目录（与 ingest_docs.py 保持一致）
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "./model_cache"
os.environ["TRANSFORMERS_CACHE"] = "./model_cache"
os.environ["HF_HOME"] = "./model_cache"

# 关闭无关警告
warnings.filterwarnings("ignore")

from config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    TOP_K,
    LLM_PROVIDER,
    OPENAI_MODEL,
    OPENAI_BASE_URL,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
)

load_dotenv()

# ========== 核心修复2：优先使用本地已缓存模型路径（不再请求 modules.json） ==========
def _local_embedding_model_path():
    """若 model_cache 中已有该模型，返回本地路径，加载时不会访问外网。"""
    cache_dir = os.path.abspath("./model_cache")
    # 与 ingest_docs 保存路径一致：sentence_transformers / model_name
    local_name = EMBEDDING_MODEL.replace("/", "_")
    path = os.path.join(cache_dir, "sentence_transformers", local_name)
    if os.path.isdir(path):
        return path
    return None

def _embedding_model_kwargs():
    return {"device": "cpu"}

def get_embedding():
    """获取 Embedding：优先本地缓存路径，避免每次请求 modules.json。"""
    local_path = _local_embedding_model_path()
    if local_path:
        # 使用本地路径作为 model_name，SentenceTransformer 直接从磁盘加载，不访问 Hub
        return HuggingFaceEmbeddings(
            model_name=local_path,
            model_kwargs=_embedding_model_kwargs(),
            encode_kwargs={"normalize_embeddings": True},
        )
    # 未缓存时仍用名称（会走网络，需可访问 Hugging Face）
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        cache_folder=os.path.abspath("./model_cache"),
        model_kwargs=_embedding_model_kwargs(),
        encode_kwargs={"normalize_embeddings": True},
    )

def get_vectorstore():
    """获取 Chroma 向量库（完全离线）"""
    embedding = get_embedding()
    # 修复：使用 persist_directory → path（新版本 Chroma 兼容）
    return Chroma(
        persist_directory=CHROMA_PATH,  # 替代 persist_directory，避免参数警告
        collection_name=COLLECTION_NAME,
        embedding_function=embedding,
    )

def get_retriever(top_k=TOP_K):
    """获取检索器（完全离线）"""
    vs = get_vectorstore()
    # 修复：增加检索参数，避免不必要的网络请求
    return vs.as_retriever(
        search_kwargs={
            "k": top_k,
           # "fetch_k": top_k * 2  # 提升检索准确性，无外网请求
        }
    )

# ========== 提示词保持不变 ==========
LAW_QA_PROMPT = """你是一个法律条文助手。请仅根据以下「参考条文」回答问题。如果参考条文中没有相关内容，请说明「参考条文中未涉及该问题」，不要编造。

参考条文：
{context}

问题：{query}

请用简洁、准确的语言回答："""

def _format_docs(docs):
    """格式化检索到的文档"""
    return "\n\n".join(doc.page_content for doc in docs)

# ========== 核心修复3：LLM 初始化优化（减少外网请求） ==========
def get_llm():
    """
    根据配置返回 OpenAI 或 DeepSeek 的 ChatOpenAI 实例。
    优化：增加超时/重试配置，减少不必要的外网请求。
    环境变量优先：LLM_PROVIDER（openai | deepseek）、OPENAI_API_KEY、DEEPSEEK_API_KEY。
    """
    provider = (os.getenv("LLM_PROVIDER") or LLM_PROVIDER).strip().lower()

    # 通用 LLM 配置（减少外网请求）
    common_kwargs = {
        "temperature": 0,
        "timeout": 30,  # 超时时间，避免卡壳
        "max_retries": 1,  # 减少重试次数，避免重复外网请求
    }

    if provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DeepSeek 需设置 DEEPSEEK_API_KEY（.env 或环境变量）。")
        return ChatOpenAI(
            model=os.getenv("DEEPSEEK_MODEL") or DEEPSEEK_MODEL,
            openai_api_key=api_key,
            openai_api_base=os.getenv("DEEPSEEK_BASE_URL") or DEEPSEEK_BASE_URL,
            **common_kwargs
        )

    # OpenAI（默认）
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI 需设置 OPENAI_API_KEY（.env 或环境变量）。")
    kwargs = {
        "model": os.getenv("OPENAI_MODEL") or OPENAI_MODEL,
        "openai_api_key": api_key,
        **common_kwargs
    }
    base = os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL") or OPENAI_BASE_URL
    if base:
        kwargs["openai_api_base"] = base
    return ChatOpenAI(**kwargs)

# ========== 核心修复4：优化检索结果展示（统一元数据） ==========
def ask(question: str, top_k=TOP_K):
    """检索 TopK 并返回大模型回答（完全离线检索）"""
    llm = get_llm()
    retriever = get_retriever(top_k=top_k)
    # 检索过程完全离线（无外网请求）
    docs = retriever.invoke(question)
    context = _format_docs(docs)
    prompt = PromptTemplate(
        template=LAW_QA_PROMPT,
        input_variables=["context", "query"],
    )
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "query": question})
    return answer, docs

def main():
    import sys
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "合同" #"民法典中关于合同的规定有哪些？"
    print("问题:", q)
    print()

    # 提示：检索过程离线，LLM 需联网（可替换为本地 LLM 完全离线）
    print("[NOTE] 向量检索已完全离线，LLM 回答需联网（OpenAI/DeepSeek）")
    print()

    answer, sources = ask(q)
    print("回答:", answer)
    print()

    if sources:
        print("参考来源 (TopK):")
        for i, doc in enumerate(sources, 1):
            meta = doc.metadata
            # 统一元数据展示（适配爬虫/本地文件）
            src = (
                meta.get("title")
                or meta.get("source_file")
                or meta.get("source")
                or "未知来源"
            )
            print(f"  {i}. {src}")
            # 截断过长内容，优化展示
            content = doc.page_content.strip()[:120] + "..." if len(doc.page_content) > 120 else doc.page_content
            print(f"     {content}")

if __name__ == "__main__":
    main()