"""
上传 PDF/Word → 切片 → 生成 Embedding → 增量存入 Chroma
修复 models.json 重复下载 + HuggingFace 镜像源配置
"""
import os
from pathlib import Path
import warnings

# ========== 1. 强制配置 HuggingFace 国内镜像（核心修复） ==========
# 设置 HF 镜像源（优先环境变量，无则默认镜像）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# 强制使用镜像源下载
os.environ["HUGGINGFACE_HUB_BASE_URL"] = "https://hf-mirror.com"
# 其他离线/缓存配置（保留）
os.environ["TRANSFORMERS_OFFLINE"] = "0"  # 首次下载需设为0，后续可改1
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "./model_cache"
os.environ["TRANSFORMERS_CACHE"] = "./model_cache"
os.environ["HF_HOME"] = "./model_cache"

# 关闭无关警告
warnings.filterwarnings("ignore")

# ========== 2. 导入依赖（需在镜像配置后） ==========
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

# ========== 3. 导入配置（保持原有） ==========
from config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

# ========== 4. 预加载模型（适配镜像源） ==========
def preload_embedding_model(force_download=False):
    """
    预加载 Embedding 模型（通过 HF 镜像源下载，修复参数错误）
    """
    cache_dir = os.path.abspath("./model_cache")
    model_cache_path = os.path.join(cache_dir, "sentence_transformers", EMBEDDING_MODEL.replace("/", "_"))

    if os.path.exists(model_cache_path) and not force_download:
        print(f"[OK] 模型 {EMBEDDING_MODEL} 已缓存，跳过下载")
        # 后续运行可切换为完全离线
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        return

    print(f"[...] 首次运行，通过镜像源下载模型 {EMBEDDING_MODEL}...")
    try:
        # 修复：移除 local_files_only 参数（SentenceTransformer 不支持）
        # 如需控制本地文件，通过 TRANSFORMERS_OFFLINE 环境变量实现
        model = SentenceTransformer(
            EMBEDDING_MODEL,
            cache_folder=cache_dir,  # 仅保留支持的参数
            trust_remote_code=True  # 必要时开启，兼容部分模型
        )
        model.save(model_cache_path)
        print(f"[OK] 模型下载完成，缓存路径：{model_cache_path}")
        # 下载完成后切换为离线模式
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
    except Exception as e:
        raise Exception(f"[ERR] 模型下载失败：{str(e)}\n建议检查：\n1. 网络是否能访问 https://hf-mirror.com\n2. 代理配置是否生效")
# 初始化预加载
preload_embedding_model()

# ========== 后续函数：优先本地缓存路径，避免重复请求 modules.json ==========
def _local_embedding_model_path():
    """若 model_cache 中已有该模型，返回本地路径。"""
    cache_dir = os.path.abspath("./model_cache")
    local_name = EMBEDDING_MODEL.replace("/", "_")
    path = os.path.join(cache_dir, "sentence_transformers", local_name)
    return path if os.path.isdir(path) else None

def _embedding_model_kwargs():
    return {"device": "cpu"}

def get_langchain_embedding():
    """获取 LangChain Embedding，优先本地路径。"""
    local_path = _local_embedding_model_path()
    if local_path:
        return HuggingFaceEmbeddings(
            model_name=local_path,
            model_kwargs=_embedding_model_kwargs(),
            encode_kwargs={"normalize_embeddings": True},
        )
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        cache_folder=os.path.abspath("./model_cache"),
        model_kwargs=_embedding_model_kwargs(),
        encode_kwargs={"normalize_embeddings": True},
    )

def get_chroma_embedding_function():
    """获取 Chroma Embedding 函数，优先本地路径。"""
    local_path = _local_embedding_model_path()
    if local_path:
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=local_path,
            device="cpu",
        )
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
        device="cpu",
        cache_folder=os.path.abspath("./model_cache"),
    )

def load_pdf(path: str):
    loader = PyPDFLoader(path)
    return loader.load()

def load_docx(path: str):
    loader = Docx2txtLoader(path)
    return loader.load()

def load_file(path: str):
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix in (".docx", ".doc"):
        return load_docx(path)
    raise ValueError(f"不支持的文件类型: {suffix}")

def split_docs(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """切片"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", "；", " ", ""],
    )
    return splitter.split_documents(docs)

def get_chroma_collection(chroma_path=CHROMA_PATH, collection_name=COLLECTION_NAME):
    """获取/创建 Chroma 集合（完全离线）"""
    # 初始化 Chroma 客户端（禁用远端检查）
    client = chromadb.PersistentClient(path=chroma_path)
    chroma_ef = get_chroma_embedding_function()
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=chroma_ef
    )
    return client, collection

def generate_unique_ids(collection, num_chunks):
    """生成唯一的文档ID（避免重复覆盖）"""
    existing_count = collection.count()
    return [f"doc_chunk_{existing_count + i + 1}" for i in range(num_chunks)]

def check_duplicate_file(collection, file_path):
    """检查文件是否已导入（通过source_file元数据）"""
    try:
        results = collection.get(
            where={"source_file": file_path},
            limit=1
        )
        return len(results["ids"]) > 0
    except Exception:
        return False

def ingest_files_incrementally(file_paths: list[str], chroma_path=CHROMA_PATH, collection_name=COLLECTION_NAME):
    """增量导入多个 PDF/Word 文件到 Chroma（完全离线）"""
    client, collection = get_chroma_collection(chroma_path, collection_name)

    all_chunks = []
    processed_files = []

    for path in file_paths:
        file_path = os.path.abspath(path)
        if check_duplicate_file(collection, file_path):
            print(f"[SKIP] 跳过已导入文件: {file_path}")
            continue

        try:
            docs = load_file(file_path)
            for d in docs:
                d.metadata["source_file"] = file_path
                d.metadata["file_name"] = Path(file_path).name
            chunks = split_docs(docs)
            all_chunks.extend(chunks)
            processed_files.append(file_path)
            print(f"[OK] 已加载并切片: {file_path} (生成 {len(chunks)} 个切片)")
        except Exception as e:
            print(f"[ERR] 处理失败 {file_path}: {str(e)}")

    if not all_chunks:
        print("[--] 没有新增的文档切片需要导入")
        return

    chunk_ids = generate_unique_ids(collection, len(all_chunks))
    print(f"[...] 待增量导入切片总数: {len(all_chunks)}")

    chunk_texts = [chunk.page_content for chunk in all_chunks]
    chunk_metadatas = [chunk.metadata for chunk in all_chunks]

    collection.add(
        ids=chunk_ids,
        documents=chunk_texts,
        metadatas=chunk_metadatas
    )

    print(f"\n[OK] 增量导入完成！")
    print(f"     新增处理文件: {len(processed_files)} 个")
    print(f"     新增切片数量: {len(all_chunks)} 个")
    print(f"     向量库总切片数: {collection.count()} 个")
    print(f"     向量库路径: {chroma_path}")

def ingest_folder_incrementally(folder: str, extensions=(".pdf", ".docx", ".doc")):
    """增量扫描文件夹内所有 PDF/Word 并入库（完全离线）"""
    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        raise NotADirectoryError(f"目录不存在: {folder}")

    file_paths = []
    for root, _, files in os.walk(folder):
        for f in files:
            if Path(f).suffix.lower() in extensions:
                file_paths.append(os.path.join(root, f))

    if not file_paths:
        print(f"[--] 目录下未找到 {extensions} 文件: {folder}")
        return

    ingest_files_incrementally(file_paths)

# 兼容原有函数名
ingest_files = ingest_files_incrementally
ingest_folder = ingest_folder_incrementally

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python ingest_docs.py <文件或目录> [更多文件/目录...]")
        print("示例: python ingest_docs.py doc.pdf  doc.docx  ./docs")
        sys.exit(1)

    paths = []
    for p in sys.argv[1:]:
        abs_path = os.path.abspath(p)
        if os.path.isfile(abs_path):
            paths.append(abs_path)
        elif os.path.isdir(abs_path):
            ingest_folder_incrementally(abs_path)
        else:
            print(f"[--] 忽略不存在的路径: {p}")

    if paths:
        ingest_files_incrementally(paths)