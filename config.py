# 统一配置：与 crawler-law 共用同一向量库
CHROMA_PATH = "./chroma_laws_db"
COLLECTION_NAME = "laws_collection"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# 仅使用本地已缓存模型，不访问 Hugging Face（网络受限或 SSL 报错时在 .env 设 EMBEDDING_LOCAL_FILES_ONLY=1）
EMBEDDING_LOCAL_FILES_ONLY = False

# 切片参数
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# 检索
TOP_K = 10

# 大模型：openai | deepseek（可通过环境变量 LLM_PROVIDER 覆盖）
LLM_PROVIDER = "openai"
OPENAI_MODEL = "gpt-3.5-turbo"
OPENAI_BASE_URL = None  # 默认 None，使用 OpenAI 官方

# DeepSeek（OpenAI 兼容 API）
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"
