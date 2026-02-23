================================================================================
  AI RAG 法律文档 Demo（Python + LangChain + Chroma）
================================================================================

功能概览
--------
1. 爬取法律文档：从国家法律法规数据库抓取条文并写入向量库
2. 上传 PDF/Word：本地文件 → 切片 → 生成 Embedding → 存入 Chroma
3. 检索 TopK：根据问题检索最相关的条文片段
4. 大模型回答：基于检索结果由 LLM 生成答案（RAG）

环境要求
--------
- Python 3.10+
- 建议使用虚拟环境 .venv

安装步骤
--------
1. 创建并激活虚拟环境（在项目根目录执行，.venv 需自行创建）：

   Windows (PowerShell):
     python -m venv .venv
     .venv\Scripts\Activate.ps1

   Windows (CMD):
     python -m venv .venv
     .venv\Scripts\activate.bat

   Linux / macOS:
     python3 -m venv .venv
     source .venv/bin/activate

2. 安装依赖：

   pip install -r requirements.txt


   首次运行会下载 Embedding 模型（all-MiniLM-L6-v2），可能较慢。若出现 SSL/连接
   Hugging Face 失败，可在 .env 中设置 EMBEDDING_LOCAL_FILES_ONLY=1 并确保模型已
   缓存，或使用镜像：HF_ENDPOINT=https://hf-mirror.com

3. 大模型问答（支持 OpenAI 与 DeepSeek）：
   - 复制 .env.example 为 .env，按需填写：
   - 使用 OpenAI：LLM_PROVIDER=openai，并设置 OPENAI_API_KEY
   - 使用 DeepSeek：LLM_PROVIDER=deepseek，并设置 DEEPSEEK_API_KEY
   - 也可通过系统环境变量设置上述变量

使用说明
--------
一、爬取法律文档（写入 Chroma）
    python crawler-law.py

二、上传 PDF/Word 并入库（切片 + Embedding）
    python ingest_docs.py  <文件或目录> [更多文件/目录...]
    示例：
      python ingest_docs.py  law.pdf
      python ingest_docs.py  law.pdf  contract.docx
      python ingest_docs.py  ./law_files
    支持 .pdf / .docx / .doc，与爬虫数据共用同一向量库。

三、检索 + 大模型回答（RAG）
    python rag_query.py
    默认问题为「合同」。
    自定义问题：
      python rag_query.py  刑法中盗窃罪如何认定？

配置说明（config.py 与 .env）
----------------------------
向量与检索：
- CHROMA_PATH: 向量库持久化目录（默认 ./chroma_laws_db）
- COLLECTION_NAME: 集合名（默认 laws_collection）
- EMBEDDING_MODEL: 本地 Embedding 模型（默认 all-MiniLM-L6-v2，来自 langchain-huggingface）
- EMBEDDING_LOCAL_FILES_ONLY: 为 True 时仅用本地缓存，不访问 Hugging Face（可设于 .env）
- CHUNK_SIZE / CHUNK_OVERLAP: 文档切片大小与重叠
- TOP_K: 检索返回条数

大模型（OpenAI / DeepSeek）：
- LLM_PROVIDER: openai 或 deepseek（可在 .env 中写 LLM_PROVIDER=deepseek）
- OpenAI: OPENAI_API_KEY 必填；可选 OPENAI_API_BASE、OPENAI_MODEL
- DeepSeek: DEEPSEEK_API_KEY 必填；可选 DEEPSEEK_BASE_URL、DEEPSEEK_MODEL（默认 deepseek-chat）

项目结构（简要）
----------------
  config.py        统一配置（Chroma、切片、TopK、LLM 等）
  crawler-law.py   爬取法律网页（Playwright）→ Chroma
  ingest_docs.py   PDF/Word 上传 → 切片 → Embedding → Chroma
  rag_query.py     检索 TopK → 大模型回答（OpenAI/DeepSeek）
  requirements.txt 依赖列表（含 playwright）
  .env.example     环境变量示例，复制为 .env 后填写 API Key
  README.txt       本说明
  .venv/           虚拟环境（需自行创建）
  chroma_laws_db/  向量库数据（运行后生成）

================================================================================
