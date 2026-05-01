# rag.py
#
# 嵌入模型加载方式（任选其一）：
# 1) 在线：依赖 HuggingFace Hub。国内建议在 .env 设置 HF_ENDPOINT=https://hf-mirror.com
#    （不设置 HF_ENDPOINT 且未设置 RAG_USE_OFFICIAL_HF=1 时，本模块会默认使用 hf-mirror）
# 2) 离线（推荐网络不稳定时）：先用镜像把模型下载到本机目录，再在 .env 设置
#    RAG_EMBEDDING_MODEL_PATH=D:\path\to\all-MiniLM-L6-v2
#    然后 HuggingFaceEmbeddings 会只从该目录加载，避免访问 huggingface.co。
#
# 手动下载示例（PowerShell；新版 huggingface_hub 已用 hf 替代 huggingface-cli）：
#   $env:HF_ENDPOINT = "https://hf-mirror.com"
#   .\.venv\Scripts\hf.exe download sentence-transformers/all-MiniLM-L6-v2 --local-dir .\models\all-MiniLM-L6-v2
# 若 hf 在 PATH 中，可直接： hf download sentence-transformers/all-MiniLM-L6-v2 --local-dir .\models\all-MiniLM-L6-v2
# 下载完成后把 RAG_EMBEDDING_MODEL_PATH 指到该目录的绝对路径即可。
#
# 国内常用 Hub 镜像： https://hf-mirror.com
#
# 检索量可调（.env）：
#   RAG_TOP_K：返回几条最相似片段，默认 8，范围 1～50（仅影响检索，不改索引）
#   RAG_CHUNK_SIZE / RAG_CHUNK_OVERLAP：建索引切块大小与重叠；修改后会因指纹变化自动重建索引

import hashlib
import json
import os


def _bootstrap_huggingface_hub_env() -> None:
    """
    在导入 HuggingFace 相关依赖之前配置 Hub 端点，避免请求 huggingface.co 长时间超时。

    执行顺序：
    1. 尝试 load_dotenv()，使 .env 中的 HF_ENDPOINT 等变量生效。
    2. 若已设置 HF_ENDPOINT，则写入进程环境（供 huggingface_hub 使用）。
    3. 若未设置且未显式声明使用官方源（RAG_USE_OFFICIAL_HF），则默认使用 hf-mirror 端点。

    说明:
        需要始终使用官方 Hub 时，在 .env 中设置 RAG_USE_OFFICIAL_HF=1，且不要设置 HF_ENDPOINT。
    """
    try:
        from dotenv import load_dotenv  # pyright: ignore[reportMissingImports]

        load_dotenv()
    except ImportError:
        pass

    ep = (os.getenv("HF_ENDPOINT") or "").strip().rstrip("/")
    if ep:
        os.environ["HF_ENDPOINT"] = ep
        return

    if (os.getenv("RAG_USE_OFFICIAL_HF") or "").strip().lower() in ("1", "true", "yes"):
        return

    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


_bootstrap_huggingface_hub_env()

from langchain_community.vectorstores import FAISS  # pyright: ignore[reportMissingImports]
from langchain_huggingface import HuggingFaceEmbeddings  # pyright: ignore[reportMissingImports]
from langchain_text_splitters import CharacterTextSplitter  # pyright: ignore[reportMissingImports]
from langchain_community.document_loaders import TextLoader  # pyright: ignore[reportMissingImports]


SUPPORTED_EXTENSIONS = (".txt", ".md", ".pdf", ".docx")


def _fingerprint_sources(data_path: str) -> str:
    """
    根据知识库目录下支持格式文件的相对名、文件大小与修改时间生成指纹字符串。

    任意源文件增删改（含内容写入导致 mtime/size 变化）都会使指纹变化，
    从而触发重新切块与向量化；未变化时可直接复用磁盘上的 FAISS 索引。

    参数:
        data_path: 知识库根目录的绝对路径。

    返回:
        用 "|" 拼接的多段描述字符串；目录不存在或无支持文件时返回空字符串。
    """
    if not os.path.isdir(data_path):
        return ""
    parts: list[str] = []
    for name in sorted(
        f
        for f in os.listdir(data_path)
        if os.path.isfile(os.path.join(data_path, f))
        and f.lower().endswith(SUPPORTED_EXTENSIONS)
    ):
        full = os.path.join(data_path, name)
        try:
            st = os.stat(full)
            parts.append(f"{name}:{st.st_size}:{int(st.st_mtime)}")
        except OSError:
            parts.append(f"{name}:missing")
    return "|".join(parts)


def _read_cached_fingerprint(index_dir: str) -> str | None:
    """
    读取上次构建索引时写入的指纹文件内容。

    参数:
        index_dir: 存放 FAISS 与元数据的目录。

    返回:
        文件中的指纹字符串；文件不存在或读取失败时返回 None。
    """
    fp_path = os.path.join(index_dir, "source_fingerprint.txt")
    if not os.path.isfile(fp_path):
        return None
    try:
        with open(fp_path, encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return None


def _write_fingerprint(index_dir: str, fingerprint: str) -> None:
    """
    将当前数据源指纹写入索引目录，供下次启动时比对。

    参数:
        index_dir: 索引目录。
        fingerprint: 与 _fingerprint_txt_sources 一致的指纹字符串。
    """
    os.makedirs(index_dir, exist_ok=True)
    fp_path = os.path.join(index_dir, "source_fingerprint.txt")
    with open(fp_path, "w", encoding="utf-8") as f:
        f.write(fingerprint)


def _faiss_artifacts_present(index_dir: str) -> bool:
    """
    判断目录中是否包含 LangChain FAISS.save_local 写入的典型文件。

    参数:
        index_dir: 索引目录。

    返回:
        同时存在 index.faiss 与 index.pkl 时为 True，否则为 False。
    """
    return os.path.isfile(os.path.join(index_dir, "index.faiss")) and os.path.isfile(
        os.path.join(index_dir, "index.pkl")
    )


def _build_loader(file_path: str):
    """
    根据文件后缀构建对应的 LangChain Loader。
    """
    suffix = os.path.splitext(file_path)[1].lower()
    if suffix == ".txt":
        return TextLoader(file_path, encoding="utf-8")
    if suffix == ".md":
        from langchain_community.document_loaders import (
            UnstructuredMarkdownLoader,  # pyright: ignore[reportMissingImports]
        )

        return UnstructuredMarkdownLoader(file_path)
    if suffix == ".pdf":
        from langchain_community.document_loaders import (
            PyPDFLoader,  # pyright: ignore[reportMissingImports]
        )

        return PyPDFLoader(file_path)
    if suffix == ".docx":
        from langchain_community.document_loaders import (
            Docx2txtLoader,  # pyright: ignore[reportMissingImports]
        )

        return Docx2txtLoader(file_path)
    raise ValueError(f"不支持的文件格式: {suffix}")


def _load_documents(data_path: str) -> list:
    """
    从目录中加载所有支持格式文件为 LangChain Document 列表。

    参数:
        data_path: 知识库根目录。

    返回:
        由各类 Loader 合并得到的文档列表；无匹配文件时为空列表。
    """
    documents: list = []
    for file in sorted(os.listdir(data_path)):
        full = os.path.join(data_path, file)
        if not os.path.isfile(full) or not file.lower().endswith(SUPPORTED_EXTENSIONS):
            continue
        loader = _build_loader(full)
        documents.extend(loader.load())
    return documents


def _resolve_embedding_model_name() -> str:
    """
    决定传给 HuggingFaceEmbeddings 的 model_name：Hub 上的模型 ID，或本机已下载目录的绝对路径。

    若环境变量 RAG_EMBEDDING_MODEL_PATH 指向一个存在的目录，则使用该目录（适合离线）；
    否则使用默认 Hub 模型 sentence-transformers/all-MiniLM-L6-v2。

    返回:
        模型 ID 字符串，或本机目录绝对路径。
    """
    raw = (os.getenv("RAG_EMBEDDING_MODEL_PATH") or "").strip().strip('"')
    if not raw:
        return "sentence-transformers/all-MiniLM-L6-v2"
    abs_path = os.path.abspath(raw)
    if os.path.isdir(abs_path):
        return abs_path
    return "sentence-transformers/all-MiniLM-L6-v2"


def _split_documents(
    documents: list, chunk_size: int = 300, chunk_overlap: int = 50
) -> list:
    """
    使用固定字符窗口对文档列表做切块。

    参数:
        documents: LangChain Document 列表。
        chunk_size: 每块最大字符数。
        chunk_overlap: 相邻块重叠字符数。

    返回:
        切块后的 Document 列表。
    """
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)


class RAG:
    """
    基于本地文档 + FAISS 的简单检索：首次构建向量索引并落盘，之后优先从磁盘加载。
    """

    def __init__(self, data_path: str = "docs") -> None:
        """
        初始化 RAG：优先加载缓存索引；指纹或文件缺失时重新加载文档、切块、向量化并保存。

        参数:
            data_path: 存放知识文件的目录（支持 txt/md/pdf/docx），相对路径则相对于进程当前工作目录。
        """
        self.data_path = os.path.abspath(data_path)
        # 索引与知识库同级，避免把向量文件混进 docs 里
        self.index_dir = os.path.join(os.path.dirname(self.data_path), ".rag_faiss_index")

        emb_name = _resolve_embedding_model_name()
        raw_emb_path = (os.getenv("RAG_EMBEDDING_MODEL_PATH") or "").strip().strip('"')
        if raw_emb_path and emb_name == "sentence-transformers/all-MiniLM-L6-v2":
            print(
                "RAG：警告 RAG_EMBEDDING_MODEL_PATH 不是有效目录，已忽略并改用 Hub 模型：",
                raw_emb_path,
            )

        try:
            chunk_size = int((os.getenv("RAG_CHUNK_SIZE") or "300").strip())
        except ValueError:
            chunk_size = 300
        chunk_size = max(128, min(chunk_size, 8000))
        try:
            chunk_overlap = int((os.getenv("RAG_CHUNK_OVERLAP") or "50").strip())
        except ValueError:
            chunk_overlap = 50
        chunk_overlap = max(0, min(chunk_overlap, max(0, chunk_size - 32)))
        source_fp = _fingerprint_sources(self.data_path)
        emb_fp = hashlib.sha256(emb_name.encode("utf-8")).hexdigest()[:24]
        # 切块参数与嵌入模型均参与缓存键，避免换模型或换 chunk 仍误用旧索引
        cache_key = f"{source_fp}||split:{chunk_size}:{chunk_overlap}||emb:{emb_fp}"

        # 1️⃣ 加载 embedding 模型（加载索引时也需要同一套 embeddings）
        hub_ep = (os.environ.get("HF_ENDPOINT") or "").strip()
        print(
            "RAG：HuggingFace Hub 端点 = "
            + (hub_ep if hub_ep else "（未设置，将使用 huggingface_hub 默认 https://huggingface.co）")
        )
        use_local = os.path.isdir(emb_name)
        if use_local:
            print("RAG：正在从本机目录加载嵌入模型（离线）：", emb_name)
            self.embedding = HuggingFaceEmbeddings(
                model_name=emb_name,
                model_kwargs={"local_files_only": True},
            )
        else:
            print(
                "RAG：正在从 Hub 加载嵌入模型（在线）：",
                emb_name,
                "（首次下载可能较慢、终端可能长时间无新输出）",
            )
            self.embedding = HuggingFaceEmbeddings(model_name=emb_name)
        cached_fp = _read_cached_fingerprint(self.index_dir)

        if (
            cached_fp is not None
            and cached_fp == cache_key
            and _faiss_artifacts_present(self.index_dir)
        ):
            print("RAG：从磁盘加载缓存索引（跳过切块与向量化）...")
            self.vectorstore = FAISS.load_local(
                self.index_dir,
                self.embedding,
                allow_dangerous_deserialization=True,
            )
            print("RAG：缓存索引加载完成")
            return

        print("RAG：未命中缓存或源文件已变更，正在重新构建索引...")
        documents = _load_documents(self.data_path)
        docs = _split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        if not docs:
            raise ValueError(
                "知识库目录中没有可用文件，支持格式为 txt/md/pdf/docx，"
                f"无法构建向量索引: {self.data_path}"
            )

        self.vectorstore = FAISS.from_documents(docs, self.embedding)
        os.makedirs(self.index_dir, exist_ok=True)
        self.vectorstore.save_local(self.index_dir)
        _write_fingerprint(self.index_dir, cache_key)
        meta_path = os.path.join(self.index_dir, "split_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "embedding_model": emb_name,
                },
                f,
                ensure_ascii=False,
            )
        print("RAG：索引已构建并保存到", self.index_dir)

    def search(self, query: str, k: int | None = None) -> str:
        """
        在向量库中做相似度检索，返回拼接后的文本片段。

        参数:
            query: 用户查询语句。
            k: 返回最相近的文档条数；为 None 时读取环境变量 RAG_TOP_K（默认 8，范围 1～50）。

        返回:
            多条检索结果 page_content 用换行拼接的字符串。
        """
        if k is None:
            try:
                k = int((os.getenv("RAG_TOP_K") or "8").strip())
            except ValueError:
                k = 8
            k = max(1, min(k, 50))
        results = self.vectorstore.similarity_search(query, k=k)
        return "\n".join(doc.page_content for doc in results)
