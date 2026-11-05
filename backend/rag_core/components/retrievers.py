"""Load or rebuild Chinese/English retrieval indexes and the shared reranker."""

import logging
import os
import pickle
import jieba
import pandas as pd
import torch
from typing import Optional, Dict, Any, List
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from ..config import (
    CSV_FILE_PATH_ZH, CSV_FILE_PATH_EN, EMBEDDING_MODEL,
    CHROMA_DB_DIR_ZH, CHROMA_DB_DIR_EN, RERANKER_MODEL_PATH,
    BM25_INDEX_PATH_ZH, BM25_INDEX_PATH_EN, OLLAMA_BASE_URL,
)

logger = logging.getLogger(__name__)

def jieba_tokenize(text: str) -> List[str]:
    return list(jieba.cut(text))

def setup_retrievers() -> Optional[Dict[str, Any]]:
    """
    构建包含多语言支持的高级检索器套件。
    """
    logger.info("开始初始化中英文混合检索索引")

    # 检测计算设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f" 🖥️  计算设备: {device.upper()}")

    # 0. 辅助函数定义
    def load_docs_from_csv(file_path, lang_prefix) -> List[Document]:
        if not os.path.exists(file_path):
            logger.info(f"❌ 文件不存在: {file_path}")
            return []
        try:
            logger.info(f" 📂 [Cold Start] 加载原始数据集: {file_path}...")
            # 优化：dtype=str 防止 pandas 将 'id' 识别为 float 导致 fillna 警告
            df = pd.read_csv(
                file_path,
                header=None,
                names=['id', 'title', 'content', 'date', 'URL'],
                dtype=str
            )
            df.fillna('', inplace=True)
            docs = [
                Document(page_content=str(row['content']),
                         metadata={'source_id': f"{lang_prefix}_{row['id']}", 'title': str(row['title']),
                                   'date': str(row['date']), 'URL': str(row['URL']), 'lang': lang_prefix})
                for _, row in df.iterrows() if str(row['content']).strip()
            ]
            return docs
        except Exception as e:
            logger.info(f"❌ 读取 CSV 失败: {e}")
            return []

    # 1. 检查持久化状态，决定是否需要加载 CSV
    def check_chroma_exists(chroma_path):
        """检查 Chroma 持久化数据库中是否已存在 SQLite 文件"""
        sqlite_path = os.path.join(chroma_path, "chroma.sqlite3")
        return os.path.exists(sqlite_path)

    vector_zh_ready = check_chroma_exists(CHROMA_DB_DIR_ZH)
    vector_en_ready = check_chroma_exists(CHROMA_DB_DIR_EN)
    bm25_zh_ready = os.path.exists(BM25_INDEX_PATH_ZH)
    bm25_en_ready = os.path.exists(BM25_INDEX_PATH_EN)

    # 只有当某个组件缺失时，才加载对应的 CSV
    raw_documents_zh = None
    if not vector_zh_ready or not bm25_zh_ready:
        raw_documents_zh = load_docs_from_csv(CSV_FILE_PATH_ZH, 'zh')
        if not raw_documents_zh: return None

    raw_documents_en = None
    if not vector_en_ready or not bm25_en_ready:
        raw_documents_en = load_docs_from_csv(CSV_FILE_PATH_EN, 'en')
        if not raw_documents_en: return None

    if vector_zh_ready and bm25_zh_ready and vector_en_ready and bm25_en_ready:
        logger.info(" ⚡ 全量索引已存在，跳过 CSV 加载 (Hot Start Mode)")

    # 2. 初始化 Embeddings
    try:
        embeddings_model = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    except Exception as e:
        logger.info(f"❌ 初始化 Embeddings 失败: {e}")
        return None

    # 3. 构建或加载 VectorStore Retriever (直接使用向量库)
    def build_vector_retriever(chroma_path, collection_name, raw_docs, csv_path, lang_prefix,
                               chunk_strategy="recursive"):
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings_model,
            persist_directory=chroma_path
        )

        try:
            has_data = vectorstore._collection.count() > 0
        except Exception:
            has_data = False

        if has_data:
            logger.info(f" 🟢 [{collection_name}] 发现本地向量索引，加载完成。")
        else:
            if not raw_docs:
                # A SQLite file can exist even when this collection has no documents.
                raw_docs = load_docs_from_csv(csv_path, lang_prefix)
            if not raw_docs:
                logger.info(f" ❌ [{collection_name}] 索引为空且未提供原始文档，无法构建。")
                return None
            logger.info(f" 🟡 [{collection_name}] 未发现完整向量索引，开始文档分块与全量编码 (耗时操作)...")

            if chunk_strategy == "tiktoken":
                splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                    model_name="gpt-4o-mini", chunk_size=1200, chunk_overlap=100
                )
            else:
                splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100)

            split_docs = splitter.split_documents(raw_docs)
            vectorstore.add_documents(split_docs)
            logger.info(f"    -> 完成写入 {len(split_docs)} 块文档。")

        return vectorstore.as_retriever(search_kwargs={"k": 25})

    vector_retriever_zh = build_vector_retriever(
        CHROMA_DB_DIR_ZH, "tech_news_zh", raw_documents_zh,
        CSV_FILE_PATH_ZH, "zh", chunk_strategy="recursive"
    )

    vector_retriever_en = build_vector_retriever(
        CHROMA_DB_DIR_EN, "tech_news_en", raw_documents_en,
        CSV_FILE_PATH_EN, "en", chunk_strategy="tiktoken"
    )

    if vector_retriever_zh is None or vector_retriever_en is None:
        return None

    # 4. 构建或加载 BM25 检索器 (Pickle 持久化)
    def get_or_create_bm25(docs, save_path, lang_tag, csv_path):
        if os.path.exists(save_path):
            logger.info(f" 🟢 [{lang_tag}] 从缓存加载 BM25 索引: {save_path}")
            try:
                with open(save_path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.info(f" ⚠️ 加载 BM25 缓存失败 ({e})，尝试重新构建...")

        if not docs:
            # Recover from an unreadable cache even when hot-start skipped the CSV.
            docs = load_docs_from_csv(csv_path, lang_tag.lower())
        if not docs:
            logger.info(f" ❌ [{lang_tag}] BM25 缓存缺失且未提供原始文档，无法构建。")
            return None

        logger.info(f" 📝 [{lang_tag}] 构建并保存 BM25 关键词索引...")

        if lang_tag == "ZH":
            retriever = BM25Retriever.from_documents(docs, preprocess_func=jieba_tokenize)
        else:
            retriever = BM25Retriever.from_documents(docs)

        retriever.k = 25

        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(retriever, f)
        return retriever

    bm25_retriever_zh = get_or_create_bm25(
        raw_documents_zh, BM25_INDEX_PATH_ZH, "ZH", CSV_FILE_PATH_ZH
    )
    bm25_retriever_en = get_or_create_bm25(
        raw_documents_en, BM25_INDEX_PATH_EN, "EN", CSV_FILE_PATH_EN
    )

    if not bm25_retriever_zh or not bm25_retriever_en:
        return None

    # 5. 初始化重排器 (Cross-Encoder)
    logger.info(f" 🎯 加载重排模型: {os.path.basename(RERANKER_MODEL_PATH)} (Device: {device})...")
    try:
        cross_encoder_model = HuggingFaceCrossEncoder(
            model_name=RERANKER_MODEL_PATH,
            model_kwargs={'device': device}
        )
        compressor = CrossEncoderReranker(model=cross_encoder_model, top_n=5)
    except Exception as e:
        logger.info(f"❌ 加载 Reranker 失败 (请检查路径或CUDA状态): {e}")
        return None

    logger.info(" ✅ 检索套件初始化完成！\n")

    return {
        "vector_retriever_zh": vector_retriever_zh,
        "vector_retriever_en": vector_retriever_en,
        "pdr_retriever_zh": vector_retriever_zh,
        "pdr_retriever_en": vector_retriever_en,
        "bm25_retriever_zh": bm25_retriever_zh,
        "bm25_retriever_en": bm25_retriever_en,
        "default_hybrid": EnsembleRetriever(
            retrievers=[vector_retriever_zh, bm25_retriever_zh],
            weights=[0.5, 0.5],
            search_type="rrf"
        ),
        "compressor": compressor
    }
