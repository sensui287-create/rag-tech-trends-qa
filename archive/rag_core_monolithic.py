# 无统计图表
# 1. 导入与环境配置
import os
import shutil
# import traceback
# import json
from typing import List, Optional, TypedDict, Literal
import pandas as pd
import dotenv
import torch
from langchain_core.stores import InMemoryStore
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Tongyi as ChatTongyi
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_tavily import TavilySearch
from langchain_classic.retrievers import (
    EnsembleRetriever,
    ParentDocumentRetriever,
)
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langgraph.graph import StateGraph, END
from functools import partial

# 2. 常量与配置
# 请根据实际文件位置调整路径
CSV_FILE_PATH_ZH = './cleaned_tech_news_robust.csv'
CSV_FILE_PATH_EN = './hacker_news_cleaned.csv'
EMBEDDING_MODEL = "qwen3-embedding:4b"
LLM_MODEL = "qwen3-max"
RERANKER_MODEL_PATH = r"./models/bge-reranker-large"
MAX_RESEARCH_ITERATIONS = 5

# 3. Pydantic 数据模型定义 ===
# 路由决策模型
class RouteQuery(BaseModel):
    """将用户查询路由到最合适的路径"""
    step: Literal["simple_answer", "complex_research"] = Field(
        description="下一步行动。'simple_answer' 用于问候、闲聊或LLM无需搜索即可回答的通用知识。"
                    "'complex_research' 用于需要检索最新新闻、具体技术细节或深入分析的问题。"
    )
    simple_response: str = Field(
        default="",
        description="如果 step 是 'simple_answer'，在这里直接提供回复内容。如果是 'complex_research'，留空。"
    )

class BilingualQuery(BaseModel):
    chinese_query: str = Field(description="优化后的中文搜索查询")
    english_query: str = Field(description="翻译并优化后的英文搜索查询")
    tech_entities: List[str] = Field(description="提取的关键技术实体、产品型号、公司名称或专有名词，例如 ['Llama 3', 'NVIDIA', 'RAG']")
    time_range: Optional[str] = Field(description="如果查询包含特定的时间限制（如'上周'、'2024年'），提取出时间描述，否则为 null。")

class RetrievalDecision(BaseModel):
    """检索主管的决策输出模型"""
    strategy: Literal["vector_search", "keyword_search", "hybrid_search", "web_search"] = Field(
        description="选择的检索策略。必须是 'vector_search', 'keyword_search', 'hybrid_search', 或 'web_search' 之一。"
    )
    justification: str = Field(description="做出此选择的理由。")

class Step(BaseModel):
    """研究计划中的一个独立步骤"""
    sub_question: str = Field(description="为这一步需要回答的具体、可操作的子问题。")
    justification: str = Field(description="简要解释为什么这一步对于回答最终问题是必要的。")
    keywords: List[str] = Field(description="用于增强检索效果的关键术语列表。")

class Plan(BaseModel):
    """完整的、多步骤的研究计划"""
    steps: List[Step] = Field(description="一个详细的、多步骤的计划，用于回答用户的复杂查询。")

class PastStep(TypedDict):
    """存储已完成步骤的结果"""
    step_index: int
    sub_question: str
    rewritten_query: str
    retrieved_docs: List[Document]
    distilled_context: str
    summary: str

class AgentState(TypedDict):
    """贯穿整个Graph的核心状态对象"""
    original_question: str
    router_decision: str
    plan: Plan
    past_steps: List[PastStep]
    current_step_index: int
    rewritten_query: str
    retrieval_decision: RetrievalDecision
    retrieved_docs: List[Document]
    distilled_context: str
    final_answer: str
    # 为最终流式生成准备的字段
    context_for_final_answer: Optional[str]
    metadata_for_final_answer: Optional[dict]

class PolicyDecision(BaseModel):
    """策略Agent的决策输出模型"""
    next_action: str = Field(description="决策结果，必须是 'CONTINUE' 或 'FINISH'。")
    justification: str = Field(description="做出此决策的理由。")

# 4. 辅助函数与工具
def setup_retrievers(
        csv_file_path_zh: str,
        csv_file_path_en: str,
) -> Optional[dict]:
    """构建包含多语言支持的高级检索器套件。"""
    print(f"\n--- 开始构建多语言检索器 ---")

    # 1. 加载中文数据
    print(" 加载中文数据集...")
    try:
        df_zh = pd.read_csv(csv_file_path_zh, header=None, names=['id', 'title', 'content', 'date', 'URL'])
        df_zh.fillna('', inplace=True)
        parent_documents_zh = [
            Document(page_content=str(row['content']),
                     metadata={'source_id': 'zh_' + str(row['id']), 'title': str(row['title']),
                               'date': str(row['date']), 'URL': str(row['URL']), 'lang': 'zh'})
            for _, row in df_zh.iterrows() if str(row['content']).strip()
        ]
        print(f" -> 中文文档数: {len(parent_documents_zh)}")
    except Exception as e:
        print(f"❌ 读取中文CSV失败: {e}")
        return None

    # 2. 加载英文数据
    print(" 加载英文数据集...")
    try:
        df_en = pd.read_csv(csv_file_path_en, header=None, names=['id', 'title', 'content', 'date', 'URL'])
        df_en.fillna('', inplace=True)
        parent_documents_en = [
            Document(page_content=str(row['content']),
                     metadata={'source_id': 'en_' + str(row['id']), 'title': str(row['title']),
                               'date': str(row['date']), 'URL': str(row['URL']), 'lang': 'en'})
            for _, row in df_en.iterrows() if str(row['content']).strip()
        ]
        print(f" -> 英文文档数: {len(parent_documents_en)}")
    except Exception as e:
        print(f"❌ 读取英文CSV失败: {e}")
        return None

    # 3. 初始化嵌入模型
    print(" 初始化嵌入模型...")
    try:
        embeddings_model = OllamaEmbeddings(model=EMBEDDING_MODEL)
    except Exception as e:
        print(f"❌ 初始化 Ollama Embeddings 失败: {e}")
        return None

    # 4. 构建向量库 (ParentDocumentRetriever)
    # 中文库
    CHROMA_DB_DIR_ZH = "./chroma_db_ollama_qwen3_zh"
    if os.path.exists(CHROMA_DB_DIR_ZH): shutil.rmtree(CHROMA_DB_DIR_ZH)
    vectorstore_zh = Chroma(collection_name="tech_news_zh", embedding_function=embeddings_model,
                            persist_directory=CHROMA_DB_DIR_ZH)
    pdr_retriever_zh = ParentDocumentRetriever(
        vectorstore=vectorstore_zh,
        docstore=InMemoryStore(),
        child_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    )
    pdr_retriever_zh.add_documents(parent_documents_zh, ids=None)

    # 英文库
    CHROMA_DB_DIR_EN = "./chroma_db_ollama_qwen3_en"
    if os.path.exists(CHROMA_DB_DIR_EN): shutil.rmtree(CHROMA_DB_DIR_EN)
    vectorstore_en = Chroma(collection_name="tech_news_en", embedding_function=embeddings_model,
                            persist_directory=CHROMA_DB_DIR_EN)
    pdr_retriever_en = ParentDocumentRetriever(
        vectorstore=vectorstore_en,
        docstore=InMemoryStore(),
        child_splitter=RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4o",  # 英文语料使用 Token 计数效果更佳
            chunk_size=500,
            chunk_overlap=50
        )
    )
    pdr_retriever_en.add_documents(parent_documents_en, ids=None)
    print("   - ParentDocumentRetriever 构建完成。")

    # 5. 构建关键词检索器 (BM25)
    bm25_retriever_zh = BM25Retriever.from_documents(parent_documents_zh)
    bm25_retriever_zh.k = 3
    bm25_retriever_en = BM25Retriever.from_documents(parent_documents_en)
    bm25_retriever_en.k = 3
    print(" BM25Retriever 构建完成。")

    # 6. 混合检索集成 (Ensemble)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[pdr_retriever_zh, pdr_retriever_en, bm25_retriever_zh, bm25_retriever_en],
        weights=[0.4, 0.4, 0.1, 0.1],
        search_type="rrf"
    )

    # 7. 重排器 (Cross Encoder)
    try:
        # 如果下载了模型在本地，使用路径；否则会自动下载
        cross_encoder_model = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL_PATH)
        compressor = CrossEncoderReranker(model=cross_encoder_model, top_n=3)
    except Exception as e:
        print(f"❌ 加载 Reranker 失败: {e}")
        return None

    print("✅ 检索器套件构建完成。")
    return {
        "vector_search": EnsembleRetriever(retrievers=[pdr_retriever_zh, pdr_retriever_en], weights=[0.5, 0.5]),
        "keyword_search": EnsembleRetriever(retrievers=[bm25_retriever_zh, bm25_retriever_en], weights=[0.5, 0.5]),
        "hybrid_search": ensemble_retriever,
        "compressor": compressor
    }

def preprocess_docs_for_final_answer(past_steps: List[PastStep]) -> (str, dict):
    """预处理所有步骤的文档，为每篇唯一文档分配索引，并格式化为上下文。"""
    all_docs = [doc for step in past_steps for doc in step['retrieved_docs']]
    unique_docs_map = {}

    for doc in all_docs:
        # 尝试寻找唯一标识
        if 'source' in doc.metadata and 'http' in doc.metadata['source']:
            key = doc.metadata['source']
        elif 'source_id' in doc.metadata:
            key = doc.metadata['source_id']
        else:
            continue  # 跳过无标识文档

        if key not in unique_docs_map:
            unique_docs_map[key] = doc

    numbered_sources = {i + 1: doc for i, doc in enumerate(unique_docs_map.values())}

    if not numbered_sources:
        return "没有找到相关的上下文信息。", {}

    context_parts = []
    for index, doc in numbered_sources.items():
        if "source" in doc.metadata and "http" in doc.metadata["source"]:
            source_str = f"来源 {index}: 《{doc.metadata.get('title', '网络文章')}》 ({doc.metadata['source']})"
        else:
            source_str = f"来源 {index}: 《{doc.metadata.get('title', '未知')}》 (日期: {doc.metadata.get('date', '未知')})"

        context_parts.append(f"--- {source_str} ---\n内容: {doc.page_content}")

    formatted_context = "\n\n".join(context_parts)
    numbered_metadata = {index: doc.metadata for index, doc in numbered_sources.items()}
    return formatted_context, numbered_metadata

def web_search_function(query: str, tavily_tool: TavilySearch) -> List[Document]:
    """执行网络搜索"""
    print(f" [Web Search] 查询: '{query}'")
    try:
        response_dict = tavily_tool.invoke({"query": query})
        results = response_dict.get('results', [])
        if not isinstance(results, list):
            return []
        return [
            Document(
                page_content=res.get("content", ""),
                metadata={"source": res.get("url", ""), "title": res.get("title", "未知标题")}
            ) for res in results if isinstance(res, dict)
        ]
    except Exception as e:
        print(f"❌ Tavily 搜索错误: {e}")
        return []

def format_docs_for_synthesis(docs: List[Document]) -> str:
    """为提炼节点格式化文档"""
    if not docs: return "没有找到相关的上下文信息。"
    formatted_strings = []
    for i, doc in enumerate(docs):
        title = doc.metadata.get('title', '未知')
        formatted_strings.append(f" 文档 {i + 1} ({title}) ---\n{doc.page_content}")
    return "\n\n".join(formatted_strings)

def get_past_context_str(past_steps: List[PastStep]) -> str:
    """格式化研究历史"""
    if not past_steps: return "还未进行任何研究。"
    return "\n\n".join([f"第 {s['step_index']} 步: {s['sub_question']}\n发现摘要: {s['summary']}" for s in past_steps])

# 5. 专用 Agent 构建函数
# 路由 Agent
def build_router_agent(llm):
    """
    构建路由Agent。
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一名专家级的意图分类器。你的任务是根据问题的复杂度，选择最高效的解决路径。

        路径选择标准:

        1.  SIMPLE_ANSWER (快速检索模式):
            - 定义: 问题具体、明确，只需要进行一次数据库检索即可找到答案，不需要多步推理或复杂的横向对比。
            - 适用场景:
                - 询问具体事实 (如 "Meta Free Transformer的架构细节", "夸克眼镜哪天预售", "Kyber项目何时量产")。
                - 简单的概念解释 (如 "什么是RAG", "介绍一下Transformer")。
            - 决策: 将 `step` 设为 "simple_answer"。即使需要查库，只要一次能查到，就选这个。

        2.  COMPLEX_RESEARCH (深度思考模式):
            - 定义: 问题模糊、庞大，或者需要跨多个时间段、多个实体进行综合分析，往往需要拆解为多个子问题。
            - 适用场景:
                - 趋势分析 (如 "分析2025年AI发展趋势")。
                - 深度对比 (如 "详细对比DeepSeek和ChatGPT在编码能力上的优劣及底层架构差异")。
                - 综述报告 (如 "总结过去一个月所有关于半导体的大事件")。
            - 决策: 将 `step` 设为 "complex_research"。

        输出格式:
        严格遵循JSON格式。
        {{
            "step": "simple_answer" 或 "complex_research",
            "simple_response": ""  <-- 永远留空，因为我们现在改用检索生成，不再由你直接回答
        }}
        {format_instructions}"""),
        ("human", "用户问题: {question}")
    ])

    parser = JsonOutputParser(pydantic_object=RouteQuery)
    prompt = prompt.partial(format_instructions=parser.get_format_instructions())
    return prompt | llm | parser


def build_planner_agent(llm):
    """构建规划Agent"""
    prompt_template = ChatPromptTemplate.from_messages([
        ("system",
         "你是一名顶级的AI查询意图分析师。你的任务是将用户的复杂问题分解为一个结构清晰、逻辑严谨的多步骤研究计划。"
         "专注于问题分解，将数据源选择等策略性任务留给下游Agent。你的输出必须是严格的JSON格式，完全符合以下Pydantic模型。\n{format_instructions}"),
        ("human", "用户问题: {question}")
    ])
    parser = JsonOutputParser(pydantic_object=Plan)
    return prompt_template.partial(format_instructions=parser.get_format_instructions()) | llm | parser

# --- [修改] 重写Agent：集成实体提取能力 ---
def build_universal_query_rewriter_agent(llm):
    """
    构建一个通用的、支持双向翻译的查询重写Agent。
    新增能力：利用LLM进行技术实体提取，替代旧的BERT NER模型。
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个为多语言数据库设计的、专家级的查询优化引擎。

        **核心任务**:
        1.  **查询优化**: 接收用户查询，生成两个优化版本（中文和英文），使其更适合数据库检索。
        2.  **实体提取**: 识别并提取查询中的关键技术实体（如公司名、AI模型、硬件型号、算法名称）和专有名词。
        3.  **时间识别**: 如果查询包含明确的时间范围（如"上周"、"2025年"），将其提取出来。

        **输出格式要求**:
        你的回答必须严格遵循以下JSON格式，不允许包含额外文本。
        ```json
        {{
            "chinese_query": "优化后的中文查询",
            "english_query": "Optimized English query",
            "tech_entities": ["实体1", "实体2"],
            "time_range": "提取的时间描述 或 null"
        }}
        ```

        **示例 1 (中文输入)**:
        - 输入: 子问题="对比DeepSeek和GPT-4在编码上的表现", 关键词="DeepSeek, GPT-4, 编码"
        - 输出:
        ```json
        {{
            "chinese_query": "DeepSeek GPT-4 编码能力 评测 对比",
            "english_query": "DeepSeek vs GPT-4 coding performance benchmark",
            "tech_entities": ["DeepSeek", "GPT-4"],
            "time_range": null
        }}
        ```

        **示例 2 (英文输入)**:
        - 输入: 子问题="What happened to NVIDIA stock last week?", 关键词="NVIDIA, stock"
        - 输出:
        ```json
        {{
            "chinese_query": "英伟达 股价 波动 上周 新闻",
            "english_query": "NVIDIA stock price fluctuations last week news",
            "tech_entities": ["NVIDIA"],
            "time_range": "last week"
        }}
        ```

        请严格遵守以上规则。"""),
        ("human", "用户查询: {user_query}\n\n相关关键词: {keywords}\n\n历史上下文:\n{past_context}")
    ])

    parser = JsonOutputParser(pydantic_object=BilingualQuery)
    return prompt | llm | parser

def build_retrieval_supervisor_agent(llm):
    """构建检索主管Agent，动态选择策略"""
    parser = JsonOutputParser(pydantic_object=RetrievalDecision)
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """你是一名检索策略专家。你的任务是根据用户的查询，决定最佳的检索信息源。

        可用信息源:
        1. 本地数据库 (Local Database)**: 这是一个包含大量科技新闻文章的知识库。当查询涉及分析、背景信息、历史事件或不一定需要实时信息的概念时，应优先使用它。它可以通过 `vector_search`, `keyword_search`, 或 `hybrid_search` 来访问。
        2. 网络搜索 (Web Search)**: 用于获取实时、突发的，以及本地数据库中明确没有的内容。

        决策规则:
        - 优先本地: 对于需要深度分析和背景的问题，即使提到了“近期”，也应该首先考虑使用 `hybrid_search` 从本地数据库获取基础信息。
        - 网络补充: 只有当问题明确指向一个非常具体、实时性极强的事件（如“今天的股价”），或者计划中明确提到需要查找外部资料时，才首选 `web_search`。
        - 综合判断: 如果一个问题既需要背景知识又需要最新动态，`hybrid_search` 通常是最好的起点。

        你的输出必须是严格的JSON格式，完全符合以下Pydantic模型。
        {format_instructions}"""),
        ("human", "重写后的查询: {rewritten_query}")
    ])
    prompt = prompt_template.partial(format_instructions=parser.get_format_instructions())
    return prompt | llm | parser


def build_distiller_agent(llm):
    """构建上下文提炼Agent"""
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "你是一位乐于助手的AI。你的任务是将以下检索到的文档片段合成为一个单一、简洁的段落。目标是提供一个清晰、连贯的上下文，直接回答问题：'{question}'。专注于消除冗余信息并按逻辑组织内容。只回答合成后的上下文。"),
        ("human", "检索到的文档:\n{context}")
    ])
    return prompt | llm | StrOutputParser()


def build_reflection_agent(llm):
    """构建反思Agent (现在基于提炼后的上下文)"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一名研究助理。根据当前子问题和提炼后的上下文，写一个简洁、事实驱动的单句摘要，总结核心发现。"),
        ("human", "当前子问题: {sub_question}\n\n提炼后的上下文:\n{distilled_context}")
    ])
    return prompt | llm | StrOutputParser()

def build_policy_agent(llm):
    """构建策略Agent，决定工作流的下一步"""
    prompt_template = ChatPromptTemplate.from_messages([
        ("system",
         "你是一位战略大师，负责分析研究进展并决定下一步行动。分析依据: 1. 原始问题, 2. 初始计划, 3. 研究历史。"
         "决策: 如果历史信息已足够全面回答原始问题，决策为 FINISH。否则，决策为 CONTINUE。你的输出必须是严格的JSON格式。\n{format_instructions}"),
        ("human", "原始问题: {question}\n\n初始计划:\n{plan}\n\n研究历史:\n{history}")
    ])
    parser = JsonOutputParser(pydantic_object=PolicyDecision)
    return prompt_template.partial(format_instructions=parser.get_format_instructions()) | llm | parser

def build_final_answer_agent(llm):
    """构建最终答案生成Agent (优化以支持精确引用)"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一位顶尖的科技动向分析师。你的任务是基于下方提供的、已编号的上下文信息，为用户的原始问题生成一份精准、深入且结构清晰的最终报告。

        ！！！核心规则 - 必须严格遵守！！！:
        1.  绝对忠于原文: 你的回答必须完全基于提供的上下文。严禁使用外部知识。
        2.  句句皆有出处: 报告中的每一个句子或独立信息点，都必须在其末尾使用 `[来源:INDEX]` 的格式清晰标注其来源的文档索引。
        3.  多来源标注: 如果一个句子综合了多个来源的信息，请使用逗号分隔索引，例如 `[来源:2, 5]`。
        4.  结构化输出: 使用Markdown标题、要点和列表来组织回答，使其清晰易读。
        5.  无来源则不答: 如果提供的上下文信息不足以回答问题的某个方面，请明确指出“根据提供的资料，无法回答关于[某方面]的问题”，不要进行猜测。
        6.  格式示例:
            - Meta在2025年10月22日进行了大规模裁员 [来源:1, 3]。
            - 新成立的TBD Lab将专注于下一代AI模型的研发 [来源:4]。

        开始分析以下上下文，并生成报告。"""),
        ("human", "原始问题: {question}\n\n已编号的上下文信息:\n{context}")
    ])
    return prompt | llm | StrOutputParser()

# === 6. LangGraph 节点定义 ===
# 路由节点
def router_node(state: AgentState, router_agent):
    """路由节点：分析意图并分流"""
    print("--- [Node] Router: Analyzing Query Intent ---")
    question = state["original_question"]
    try:
        decision = router_agent.invoke({"question": question})
        # 决策为简单回答，直接把结果放入 final_answer
        if decision.get("step") == "simple_answer":
            return {
                "router_decision": "simple_answer",
                "final_answer": decision.get("simple_response", "")
            }
        else:
            return {"router_decision": "complex_research"}

    except Exception as e:
        print(f"❌ 路由判断出错，默认进入复杂模式: {e}")
        return {"router_decision": "complex_research"}


# 路由逻辑
def route_logic(state: AgentState):
    """条件边逻辑：决定下一步去哪里"""
    decision = state.get("router_decision", "complex_research")

    if decision == "simple_answer":
        print("   -> 判定为简单问题，进入快速检索模式。")
        return "fast_path"  # 修改：使用更清晰的名称
    else:
        print("   -> 判定为复杂问题，进入深度思考模式。")
        return "deep_path"  # 修改：使用更清晰的名称


def plan_node(state: AgentState, planner_agent):
    print("--- [Node] Planning ---")
    plan_dict = planner_agent.invoke({"question": state["original_question"]})
    return {"plan": Plan(**plan_dict), "current_step_index": 0, "past_steps": []}


def research_step_node(state: AgentState, query_rewriter_agent, supervisor_agent, retrievers, web_search_tool):
    idx = state["current_step_index"]
    step = state["plan"].steps[idx]
    print(f"--- [Node] Research Step {idx + 1}: {step.sub_question} ---")

    # 1. 重写查询
    past_context = get_past_context_str(state['past_steps'])
    rewritten = query_rewriter_agent.invoke({
        "user_query": step.sub_question,
        "keywords": step.keywords,
        "past_context": past_context
    })
    tech_entities = rewritten.get('tech_entities', [])
    time_range = rewritten.get('time_range', None)
    if tech_entities:
        print(f"   -> [NER via LLM] 识别实体: {tech_entities}")
    if time_range:
        print(f"   -> [Time via LLM] 时间范围: {time_range}")

    # 2. 决策
    decision_dict = supervisor_agent.invoke({"rewritten_query": rewritten.get('chinese_query')})
    decision = RetrievalDecision(**decision_dict)

    # 3. 检索
    final_query = rewritten.get('english_query') if decision.strategy == "web_search" else rewritten.get(
        'chinese_query')

    docs = []
    if decision.strategy == "web_search" and web_search_tool:
        docs = web_search_function(final_query, web_search_tool)
    elif decision.strategy != "web_search":
        base_retriever = retrievers[decision.strategy]
        docs = base_retriever.invoke(final_query)

    # 4. 重排
    if docs and decision.strategy != "web_search":
        compressor = retrievers["compressor"]
        docs = compressor.compress_documents(documents=docs, query=final_query)

    return {"rewritten_query": final_query, "retrieval_decision": decision, "retrieved_docs": docs}

def distill_node(state: AgentState, distiller_agent):
    print("--- [Node] Distill ---")
    if not state["retrieved_docs"]:
        return {"distilled_context": "未找到相关信息。"}

    context = format_docs_for_synthesis(state["retrieved_docs"])
    distilled = distiller_agent.invoke({"question": state["rewritten_query"], "context": context})
    return {"distilled_context": distilled}

def reflection_node(state: AgentState, reflection_agent):
    print("--- [Node] Reflect ---")
    step = state["plan"].steps[state["current_step_index"]]
    summary = reflection_agent.invoke({
        "sub_question": step.sub_question,
        "distilled_context": state["distilled_context"]
    })

    new_step = PastStep(
        step_index=state["current_step_index"] + 1,
        sub_question=step.sub_question,
        rewritten_query=state["rewritten_query"],
        retrieved_docs=state["retrieved_docs"],
        distilled_context=state["distilled_context"],
        summary=summary
    )
    return {"past_steps": state["past_steps"] + [new_step], "current_step_index": state["current_step_index"] + 1}

# def final_answer_node(state: AgentState, final_answer_agent):
#     print("--- [Node] Final Answer ---")
#     if not state.get('past_steps'):
#         return {"final_answer": "无法生成答案，未收集到信息。"}
#
#     context, metadata = preprocess_docs_for_final_answer(state['past_steps'])
#
#     answer_chunks = []
#     for chunk in final_answer_agent.stream({"question": state["original_question"], "context": context}):
#         answer_chunks.append(chunk)
#
#     full_answer = "".join(answer_chunks)
#
#     # 添加引用
#     citations = "\n\n### 引用来源\n"
#     for idx, meta in sorted(metadata.items()):
#         source = meta.get('source') or meta.get('URL') or '本地归档'
#         title = meta.get('title', '未知标题')
#         citations += f"- [来源 {idx}] 《{title}》 ({source})\n"
#
#     return {"final_answer": full_answer + citations}
def final_answer_node(state: AgentState):
    """
    准备最终答案所需的上下文和元数据。
    这个节点本身不调用 LLM，只做数据准备工作。
    """
    print("--- [Node] Final Answer Preparation ---")

    # 使用 .get() 来安全地访问 past_steps
    past_steps = state.get('past_steps')

    if not past_steps:
        # 如果没有历史步骤，准备一个错误信息并返回
        return {
            "final_answer": "无法生成答案，因为研究过程中未能收集到任何信息。",
            "context_for_final_answer": None,
            "metadata_for_final_answer": None
        }

    # 调用预处理函数来格式化所有文档
    context, metadata = preprocess_docs_for_final_answer(past_steps)

    # 核心：确保显式地返回一个包含准备好的数据的字典
    return {
        "context_for_final_answer": context,
        "metadata_for_final_answer": metadata
    }


def should_continue(state: AgentState, policy_agent):
    """
    决定是继续还是结束。
    """
    print("--- [Edge] Policy Check ---")

    if state["current_step_index"] >= MAX_RESEARCH_ITERATIONS:
        print("   -> 达到最大迭代次数，强制结束。")
        return "finish"

    if state["current_step_index"] >= len(state["plan"].steps):
        print("   -> 计划步骤已全部完成，结束。")
        return "finish"

    total_steps = len(state["plan"].steps)
    current_step = state["current_step_index"]
    print(f"   -> 进度: {current_step}/{total_steps}。通常情况下继续执行...")
    return "continue"


def instant_rag_node(state: AgentState, query_rewriter_agent, retrievers):
    """
    快速模式下的检索节点：执行一次标准的改写+混合检索+重排。
    """
    print("--- [Node] Instant RAG: One-shot Retrieval ---")
    question = state["original_question"]

    # 1. 简单重写 (提取关键词)
    # 为了复用 rewriter，我们需要构造一个假的 past_context
    rewritten = query_rewriter_agent.invoke({
        "user_query": question,
        "keywords": [],
        "past_context": "Current mode: Instant Retrieval."
    })

    # 2. 默认使用混合检索 (Hybrid) 获取最全信息
    # 注意：这里我们跳过了 Supervisor 的决策，直接用最稳健的 hybrid_search
    retriever = retrievers["hybrid_search"]
    query_zh = rewritten.get('chinese_query', question)

    print(f"   -> 检索词: {query_zh}")
    docs = retriever.invoke(query_zh)

    # 3. 重排 (Rerank)
    if docs:
        compressor = retrievers["compressor"]
        docs = compressor.compress_documents(documents=docs, query=query_zh)

    print(f"   -> 检索到 {len(docs)} 篇文档")

    return {
        "rewritten_query": query_zh,
        "retrieved_docs": docs,
        # 兼容 AgentState，虽然这些字段在 Simple 模式下不一定用到
        "current_step_index": 0
    }


# 快速生成节点
def simple_gen_node(state: AgentState):
    """
    快速模式下的生成准备节点：只准备上下文和元数据。
    """
    print("--- [Node] Simple Gen Preparation ---")

    docs = state.get("retrieved_docs", [])

    if not docs:
        return {
            "final_answer": "抱歉，在数据库中没有找到关于该问题的具体信息。",
            "context_for_final_answer": None,
            "metadata_for_final_answer": None
        }

    # 手动格式化文档和元数据
    context_parts = []
    numbered_metadata = {}
    for i, doc in enumerate(docs):
        idx = i + 1
        source_str = doc.metadata.get('source') or doc.metadata.get('URL') or "本地数据"
        title = doc.metadata.get('title', '未命名文档')
        context_parts.append(f"--- 来源 {idx}: 《{title}》 ---\n{doc.page_content}")
        numbered_metadata[idx] = doc.metadata

    formatted_context = "\n\n".join(context_parts)

    # 返回准备好的数据，供 server.py 使用
    return {
        "context_for_final_answer": formatted_context,
        "metadata_for_final_answer": numbered_metadata
    }


# 7. 系统初始化入口
def initialize_rag_system():
    """
    初始化函数：被 server.py 调用。
    加载模型、建立索引、编译图。
    """
    print("\n🚀 正在启动 RAG 系统后端引擎...")

    # 1. 环境与 API Key 检查
    dotenv.load_dotenv()
    os.environ["http_proxy"] = "http://127.0.0.1:11434"
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("⚠️ 警告: DASHSCOPE_API_KEY 未设置，LLM 可能无法运行。")

    tavily_api_key = os.getenv("TAVILY_API_KEY")
    web_search_tool = TavilySearch(k=5) if tavily_api_key else None
    if not web_search_tool:
        print("⚠️ 警告: TAVILY_API_KEY 未设置，网络搜索功能已禁用。")

    # 2. 基础模型初始化
    if not os.path.exists(CSV_FILE_PATH_ZH) or not os.path.exists(CSV_FILE_PATH_EN):
        raise FileNotFoundError(f"找不到数据文件。请确保 {CSV_FILE_PATH_ZH} 和 {CSV_FILE_PATH_EN} 存在。")

    # 3. 检索器构建 (最耗时步骤)
    retrievers = setup_retrievers(CSV_FILE_PATH_ZH, CSV_FILE_PATH_EN)
    if not retrievers:
        raise RuntimeError("检索器初始化失败，无法继续。")

    llm = ChatTongyi(model_name=LLM_MODEL, temperature=0.3, request_timeout=120)

    # 找到你构建 final_writer 的地方
    final_writer = build_final_answer_agent(llm)

    # 我们需要把用于构建它的 prompt 也传出去
    final_writer_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一位顶尖的科技动向分析师。你的任务是基于下方提供的、已编号的上下文信息，为用户的原始问题生成一份精准、深入且结构清晰的最终报告。

        ！！！核心规则 - 必须严格遵守！！！:
        1.  绝对忠于原文: 你的回答必须完全基于提供的上下文。严禁使用外部知识。
        2.  句句皆有出处: 报告中的**每一个句子或独立信息点**，都必须在其末尾使用 `[来源:INDEX]` 的格式清晰标注其来源的文档索引。
        3.  多来源标注: 如果一个句子综合了多个来源的信息，请使用逗号分隔索引，例如 `[来源:2, 5]`。
        4.  结构化输出: 使用Markdown标题、要点和列表来组织回答，使其清晰易读。
        5.  无来源则不答: 如果提供的上下文信息不足以回答问题的某个方面，请明确指出“根据提供的资料，无法回答关于[某方面]的问题”，不要进行猜测。
        6.  格式示例:
            - Meta在2025年10月22日进行了大规模裁员 [来源:1, 3]。
            - 新成立的TBD Lab将专注于下一代AI模型的研发 [来源:4]。

        开始分析以下上下文，并生成报告。"""),
        ("human", "原始问题: {question}\n\n已编号的上下文信息:\n{context}")
    ])

    # 4. Agent 构建
    print("   - 构建智能体 (Agents)...")
    router = build_router_agent(llm)  # --- [NEW]
    planner = build_planner_agent(llm)
    rewriter = build_universal_query_rewriter_agent(llm)
    supervisor = build_retrieval_supervisor_agent(llm)
    distiller = build_distiller_agent(llm)
    reflector = build_reflection_agent(llm)
    policy = build_policy_agent(llm)
    final_writer = build_final_answer_agent(llm)

    # 5. 图 (Graph) 组装
    print("   - 编译 LangGraph 工作流...")
    graph = StateGraph(AgentState)

    # --- 节点注册 ---
    graph.add_node("router", partial(router_node, router_agent=router))

    # Fast RAG 路径节点
    graph.add_node("instant_rag", partial(instant_rag_node, query_rewriter_agent=rewriter, retrievers=retrievers))
    graph.add_node("simple_gen", simple_gen_node) # 新的

    # Deep RAG 路径节点
    graph.add_node("plan", partial(plan_node, planner_agent=planner))
    graph.add_node("research_step", partial(
        research_step_node,
        query_rewriter_agent=rewriter,
        supervisor_agent=supervisor,
        retrievers=retrievers,
        web_search_tool=web_search_tool
    ))
    graph.add_node("distill", partial(distill_node, distiller_agent=distiller))
    graph.add_node("reflect", partial(reflection_node, reflection_agent=reflector))
    graph.add_node("generate_final_answer", final_answer_node)

    # --- 边连接 ---
    graph.set_entry_point("router")

    # 核心分流逻辑
    graph.add_conditional_edges(
        "router",
        route_logic,
        {
            "fast_path": "instant_rag",  # 修改
            "deep_path": "plan"  # 修改
        }
    )

    # Fast RAG 路径
    graph.add_edge("instant_rag", "simple_gen")
    graph.add_edge("simple_gen", END)  # simple_gen 准备完数据后，图就结束了

    # Deep RAG 路径
    graph.add_edge("plan", "research_step")
    graph.add_edge("research_step", "distill")
    graph.add_edge("distill", "reflect")
    graph.add_conditional_edges(
        "reflect",
        partial(should_continue, policy_agent=policy),
        {"continue": "research_step", "finish": "generate_final_answer"}
    )
    graph.add_edge("generate_final_answer", END)  # generate_final_answer 准备完数据后，图也结束了

    app = graph.compile()
    print("✅ RAG 引擎初始化成功，支持 Fast/Deep 双模式。")
    return app, final_writer_prompt, llm





# import os
# import shutil
# import traceback
# import json
# from typing import List, Optional, TypedDict, Literal
#
# # --- 第三方库 ---
# import pandas as pd
# import dotenv
# import torch
# from langchain_core.stores import InMemoryStore
# from pydantic import BaseModel, Field
# from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
#
# # --- LangChain 核心组件 ---
# from langchain_core.documents import Document
# from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate
#
# # --- LangChain 社区与集成组件 ---
# from langchain_community.llms import Tongyi as ChatTongyi
# from langchain_community.retrievers import BM25Retriever
# from langchain_community.cross_encoders import HuggingFaceCrossEncoder
# from langchain_tavily import TavilySearch
#
# # --- LangChain Classic (高级检索) ---
# from langchain_classic.retrievers import (
#     ContextualCompressionRetriever,
#     EnsembleRetriever,
#     ParentDocumentRetriever,
# )
# from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
#
# # --- LangChain 其他专用库 ---
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma
# from langchain_ollama import OllamaEmbeddings
#
# # --- LangGraph (Agentic Workflow核心) ---
# from langgraph.graph import StateGraph, END
# from functools import partial
#
# # === 2. 常量与配置 ===
# # 请根据实际文件位置调整路径
# CSV_FILE_PATH_ZH = './cleaned_tech_news_robust.csv'
# CSV_FILE_PATH_EN = './hacker_news_cleaned.csv'
# EMBEDDING_MODEL = "qwen3-embedding:4b"
# LLM_MODEL = "qwen3-max"
# RERANKER_MODEL_PATH = r"./models/bge-reranker-large"  # 或者使用 HuggingFace ID: "BAAI/bge-reranker-large"
# NER_MODEL = "dslim/bert-base-NER"
# MAX_RESEARCH_ITERATIONS = 5
#
#
# # === 3. Pydantic 数据模型定义 ===
#
# # --- [NEW] 路由决策模型 ---
# class RouteQuery(BaseModel):
#     """将用户查询路由到最合适的路径"""
#     step: Literal["simple_answer", "complex_research"] = Field(
#         description="下一步行动。'simple_answer' 用于问候、闲聊或LLM无需搜索即可回答的通用知识。'complex_research' 用于需要检索最新新闻、具体技术细节或深入分析的问题。"
#     )
#     simple_response: str = Field(
#         default="",
#         description="如果 step 是 'simple_answer'，在这里直接提供回复内容。如果是 'complex_research'，留空。"
#     )
#
#
# class BilingualQuery(BaseModel):
#     chinese_query: str = Field(description="优化后的中文搜索查询")
#     english_query: str = Field(description="翻译并优化后的英文搜索查询")
#
#
# class RetrievalDecision(BaseModel):
#     """检索主管的决策输出模型"""
#     strategy: Literal["vector_search", "keyword_search", "hybrid_search", "web_search"] = Field(
#         description="选择的检索策略。必须是 'vector_search', 'keyword_search', 'hybrid_search', 或 'web_search' 之一。"
#     )
#     justification: str = Field(description="做出此选择的理由。")
#
#
# class Step(BaseModel):
#     """研究计划中的一个独立步骤"""
#     sub_question: str = Field(description="为这一步需要回答的具体、可操作的子问题。")
#     justification: str = Field(description="简要解释为什么这一步对于回答最终问题是必要的。")
#     keywords: List[str] = Field(description="用于增强检索效果的关键术语列表。")
#
#
# class Plan(BaseModel):
#     """完整的、多步骤的研究计划"""
#     steps: List[Step] = Field(description="一个详细的、多步骤的计划，用于回答用户的复杂查询。")
#
#
# class PastStep(TypedDict):
#     """存储已完成步骤的结果"""
#     step_index: int
#     sub_question: str
#     rewritten_query: str
#     retrieved_docs: List[Document]
#     distilled_context: str
#     summary: str
#
#
# class AgentState(TypedDict):
#     """贯穿整个Graph的核心状态对象"""
#     original_question: str
#     router_decision: str
#     plan: Plan
#     past_steps: List[PastStep]
#     current_step_index: int
#     rewritten_query: str
#     retrieval_decision: RetrievalDecision
#     retrieved_docs: List[Document]
#     distilled_context: str
#     final_answer: str
#
#     # --- [新增] 为最终的流式生成准备的字段 ---
#     context_for_final_answer: Optional[str]
#     metadata_for_final_answer: Optional[dict]
#
#
# class PolicyDecision(BaseModel):
#     """策略Agent的决策输出模型"""
#     next_action: str = Field(description="决策结果，必须是 'CONTINUE' 或 'FINISH'。")
#     justification: str = Field(description="做出此决策的理由。")
#
#
# # === 4. 辅助函数与工具 ===
#
# def initialize_ner_pipeline() -> Optional[pipeline]:
#     """初始化 Hugging Face 的 NER pipeline。"""
#     print(f"正在初始化 NER 模型: {NER_MODEL}...")
#     try:
#         device_id = 0 if torch.cuda.is_available() else -1
#         tokenizer = AutoTokenizer.from_pretrained(NER_MODEL)
#         model = AutoModelForTokenClassification.from_pretrained(NER_MODEL)
#         ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple",
#                                 device=device_id)
#         print("✅ NER 模型初始化完成。")
#         return ner_pipeline
#     except Exception as e:
#         print(f"❌ 初始化 NER 模型失败 (非致命): {e}")
#         return None
#
#
# def setup_retrievers(
#         csv_file_path_zh: str,
#         csv_file_path_en: str,
#         ner_pipeline: Optional[pipeline]
# ) -> Optional[dict]:
#     """构建包含多语言支持的高级检索器套件。"""
#     print(f"\n--- 开始构建多语言检索器 ---")
#
#     # --- 1. 加载中文数据 ---
#     print("   - 加载中文数据集...")
#     try:
#         df_zh = pd.read_csv(csv_file_path_zh, header=None, names=['id', 'title', 'content', 'date', 'URL'])
#         df_zh.fillna('', inplace=True)
#         parent_documents_zh = [
#             Document(page_content=str(row['content']),
#                      metadata={'source_id': 'zh_' + str(row['id']), 'title': str(row['title']),
#                                'date': str(row['date']), 'URL': str(row['URL']), 'lang': 'zh'})
#             for _, row in df_zh.iterrows() if str(row['content']).strip()
#         ]
#         print(f"     -> 中文文档数: {len(parent_documents_zh)}")
#     except Exception as e:
#         print(f"❌ 读取中文CSV失败: {e}")
#         return None
#
#     # --- 2. 加载英文数据 ---
#     print("   - 加载英文数据集...")
#     try:
#         # 假设英文CSV格式兼容: id, title, content, date, URL
#         df_en = pd.read_csv(csv_file_path_en, header=None, names=['id', 'title', 'content', 'date', 'URL'])
#         df_en.fillna('', inplace=True)
#         parent_documents_en = [
#             Document(page_content=str(row['content']),
#                      metadata={'source_id': 'en_' + str(row['id']), 'title': str(row['title']),
#                                'date': str(row['date']), 'URL': str(row['URL']), 'lang': 'en'})
#             for _, row in df_en.iterrows() if str(row['content']).strip()
#         ]
#         print(f"     -> 英文文档数: {len(parent_documents_en)}")
#     except Exception as e:
#         print(f"❌ 读取英文CSV失败: {e}")
#         return None
#
#     # --- 3. 初始化嵌入模型 ---
#     print("   - 初始化嵌入模型...")
#     try:
#         embeddings_model = OllamaEmbeddings(model=EMBEDDING_MODEL)
#     except Exception as e:
#         print(f"❌ 初始化 Ollama Embeddings 失败: {e}")
#         return None
#
#     # --- 4. 构建向量库 (ParentDocumentRetriever) ---
#     # 中文库
#     CHROMA_DB_DIR_ZH = "./chroma_db_ollama_qwen3_zh"
#     if os.path.exists(CHROMA_DB_DIR_ZH): shutil.rmtree(CHROMA_DB_DIR_ZH)
#     vectorstore_zh = Chroma(collection_name="tech_news_zh", embedding_function=embeddings_model,
#                             persist_directory=CHROMA_DB_DIR_ZH)
#     pdr_retriever_zh = ParentDocumentRetriever(
#         vectorstore=vectorstore_zh,
#         docstore=InMemoryStore(),
#         child_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     )
#     pdr_retriever_zh.add_documents(parent_documents_zh, ids=None)
#
#     # 英文库
#     CHROMA_DB_DIR_EN = "./chroma_db_ollama_qwen3_en"
#     if os.path.exists(CHROMA_DB_DIR_EN): shutil.rmtree(CHROMA_DB_DIR_EN)
#     vectorstore_en = Chroma(collection_name="tech_news_en", embedding_function=embeddings_model,
#                             persist_directory=CHROMA_DB_DIR_EN)
#     pdr_retriever_en = ParentDocumentRetriever(
#         vectorstore=vectorstore_en,
#         docstore=InMemoryStore(),
#         child_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     )
#     pdr_retriever_en.add_documents(parent_documents_en, ids=None)
#     print("   - ParentDocumentRetriever 构建完成。")
#
#     # --- 5. 构建关键词检索器 (BM25) ---
#     bm25_retriever_zh = BM25Retriever.from_documents(parent_documents_zh)
#     bm25_retriever_zh.k = 3
#     bm25_retriever_en = BM25Retriever.from_documents(parent_documents_en)
#     bm25_retriever_en.k = 3
#     print("   - BM25Retriever 构建完成。")
#
#     # --- 6. 混合检索集成 (Ensemble) ---
#     ensemble_retriever = EnsembleRetriever(
#         retrievers=[pdr_retriever_zh, pdr_retriever_en, bm25_retriever_zh, bm25_retriever_en],
#         weights=[0.4, 0.4, 0.1, 0.1],
#         search_type="rrf"
#     )
#
#     # --- 7. 重排器 (Cross Encoder) ---
#     try:
#         # 如果下载了模型在本地，使用路径；否则会自动下载
#         cross_encoder_model = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL_PATH)
#         compressor = CrossEncoderReranker(model=cross_encoder_model, top_n=3)
#     except Exception as e:
#         print(f"❌ 加载 Reranker 失败: {e}")
#         return None
#
#     print("✅ 检索器套件构建完成。")
#     return {
#         "vector_search": EnsembleRetriever(retrievers=[pdr_retriever_zh, pdr_retriever_en], weights=[0.5, 0.5]),
#         "keyword_search": EnsembleRetriever(retrievers=[bm25_retriever_zh, bm25_retriever_en], weights=[0.5, 0.5]),
#         "hybrid_search": ensemble_retriever,
#         "compressor": compressor
#     }
#
#
# def preprocess_docs_for_final_answer(past_steps: List[PastStep]) -> (str, dict):
#     """预处理所有步骤的文档，为每篇唯一文档分配索引，并格式化为上下文。"""
#     all_docs = [doc for step in past_steps for doc in step['retrieved_docs']]
#     unique_docs_map = {}
#
#     for doc in all_docs:
#         # 尝试寻找唯一标识
#         if 'source' in doc.metadata and 'http' in doc.metadata['source']:
#             key = doc.metadata['source']
#         elif 'source_id' in doc.metadata:
#             key = doc.metadata['source_id']
#         else:
#             continue  # 跳过无标识文档
#
#         if key not in unique_docs_map:
#             unique_docs_map[key] = doc
#
#     numbered_sources = {i + 1: doc for i, doc in enumerate(unique_docs_map.values())}
#
#     if not numbered_sources:
#         return "没有找到相关的上下文信息。", {}
#
#     context_parts = []
#     for index, doc in numbered_sources.items():
#         if "source" in doc.metadata and "http" in doc.metadata["source"]:
#             source_str = f"来源 {index}: 《{doc.metadata.get('title', '网络文章')}》 ({doc.metadata['source']})"
#         else:
#             source_str = f"来源 {index}: 《{doc.metadata.get('title', '未知')}》 (日期: {doc.metadata.get('date', '未知')})"
#
#         context_parts.append(f"--- {source_str} ---\n内容: {doc.page_content}")
#
#     formatted_context = "\n\n".join(context_parts)
#     numbered_metadata = {index: doc.metadata for index, doc in numbered_sources.items()}
#     return formatted_context, numbered_metadata
#
#
# def web_search_function(query: str, tavily_tool: TavilySearch) -> List[Document]:
#     """执行网络搜索"""
#     print(f"   - [Web Search] 查询: '{query}'")
#     try:
#         response_dict = tavily_tool.invoke({"query": query})
#         results = response_dict.get('results', [])
#         if not isinstance(results, list):
#             return []
#         return [
#             Document(
#                 page_content=res.get("content", ""),
#                 metadata={"source": res.get("url", ""), "title": res.get("title", "未知标题")}
#             ) for res in results if isinstance(res, dict)
#         ]
#     except Exception as e:
#         print(f"❌ Tavily 搜索错误: {e}")
#         return []
#
#
# def format_docs_for_synthesis(docs: List[Document]) -> str:
#     """为提炼节点格式化文档"""
#     if not docs: return "没有找到相关的上下文信息。"
#     formatted_strings = []
#     for i, doc in enumerate(docs):
#         title = doc.metadata.get('title', '未知')
#         formatted_strings.append(f"--- 文档 {i + 1} ({title}) ---\n{doc.page_content}")
#     return "\n\n".join(formatted_strings)
#
#
# def get_past_context_str(past_steps: List[PastStep]) -> str:
#     """格式化研究历史"""
#     if not past_steps: return "还未进行任何研究。"
#     return "\n\n".join([f"第 {s['step_index']} 步: {s['sub_question']}\n发现摘要: {s['summary']}" for s in past_steps])
#
#
# # === 5. 专用 Agent 构建函数 ===
#
# # --- [NEW] 路由 Agent ---
# def build_router_agent(llm):
#     """
#     构建路由Agent。
#     修改逻辑：Simple 模式不再是纯闲聊，而是"单次检索即可回答"的问题。
#     """
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", """你是一名专家级的意图分类器。你的任务是根据问题的复杂度，选择最高效的解决路径。
#
#         **路径选择标准**:
#
#         1.  **SIMPLE_ANSWER (快速检索模式)**:
#             - **定义**: 问题具体、明确，只需要进行**一次**数据库检索即可找到答案，不需要多步推理或复杂的横向对比。
#             - **适用场景**:
#                 - 询问具体事实 (如 "Meta Free Transformer的架构细节", "夸克眼镜哪天预售", "Kyber项目何时量产")。
#                 - 简单的概念解释 (如 "什么是RAG", "介绍一下Transformer")。
#             - **决策**: 将 `step` 设为 "simple_answer"。即使需要查库，只要一次能查到，就选这个。
#
#         2.  **COMPLEX_RESEARCH (深度思考模式)**:
#             - **定义**: 问题模糊、庞大，或者需要跨多个时间段、多个实体进行综合分析，往往需要拆解为多个子问题。
#             - **适用场景**:
#                 - 趋势分析 (如 "分析2025年AI发展趋势")。
#                 - 深度对比 (如 "详细对比DeepSeek和ChatGPT在编码能力上的优劣及底层架构差异")。
#                 - 综述报告 (如 "总结过去一个月所有关于半导体的大事件")。
#             - **决策**: 将 `step` 设为 "complex_research"。
#
#         **输出格式**:
#         严格遵循JSON格式。
#         {{
#             "step": "simple_answer" 或 "complex_research",
#             "simple_response": ""  <-- 永远留空，因为我们现在改用检索生成，不再由你直接回答
#         }}
#         {format_instructions}"""),
#         ("human", "用户问题: {question}")
#     ])
#
#     parser = JsonOutputParser(pydantic_object=RouteQuery)
#     prompt = prompt.partial(format_instructions=parser.get_format_instructions())
#
#     return prompt | llm | parser
#
#
# def build_planner_agent(llm):
#     """构建规划Agent"""
#     prompt_template = ChatPromptTemplate.from_messages([
#         ("system",
#          "你是一名顶级的AI查询意图分析师。你的任务是将用户的复杂问题分解为一个结构清晰、逻辑严谨的多步骤研究计划。"
#          "专注于问题分解，将数据源选择等策略性任务留给下游Agent。你的输出必须是严格的JSON格式，完全符合以下Pydantic模型。\n{format_instructions}"),
#         ("human", "用户问题: {question}")
#     ])
#     parser = JsonOutputParser(pydantic_object=Plan)
#     return prompt_template.partial(format_instructions=parser.get_format_instructions()) | llm | parser
#
#
# def build_universal_query_rewriter_agent(llm):
#     """
#     构建一个通用的、支持双向翻译的查询重写Agent。
#     """
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", """你是一个为多语言数据库设计的、专家级的查询优化引擎。
#
#         **核心任务**:
#         接收一个可能是中文或英文的用户查询，然后生成两个针对数据库检索的优化版本：一个中文版和一个英文版。
#
#         **处理流程**:
#         1.  识别用户查询的原始语言。
#         2.  对原始语言的查询进行优化，提取核心关键词，使其更适合机器检索。
#         3.  将优化后的查询精确地翻译成另一种语言。
#         4.  确保翻译后的版本也针对该语言的数据库检索进行了优化。
#
#         **输出格式要求**:
#         你的回答必须严格遵循以下JSON格式，不允许包含任何额外的文本、解释或Markdown标记。
#         ```json
#         {{
#             "chinese_query": "优化后的中文查询",
#             "english_query": "Optimized English query"
#         }}
#         ```
#
#         **示例 1 (中文输入)**:
#         - 输入: 子问题="什么是“聊天控制2.0”提案？", 关键词="欧盟, 加密, 隐私"
#         - 输出:
#         ```json
#         {{
#             "chinese_query": "欧盟 聊天控制2.0 提案 详情",
#             "english_query": "EU chat control 2.0 proposal details"
#         }}
#         ```
#
#         **示例 2 (英文输入)**:
#         - 输入: 子问题="What is LCEL in LangChain and how does it differ from a normal chain?", 关键词="LangChain, LCEL, performance"
#         - 输出:
#         ```json
#         {{
#             "chinese_query": "LangChain LCEL 与普通链的区别和优势",
#             "english_query": "LangChain LCEL vs normal chain differences and advantages"
#         }}
#         ```
#
#         请严格遵守以上规则。"""),
#         ("human", "用户查询: {user_query}\n\n相关关键词: {keywords}\n\n历史上下文:\n{past_context}")
#     ])
#
#     parser = JsonOutputParser(pydantic_object=BilingualQuery)
#     return prompt | llm | parser
#
#
# def build_retrieval_supervisor_agent(llm):
#     """构建检索主管Agent，动态选择策略"""
#     parser = JsonOutputParser(pydantic_object=RetrievalDecision)
#     prompt_template = ChatPromptTemplate.from_messages([
#         ("system", """你是一名检索策略专家。你的任务是根据用户的查询，决定最佳的检索信息源。
#
#         **可用信息源**:
#         1.  **本地数据库 (Local Database)**: 这是一个包含大量科技新闻文章的知识库。当查询涉及分析、背景信息、历史事件或不一定需要实时信息的概念时，应优先使用它。它可以通过 `vector_search`, `keyword_search`, 或 `hybrid_search` 来访问。
#         2.  **网络搜索 (Web Search)**: 用于获取实时、突发的，以及本地数据库中明确没有的内容。
#
#         **决策规则**:
#         - **优先本地**: 对于需要深度分析和背景的问题，即使提到了“近期”，也应该首先考虑使用 `hybrid_search` 从本地数据库获取基础信息。
#         - **网络补充**: 只有当问题明确指向一个非常具体、实时性极强的事件（如“今天的股价”），或者计划中明确提到需要查找外部资料时，才首选 `web_search`。
#         - **综合判断**: 如果一个问题既需要背景知识又需要最新动态，`hybrid_search` 通常是最好的起点。
#
#         **你的输出必须是严格的JSON格式，完全符合以下Pydantic模型。**
#         {format_instructions}"""),
#         ("human", "重写后的查询: {rewritten_query}")
#     ])
#     prompt = prompt_template.partial(format_instructions=parser.get_format_instructions())
#     return prompt | llm | parser
#
#
# def build_distiller_agent(llm):
#     """构建上下文提炼Agent"""
#     prompt = ChatPromptTemplate.from_messages([
#         ("system",
#          "你是一位乐于助手的AI。你的任务是将以下检索到的文档片段合成为一个单一、简洁的段落。目标是提供一个清晰、连贯的上下文，直接回答问题：'{question}'。专注于消除冗余信息并按逻辑组织内容。只回答合成后的上下文。"),
#         ("human", "检索到的文档:\n{context}")
#     ])
#     return prompt | llm | StrOutputParser()
#
#
# def build_reflection_agent(llm):
#     """构建反思Agent (现在基于提炼后的上下文)"""
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", "你是一名研究助理。根据当前子问题和**提炼后**的上下文，写一个简洁、事实驱动的单句摘要，总结核心发现。"),
#         ("human", "当前子问题: {sub_question}\n\n提炼后的上下文:\n{distilled_context}")
#     ])
#     return prompt | llm | StrOutputParser()
#
#
# def build_policy_agent(llm):
#     """构建策略Agent，决定工作流的下一步"""
#     prompt_template = ChatPromptTemplate.from_messages([
#         ("system",
#          "你是一位战略大师，负责分析研究进展并决定下一步行动。分析依据: 1. 原始问题, 2. 初始计划, 3. 研究历史。"
#          "决策: 如果历史信息已足够全面回答原始问题，决策为 FINISH。否则，决策为 CONTINUE。你的输出必须是严格的JSON格式。\n{format_instructions}"),
#         ("human", "原始问题: {question}\n\n初始计划:\n{plan}\n\n研究历史:\n{history}")
#     ])
#     parser = JsonOutputParser(pydantic_object=PolicyDecision)
#     return prompt_template.partial(format_instructions=parser.get_format_instructions()) | llm | parser
#
#
# def build_final_answer_agent(llm):
#     """构建最终答案生成Agent (优化以支持图表和表格)"""
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", """你是一位顶尖的科技动向分析师。你的任务是基于下方提供的上下文信息，为用户的原始问题生成一份精准、深入的最终报告。
#
#         **！！！核心规则！！！**:
#         1.  **绝对忠于原文**: 严禁使用外部知识。
#         2.  **来源标注**: 每一句话后必须标注 `[来源:INDEX]`。
#         3.  **结构化输出**: 使用 Markdown 组织内容。
#
#         **！！！可视化增强规则 (重要)！！！**:
#         如果数据适合展示为图表（如股价趋势、参数对比、占比分布）或结构化表格（如会议清单、硬件参数表），请按以下格式输出：
#
#         **规则 A: 数据图表 (ECharts)**
#         请在回答中插入一个代码块，语言标记为 `echarts`，内容为 **纯 JSON 格式** 的 ECharts option 配置对象。
#         示例：
#         ```echarts
#         {{
#             "title": {{ "text": "DeepSeek vs GPT-4 编码能力对比" }},
#             "tooltip": {{ "trigger": "axis" }},
#             "xAxis": {{ "type": "category", "data": ["Python", "Java", "C++"] }},
#             "yAxis": {{ "type": "value" }},
#             "series": [
#                 {{ "name": "DeepSeek", "type": "bar", "data": [85, 90, 88] }},
#                 {{ "name": "GPT-4", "type": "bar", "data": [82, 88, 95] }}
#             ]
#         }}
#         ```
#
#         **规则 B: 结构化数据 (JSON 表格)**
#         如果用户要求“列出清单”或数据适合表格展示，除了 Markdown 表格，你也可以提供 JSON 数据供前端渲染，语言标记为 `datatable`。
#         示例：
#         ```datatable
#         [
#             {{ "会议名称": "CVPR 2025", "时间": "6月", "地点": "美国" }},
#             {{ "会议名称": "NeurIPS 2025", "时间": "12月", "地点": "加拿大" }}
#         ]
#         ```
#
#         请根据上下文判断是否需要生成图表。如果不需要，仅输出文本即可。
#         """),
#         ("human", "原始问题: {question}\n\n已编号的上下文信息:\n{context}")
#     ])
#     return prompt | llm | StrOutputParser()
#
#
# # === 6. LangGraph 节点定义 ===
#
# # --- [NEW] 路由节点 ---
# def router_node(state: AgentState, router_agent):
#     """路由节点：分析意图并分流"""
#     print("--- [Node] Router: Analyzing Query Intent ---")
#     question = state["original_question"]
#     try:
#         decision = router_agent.invoke({"question": question})
#         # 决策为简单回答，直接把结果放入 final_answer
#         if decision.get("step") == "simple_answer":
#             return {
#                 "router_decision": "simple_answer",
#                 "final_answer": decision.get("simple_response", "")
#             }
#         else:
#             return {"router_decision": "complex_research"}
#
#     except Exception as e:
#         print(f"❌ 路由判断出错，默认进入复杂模式: {e}")
#         return {"router_decision": "complex_research"}
#
#
# # --- [NEW] 路由逻辑 ---
# def route_logic(state: AgentState):
#     """条件边逻辑：决定下一步去哪里"""
#     decision = state.get("router_decision", "complex_research")
#
#     if decision == "simple_answer":
#         print("   -> 判定为简单问题，进入快速检索模式。")
#         return "fast_path"  # 修改：使用更清晰的名称
#     else:
#         print("   -> 判定为复杂问题，进入深度思考模式。")
#         return "deep_path"  # 修改：使用更清晰的名称
#
#
# def plan_node(state: AgentState, planner_agent):
#     print("--- [Node] Planning ---")
#     plan_dict = planner_agent.invoke({"question": state["original_question"]})
#     return {"plan": Plan(**plan_dict), "current_step_index": 0, "past_steps": []}
#
#
# def research_step_node(state: AgentState, query_rewriter_agent, supervisor_agent, retrievers, web_search_tool):
#     idx = state["current_step_index"]
#     step = state["plan"].steps[idx]
#     print(f"--- [Node] Research Step {idx + 1}: {step.sub_question} ---")
#
#     # 1. 重写查询
#     past_context = get_past_context_str(state['past_steps'])
#     rewritten = query_rewriter_agent.invoke({
#         "user_query": step.sub_question,
#         "keywords": step.keywords,
#         "past_context": past_context
#     })
#
#     # 2. 决策
#     decision_dict = supervisor_agent.invoke({"rewritten_query": rewritten.get('chinese_query')})
#     decision = RetrievalDecision(**decision_dict)
#
#     # 3. 检索
#     final_query = rewritten.get('english_query') if decision.strategy == "web_search" else rewritten.get(
#         'chinese_query')
#
#     docs = []
#     if decision.strategy == "web_search" and web_search_tool:
#         docs = web_search_function(final_query, web_search_tool)
#     elif decision.strategy != "web_search":
#         base_retriever = retrievers[decision.strategy]
#         docs = base_retriever.invoke(final_query)
#
#     # 4. 重排
#     if docs and decision.strategy != "web_search":
#         compressor = retrievers["compressor"]
#         docs = compressor.compress_documents(documents=docs, query=final_query)
#
#     return {"rewritten_query": final_query, "retrieval_decision": decision, "retrieved_docs": docs}
#
#
# def distill_node(state: AgentState, distiller_agent):
#     print("--- [Node] Distill ---")
#     if not state["retrieved_docs"]:
#         return {"distilled_context": "未找到相关信息。"}
#
#     context = format_docs_for_synthesis(state["retrieved_docs"])
#     distilled = distiller_agent.invoke({"question": state["rewritten_query"], "context": context})
#     return {"distilled_context": distilled}
#
#
# def reflection_node(state: AgentState, reflection_agent):
#     print("--- [Node] Reflect ---")
#     step = state["plan"].steps[state["current_step_index"]]
#     summary = reflection_agent.invoke({
#         "sub_question": step.sub_question,
#         "distilled_context": state["distilled_context"]
#     })
#
#     new_step = PastStep(
#         step_index=state["current_step_index"] + 1,
#         sub_question=step.sub_question,
#         rewritten_query=state["rewritten_query"],
#         retrieved_docs=state["retrieved_docs"],
#         distilled_context=state["distilled_context"],
#         summary=summary
#     )
#     return {"past_steps": state["past_steps"] + [new_step], "current_step_index": state["current_step_index"] + 1}
#
#
# # def final_answer_node(state: AgentState, final_answer_agent):
# #     print("--- [Node] Final Answer ---")
# #     if not state.get('past_steps'):
# #         return {"final_answer": "无法生成答案，未收集到信息。"}
# #
# #     context, metadata = preprocess_docs_for_final_answer(state['past_steps'])
# #
# #     answer_chunks = []
# #     for chunk in final_answer_agent.stream({"question": state["original_question"], "context": context}):
# #         answer_chunks.append(chunk)
# #
# #     full_answer = "".join(answer_chunks)
# #
# #     # 添加引用
# #     citations = "\n\n### 引用来源\n"
# #     for idx, meta in sorted(metadata.items()):
# #         source = meta.get('source') or meta.get('URL') or '本地归档'
# #         title = meta.get('title', '未知标题')
# #         citations += f"- [来源 {idx}] 《{title}》 ({source})\n"
# #
# #     return {"final_answer": full_answer + citations}
# def final_answer_node(state: AgentState):
#     """
#     准备最终答案所需的上下文和元数据。
#     这个节点本身不调用 LLM，只做数据准备工作。
#     """
#     print("--- [Node] Final Answer Preparation ---")
#
#     # 使用 .get() 来安全地访问 past_steps
#     past_steps = state.get('past_steps')
#
#     if not past_steps:
#         # 如果没有历史步骤，准备一个错误信息并返回
#         return {
#             "final_answer": "无法生成答案，因为研究过程中未能收集到任何信息。",
#             "context_for_final_answer": None,
#             "metadata_for_final_answer": None
#         }
#
#     # 调用预处理函数来格式化所有文档
#     context, metadata = preprocess_docs_for_final_answer(past_steps)
#
#     # 核心：确保显式地返回一个包含准备好的数据的字典
#     return {
#         "context_for_final_answer": context,
#         "metadata_for_final_answer": metadata
#     }
#
#
# def should_continue(state: AgentState, policy_agent):
#     """
#     决定是继续还是结束。
#     """
#     print("--- [Edge] Policy Check ---")
#
#     if state["current_step_index"] >= MAX_RESEARCH_ITERATIONS:
#         print("   -> 达到最大迭代次数，强制结束。")
#         return "finish"
#
#     if state["current_step_index"] >= len(state["plan"].steps):
#         print("   -> 计划步骤已全部完成，结束。")
#         return "finish"
#
#     total_steps = len(state["plan"].steps)
#     current_step = state["current_step_index"]
#     print(f"   -> 进度: {current_step}/{total_steps}。通常情况下继续执行...")
#     return "continue"
#
#
# def instant_rag_node(state: AgentState, query_rewriter_agent, retrievers):
#     """
#     快速模式下的检索节点：执行一次标准的改写+混合检索+重排。
#     """
#     print("--- [Node] Instant RAG: One-shot Retrieval ---")
#     question = state["original_question"]
#
#     # 1. 简单重写 (提取关键词)
#     # 为了复用 rewriter，我们需要构造一个假的 past_context
#     rewritten = query_rewriter_agent.invoke({
#         "user_query": question,
#         "keywords": [],
#         "past_context": "Current mode: Instant Retrieval."
#     })
#
#     # 2. 默认使用混合检索 (Hybrid) 获取最全信息
#     # 注意：这里我们跳过了 Supervisor 的决策，直接用最稳健的 hybrid_search
#     retriever = retrievers["hybrid_search"]
#     query_zh = rewritten.get('chinese_query', question)
#
#     print(f"   -> 检索词: {query_zh}")
#     docs = retriever.invoke(query_zh)
#
#     # 3. 重排 (Rerank)
#     if docs:
#         compressor = retrievers["compressor"]
#         docs = compressor.compress_documents(documents=docs, query=query_zh)
#
#     print(f"   -> 检索到 {len(docs)} 篇文档")
#
#     return {
#         "rewritten_query": query_zh,
#         "retrieved_docs": docs,
#         # 兼容 AgentState，虽然这些字段在 Simple 模式下不一定用到
#         "current_step_index": 0
#     }
#
#
# # --- [NEW] 快速生成节点 ---
# def simple_gen_node(state: AgentState):
#     """
#     快速模式下的生成准备节点：只准备上下文和元数据。
#     """
#     print("--- [Node] Simple Gen Preparation ---")
#
#     docs = state.get("retrieved_docs", [])
#
#     if not docs:
#         return {
#             "final_answer": "抱歉，在数据库中没有找到关于该问题的具体信息。",
#             "context_for_final_answer": None,
#             "metadata_for_final_answer": None
#         }
#
#     # 手动格式化文档和元数据
#     context_parts = []
#     numbered_metadata = {}
#     for i, doc in enumerate(docs):
#         idx = i + 1
#         source_str = doc.metadata.get('source') or doc.metadata.get('URL') or "本地数据"
#         title = doc.metadata.get('title', '未命名文档')
#         context_parts.append(f"--- 来源 {idx}: 《{title}》 ---\n{doc.page_content}")
#         numbered_metadata[idx] = doc.metadata
#
#     formatted_context = "\n\n".join(context_parts)
#
#     # 返回准备好的数据，供 server.py 使用
#     return {
#         "context_for_final_answer": formatted_context,
#         "metadata_for_final_answer": numbered_metadata
#     }
#
#
# # === 7. 系统初始化入口 ===
#
# def initialize_rag_system():
#     """
#     初始化函数：被 server.py 调用。
#     加载模型、建立索引、编译图。
#     """
#     print("\n🚀 正在启动 RAG 系统后端引擎...")
#
#     # 1. 环境与 API Key 检查
#     dotenv.load_dotenv()
#     os.environ["http_proxy"] = "http://127.0.0.1:11434"
#     if not os.getenv("DASHSCOPE_API_KEY"):
#         print("⚠️ 警告: DASHSCOPE_API_KEY 未设置，LLM 可能无法运行。")
#
#     tavily_api_key = os.getenv("TAVILY_API_KEY")
#     web_search_tool = TavilySearch(k=5) if tavily_api_key else None
#     if not web_search_tool:
#         print("⚠️ 警告: TAVILY_API_KEY 未设置，网络搜索功能已禁用。")
#
#     # 2. 基础模型初始化
#     ner_pipeline = initialize_ner_pipeline()
#
#     if not os.path.exists(CSV_FILE_PATH_ZH) or not os.path.exists(CSV_FILE_PATH_EN):
#         raise FileNotFoundError(f"找不到数据文件。请确保 {CSV_FILE_PATH_ZH} 和 {CSV_FILE_PATH_EN} 存在。")
#
#     # 3. 检索器构建 (最耗时步骤)
#     retrievers = setup_retrievers(CSV_FILE_PATH_ZH, CSV_FILE_PATH_EN, ner_pipeline)
#     if not retrievers:
#         raise RuntimeError("检索器初始化失败，无法继续。")
#
#     llm = ChatTongyi(model_name=LLM_MODEL, temperature=0.3, request_timeout=120)
#
#     # 找到你构建 final_writer 的地方
#     final_writer = build_final_answer_agent(llm)
#
#     # 我们需要把用于构建它的 prompt 也传出去
#     final_writer_prompt = ChatPromptTemplate.from_messages([
#         ("system", """你是一位顶尖的科技动向分析师。你的任务是基于下方提供的、已编号的上下文信息，为用户的原始问题生成一份精准、深入且结构清晰的最终报告。
#
#         **！！！核心规则 - 必须严格遵守！！！**:
#         1.  **绝对忠于原文**: 你的回答必须完全基于提供的上下文。严禁使用外部知识。
#         2.  **句句皆有出处**: 报告中的**每一个句子或独立信息点**，都必须在其末尾使用 `[来源:INDEX]` 的格式清晰标注其来源的文档索引。
#         3.  **多来源标注**: 如果一个句子综合了多个来源的信息，请使用逗号分隔索引，例如 `[来源:2, 5]`。
#         4.  **结构化输出**: 使用Markdown标题、要点和列表来组织回答，使其清晰易读。
#         5.  **无来源则不答**: 如果提供的上下文信息不足以回答问题的某个方面，请明确指出“根据提供的资料，无法回答关于[某方面]的问题”，不要进行猜测。
#         6.  **格式示例**:
#             - Meta在2025年10月22日进行了大规模裁员 [来源:1, 3]。
#             - 新成立的TBD Lab将专注于下一代AI模型的研发 [来源:4]。
#
#         开始分析以下上下文，并生成报告。"""),
#         ("human", "原始问题: {question}\n\n已编号的上下文信息:\n{context}")
#     ])
#
#     # 4. Agent 构建
#     print("   - 构建智能体 (Agents)...")
#     router = build_router_agent(llm)  # --- [NEW]
#     planner = build_planner_agent(llm)
#     rewriter = build_universal_query_rewriter_agent(llm)
#     supervisor = build_retrieval_supervisor_agent(llm)
#     distiller = build_distiller_agent(llm)
#     reflector = build_reflection_agent(llm)
#     policy = build_policy_agent(llm)
#     final_writer = build_final_answer_agent(llm)
#
#     # 5. 图 (Graph) 组装
#     print("   - 编译 LangGraph 工作流...")
#     graph = StateGraph(AgentState)
#
#     # --- 节点注册 ---
#     graph.add_node("router", partial(router_node, router_agent=router))
#
#     # Fast RAG 路径节点
#     graph.add_node("instant_rag", partial(instant_rag_node, query_rewriter_agent=rewriter, retrievers=retrievers))
#     graph.add_node("simple_gen", simple_gen_node) # 新的
#
#     # Deep RAG 路径节点
#     graph.add_node("plan", partial(plan_node, planner_agent=planner))
#     graph.add_node("research_step", partial(
#         research_step_node,
#         query_rewriter_agent=rewriter,
#         supervisor_agent=supervisor,
#         retrievers=retrievers,
#         web_search_tool=web_search_tool
#     ))
#     graph.add_node("distill", partial(distill_node, distiller_agent=distiller))
#     graph.add_node("reflect", partial(reflection_node, reflection_agent=reflector))
#     graph.add_node("generate_final_answer", final_answer_node)
#
#     # --- 边连接 ---
#     graph.set_entry_point("router")
#
#     # 核心分流逻辑
#     graph.add_conditional_edges(
#         "router",
#         route_logic,
#         {
#             "fast_path": "instant_rag",  # 修改
#             "deep_path": "plan"  # 修改
#         }
#     )
#
#     # Fast RAG 路径
#     graph.add_edge("instant_rag", "simple_gen")
#     graph.add_edge("simple_gen", END)  # simple_gen 准备完数据后，图就结束了
#
#     # Deep RAG 路径
#     graph.add_edge("plan", "research_step")
#     graph.add_edge("research_step", "distill")
#     graph.add_edge("distill", "reflect")
#     graph.add_conditional_edges(
#         "reflect",
#         partial(should_continue, policy_agent=policy),
#         {"continue": "research_step", "finish": "generate_final_answer"}
#     )
#     graph.add_edge("generate_final_answer", END)  # generate_final_answer 准备完数据后，图也结束了
#
#     app = graph.compile()
#     print("✅ RAG 引擎初始化成功，支持 Fast/Deep 双模式。")
#     return app, final_writer_prompt, llm