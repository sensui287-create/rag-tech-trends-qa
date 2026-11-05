"""Research graph nodes; final nodes prepare context for the caller's writer."""

import hashlib
import logging
import time
from ..models import AgentState, PastStep, Plan
from ..utils import (
    get_past_context_str, format_docs_for_synthesis,
    preprocess_docs_for_final_answer
)
logger = logging.getLogger(__name__)

def router_node(state: AgentState, router_agent):
    """路由节点"""
    logger.info("--- [Node] Router ---")
    question = state["original_question"]
    decision = router_agent.invoke({"question": question})
    if decision.get("step") == "simple_answer":
        return {"router_decision": "simple_answer"}
    else:
        return {"router_decision": "complex_research"}

def plan_node(state: AgentState, planner_agent):
    """规划节点"""
    logger.info("--- [Node] Planning ---")
    start_t = time.time()
    plan_dict = planner_agent.invoke({"question": state["original_question"]})
    cost_t = time.time() - start_t
    logger.info(f"📊 [PERF_METRIC] 任务拆解与规划耗时: {cost_t:.3f} 秒")
    return {"plan": Plan(**plan_dict), "current_step_index": 0, "past_steps": []}

def global_research_node(state: AgentState, query_rewriter_agent, retrievers):
    """
    [混合模式第一步]
    对原始问题进行一次大规模、高召回的混合检索，构建候选文档池。
    """
    logger.info("--- [Node] Global Research (Hybrid Mode) ---")
    question = state["original_question"]

    # 1. 查询优化
    start_t1 = time.time()
    rewritten = query_rewriter_agent.invoke({
        "user_query": question, "keywords": [], "past_context": "Global Search for Main Question"
    })
    query_zh = rewritten.get('chinese_query', question)
    cost_t1 = time.time() - start_t1
    logger.info(f"📊 [PERF_METRIC] 全局查询重写耗时: {cost_t1:.3f} 秒")
    logger.info(f"   -> [Global Rewriter] 优化后查询: {query_zh}")

    # 2. 大规模混合召回
    start_t2 = time.time()
    hybrid_retriever = retrievers["default_hybrid"]
    candidate_docs = hybrid_retriever.invoke(query_zh)
    cost_t2 = time.time() - start_t2
    logger.info(f"📊 [PERF_METRIC] 全局混合召回(I/O密集)耗时: {cost_t2:.3f} 秒")
    logger.info(f"   -> [Global Hybrid Retrieval] 召回 {len(candidate_docs)} 篇候选文档")

    return {"candidate_docs": candidate_docs}


def research_step_node(state: AgentState, query_rewriter_agent, retrievers):
    """
    [混合模式第二步]
    针对当前子问题，在全局候选池中进行重排，筛选出最相关的文档。
    """
    idx = state["current_step_index"]
    step = state["plan"].steps[idx]
    original_query = step.sub_question
    logger.info(f"--- [Node] Research Step {idx + 1} (Reranking): {original_query} ---")

    # 1. 对子问题进行优化
    start_t1 = time.time()
    past_context = get_past_context_str(state['past_steps'])
    rewritten = query_rewriter_agent.invoke({
        "user_query": original_query,
        "keywords": step.keywords,
        "past_context": past_context
    })
    final_query = rewritten.get('chinese_query', original_query)
    cost_t1 = time.time() - start_t1
    logger.info(f"📊 [PERF_METRIC] 子问题重写耗时: {cost_t1:.3f} 秒")
    logger.info(f"   -> [Sub-query Rewriter] 优化后查询: {final_query}")

    # 2. 在候选池中进行深度重排
    start_t2 = time.time()
    compressor = retrievers["compressor"]
    reranked_docs = []
    candidate_docs = state["candidate_docs"]

    if candidate_docs:
        try:
            reranked_docs = compressor.compress_documents(documents=candidate_docs, query=final_query)
            logger.info(f"   -> [Reranking Pool] 从候选池中筛选出 {len(reranked_docs)} 篇核心文档")
        except Exception as e:
            logger.error(f"   -> [Rerank Error] 重排失败，使用空结果: {e}")
            reranked_docs = []
    cost_t2 = time.time() - start_t2
    logger.info(f"📊 [PERF_METRIC] 深度特征重排耗时: {cost_t2:.3f} 秒")

    # 去重逻辑
    seen_ids = set()
    for past_step in state['past_steps']:
        for doc in past_step['retrieved_docs']:
            content_hash = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
            seen_ids.add(content_hash)

    unique_docs = []
    for doc in reranked_docs:
        content_hash = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
        if content_hash not in seen_ids:
            unique_docs.append(doc)
            seen_ids.add(content_hash)

    return {
        "current_rewritten_query": final_query,
        "retrieved_docs": unique_docs
    }


def distill_node(state: AgentState, distiller_agent):
    """信息蒸馏节点"""
    logger.info("--- [Node] Distill ---")
    start_t = time.time()
    if not state["retrieved_docs"]:
        return {"distilled_context": "未找到与此步骤相关的具体信息。"}
    context = format_docs_for_synthesis(state["retrieved_docs"])
    distilled = distiller_agent.invoke({"question": state["current_rewritten_query"], "context": context})
    cost_t = time.time() - start_t
    logger.info(f"📊 [PERF_METRIC] 信息蒸馏耗时: {cost_t:.3f} 秒")
    return {"distilled_context": distilled}

def reflection_node(state: AgentState, reflection_agent):
    """反思节点"""
    logger.info("--- [Node] Reflect ---")
    step = state["plan"].steps[state["current_step_index"]]
    summary = reflection_agent.invoke({
        "sub_question": step.sub_question,
        "distilled_context": state["distilled_context"]
    })
    new_step = PastStep(
        step_index=state["current_step_index"] + 1,
        sub_question=step.sub_question,
        sub_question_rewritten=state["current_rewritten_query"],
        retrieved_docs=state["retrieved_docs"],
        distilled_context=state["distilled_context"],
        summary=summary
    )
    return {"past_steps": state["past_steps"] + [new_step], "current_step_index": state["current_step_index"] + 1}

def instant_rag_node(state: AgentState, query_rewriter_agent, retrievers):
    """快速模式：单次两阶段检索"""
    logger.info("--- [Node] Instant RAG ---")
    question = state["original_question"]

    rewritten = query_rewriter_agent.invoke({
        "user_query": question, "keywords": [], "past_context": "Instant Mode"
    })
    query_zh = rewritten.get('chinese_query', question)

    retriever = retrievers["default_hybrid"]
    docs = retriever.invoke(query_zh)

    if docs:
        compressor = retrievers["compressor"]
        docs = compressor.compress_documents(documents=docs, query=query_zh)

    return {"retrieved_docs": docs}


def simple_gen_node(state: AgentState):
    """为快速模式准备编号上下文，由调用方执行最终答案生成。"""
    logger.info("--- [Node] Simple Answer Generation ---")
    docs = state.get("retrieved_docs", [])
    if not docs:
        return {"final_answer": "根据现有信息，无法回答您的问题。"}

    context_parts = []
    numbered_metadata = {}
    for i, doc in enumerate(docs):
        idx = i + 1
        title = doc.metadata.get('title', '文档')
        context_parts.append(f"--- 来源 {idx}: 《{title}》 ---\n{doc.page_content}")
        numbered_metadata[idx] = doc.metadata

    context_str = "\n\n".join(context_parts)

    return {
        "final_answer": "",  # 留空，由服务端或实验脚本调用统一的 writer
        "context_for_final_answer": context_str,
        "metadata_for_final_answer": numbered_metadata
    }


def final_answer_node(state: AgentState):
    """为深度模式准备编号上下文，由调用方执行最终答案生成。"""
    logger.info("--- [Node] Final Answer Generation ---")
    past_steps = state.get('past_steps')
    plan = state.get('plan')

    if not past_steps or not plan:
        return {"final_answer": "研究未能完成，无法生成最终报告。"}

    # 预处理上下文
    context, metadata = preprocess_docs_for_final_answer(plan, past_steps)

    return {
        "final_answer": "",
        "context_for_final_answer": context,
        "metadata_for_final_answer": metadata
    }
