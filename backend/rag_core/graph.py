"""Assemble the shared RAG research graph and final response prompt."""

import logging
from functools import partial
from typing import Any, Dict, Optional
from langgraph.graph import StateGraph, END
from langchain_community.llms import Tongyi as ChatTongyi
from .utils import get_past_context_str
from .config import LLM_MODEL, FAST_LLM_MODEL, MAX_RESEARCH_ITERATIONS
from .models import AgentState
from .components.retrievers import setup_retrievers
from .components import agents
from .components import nodes

logger = logging.getLogger(__name__)

def route_logic(state: AgentState):
    """路由决策逻辑"""
    if state.get("router_decision") == "simple_answer":
        return "fast_path"
    return "deep_path"

def should_continue(state: AgentState, policy_agent):
    """判断是否继续研究的决策逻辑"""
    if state["current_step_index"] >= MAX_RESEARCH_ITERATIONS:
        return "finish"

    if state["current_step_index"] >= len(state["plan"].steps):
        return "finish"

    decision = policy_agent.invoke({
        "question": state["original_question"],
        "plan": state["plan"].model_dump_json(),
        "history": get_past_context_str(state["past_steps"])
    })

    if decision.get("next_action") == "FINISH":
        return "finish"
    return "continue"


def initialize_rag_system(retrievers: Optional[Dict[str, Any]] = None):
    """Build the graph, reusing an existing retriever suite when supplied.

    The graph prepares numbered context; callers render the final answer using
    the returned prompt and LLM. Experiments can share one loaded retriever suite
    between their baseline and agentic runs.
    """
    logger.info("\n🚀 RAG 系统后端引擎启动中...")

    if retrievers is None:
        retrievers = setup_retrievers()
    if not retrievers:
        raise RuntimeError("检索器初始化失败")

    # 使用不同模型执行不同任务
    llm = ChatTongyi(model_name=LLM_MODEL, temperature=0.3)
    fast_llm = ChatTongyi(model_name=FAST_LLM_MODEL, temperature=0.1)

    # 构建所有Agent
    router_agent = agents.build_router_agent(fast_llm)
    planner_agent = agents.build_planner_agent(llm)
    query_rewriter_agent = agents.build_universal_query_rewriter_agent(llm)
    distiller_agent = agents.build_distiller_agent(llm)
    reflection_agent = agents.build_reflection_agent(llm)
    policy_agent = agents.build_policy_agent(fast_llm)

    # 构建图
    graph = StateGraph(AgentState)

    # 定义所有节点
    graph.add_node("router", partial(nodes.router_node, router_agent=router_agent))
    graph.add_node("instant_rag",
                   partial(nodes.instant_rag_node, query_rewriter_agent=query_rewriter_agent, retrievers=retrievers))
    graph.add_node("simple_gen", nodes.simple_gen_node)

    # 深度路径的节点
    graph.add_node("plan", partial(nodes.plan_node, planner_agent=planner_agent))
    graph.add_node("global_research", partial(nodes.global_research_node, query_rewriter_agent=query_rewriter_agent,
                                              retrievers=retrievers))
    graph.add_node("research_step",
                   partial(nodes.research_step_node, query_rewriter_agent=query_rewriter_agent, retrievers=retrievers))
    graph.add_node("distill", partial(nodes.distill_node, distiller_agent=distiller_agent))
    graph.add_node("reflect", partial(nodes.reflection_node, reflection_agent=reflection_agent))
    graph.add_node("generate_final_answer", nodes.final_answer_node)

    # 定义图的边
    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        route_logic,
        {"fast_path": "instant_rag", "deep_path": "plan"}
    )

    # 快速路径
    graph.add_edge("instant_rag", "simple_gen")
    graph.add_edge("simple_gen", END)

    # 深度路径（混合模式）
    graph.add_edge("plan", "global_research")
    graph.add_edge("global_research", "research_step")
    graph.add_edge("research_step", "distill")
    graph.add_edge("distill", "reflect")

    graph.add_conditional_edges(
        "reflect",
        partial(should_continue, policy_agent=policy_agent),
        {"continue": "research_step", "finish": "generate_final_answer"}
    )

    graph.add_edge("generate_final_answer", END)

    # 编译图
    # Each research iteration traverses research_step, distill and reflect.
    # Include router, planning, global retrieval, final preparation and termination
    # so larger MAX_RESEARCH_ITERATIONS values do not hit LangGraph's default cap.
    app = graph.compile().with_config(
        {"recursion_limit": max(25, 3 * MAX_RESEARCH_ITERATIONS + 5)}
    )
    logger.info("✅ RAG 系统图编译完成，进入待命状态。")
    final_writer_prompt = agents.get_final_answer_prompt()
    return app, final_writer_prompt, llm
