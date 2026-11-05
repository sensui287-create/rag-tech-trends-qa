"""Generate CoT, Naive RAG and Agentic RAG answers for local TXT/JSON inputs.

Run: python -m experiments.agent_experiment --help
TXT question lists do not contain the evidence needed by agent_results. For
supported judging, use JSON records containing contexts and ground_truth.
"""

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from experiments._shared import LOCAL_DATA_DIR, RESULTS_DIR, save_json

INPUT_DATASET = LOCAL_DATA_DIR / "ceshi.txt"
OUTPUT_RESULTS = RESULTS_DIR / "Experiment_Results.json"

def answer_text(value):
    return str(getattr(value, "content", value) or "").strip()

def load_dataset(path):
    path = Path(path)
    if path.suffix.lower() == ".json":
        dataset = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(dataset, list):
            raise ValueError("Dataset must be a JSON array")
    else:
        dataset = []
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            match = re.match(r"\s*-\s*Question\s+\d+\s*[:：]\s*(.+)", line)
            if match:
                dataset.append({"id": len(dataset) + 1, "question": match.group(1).strip()})
    if not dataset:
        raise ValueError("Dataset contains no questions")
    for index, item in enumerate(dataset, 1):
        if not isinstance(item, dict) or not isinstance(item.get("question"), str) or not item["question"].strip():
            raise ValueError(f"Record {index} must contain a nonempty question")
        if "contexts" in item and (not isinstance(item["contexts"], list) or any(not isinstance(c, str) for c in item["contexts"])):
            raise ValueError(f"Record {index}: contexts must be a list of strings")
        if "ground_truth" in item and not isinstance(item["ground_truth"], str):
            raise ValueError(f"Record {index}: ground_truth must be a string")
    return dataset

def run_cot_baseline(question: str, llm) -> str:
    """Baseline 1: 无检索，纯大模型 Chain-of-Thought"""
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一位全球顶尖的战略分析师和科技顾问。你的任务是凭借你的专业知识，通过严密的逻辑推理，对用户的原始问题给出一份【高度整合、极具启发性】的分析报告。

     核心痛点警告（必须克服）：
     不要做“信息的搬运工”！如果你只是按资料分段罗列、拼凑，报告会显得极其松散且缺乏逻辑。你必须将所有碎片化的信息“融毁重铸”，抽丝剥茧，形成一条贯穿始终的逻辑主线。

     强制输出结构（严格遵循）：
     1. 【核心定调】（1段）：开门见山，直接给出针对原始问题的最核心、最具穿透力的结论。拒绝废话和套话。
     2. 【深度综合剖析】（主体）：
         - 严禁按“关于A...关于B...”的机械结构写作。
         - 必须围绕核心结论，建立清晰的论证逻辑（如从底层技术->产品形态->市场商业模式->未来影响）。
         - 将不同来源的信息有机缝合在你的逻辑链条中。
     3. 【战略洞察与启示】：提取具有前瞻性、可操作性的见解。告诉读者“这本质上意味什么”、“未来的演进节点在哪里”或“应当如何应对”。这部分是体现你专家价值的关键！

注意：作为专业顾问，请确保即使没有外部参考资料，你的回答也具有极高的逻辑连贯性和洞察力，拒绝废话。"""),
    ("human", "原始问题: {question}")
])
    chain = prompt | llm
    try:
        return chain.invoke({"question": question})
    except Exception as e:
        raise RuntimeError(f"CoT generation failed: {e}") from e

def run_naive_rag_baseline(question: str, hybrid_retriever, reranker, llm) -> str:
    """Baseline 2: Naive RAG (单次混合检索 -> 按配置重排（默认 Top-5，最多取 10） -> 直接生成)"""
    from langchain_core.prompts import ChatPromptTemplate

    try:
        initial_docs = hybrid_retriever.invoke(question)
        if initial_docs:
            reranked_docs = reranker.compress_documents(documents=initial_docs, query=question)
            top_docs = reranked_docs[:10]  # 强行塞入 Top-10
        else:
            top_docs = []

        context_parts = []
        for i, doc in enumerate(top_docs):
            title = doc.metadata.get('title', '未知')
            context_parts.append(f"[文档 {i + 1}]: {title}\n{doc.page_content}")
        context_str = "\n\n".join(context_parts) if context_parts else "未检索到相关信息。"

        prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一位全球顶尖的战略分析师和科技顾问。你的任务是综合下方检索到的研究资料，对用户的原始问题给出一份【逻辑严密、高度整合、极具启发性】的最终分析报告。

     核心痛点警告（必须克服）：
     不要做“信息的搬运工”！如果你只是按资料分段罗列、拼凑，报告会显得极其松散且缺乏逻辑。你必须将所有碎片化的信息“融毁重铸”，抽丝剥茧，形成一条贯穿始终的逻辑主线。

     强制输出结构（严格遵循）：
     1. 【核心定调】（1段）：开门见山，直接给出针对原始问题的最核心、最具穿透力的结论。拒绝废话和套话。
     2. 【深度综合剖析】（主体）：
         - 严禁按“关于A...关于B...”的机械结构写作。
         - 必须围绕核心结论，建立清晰的论证逻辑（如从底层技术->产品形态->市场商业模式->未来影响）。
         - 将不同来源的信息有机缝合在你的逻辑链条中。
     3. 【战略洞察与启示】：提取具有前瞻性、可操作性的见解。告诉读者“这本质上意味什么”、“未来的演进节点在哪里”或“应当如何应对”。这部分是体现你专家价值的关键！

     ！！！引用与合规规则（不可违背）！！！:
     1. 绝对忠于原文，严禁引入任何外部知识或进行无事实依据的幻觉推演。
     2. 句句皆有出处：所有观点和数据必须在句子末尾使用`[来源:INDEX]` 标注，如 `[来源:1, 3]`。
     3. 无来源则不答：如果资料完全不支持问题的某个部分，明确指出“现有资料不足以回答某部分”。"""),
    ("human", "原始问题: {question}\n\n检索到的支撑材料:\n{context}")
])
        chain = prompt | llm
        return chain.invoke({"context": context_str, "question": question})
    except Exception as e:
        raise RuntimeError(f"Naive RAG generation failed: {e}") from e

def run_agentic_rag(question, graph, final_writer_prompt, llm):
    """Finish the graph's deferred writer, matching the server generation path."""
    initial_state = {
        "original_question": question, "router_decision": "", "past_steps": [],
        "current_step_index": 0, "current_rewritten_query": "", "retrieved_docs": [],
        "candidate_docs": [], "distilled_context": "", "final_answer": "",
        "context_for_final_answer": None, "metadata_for_final_answer": None,
    }
    final_state = graph.invoke(initial_state)
    context = final_state.get("context_for_final_answer")
    if context:
        result = (final_writer_prompt | llm).invoke({"question": question, "context": context})
    else:
        result = final_state.get("final_answer", "")
    result = answer_text(result)
    if not result:
        raise RuntimeError("Agentic RAG returned no answer")
    return result

def process_single_item(item, idx, llm, hybrid_retriever, reranker, graph, final_writer_prompt):
    question = item["question"]
    answers, errors = {}, {}
    methods = {
        "CoT": lambda: run_cot_baseline(question, llm),
        "Naive_RAG": lambda: run_naive_rag_baseline(question, hybrid_retriever, reranker, llm),
        "Agentic_RAG": lambda: run_agentic_rag(question, graph, final_writer_prompt, llm),
    }
    for name, generate in methods.items():
        try:
            answers[name] = answer_text(generate())
            if not answers[name]:
                raise RuntimeError("Model returned an empty answer")
        except Exception as exc:
            answers[name] = ""
            errors[name] = str(exc)
    record = {
        "id": item.get("id", idx + 1), "question": question,
        "contexts": item.get("contexts", []), "ground_truth": item.get("ground_truth", ""),
        "answers": answers,
    }
    if errors:
        record["errors"] = errors
    return record

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_DATASET)
    parser.add_argument("--output", type=Path, default=OUTPUT_RESULTS)
    parser.add_argument("--workers", type=int, default=1, help="Shared retrievers/models: use 1 unless their thread safety is verified")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    dataset = load_dataset(args.input)
    from backend.rag_core.components.retrievers import setup_retrievers
    from backend.rag_core.graph import initialize_rag_system

    retrievers = setup_retrievers()
    if not retrievers:
        raise RuntimeError("Retriever initialization failed")
    graph, prompt, writer_llm = initialize_rag_system(retrievers=retrievers)
    llm = writer_llm
    indexed_results = {}
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_single_item, item, index, llm,
                                retrievers["default_hybrid"], retrievers["compressor"], graph, prompt): index
                for index, item in enumerate(dataset)
            }
            for future in as_completed(futures):
                index = futures[future]
                indexed_results[index] = future.result()
                save_json(args.output, [indexed_results[i] for i in sorted(indexed_results)])
                print(f"Saved {len(indexed_results)}/{len(dataset)}")
    finally:
        if indexed_results:
            save_json(args.output, [indexed_results[i] for i in sorted(indexed_results)])
    failures = sum(bool(record.get("errors")) for record in indexed_results.values())
    print(f"Saved {len(indexed_results)} records to {args.output}; failures: {failures}")
    return 1 if failures else 0

if __name__ == "__main__":
    raise SystemExit(main())
