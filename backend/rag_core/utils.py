"""Format research history and deduplicated, numbered source context."""

import hashlib
from typing import List, Tuple, Dict
from .models import PastStep, Plan
from langchain_core.documents import Document

def preprocess_docs_for_final_answer(plan: Plan, past_steps: List[PastStep]) -> Tuple[str, dict]:
    """
    根据研究计划和历史步骤，预处理所有文档，去重并格式化为带有子问题标题的结构化上下文。
    """
    if not past_steps:
        return "未能完成任何研究步骤，无法生成报告。", {}

    # 使用内容哈希进行全局去重
    seen_doc_hashes = set()
    # 建立一个从子问题到其相关文档的映射
    step_to_docs_map: Dict[str, List[Document]] = {step.sub_question: [] for step in plan.steps}

    # 填充映射，并进行去重
    for past_step in past_steps:
        sub_question = past_step['sub_question']
        if sub_question in step_to_docs_map:
            for doc in past_step['retrieved_docs']:
                content_hash = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
                if content_hash not in seen_doc_hashes:
                    step_to_docs_map[sub_question].append(doc)
                    seen_doc_hashes.add(content_hash)

    # 为所有唯一文档进行全局编号
    all_unique_docs = [doc for docs in step_to_docs_map.values() for doc in docs]

    numbered_sources = {i + 1: doc for i, doc in enumerate(all_unique_docs)}
    # 创建一个从文档内容哈希到其全局索引的映射，以便后续引用
    hash_to_id_map = {hashlib.md5(doc.page_content.encode('utf-8')).hexdigest(): i + 1 for i, doc in
                      enumerate(all_unique_docs)}

    context_parts = ["以下是本次研究报告的支撑材料，按研究步骤组织：\n"]

    # 按计划的步骤顺序构建上下文
    for step in plan.steps:
        sub_question = step.sub_question
        docs_for_this_step = step_to_docs_map.get(sub_question, [])

        if docs_for_this_step:
            context_parts.append(f"--- 研究步骤: “{sub_question}” 的相关信息 ---")
            for doc in docs_for_this_step:
                content_hash = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
                doc_id = hash_to_id_map[content_hash]
                title = doc.metadata.get('title', '未知文档')
                context_parts.append(f"来源 {doc_id}: 《{title}》\n内容: {doc.page_content}\n")
            context_parts.append("\n")

    if len(context_parts) == 1:
        return "在所有研究步骤中均未找到有效的、唯一的上下文信息。", {}

    formatted_context = "\n".join(context_parts)
    numbered_metadata = {index: doc.metadata for index, doc in numbered_sources.items()}

    return formatted_context, numbered_metadata

def format_docs_for_synthesis(docs: List[Document]) -> str:
    if not docs: return "没有找到相关的上下文信息。"
    formatted_strings = []
    for i, doc in enumerate(docs):
        title = doc.metadata.get('title', '未知')
        formatted_strings.append(f" 文档 {i + 1} ({title}) ---\n{doc.page_content}")
    return "\n\n".join(formatted_strings)

def get_past_context_str(past_steps: List[PastStep]) -> str:
    if not past_steps: return "还未进行任何研究。"
    return "\n\n".join([f"第 {s['step_index']} 步: {s['sub_question']}\n发现摘要: {s['summary']}" for s in past_steps])
