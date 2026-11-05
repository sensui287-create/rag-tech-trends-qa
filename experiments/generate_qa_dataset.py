"""Generate two retrieval evaluation subsets from authorized local CSV files."""

import argparse
import json
import random
from pathlib import Path
from experiments._shared import LOCAL_DATA_DIR, RESULTS_DIR, load_news_csv, save_json

FILE_ZH = LOCAL_DATA_DIR / "cleaned_tech_news_rag_ready.csv"
FILE_EN = LOCAL_DATA_DIR / "hacker_news.csv"
OUTPUT_FILE = RESULTS_DIR / "thesis_benchmark_dataset_500.json"


def load_documents(zh_path=FILE_ZH, en_path=FILE_EN):
    documents = load_news_csv(zh_path, "zh") + load_news_csv(en_path, "en")
    return [doc for doc in documents if len(doc["text"]) > 50 and doc["content"]]


def call_llm_for_question(system_prompt: str, user_prompt: str) -> dict:
    """调用大模型并强制返回 JSON 格式"""
    from experiments.experiment_config import EVAL_MODEL as MODEL_NAME, create_eval_client

    client = create_eval_client()
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.8,  # 稍微调高温度，增加问题的多样性
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        if "```json" in content:
            content = content.replace("```json", "").replace("```", "")
        return json.loads(content)
    except Exception as e:
        print(f"\n[LLM Error] 生成失败: {e}")
        return None

def generate_qa_pair(doc: dict, subset_type: str) -> dict:
    """根据论文定义的子集类型生成特定风格的问题"""

    base_system = """你是一个严谨的 RAG 系统评测专家。你的任务是根据提供的文档内容，生成一个高质量的中文测试问题。
    输出必须是严格的 JSON 格式：{"question": "你的问题", "answer": "简短的参考答案"}
    """

    if subset_type == "subset_a":
        instruction = """
        【当前任务】：生成一个属于“子集 A（实体约束型）”的测试问题。

        【设计要求】：
        1. **关注细节**：问题必须针对文档中的具体实体（如人名、公司名、芯片型号、软件版本号）、具体日期（年月日）或具体数据（价格、参数）。
        2. **关键字匹配**：这类问题通常依赖关键词匹配。问题中必须显式包含文档中的某个专有名词。
        3. **拒绝模糊**：不要问“讲了什么”、“有什么意义”，要问“是多少”、“是哪天”、“是谁”。
        4. **语言要求**：无论参考文档是中文还是英文，**必须用中文提问**。

        【参考范例】：
        - "2024年11月发布的 Qwen3-Max 模型的上下文窗口具体是多少？"
        - "英伟达 RTX 5090 显卡的显存位宽参数是多少？"
        - "OpenAI 的 o1 模型是在哪一天正式上线的？"
        """
    else:
        instruction = """
        【当前任务】：生成一个属于“子集 B（语义概念型）”的测试问题。

        【设计要求】：
        1. **关注逻辑与概念**：问题应侧重于原理解释、趋势分析、优缺点对比、或者对某个抽象概念的理解。
        2. **口语化/模糊化**：模拟用户不知道具体专业术语时的提问方式，或者使用稍微口语化的表达。
        3. **语义匹配**：这类问题依赖向量语义检索。不要直接照搬文档中的原句，要进行同义改写或宏观提问。
        4. **语言要求**：无论参考文档是中文还是英文，**必须用中文提问**。

        【参考范例】：
        - "介绍一下近年来开源大语言模型在多模态对齐上的主要发展瓶颈。"
        - "为什么最近大家都在讨论 Transformer 架构的线性化改进？"
        - "RAG 技术相比于微调，在处理私有数据方面有什么本质的区别？"
        """

    user_prompt = f"{instruction}\n\n【参考文档】（ID: {doc['id']}）：\n{doc['text'][:2500]}"

    res = call_llm_for_question(base_system, user_prompt)

    if res and "question" in res:
        return {
            "query": res["question"],
            "reference_answer": res.get("answer", "详见文档"),
            "golden_doc_id": doc['id'],  # 记录由于哪个文档生成的，作为 Ground Truth
            "subset": subset_type
        }
    return None

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zh", type=Path, default=FILE_ZH)
    parser.add_argument("--en", type=Path, default=FILE_EN)
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    parser.add_argument("--per-subset", type=int, default=250)
    parser.add_argument("--max-attempts", type=int, default=2500, help="Maximum calls per subset")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    if args.per_subset < 1 or args.max_attempts < 1:
        parser.error("--per-subset and --max-attempts must be positive")
    documents = load_documents(args.zh, args.en)
    if not documents:
        parser.error("No usable documents found")
    if args.seed is not None:
        random.seed(args.seed)
    from experiments.experiment_config import create_eval_client
    create_eval_client()
    dataset, seen = [], set()
    try:
        for subset in ("subset_a", "subset_b"):
            count = 0
            for attempt in range(args.max_attempts):
                if count >= args.per_subset:
                    break
                item = generate_qa_pair(random.choice(documents), subset)
                if item and isinstance(item.get("query"), str) and item["query"].strip() and item["query"] not in seen:
                    item["id"] = len(dataset) + 1
                    dataset.append(item)
                    seen.add(item["query"])
                    count += 1
                    save_json(args.output, dataset)
                    print(f"{subset}: {count}/{args.per_subset}")
    finally:
        save_json(args.output, dataset)
    print(f"Saved {len(dataset)} records to {args.output}")
    return 0 if len(dataset) == args.per_subset * 2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
