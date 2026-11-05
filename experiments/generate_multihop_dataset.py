"""Generate evidence-grounded multi-document questions with bounded attempts."""

import argparse
import json
import random
from pathlib import Path
from experiments._shared import LOCAL_DATA_DIR, RESULTS_DIR, load_news_csv, save_json

FILE_ZH = LOCAL_DATA_DIR / "cleaned_tech_news_rag_ready.csv"
FILE_EN = LOCAL_DATA_DIR / "hacker_news.csv"
OUTPUT_FILE = RESULTS_DIR / "Tech_MultiHop_QA_RAGAS_200.json"


def sample_documents(documents_zh, documents_en, sample_size=3):
    full, snippets = [], []
    if not documents_zh or not documents_en:
        raise ValueError("Both language corpora must contain documents")
    for _ in range(sample_size):
        language = "zh" if random.random() > 0.5 else "en"
        doc = random.choice(documents_zh if language == "zh" else documents_en)
        full.append(f"{doc['title']}\n{doc['content']}")
        label = "中文报道" if language == "zh" else "English News"
        snippets.append(f"[{label} - {doc['date']}] {doc['title']}: {doc['content'][:500]}")
    return full, snippets


def load_and_sample_docs(zh_path, en_path, sample_size=3):
    return sample_documents(load_news_csv(zh_path, "zh"), load_news_csv(en_path, "en"), sample_size)


def generate_multihop_qa_for_ragas(docs_for_prompt, num_questions=2):
    """
    调用 gpt-4o-mini 基于文档生成多跳问题和答案，格式适配RAGAS。

    Args:
        docs_for_prompt (list[str]): 用于构建提示词的格式化文档片段列表。
        num_questions (int): 希望生成的问题数量。

    Returns:
        list[dict]: 包含 'question' 和 'ground_truth' 的字典列表。
    """
    context = "\n\n".join(docs_for_prompt)

    prompt = f"""
    你是一位顶级的科技分析师，任务是基于以下几段可能互不相关的科技新闻，设计出需要综合、推理、比较才能回答的【多跳问题】。

    # 指令
    1.  **生成复杂问题**：问题必须跨越至少两段提供的文档信息，不能通过单一文档直接回答。问题类型可以是趋势对比、因果推导或技术影响分析。
    2.  **提供精准答案**：为每个问题提供一个简明扼要的答案（`ground_truth`）。这个答案必须且只能从我提供的文档内容中推导出来，不能包含任何外部知识。
    3.  **严格的JSON输出**：返回一个JSON对象，其`questions`键对应一个数组，不包含解释性文字。数组中的每个对象包含`question`和`ground_truth`两个键。

    # 提供的文档片段
    {context}

    # 输出格式示例
    {{"questions": [
        {{
            "question": "结合A公司的硬件发布和B公司的软件策略，分析未来智能穿戴市场的潜在竞争格局。",
            "ground_truth": "A公司推出的新硬件专注于长续航，而B公司的软件策略则侧重于开放生态。这预示着未来市场竞争将围绕硬件续航和软件生态的整合展开，而非单一功能竞争。"
        }},
        {{
            "question": "...",
            "ground_truth": "..."
        }}
    ]}}

    请严格按照以上要求，生成 {num_questions} 个问答对。
    """

    from experiments.experiment_config import EVAL_MODEL, create_eval_client

    client = create_eval_client()
    try:
        response = client.chat.completions.create(
            model=EVAL_MODEL,
            messages=[
                {"role": "system",
                 "content": "You are an expert in creating complex questions and answers from provided texts, and you strictly output in JSON format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content

        data = json.loads(content)

        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, list):
                    return value
        elif isinstance(data, list):
            return data

        print(f"警告：收到了非预期的JSON结构: {content}")
        return []

    except json.JSONDecodeError:
        print(f"错误：无法解析LLM返回的JSON。响应内容: {content}")
        return []
    except Exception as e:
        print(f"调用API时发生错误: {e}")
        return []

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zh", type=Path, default=FILE_ZH)
    parser.add_argument("--en", type=Path, default=FILE_EN)
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    parser.add_argument("--target", type=int, default=200)
    parser.add_argument("--max-attempts", type=int, default=2000)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    if args.target < 1 or args.max_attempts < 1:
        parser.error("--target and --max-attempts must be positive")
    zh = [doc for doc in load_news_csv(args.zh, "zh") if doc["content"]]
    en = [doc for doc in load_news_csv(args.en, "en") if doc["content"]]
    if not zh or not en:
        parser.error("Both language corpora must contain usable documents")
    if args.seed is not None:
        random.seed(args.seed)
    from experiments.experiment_config import create_eval_client
    create_eval_client()
    dataset, seen = [], set()
    try:
        for attempt in range(args.max_attempts):
            if len(dataset) >= args.target:
                break
            full, snippets = sample_documents(zh, en)
            items = generate_multihop_qa_for_ragas(snippets, min(3, args.target-len(dataset)))
            for item in items:
                if len(dataset) >= args.target:
                    break
                if not isinstance(item, dict):
                    continue
                question, answer = item.get("question"), item.get("ground_truth")
                if not isinstance(question, str) or not question.strip() or question in seen:
                    continue
                if not isinstance(answer, str) or not answer.strip():
                    continue
                dataset.append({"id": f"multihop_{len(dataset)+1}", "subset": "subset_c_multihop",
                                "question": question, "contexts": full, "ground_truth": answer})
                seen.add(question)
                save_json(args.output, dataset)
                print(f"Saved {len(dataset)}/{args.target}")
    finally:
        save_json(args.output, dataset)
    return 0 if len(dataset) == args.target else 1


if __name__ == "__main__":
    raise SystemExit(main())
