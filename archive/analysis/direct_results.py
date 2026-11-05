"""Historical order-aligned answer comparison with missing external inputs.

Reference only. Prefer experiments.agent_results with evidence and stable IDs.
Run from the project root as python -m archive.analysis.direct_results only
when the three original, correctly aligned input files have been prepared.
"""

import re
import json
import time
from experiments.experiment_config import EVAL_MODEL, create_eval_client
from tqdm import tqdm  # 如果没有安装，请运行: pip install tqdm

QUERY_FILE = "cs_questions_xiaorong.txt"
RESULT1_FILE = "cs_zoomrag_xiaorong.json"
RESULT2_FILE = "cs_zoomrag_no_tiaohe.json"
OUTPUT_RESULTS_FILE = "eval_direct_details.json"

def get_client():
    """Use the separately configured evaluation provider."""
    return create_eval_client()

def direct_eval():
    client = get_client()

    print("正在加载输入文件...")
    with open(QUERY_FILE, "r", encoding="utf-8") as f:
        data = f.read()
    queries = re.findall(r"- Question \d+: (.+)", data)

    with open(RESULT1_FILE, "r", encoding="utf-8") as f:
        answers1 = [i["result"] for i in json.load(f)]

    with open(RESULT2_FILE, "r", encoding="utf-8") as f:
        answers2 = [i["result"] for i in json.load(f)]

    total_tasks = min(len(queries), len(answers1), len(answers2))
    print(f"共加载 {total_tasks} 条评测任务。")

    categories = ["Comprehensiveness", "Diversity", "Empowerment", "Overall Winner"]
    stats = {cat: {"Model A": 0, "Model B": 0, "Error/Tie": 0} for cat in categories}
    detailed_results = []

    print("\n开始实时评测 (使用 gpt-4o-mini)...")
    for i in tqdm(range(total_tasks)):
        query = queries[i]
        ans1 = answers1[i]
        ans2 = answers2[i]

        is_swapped = (i % 2 == 1)
        if is_swapped:
            final_a1, final_a2 = ans2, ans1  # Answer 1 是模型 B, Answer 2 是模型 A
        else:
            final_a1, final_a2 = ans1, ans2  # Answer 1 是模型 A, Answer 2 是模型 B

        sys_prompt = """
                ---Role---
                You are an expert tasked with evaluating two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.
                """

        prompt = f"""
                You will evaluate two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.

                - **Comprehensiveness**: How much detail does the answer provide to cover all aspects and details of the question?
                - **Diversity**: How varied and rich is the answer in providing different perspectives and insights on the question?
                - **Empowerment**: How well does the answer help the reader understand and make informed judgments about the topic?

                For each criterion, choose the better answer (either Answer 1 or Answer 2) and explain why. Then, select an overall winner based on these three categories.

                Here is the question:
                {query}

                Here are the two answers:

                **Answer 1:**
                {final_a1}

                **Answer 2:**
                {final_a2}

                Evaluate both answers using the three criteria listed above and provide detailed explanations for each criterion.

                Output your evaluation in the following JSON format:

                {{
                    "Comprehensiveness": {{
                        "Winner": "[Answer 1 or Answer 2]",
                        "Explanation": "[Provide explanation here]"
                    }},
                    "Diversity": {{
                        "Winner": "[Answer 1 or Answer 2]",
                        "Explanation": "[Provide explanation here]"
                    }},
                    "Empowerment": {{
                        "Winner": "[Answer 1 or Answer 2]",
                        "Explanation": "[Provide explanation here]"
                    }},
                    "Overall Winner": {{
                        "Winner": "[Answer 1 or Answer 2]",
                        "Explanation": "[Summarize why this answer is the overall winner based on the three criteria]"
                    }}
                }}
                """
        try:
            response = client.chat.completions.create(
                model=EVAL_MODEL,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                response_format={"type": "json_object"}  # 强制 JSON 输出
            )

            res_content = response.choices[0].message.content
            eval_result = json.loads(res_content)

            for cat in categories:
                winner_text = ""
                if cat in eval_result:
                    winner_text = str(eval_result[cat].get("Winner", "")).lower()

                real_winner = "Error"
                if "answer 1" in winner_text:
                    real_winner = "Model B" if is_swapped else "Model A"
                elif "answer 2" in winner_text:
                    real_winner = "Model A" if is_swapped else "Model B"

                if real_winner in stats[cat]:
                    stats[cat][real_winner] += 1
                else:
                    stats[cat]["Error/Tie"] += 1

            detailed_results.append({
                "index": i,
                "is_swapped": is_swapped,
                "raw_response": eval_result
            })

        except Exception as e:
            print(f"\n[Error] Task {i} failed: {e}")
            for cat in categories:
                stats[cat]["Error/Tie"] += 1

    print("\n" + "=" * 65)
    print(f"{'CATEGORY':<20} | {'MODEL A':<10} | {'MODEL B':<10} | {'WIN RATE A':<10}")
    print("=" * 65)

    for cat in categories:
        a_wins = stats[cat]["Model A"]
        b_wins = stats[cat]["Model B"]
        total_valid = a_wins + b_wins
        win_rate = (a_wins / total_valid * 100) if total_valid > 0 else 0
        print(f"{cat:<20} | {a_wins:<10} | {b_wins:<10} | {win_rate:.1f}%")
    print("-" * 65)

    with open(OUTPUT_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=4)
    print(f"详细评测结果已保存至: {OUTPUT_RESULTS_FILE}")

if __name__ == "__main__":
    direct_eval()
