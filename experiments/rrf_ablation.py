"""Scan BM25/vector fusion weights while retaining the historical RRF formula."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, TYPE_CHECKING
from experiments._shared import LOCAL_DATA_DIR, RESULTS_DIR, load_records, normalize_id, save_json

if TYPE_CHECKING:
    from langchain_core.documents import Document


class WeightAblationEvaluator:
    def __init__(self, components=None):
        print("Initializing retrieval weight ablation")
        if components is None:
            from backend.rag_core.components.retrievers import setup_retrievers
            components = setup_retrievers()
        self.components = components
        if not self.components:
            raise RuntimeError("检索器初始化失败。")

        self.retrievers = {
            "v_zh": self.components["vector_retriever_zh"],
            "v_en": self.components["vector_retriever_en"],
            "b_zh": self.components["bm25_retriever_zh"],
            "b_en": self.components["bm25_retriever_en"]
        }
        print("Retrieval suite ready")

    def _normalize_id(self, doc_id):
        return normalize_id(doc_id)

    def _extract_ids(self, docs: List[Document], top_k: int) -> List[str]:
        ids = []
        for doc in docs:
            raw_id = doc.metadata.get('source_id') or doc.metadata.get('id')
            if raw_id:
                ids.append(self._normalize_id(raw_id))
        return ids[:top_k]

    def _weighted_rrf_merge(self, doc_lists: List[List[Document]], weights: List[float], top_k: int = 5) -> List[
        Document]:
        """带权重的 RRF 融合实现 (用于消融实验)"""
        scores = {}
        doc_map = {}

        for docs, weight in zip(doc_lists, weights):
            if weight == 0:
                continue  # 如果权重为0，直接跳过该路

            for rank, doc in enumerate(docs):
                d_id = self._normalize_id(doc.metadata.get('source_id') or doc.metadata.get('id'))
                if d_id not in doc_map:
                    doc_map[d_id] = doc
                scores[d_id] = scores.get(d_id, 0) + weight * (1.0 / (60 + rank))

        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return [doc_map[sid] for sid in sorted_ids[:top_k]]

    def run_ablation_step(self, query: str, alpha: float) -> List[str]:
        """执行单词查询的消融评测，alpha 为 BM25 的权重"""
        weight_bm25 = alpha
        weight_vector = 1.0 - alpha

        self.retrievers["v_zh"].search_kwargs['k'] = 20
        self.retrievers["v_en"].search_kwargs['k'] = 20
        self.retrievers["b_zh"].k = 20
        self.retrievers["b_en"].k = 20

        docs_v_zh = self.retrievers["v_zh"].invoke(query)
        docs_v_en = self.retrievers["v_en"].invoke(query)
        docs_b_zh = self.retrievers["b_zh"].invoke(query)
        docs_b_en = self.retrievers["b_en"].invoke(query)

        doc_lists = [docs_v_zh, docs_v_en, docs_b_zh, docs_b_en]
        weights = [weight_vector, weight_vector, weight_bm25, weight_bm25]

        merged_docs = self._weighted_rrf_merge(doc_lists, weights, top_k=5)
        return self._extract_ids(merged_docs, 5)

def run_evaluation(dataset_path, output_path):
    dataset = load_records(dataset_path, ("query", "golden_doc_id"))
    if any(not isinstance(row["query"], str) for row in dataset):
        raise ValueError("Every query must be a string")
    from tqdm import tqdm
    import pandas as pd
    evaluator = WeightAblationEvaluator()
    alphas = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    results = []
    detail_path = Path(output_path).with_suffix(".details.json")
    try:
        for row in tqdm(dataset, desc="Processing Queries"):
            golden = normalize_id(row["golden_doc_id"])
            for alpha in alphas:
                record = {"Query ID": row.get("id"), "Alpha (BM25)": alpha,
                          "Weight (Vector)": 1-alpha, "Recall@5": 0.0, "Status": "ok"}
                try:
                    record["Recall@5"] = float(golden in evaluator.run_ablation_step(row["query"], alpha))
                except Exception as exc:
                    record.update({"Status": "error", "Error": str(exc)})
                results.append(record)
            save_json(detail_path, results)
    finally:
        save_json(detail_path, results)
    frame = pd.DataFrame(results)
    summary = frame.groupby(["Alpha (BM25)", "Weight (Vector)"])[["Recall@5"]].mean().reset_index()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    print(summary.to_markdown(index=False))
    failures = sum(row["Status"] == "error" for row in results)
    print(f"Failed weight/query pairs: {failures}/{len(results)} (included as zero scores)")
    return failures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=LOCAL_DATA_DIR / "thesis_benchmark_dataset_500.json")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "ablation_weights_recall5.csv")
    args = parser.parse_args(argv)
    return 1 if run_evaluation(args.input, args.output) else 0


if __name__ == "__main__":
    raise SystemExit(main())
