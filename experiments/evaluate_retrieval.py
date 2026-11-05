"""Evaluate four retrieval pipelines; keep failed queries in the denominator."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, TYPE_CHECKING
from experiments._shared import LOCAL_DATA_DIR, RESULTS_DIR, load_records, normalize_id, save_json

if TYPE_CHECKING:
    from langchain_core.documents import Document


class ThesisExperimentEvaluator:
    def __init__(self, components=None):
        print("Initializing retrieval evaluation")
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
        self.compressor = self.components["compressor"]
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

    def _rrf_merge(self, doc_lists: List[List[Document]], top_k: int = 5) -> List[Document]:
        """手动 RRF 融合，确保不出现空列表异常"""
        scores = {}
        doc_map = {}
        for docs in doc_lists:
            for rank, doc in enumerate(docs):
                d_id = self._normalize_id(doc.metadata.get('source_id') or doc.metadata.get('id'))
                if d_id not in doc_map: doc_map[d_id] = doc
                scores[d_id] = scores.get(d_id, 0) + 1 / (60 + rank)

        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return [doc_map[sid] for sid in sorted_ids[:top_k]]

    def run_group_1_bm25(self, query: str) -> List[str]:
        self.retrievers["b_zh"].k = 20
        self.retrievers["b_en"].k = 20
        docs = self._rrf_merge([
            self.retrievers["b_zh"].invoke(query),
            self.retrievers["b_en"].invoke(query)
        ], top_k=5)
        return self._extract_ids(docs, 5)

    def run_group_2_vector(self, query: str) -> List[str]:
        self.retrievers["v_zh"].search_kwargs['k'] = 20
        self.retrievers["v_en"].search_kwargs['k'] = 20
        docs = self._rrf_merge([
            self.retrievers["v_zh"].invoke(query),
            self.retrievers["v_en"].invoke(query)
        ], top_k=5)
        return self._extract_ids(docs, 5)

    def run_group_3_hybrid(self, query: str) -> List[str]:
        """Group 3: Hybrid 单阶段"""
        self.retrievers["v_zh"].search_kwargs['k'] = 20
        self.retrievers["v_en"].search_kwargs['k'] = 20
        self.retrievers["b_zh"].k = 20
        self.retrievers["b_en"].k = 20

        docs = self._rrf_merge([
            self.retrievers["v_zh"].invoke(query),
            self.retrievers["v_en"].invoke(query),
            self.retrievers["b_zh"].invoke(query),
            self.retrievers["b_en"].invoke(query)
        ], top_k=5)
        return self._extract_ids(docs, 5)

    def run_group_4_two_stage(self, query: str) -> List[str]:
        """Group 4: Two-Stage (召回 Top-20，再通过重排取前 5)"""
        self.retrievers["v_zh"].search_kwargs['k'] = 20
        self.retrievers["v_en"].search_kwargs['k'] = 20
        self.retrievers["b_zh"].k = 20
        self.retrievers["b_en"].k = 20

        initial_docs = self._rrf_merge([
            self.retrievers["v_zh"].invoke(query), self.retrievers["v_en"].invoke(query),
            self.retrievers["b_zh"].invoke(query), self.retrievers["b_en"].invoke(query)
        ], top_k=20)

        if not initial_docs: return []
        try:
            self.compressor.top_n = 5
            reranked = self.compressor.compress_documents(initial_docs, query)
            return self._extract_ids(reranked, 5)
        except Exception as exc:
            raise RuntimeError("Reranking failed") from exc

def run_evaluation(dataset_path, output_path):
    dataset = load_records(dataset_path, ("query", "golden_doc_id"))
    if any(not isinstance(row["query"], str) for row in dataset):
        raise ValueError("Every query must be a string")
    from tqdm import tqdm
    import pandas as pd
    evaluator = ThesisExperimentEvaluator()
    methods = {
        "Group 1 (BM25)": evaluator.run_group_1_bm25,
        "Group 2 (Vector)": evaluator.run_group_2_vector,
        "Group 3 (Hybrid)": evaluator.run_group_3_hybrid,
        "Group 4 (Two-Stage)": evaluator.run_group_4_two_stage,
    }
    results = []
    try:
        for row in tqdm(dataset):
            golden = normalize_id(row["golden_doc_id"])
            for name, retrieve in methods.items():
                start = time.perf_counter()
                record = {"Query ID": row.get("id"), "Pipeline": name, "Subset": row.get("subset", "unknown"),
                          "Recall@1": 0.0, "Recall@3": 0.0, "Recall@5": 0.0, "MRR@5": 0.0, "Status": "ok"}
                try:
                    ids = retrieve(row["query"])
                    for k in (1, 3, 5):
                        record[f"Recall@{k}"] = float(golden in ids[:k])
                    if golden in ids[:5]:
                        record["MRR@5"] = 1.0 / (ids.index(golden)+1)
                except Exception as exc:
                    record.update({"Status": "error", "Error": str(exc)})
                record["Latency (ms)"] = (time.perf_counter()-start)*1000
                results.append(record)
            save_json(output_path, results)
    finally:
        save_json(output_path, results)
    frame = pd.DataFrame(results)
    metrics = ["Recall@1", "Recall@3", "Recall@5", "MRR@5", "Latency (ms)"]
    print(frame.groupby("Pipeline")[metrics].mean().round(3).to_markdown())
    for metric in ("Recall@3", "Recall@5"):
        print(frame.pivot_table(index="Pipeline", columns="Subset", values=metric, aggfunc="mean").round(3).to_markdown())
    failures = sum(row["Status"] == "error" for row in results)
    print(f"Failed pipeline/query pairs: {failures}/{len(results)} (included as zero scores)")
    return failures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=LOCAL_DATA_DIR / "thesis_benchmark_dataset_500.json")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "retrieval_detailed.json")
    args = parser.parse_args(argv)
    return 1 if run_evaluation(args.input, args.output) else 0


if __name__ == "__main__":
    raise SystemExit(main())
