"""Offline regression checks for migrated experiment commands and data contracts."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from experiments import agent_experiment, agent_results, generate_qa_dataset, generate_multihop_dataset
from experiments._shared import load_news_csv, normalize_id
from experiments.evaluate_retrieval import ThesisExperimentEvaluator
from experiments.rrf_ablation import WeightAblationEvaluator
from experiments.measure_performance import collect_metrics, iter_sse_events

ROOT = Path(__file__).resolve().parents[1]


def valid_record():
    return {"id": 1, "question": "demo", "contexts": ["source evidence"], "ground_truth": "reference",
            "answers": {"Naive_RAG": "baseline answer", "Agentic_RAG": "agent answer"}}


class ExperimentTests(unittest.TestCase):
    def test_all_active_commands_show_help_without_model_dependencies(self):
        modules = ("agent_experiment", "agent_results", "evaluate_retrieval", "rrf_ablation",
                   "generate_qa_dataset", "generate_multihop_dataset", "measure_performance")
        for module in modules:
            with self.subTest(module=module):
                completed = subprocess.run([sys.executable, "-m", "experiments." + module, "--help"],
                                           cwd=ROOT, capture_output=True, timeout=20)
                self.assertEqual(completed.returncode, 0, completed.stderr.decode(errors="replace"))
                self.assertIn(b"usage:", completed.stdout)

    def test_agentic_generation_finishes_deferred_writer(self):
        graph = Mock()
        graph.invoke.return_value = {"final_answer": "", "context_for_final_answer": "numbered evidence"}
        chain = Mock()
        chain.invoke.return_value = SimpleNamespace(content="final answer")

        class Prompt:
            def __or__(self, other):
                return chain

        self.assertEqual(agent_experiment.run_agentic_rag("question", graph, Prompt(), object()), "final answer")
        chain.invoke.assert_called_once_with({"question": "question", "context": "numbered evidence"})
        graph.invoke.return_value = {"final_answer": "No documents"}
        self.assertEqual(agent_experiment.run_agentic_rag("question", graph, None, None), "No documents")
        graph.invoke.return_value = {"final_answer": ""}
        with self.assertRaisesRegex(RuntimeError, "no answer"):
            agent_experiment.run_agentic_rag("question", graph, None, None)

    def test_json_questions_preserve_evidence_and_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            data = [valid_record()]
            path.write_text(json.dumps(data), encoding="utf-8")
            self.assertEqual(agent_experiment.load_dataset(path), data)
        self.assertEqual(agent_experiment.OUTPUT_RESULTS, agent_results.INPUT_FILE)

    def test_incomplete_judging_inputs_fail_before_client_creation(self):
        fake_config = ModuleType("experiments.experiment_config")
        fake_config.create_eval_client = Mock()
        fake_config.EVAL_MODEL = "offline"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            for field, bad_value in (("contexts", []), ("ground_truth", ""), ("answers", {"Naive_RAG": ""})):
                row = valid_record()
                row[field] = bad_value
                path.write_text(json.dumps([row]), encoding="utf-8")
                with patch.dict(sys.modules, {"experiments.experiment_config": fake_config}):
                    with self.assertRaises(ValueError):
                        agent_results.evaluate_experiments(path, Path(directory) / "out.json")
            fake_config.create_eval_client.assert_not_called()
        records = [valid_record()]
        self.assertEqual(agent_results.validate_experiments(records), records)

    def test_news_csv_handles_headers_no_headers_and_empty_content(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            path.write_text('id,title,content,date,URL\n1.0,Title,"Content, with comma",2026-01-01,https://example.org\n2,Empty,,2026-01-01,\n', encoding="utf-8")
            headed = load_news_csv(path, "en")
            path.write_text('1.0,Title,"Content, with comma",2026-01-01,https://example.org\n', encoding="utf-8")
            self.assertEqual(load_news_csv(path, "en"), headed)
            self.assertEqual(headed[0]["id"], "en_1")
            self.assertEqual(headed[0]["content"], "Content, with comma")
        self.assertEqual(normalize_id("zh_1.0"), "zh_1")
        self.assertEqual(normalize_id("section1.02"), "section1.02")

    def test_fusion_preserves_ids_deduplicates_and_obeys_weights(self):
        a = SimpleNamespace(metadata={"source_id": "zh_1.0"})
        b = SimpleNamespace(metadata={"source_id": "section1.02"})
        evaluation = ThesisExperimentEvaluator.__new__(ThesisExperimentEvaluator)
        merged = evaluation._rrf_merge([[a], [a, b]], top_k=5)
        self.assertEqual(evaluation._extract_ids(merged, 5), ["zh_1", "section1.02"])
        ablation = WeightAblationEvaluator.__new__(WeightAblationEvaluator)
        self.assertEqual(ablation._weighted_rrf_merge([[a], [b]], [0, 1]), [b])
        self.assertEqual(ablation._weighted_rrf_merge([[a], [b]], [1, 0]), [a])

    def test_generation_attempt_limits_save_partial_results_without_api(self):
        config = ModuleType("experiments.experiment_config")
        config.create_eval_client = Mock()
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"experiments.experiment_config": config}):
            output = Path(directory) / "qa.json"
            with patch.object(generate_qa_dataset, "load_documents", return_value=[{"text": "demo"}]), \
                 patch.object(generate_qa_dataset, "generate_qa_pair", return_value=None) as generate, \
                 contextlib.redirect_stdout(io.StringIO()):
                status = generate_qa_dataset.main(["--per-subset", "1", "--max-attempts", "2", "--output", str(output)])
            self.assertEqual(status, 1)
            self.assertEqual(generate.call_count, 4)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), [])
            output = Path(directory) / "multi.json"
            doc = {"title": "demo", "content": "source", "date": "2026-01-01"}
            with patch.object(generate_multihop_dataset, "load_news_csv", return_value=[doc]), \
                 patch.object(generate_multihop_dataset, "generate_multihop_qa_for_ragas", return_value=[]) as generate:
                status = generate_multihop_dataset.main(["--target", "1", "--max-attempts", "3", "--output", str(output)])
            self.assertEqual(status, 1)
            self.assertEqual(generate.call_count, 3)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), [])

    def test_performance_ignores_citations_and_handles_no_answer(self):
        events = [("process", "{}"), ("answer_chunk", '"abc"'),
                  ("answer_citations", '{"1":{"title":"source"}}'), ("answer_chunk", '"def"'), ("done", "[DONE]")]
        clock = iter((1, 2, 3, 4, 5, 6))
        metrics = collect_metrics(events, 0, lambda: next(clock))
        self.assertEqual(metrics["answer_characters"], 6)
        self.assertEqual(metrics["answer_ttfa_seconds"], 2)
        self.assertEqual(metrics["generation_seconds"], 2)
        self.assertTrue(metrics["completed"])
        clock = iter((1, 2))
        empty = collect_metrics([("done", "[DONE]")], 0, lambda: next(clock))
        self.assertIsNone(empty["answer_ttfa_seconds"])
        self.assertIsNone(empty["estimated_tpot_ms"])

    def test_python_sse_framing_handles_comments_multiline_and_eof(self):
        lines = [b":ping", b"", b"event: detail", b"data: first", b"data: second", b"", b"event: done", b"data: [DONE]"]
        self.assertEqual(list(iter_sse_events(lines)), [("detail", "first\nsecond"), ("done", "[DONE]")])


if __name__ == "__main__":
    unittest.main()
