"""Offline core checks with model, retriever and LangGraph boundaries stubbed."""

import importlib
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from pydantic import ValidationError


CORE = Path(__file__).resolve().parents[1] / "backend" / "rag_core"
PACKAGE = "offline_test_rag_core"


class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}


class CompiledGraph:
    def with_config(self, config):
        self.config = config
        return self


class StateGraph:
    def __init__(self, state_schema):
        self.nodes = {}
        self.compiled = CompiledGraph()

    def add_node(self, name, action):
        self.nodes[name] = action

    def set_entry_point(self, name):
        pass

    def add_edge(self, *args):
        pass

    def add_conditional_edges(self, *args):
        pass

    def compile(self):
        return self.compiled


def module(name, **attributes):
    result = ModuleType(name)
    result.__dict__.update(attributes)
    return result


class RagCoreTests(unittest.TestCase):
    def setUp(self):
        self.suite = {"default_hybrid": Mock(), "compressor": Mock()}
        self.setup_retrievers = Mock(return_value=self.suite)
        self.config = module(
            f"{PACKAGE}.config", LLM_MODEL="offline-main", FAST_LLM_MODEL="offline-fast",
            MAX_RESEARCH_ITERATIONS=30,
        )
        self.agents = module(f"{PACKAGE}.components.agents")
        for name in (
            "router", "planner", "universal_query_rewriter", "distiller", "reflection", "policy",
        ):
            setattr(self.agents, f"build_{name}_agent", Mock(return_value=Mock()))
        self.agents.get_final_answer_prompt = Mock(return_value=object())
        self.llm = Mock(side_effect=lambda **kwargs: object())
        stubs = {
            PACKAGE: module(PACKAGE, __path__=[str(CORE)]),
            f"{PACKAGE}.components": module(
                f"{PACKAGE}.components", __path__=[str(CORE / "components")],
            ),
            f"{PACKAGE}.config": self.config,
            f"{PACKAGE}.components.agents": self.agents,
            f"{PACKAGE}.components.retrievers": module(
                f"{PACKAGE}.components.retrievers", setup_retrievers=self.setup_retrievers,
            ),
            "langchain_core": module("langchain_core", __path__=[]),
            "langchain_core.documents": module("langchain_core.documents", Document=Document),
            "langchain_community": module("langchain_community", __path__=[]),
            "langchain_community.llms": module("langchain_community.llms", Tongyi=self.llm),
            "langgraph": module("langgraph", __path__=[]),
            "langgraph.graph": module("langgraph.graph", StateGraph=StateGraph, END="__end__"),
        }
        modules = patch.dict(sys.modules, stubs)
        modules.start()
        self.addCleanup(modules.stop)
        self.models = importlib.import_module(f"{PACKAGE}.models")
        self.nodes = importlib.import_module(f"{PACKAGE}.components.nodes")
        self.utils = importlib.import_module(f"{PACKAGE}.utils")
        self.graph = importlib.import_module(f"{PACKAGE}.graph")

    def plan(self, count=1):
        return self.models.Plan(steps=[{
            "sub_question": f"Question {index}", "justification": "evidence", "keywords": [],
        } for index in range(count)])

    def test_empty_model_plan_is_rejected_at_planning_boundary(self):
        planner = Mock()
        planner.invoke.return_value = {"steps": []}
        with self.assertRaises(ValidationError) as error:
            self.nodes.plan_node({"original_question": "demo"}, planner)
        self.assertEqual(error.exception.errors()[0]["loc"], ("steps",))
        self.assertEqual(error.exception.errors()[0]["type"], "too_short")

    def test_injected_retrievers_are_shared_without_second_initialization(self):
        app, prompt, llm = self.graph.initialize_rag_system(retrievers=self.suite)
        self.setup_retrievers.assert_not_called()
        self.assertIs(prompt, self.agents.get_final_answer_prompt.return_value)
        self.assertIsNotNone(llm)
        # Thirty three-node research rounds plus fixed setup/finalization nodes.
        self.assertGreaterEqual(app.config["recursion_limit"], 3 * 30 + 5)

    def test_default_initialization_loads_retrievers_and_rejects_empty_suite(self):
        self.graph.initialize_rag_system()
        self.setup_retrievers.assert_called_once_with()
        self.setup_retrievers.reset_mock()
        with self.assertRaisesRegex(RuntimeError, "检索器初始化失败"):
            self.graph.initialize_rag_system(retrievers={})
        self.setup_retrievers.assert_not_called()

    def test_research_cap_and_completed_plan_do_not_call_policy(self):
        policy = Mock()
        for steps, current_index in ((40, 30), (1, 1)):
            state = {"plan": self.plan(steps), "current_step_index": current_index}
            self.assertEqual(self.graph.should_continue(state, policy), "finish")
        policy.invoke.assert_not_called()

    def test_policy_can_finish_before_plan_is_complete(self):
        policy = Mock()
        policy.invoke.return_value = {"next_action": "FINISH"}
        state = {
            "plan": self.plan(3), "current_step_index": 1,
            "original_question": "demo", "past_steps": [],
        }
        self.assertEqual(self.graph.should_continue(state, policy), "finish")
        policy.invoke.return_value = {"next_action": "CONTINUE"}
        self.assertEqual(self.graph.should_continue(state, policy), "continue")

    def test_fast_answer_prepares_numbered_sources_without_llm(self):
        first = Document("First evidence", {"title": "One", "URL": "https://example.org/1"})
        second = Document("Second evidence", {"title": "Two"})
        result = self.nodes.simple_gen_node({"retrieved_docs": [first, second]})
        self.assertEqual(result["final_answer"], "")
        self.assertIn("来源 1: 《One》", result["context_for_final_answer"])
        self.assertIn("来源 2: 《Two》", result["context_for_final_answer"])
        self.assertEqual(result["metadata_for_final_answer"], {1: first.metadata, 2: second.metadata})
        empty = self.nodes.simple_gen_node({"retrieved_docs": []})
        self.assertTrue(empty["final_answer"])
        self.assertNotIn("context_for_final_answer", empty)

    def test_deep_answer_deduplicates_documents_and_keeps_source_order(self):
        plan = self.plan(2)
        first = Document("Shared evidence", {"title": "Shared"})
        duplicate = Document("Shared evidence", {"title": "Other copy"})
        second = Document("New evidence", {"title": "New"})
        history = [
            {"sub_question": plan.steps[0].sub_question, "retrieved_docs": [first]},
            {"sub_question": plan.steps[1].sub_question, "retrieved_docs": [duplicate, second]},
        ]
        result = self.nodes.final_answer_node({"plan": plan, "past_steps": history})
        context = result["context_for_final_answer"]
        self.assertEqual(context.count("Shared evidence"), 1)
        self.assertLess(context.index("来源 1"), context.index("来源 2"))
        self.assertEqual(result["metadata_for_final_answer"], {1: first.metadata, 2: second.metadata})

    def test_failed_subquery_reranking_returns_empty_evidence(self):
        rewriter = Mock()
        rewriter.invoke.return_value = {"chinese_query": "rewritten"}
        self.suite["compressor"].compress_documents.side_effect = RuntimeError("offline failure")
        state = {
            "plan": self.plan(), "current_step_index": 0, "past_steps": [],
            "candidate_docs": [Document("evidence")],
        }
        with self.assertLogs(self.nodes.logger, level="ERROR"):
            result = self.nodes.research_step_node(state, rewriter, self.suite)
        self.assertEqual(result, {"current_rewritten_query": "rewritten", "retrieved_docs": []})


if __name__ == "__main__":
    unittest.main()
