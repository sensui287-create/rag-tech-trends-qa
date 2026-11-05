"""Exercise the real HTTP/SSE layer using an explicitly fake RAG backend."""

import importlib
import asyncio
import os
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient


class FakeGraph:
    async def astream(self, state):
        if state["original_question"] == "raise-error":
            raise RuntimeError("simulated backend failure")
        yield {"router": {"router_decision": "simple_answer"}}
        if state["original_question"] == "no-docs":
            yield {"simple_gen": {"final_answer": "没有资料"}}
        else:
            yield {"research_step": {"current_rewritten_query": "改写查询", "retrieved_docs": []}}
            yield {"simple_gen": {
                "context_for_final_answer": "演示文本",
                "metadata_for_final_answer": {1: {"title": "演示来源", "URL": "https://example.org/demo"}},
            }}


class FakePrompt:
    def __or__(self, other):
        return self

    async def astream(self, values):
        yield "回答"
        yield " [来源:1]"


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stub = ModuleType("backend.rag_core.graph")
        cls.stub.initialize_rag_system = lambda: (FakeGraph(), FakePrompt(), object())
        cls.modules = patch.dict(sys.modules, {"backend.rag_core.graph": cls.stub})
        cls.modules.start()
        cls.env = patch.dict(os.environ, {"CORS_ORIGINS": "http://127.0.0.1:8000"})
        cls.env.start()
        cls.server = importlib.import_module("backend.server")

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop("backend.server", None)
        cls.env.stop()
        cls.modules.stop()

    def test_health_frontend_and_lifespan_cleanup(self):
        with TestClient(self.server.app) as client:
            self.assertEqual(client.get("/health").json(), {"status": "ok", "rag_ready": True})
            page = client.get("/")
            self.assertEqual(page.status_code, 200)
            self.assertIn("text/html", page.headers["content-type"])
            self.assertIn("fetch('/chat'", page.text)
            self.assertEqual(client.get("/static/stream.js").status_code, 200)
        self.assertIsNone(self.server.rag_app)

    def test_stream_contains_text_citations_rewritten_query_and_done(self):
        with TestClient(self.server.app) as client:
            response = client.post("/chat", json={"message": "demo"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers["content-type"])
        self.assertIn('data: "回答"', response.text)
        self.assertIn('data: " [来源:1]"', response.text)
        self.assertIn("改写查询", response.text)
        self.assertIn("event: answer_citations", response.text)
        self.assertTrue(response.text.endswith("event: done\ndata: [DONE]\n\n"))

    def test_empty_result_and_stream_failure_have_terminal_events(self):
        with TestClient(self.server.app) as client:
            empty = client.post("/chat", json={"message": "no-docs"})
            with self.assertLogs(self.server.logger, level="ERROR"):
                failed = client.post("/chat", json={"message": "raise-error"})
        self.assertIn('event: answer\ndata: "没有资料"', empty.text)
        self.assertIn("event: error", failed.text)
        self.assertIn("event: done", failed.text)

    def test_invalid_requests_are_rejected_before_rag(self):
        with TestClient(self.server.app) as client:
            for payload in ({}, {"message": ""}, {"message": "   "}, {"message": "x" * 10001}):
                self.assertEqual(client.post("/chat", json=payload).status_code, 422)

    def test_unlisted_origins_are_not_allowed(self):
        with TestClient(self.server.app) as client:
            response = client.options("/chat", headers={
                "Origin": "https://untrusted.example", "Access-Control-Request-Method": "POST",
            })
        self.assertNotIn("access-control-allow-origin", response.headers)

    def test_closing_stream_cancels_background_research(self):
        async def scenario():
            cancelled = asyncio.Event()

            class WaitingGraph:
                async def astream(self, state):
                    try:
                        yield {"router": {"router_decision": "complex_research"}}
                        await asyncio.Event().wait()
                    finally:
                        cancelled.set()

            with patch.object(self.server, "rag_app", WaitingGraph()):
                response = await self.server.chat_endpoint(self.server.QueryRequest(message="demo"))
                self.assertIn("event: process", await anext(response.body_iterator))
                await response.body_iterator.aclose()
                self.assertTrue(cancelled.is_set())

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
