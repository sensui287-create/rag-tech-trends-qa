"""FastAPI entry point for the local LongChain demo and SSE chat API."""

import json
import logging
import os
from pathlib import Path
import uvicorn
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from contextlib import asynccontextmanager

# 导入核心逻辑
if __package__:
    from .rag_core.graph import initialize_rag_system
else:
    from rag_core.graph import initialize_rag_system

FRONTEND_PATH = Path(__file__).resolve().with_name("index.html")
logger = logging.getLogger(__name__)

# --- 全局变量 ---
rag_app = None
final_writer_prompt = None
llm = None

# === FastAPI 生命周期管理 ===
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    在 FastAPI 应用启动时加载模型和 RAG 系统，
    在关闭时清理资源。
    """
    logger.info("Initializing the RAG engine")
    global rag_app, final_writer_prompt, llm
    try:
        rag_app, final_writer_prompt, llm = initialize_rag_system()
        logger.info("RAG engine initialized")
    except Exception:
        logger.exception("RAG initialization failed")
        raise

    try:
        yield
    finally:
        logger.info("Shutting down the RAG engine")
        rag_app = None
        final_writer_prompt = None
        llm = None


# === FastAPI 应用定义 ===
app = FastAPI(
    title="Deep Thinking RAG API",
    description="一个支持流式响应和深度思考的多智能体 RAG API",
    lifespan=lifespan
)
app.mount("/static", StaticFiles(directory=FRONTEND_PATH.parent / "static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.getenv(
        "CORS_ORIGINS", "http://127.0.0.1:8000,http://localhost:8000"
    ).split(",") if origin.strip()],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

class QueryRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    message: str = Field(min_length=1, max_length=10000)


@app.get("/", include_in_schema=False)
async def frontend():
    return FileResponse(FRONTEND_PATH)


@app.get("/health")
async def health():
    return {"status": "ok", "rag_ready": rag_app is not None}

# === 核心流式接口 ===
@app.post("/chat")
async def chat_endpoint(request: QueryRequest):
    async def event_generator():
        queue = asyncio.Queue()
        is_rag_finished = False

        async def run_rag_and_stream():
            nonlocal is_rag_finished
            try:
                # 1. 运行 RAG 核心流程
                async for output in rag_app.astream({"original_question": request.message}):
                    await queue.put(output)

                    # 2. 检查是否到达最终答案生成节点
                    is_final_prep_node = "generate_final_answer" in output or "simple_gen" in output

                    if is_final_prep_node:
                        node_name = "generate_final_answer" if "generate_final_answer" in output else "simple_gen"
                        state_update = output[node_name]

                        context = state_update.get("context_for_final_answer") if state_update else None
                        metadata = state_update.get("metadata_for_final_answer") if state_update else None

                        if not context:
                            final_answer = state_update.get("final_answer", "无法回答。") if state_update else "处理流程出现未知错误。"
                            await queue.put({"final_answer_simple": final_answer})
                        else:
                            # 3. 执行真正的 LLM 流式调用
                            final_chain = final_writer_prompt | llm
                            async for chunk in final_chain.astream({
                                "question": request.message,
                                "context": context
                            }):
                                await queue.put({"answer_chunk": chunk})

                            # 4. 流式调用结束后，放入引用元数据
                            await queue.put({"citations": metadata})

            except Exception:
                logger.exception("RAG request failed")
                await queue.put({"error": "请求执行失败，请查看服务端日志。"})
            finally:
                is_rag_finished = True
                await queue.put(None)

        rag_task = asyncio.create_task(run_rag_and_stream())

        try:
            while not (is_rag_finished and queue.empty()):
                try:
                    output = await asyncio.wait_for(queue.get(), timeout=15.0)

                    if output is None:
                        break

                    if "error" in output:
                        yield f"event: error\ndata: {json.dumps(output['error'], ensure_ascii=False)}\n\n"
                        break

                    elif "answer_chunk" in output:
                        chunk = output["answer_chunk"]
                        yield f"event: answer_chunk\ndata: {json.dumps(chunk, ensure_ascii=False)}\n\n"

                    elif "citations" in output:
                        metadata = output["citations"]
                        # 【核心修改】：直接返回字典数据给前端，让前端既能渲染列表，又能用于弹窗
                        yield f"event: answer_citations\ndata: {json.dumps(metadata, ensure_ascii=False)}\n\n"

                    elif "final_answer_simple" in output:
                        yield f"event: answer\ndata: {json.dumps(output['final_answer_simple'], ensure_ascii=False)}\n\n"

                    else:
                        # 处理 RAG 的中间步骤 (process event)
                        for node_name, state_update in output.items():
                            event_data = {"node": node_name, "details": {}}

                            # --- 1. 意图路由节点 ---
                            if node_name == "router":
                                decision = state_update.get("router_decision", "unknown")
                                desc = "简单问题 -> 快速检索模式" if decision == "simple_answer" else "复杂问题 -> 深度思考模式"
                                event_data["details"] = {
                                    "summary": f"意图识别: {desc}",
                                    "decision": decision
                                }

                            # --- 2. 快速检索节点 ---
                            elif node_name == "instant_rag":
                                docs = state_update.get("retrieved_docs", [])
                                event_data["details"] = {
                                    "summary": f"快速检索完成: 找到 {len(docs)} 篇相关文档",
                                    "docs_count": len(docs)
                                }

                            # --- 3. 规划节点 ---
                            elif node_name == "plan":
                                plan = state_update.get("plan")
                                steps = plan.steps if plan else []
                                event_data["details"] = {
                                    "summary": f"深度思考计划: 共 {len(steps)} 步",
                                    "steps": [s.sub_question for s in steps]
                                }

                            # --- 4. 核心研究节点 (此处为关键修改) ---
                            elif node_name == "research_step":
                                query = state_update.get("current_rewritten_query", "")
                                docs = state_update.get("retrieved_docs", [])
                                event_data["details"] = {
                                    "summary": f"候选资料重排: {query}",
                                    "docs_count": len(docs),
                                }

                            elif node_name == "global_research":
                                docs = state_update.get("candidate_docs", [])
                                event_data["details"] = {
                                    "summary": f"全局候选召回: {len(docs)} 篇文档",
                                    "docs_count": len(docs),
                                }

                            elif node_name == "distill":
                                event_data["details"] = {"summary": "阅读并提炼文档核心信息..."}

                            # --- 6. 反思节点 ---
                            elif node_name == "reflect":
                                past_steps = state_update.get("past_steps", [])
                                if past_steps:
                                    last_step = past_steps[-1]
                                    event_data["details"] = {
                                        "summary": f"步骤 {last_step['step_index']} 反思完成",
                                        "step_summary": last_step['summary']
                                    }

                            # 发送事件 (排除最终生成节点，因为上面已经单独处理了)
                            if node_name != "generate_final_answer" and node_name != "simple_gen":
                                yield f"event: process\ndata: {json.dumps(event_data, ensure_ascii=False)}\n\n"


                except asyncio.TimeoutError:
                    yield ":ping\n\n"
        finally:
            rag_task.cancel()
            try:
                await rag_task
            except asyncio.CancelledError:
                pass

        yield "event: done\ndata: [DONE]\n\n"

    return StreamingResponse(
        event_generator(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
