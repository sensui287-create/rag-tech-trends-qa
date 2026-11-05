# HTTP API

推荐从项目根目录启动 `python -m uvicorn backend.server:app --host 127.0.0.1 --port 8000`。服务在 RAG 初始化成功后接受请求。交互式接口文档位于 `/docs`。

## 路由

| 方法 | 路径 | 返回 |
| --- | --- | --- |
| GET | `/` | 推荐演示页面 `backend/index.html` |
| GET | `/static/stream.js` | 页面使用的 SSE 数据块解析器 |
| GET | `/health` | `{"status":"ok","rag_ready":true}` |
| POST | `/chat` | `text/event-stream` 响应 |

## 聊天请求

请求体为 JSON，`message` 去除首尾空白后是长度为 1–10000 的字符串。每次调用独立处理一个问题；网页历史会话不会作为多轮上下文发送。

```json
{"message":"演示系统如何结合关键词检索和向量检索？"}
```

macOS / Linux 的 curl 示例：

```bash
curl -N http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"演示系统如何结合关键词检索和向量检索？"}'
```

Windows 可直接使用 `/docs` 的 **Try it out**；网页演示通过 `fetch` 发送 POST 并逐块读取响应，不能直接用仅支持 GET 的浏览器 `EventSource` 代替。

## SSE 事件

每个事件由 `event:` 与 `data:` 行组成，事件间有一个空行。除结束标记外，`data` 使用 JSON 编码；接收端必须累积跨网络分块的数据，按空行拆分完整事件。

| event | data 内容 | 含义 |
| --- | --- | --- |
| `process` | `{"node":"router","details":{...}}` | 路由、规划、研究、提炼或反思过程 |
| `answer_chunk` | JSON 字符串 | 回答的增量文本 |
| `answer_citations` | 来源编号到元数据的对象 | 文档标题、URL、日期等；JSON 对象键为字符串 |
| `answer` | JSON 字符串 | 无资料或无法生成时的一次性回答 |
| `error` | JSON 字符串 | 流启动后的处理错误 |
| `done` | `[DONE]` | 结束标记，不是 JSON |

空闲期间可能发送 `:ping` 注释作为心跳。参数格式错误返回 HTTP 422；流内异常可能通过 `error` 返回，此时 HTTP 状态已是 200。正常回答中的 `[来源:1]` 等标记对应 `answer_citations` 的编号。

简化示例（文本仅用于说明协议）：

```text
event: process
data: {"node":"router","details":{"decision":"simple_answer","summary":"意图识别: 简单问题 -> 快速检索模式"}}

event: answer_chunk
data: "系统使用混合检索 [来源:1]。"

event: answer_citations
data: {"1":{"title":"演示文档","URL":"https://example.org/demo"}}

event: done
data: [DONE]

```

深度研究返回的 `research_step.details.summary` 展示用于候选资料重排的查询，`global_research` 事件给出候选文档数。当前主界面不再提供未接入服务的联网/深度模式开关或动态权重展示；问题复杂度由图自动路由。流内错误返回通用说明，具体异常记录在服务端日志。
