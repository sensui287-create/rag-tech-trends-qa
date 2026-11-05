# 最终归档验证记录

日期：2026-10-16。范围为项目自有源码、公开合成示例、配置及发布文件。逐代码文件职责与哈希记录在 [code_inventory.json](code_inventory.json)；目录迁移见 [ARCHIVE.md](ARCHIVE.md)。

## 本地验证环境

| 项目 | 实际版本 |
| --- | --- |
| 操作系统 | Windows |
| Python | 3.11.7 |
| Node.js | 24.21.0 |
| FastAPI / Starlette | 0.143.2 / 1.8.0 |
| Pydantic | 2.14.1 |
| HTTPX | 0.28.1 |
| python-dotenv / Uvicorn | 1.2.4 / 0.54.0 |
| Pyflakes | 3.4.0 |

这些版本描述本次轻量验证环境，不是完整推理依赖锁文件。GitHub Actions 配置 Python 3.11/3.12 与 Node 22，远程工作流需要发布后才会实际运行。

## 已执行的检查

- 42 个代码文件均有静态审阅职责和 SHA-256 条目；覆盖 Python、HTML、JavaScript、GitHub YAML，以及历史参考源码。
- 发布检查通过：必需文件、所有 Python 语法、常见凭证模式、本地 `.env` 密钥匹配、原始数据/模型/缓存排除和代码审阅哈希。
- `python -m pyflakes backend experiments scripts tests` 通过；不把历史快照当作现行代码质量或兼容性承诺。
- 三个 HTML 内联脚本及共用 JavaScript 的 Node 语法检查通过；GitHub YAML 可解析。
- 全部本地 Markdown 相对文件链接已核对。
- 最终 ZIP 解压后重新执行发布检查、离线测试及活跃代码静态检查；校验 ZIP CRC、逐文件 manifest 与 ZIP SHA-256。

## 离线回归测试：39 项

| 文件 | 项数 | 主要验证 |
| --- | --- | --- |
| `tests/test_config.py` | 7 | 根目录路径、环境优先级、模型路径、非法迭代数 |
| `tests/test_experiments.py` | 9 | 七个命令的帮助、延迟最终生成、证据校验、CSV/RRF、尝试上限、SSE 计时 |
| `tests/test_frontend.py` | 1 | 实际 JavaScript SSE 解析器的全部字符分割点、CRLF、心跳、多行及 EOF |
| `tests/test_publication.py` | 8 | 凭证与本地资源排除、强制跟踪文件、审阅清单、归档可重复性及防覆盖 |
| `tests/test_rag_core.py` | 8 | 非空计划、共享检索器、循环预算、来源去重及重排异常 |
| `tests/test_server.py` | 6 | HTTP/SSE、静态文件、状态、输入校验、跨域与断连后台任务清理 |

模型、检索器与图编排的边界使用明确的替身。测试不需要真实密钥、Ollama 服务、Chroma 数据或模型权重。前端解析测试需要 Node；CI 显式安装 Node。

## 未验证与保留限制

- 没有调用 DashScope、OpenAI 兼容服务、Tavily 或任何付费 API。
- 没有安装并验证完整的 LangChain/PyTorch/重排推理环境，没有下载模型或启动真实 Ollama。
- 没有重跑原始实验、验证研究指标或声称复现论文结果；合成数据只用于演示与流程说明。
- 没有执行历史单文件中的索引删除/重建、旧联网分支或固定分数绘图。
- 没有通过联网浏览器完成 CDN 依赖和整页交互验收；前端验证范围是源码阅读、JavaScript 语法、实际 SSE 解析器与 HTTP 层。
- 测试环境的 Starlette 对 HTTPX TestClient 发出弃用提示，当前 39 项测试仍通过；更新框架时需重新验证兼容性。

代码检查与文件完整性验证不等于模型效果或生产部署验证。当前服务保持本地研究演示定位。
