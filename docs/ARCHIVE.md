# 最终代码归档记录

归档日期：2026-10-16。公开源码覆盖项目自有 Python、HTML、JavaScript 与 GitHub YAML；每个代码文件都有职责、归类和内容哈希记录。逐文件清单以 [code_inventory.json](code_inventory.json) 为机器可核验版本，下面是便于阅读的职责索引。

## 目录和处理原则

- `backend/`：当前服务、RAG 图和唯一推荐浏览器入口。
- `experiments/`：独立的实验命令，统一从根目录用 `python -m` 执行。
- `archive/`：历史单文件、旧页面与旧分析；保留演进证据，脱离当前运行链路。
- `tests/`、`scripts/`、`.github/`：离线回归、发布检查和归档完整性工具。

活跃脚本中注释掉的整段旧实现、重复导入及无用最终 agent 构造已经清理。原始算法主体与主要提示词保留；修复范围包括路径与导入、实验最终 writer、证据校验、重复初始化、空计划边界、流式解析、有限尝试和错误可见性。检索实验现在将失败记为零分并保留在统计分母，因此其结果不能不加说明地与历史静默跳过失败的数字比较。

## 旧路径迁移

| 原位置 | 最终位置 |
| --- | --- |
| `backend/agent_experiment.py` | `experiments/agent_experiment.py` |
| `backend/agent_results.py` | `experiments/agent_results.py` |
| `backend/evaluate_retrieval.py` | `experiments/evaluate_retrieval.py` |
| `backend/experiment_config.py` | `experiments/experiment_config.py` |
| `backend/generate_multihop_dataset.py` | `experiments/generate_multihop_dataset.py` |
| `backend/generate_qa_dataset.py` | `experiments/generate_qa_dataset.py` |
| `backend/test_performance.py` | `experiments/measure_performance.py` |
| `backend/RRF.py` | `experiments/rrf_ablation.py` |
| `backend/direct_results.py` | `archive/analysis/direct_results.py` |
| `backend/picture.py` | `archive/analysis/picture.py` |
| `backend/liushi.html` | `archive/frontend/liushi.html` |
| `backend/new.html` | `archive/frontend/new.html` |
| `backend/rag_core.py` | `archive/rag_core_monolithic.py` |

原代码整理前的公开源码备份保留在本地 `dist/backups/LongChain-before-final-cleanup.zip`。该备份已经清除上一轮发现的硬编码凭证，不包含真实 `.env`、模型或原始数据；它不进入新的公开源码包。原本地 CSV/JSON/图表、模型、索引、IDE 配置和 `backend/rag_core.zip` 未删除，继续由 Git 忽略。旧 ZIP 的 9 个 Python 成员已作语法读取，属于本地历史备份，不重复混入发布目录。

旧 HTML 和单文件快照按原内容保留，仅静态阅读和语法检查，不作为受支持入口。旧分析脚本清理了注释和导入副作用，但没有伪装成可复现实验；已知限制见 [archive/README.md](../archive/README.md)。

## 逐代码文件职责

| 文件 | 类别 | 职责 |
| --- | --- | --- |
| [.github/ISSUE_TEMPLATE/bug_report.yml](../.github/ISSUE_TEMPLATE/bug_report.yml) | automation | 缺陷复现信息与脱敏日志表单 |
| [.github/ISSUE_TEMPLATE/feature_request.yml](../.github/ISSUE_TEMPLATE/feature_request.yml) | automation | 功能场景、目标行为与验收方式表单 |
| [.github/workflows/ci.yml](../.github/workflows/ci.yml) | automation | Python 3.11/3.12 与 Node 22 的离线检查配置 |
| [archive/__init__.py](../archive/__init__.py) | historical | 历史代码包标记，不初始化旧资源 |
| [archive/analysis/__init__.py](../archive/analysis/__init__.py) | historical | 历史分析工具包标记 |
| [archive/analysis/direct_results.py](../archive/analysis/direct_results.py) | historical | 保留依赖外部输入的旧答案顺序比较算法，明确不作为现行评审入口 |
| [archive/analysis/picture.py](../archive/analysis/picture.py) | historical | 保留手填分数雷达示意图，main 保护，导入无绘图副作用 |
| [archive/frontend/liushi.html](../archive/frontend/liushi.html) | historical | 原样保留较早 SSE 界面，记录协议兼容限制 |
| [archive/frontend/new.html](../archive/frontend/new.html) | historical | 原样保留带取消功能的历史界面，记录旧渲染与固定地址限制 |
| [archive/rag_core_monolithic.py](../archive/rag_core_monolithic.py) | historical | 原样保留旧父子检索、联网主管与图实现，标注索引重建风险 |
| [backend/__init__.py](../backend/__init__.py) | runtime | 当前服务 Python 包标记 |
| [backend/index.html](../backend/index.html) | runtime | 当前 Vue 演示、会话、来源与流式文本展示 |
| [backend/rag_core/__init__.py](../backend/rag_core/__init__.py) | runtime | 当前 RAG 核心包标记 |
| [backend/rag_core/components/__init__.py](../backend/rag_core/components/__init__.py) | runtime | 提示词、节点与检索组件包标记 |
| [backend/rag_core/components/agents.py](../backend/rag_core/components/agents.py) | runtime | 路由、规划、改写、提炼、反思、策略及统一最终生成提示词 |
| [backend/rag_core/components/nodes.py](../backend/rag_core/components/nodes.py) | runtime | 图节点执行与去重、候选重排、最终上下文准备 |
| [backend/rag_core/components/retrievers.py](../backend/rag_core/components/retrievers.py) | runtime | 中英文 Chroma/BM25、中文默认融合、BGE 重排与缓存恢复 |
| [backend/rag_core/config.py](../backend/rag_core/config.py) | runtime | 根 .env、路径、模型与研究迭代数配置 |
| [backend/rag_core/graph.py](../backend/rag_core/graph.py) | runtime | 图路由与循环边，支持共享检索器和执行步数预算 |
| [backend/rag_core/models.py](../backend/rag_core/models.py) | runtime | 结构化输出模型、非空计划校验和图状态类型 |
| [backend/rag_core/utils.py](../backend/rag_core/utils.py) | runtime | 研究历史格式化、文档全局去重与来源编号 |
| [backend/server.py](../backend/server.py) | runtime | FastAPI 生命周期、静态资产、状态接口和 SSE 聊天 |
| [backend/static/stream.js](../backend/static/stream.js) | runtime | 浏览器/Node 共用 SSE 分块、CRLF、多行数据与结束解析 |
| [experiments/__init__.py](../experiments/__init__.py) | experiments | 可选实验包标记，导入不加载模型 |
| [experiments/_shared.py](../experiments/_shared.py) | experiments | 标准库 CSV/JSON、ID 归一化和原子检查点保存 |
| [experiments/agent_experiment.py](../experiments/agent_experiment.py) | experiments | 三方法答案生成、最终 writer 衔接和失败记录 |
| [experiments/agent_results.py](../experiments/agent_results.py) | experiments | 调用前证据/答案校验、交替 A/B 裁判与胜率统计 |
| [experiments/evaluate_retrieval.py](../experiments/evaluate_retrieval.py) | experiments | 四种检索流程的单相关文档指标与失败明细 |
| [experiments/experiment_config.py](../experiments/experiment_config.py) | experiments | 独立 EVAL 服务与延迟客户端配置 |
| [experiments/generate_multihop_dataset.py](../experiments/generate_multihop_dataset.py) | experiments | 保留多文档问答提示词，限制尝试次数并保存进度 |
| [experiments/generate_qa_dataset.py](../experiments/generate_qa_dataset.py) | experiments | 实体/语义单文档评测问题生成与进度保存 |
| [experiments/measure_performance.py](../experiments/measure_performance.py) | experiments | SSE 手动计时、无答案处理和明确标注的 token 估算 |
| [experiments/rrf_ablation.py](../experiments/rrf_ablation.py) | experiments | 保留原权重/RRF 公式，导出汇总与错误明细 |
| [scripts/check_repository.py](../scripts/check_repository.py) | checks | Git 发布集合、敏感信息、语法和逐文件审阅哈希检查 |
| [scripts/prepare_release.py](../scripts/prepare_release.py) | checks | 可重复 ZIP、逐文件完整性清单和 ZIP 哈希 sidecar |
| [scripts/update_inventory.py](../scripts/update_inventory.py) | checks | 审阅后刷新有职责说明的代码清单哈希 |
| [tests/test_config.py](../tests/test_config.py) | checks | 环境优先级、路径与模型选择的离线验证 |
| [tests/test_experiments.py](../tests/test_experiments.py) | checks | 迁移命令、证据契约、最终生成、CSV、RRF、尝试上限与计时回归 |
| [tests/test_frontend.py](../tests/test_frontend.py) | checks | 对实际 JavaScript SSE 解析器作跨边界 Node 验证 |
| [tests/test_publication.py](../tests/test_publication.py) | checks | 敏感文件排除、已跟踪忽略文件、清单与归档完整性回归 |
| [tests/test_rag_core.py](../tests/test_rag_core.py) | checks | 非空计划、共享检索器、循环预算、来源去重与节点异常回归 |
| [tests/test_server.py](../tests/test_server.py) | checks | 真实 HTTP/SSE 层与替身核心的状态、校验、跨域和断连清理回归 |

## 归档与校验

运行 `python scripts/check_repository.py` 和离线测试后，执行 `python scripts/prepare_release.py`。生成 ZIP、逐文件 manifest 和 ZIP SHA-256 sidecar，详见 [发布说明](PUBLISHING.md)。清单不包含其自身哈希，整个 ZIP 由外部 SHA-256 覆盖。源代码有改动时，审阅后运行 `python scripts/update_inventory.py --review-date YYYY-MM-DD` 刷新现有条目；新增文件必须先填写职责与归类。

本次验证没有调用真实模型/API，也没有重跑原始研究数据。实际验证范围与限制见 [VALIDATION.md](VALIDATION.md)。
