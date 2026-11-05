# 实验与评测工具

所有活跃实验代码位于 `experiments/`。在项目根目录使用 `python -m experiments.<模块>`；先加 `--help` 查看参数，查看帮助不会加载模型或创建 API 客户端。只有 `tests/` 是离线测试集。

安装实验依赖：

```bash
python -m pip install -r requirements-research.txt
```

运行实验需要自己有权使用的数据、对应模型和服务，可能产生 API 费用或较长计算。以下示例命令供使用者手动执行，本次归档未运行真实模型或重新生成实验分数。

## 模块与输入输出

默认数据路径以项目根目录定位，保留对原本地 `backend/` 数据的兼容；这些原始文件不随公开源码分发。命令行显式传入的相对路径相对于终端当前目录，以下命令均在项目根目录执行。默认结果均写入被 Git 忽略的 `data/results/`。

| 模块 | 用途 | 默认输入 | 默认输出 |
| --- | --- | --- | --- |
| `agent_experiment` | 生成 CoT、Naive RAG、Agentic RAG 三组答案 | `backend/ceshi.txt` | `Experiment_Results.json` |
| `agent_results` | 基于支持资料与参考答案比较两个方法 | 上一行结果文件 | `Evaluation_Results_Detailed.json` |
| `evaluate_retrieval` | BM25、向量、混合、两阶段检索的 Recall@1/3/5、MRR@5 与耗时 | `backend/thesis_benchmark_dataset_500.json` | `retrieval_detailed.json` |
| `rrf_ablation` | 扫描 BM25 与向量融合权重，计算 Recall@5 | 同上 | `ablation_weights_recall5.csv` 及 `.details.json` |
| `generate_qa_dataset` | 实体约束与语义概念两个检索子集 | 原本地中英文 CSV | `thesis_benchmark_dataset_500.json` |
| `generate_multihop_dataset` | 带支持文档与参考答案的多文档问答 | 原本地中英文 CSV | `Tech_MultiHop_QA_RAGAS_200.json` |
| `measure_performance` | 请求 `/chat` 并测量 SSE 到达时间 | `--message` 与 `--url` | `performance.json` |

`experiment_config.py` 只管理独立的评审/生成服务配置；`_shared.py` 提供 CSV/JSON、ID 处理与结果保存，不执行模型调用。

## 服务配置

主问答图和答案生成使用根 `.env` 的 `DASHSCOPE_API_KEY`、`LLM_MODEL` 等主系统配置，以及 Ollama、语料与重排模型。

独立的裁判模型和数据生成器使用：

| 变量 | 行为 |
| --- | --- |
| `EVAL_API_KEY` | 必须显式设置；不复用 `OPENAI_API_KEY` 或主系统密钥 |
| `EVAL_BASE_URL` | 空值默认 `https://api.openai.com/v1`；也可显式配置兼容服务 |
| `EVAL_MODEL` | 默认 `gpt-4o-mini`，按服务商账户权限配置 |
| `EVAL_PROXY_URL` | 可选专用代理；底层客户端仍可能遵循系统代理环境变量 |

系统环境变量优先于根 `.env`；客户端在实际调用时才创建，接口超时为 60 秒。模型名称相同不代表不同服务商的实现相同。

## 生成答案，然后评审

```bash
python -m experiments.agent_experiment --input examples/data/qa_demo.json --workers 1
python -m experiments.agent_results
```

第一个命令会执行三种方法并保存完整答案。Agentic 方法先运行图，再使用图返回的统一最终提示词和模型生成回答，与 HTTP 服务的生成契约一致；基线和图共享一次加载的检索器。默认单线程，只有确认所用客户端、检索器和模型线程安全时才提高 `--workers`。

每条问题的三种方法各自记录失败；失败答案留空并写入 `errors`，不把异常文字作为有效答案。结果按原输入顺序保存，每完成一题写入检查点，异常退出时保留已完成记录。没有自动断点续跑功能，重复运行会替换指定输出文件。

`--input` 可以是包含 `- Question N: ...` 行的 TXT，或以下 JSON 数组：

```json
[
  {
    "id": 1,
    "question": "示例问题",
    "contexts": ["有权使用的支持文档原文"],
    "ground_truth": "由支持文档验证的参考答案"
  }
]
```

TXT 只有问题，不能凭空补充可信的支持资料。评审前需要准备 JSON 中的 `contexts` 和 `ground_truth`。`examples/data/qa_demo.json` 是一个与合成演示语料相符的小例子，不是研究基准。

评审器默认比较 `Naive_RAG` 与 `Agentic_RAG`；可用 `--model-1`、`--model-2` 选择结果中的其他方法，例如 CoT。它先完整校验所有输入：拒绝空问题、空参考答案、空支持文档、空答案或已标记失败的方法，然后才创建 API 客户端。默认输出输入文件名已经与生成器对齐。

裁判比较完整性、逻辑性、决策安全性、准确性及总体胜出者；相邻样本交替交换 A/B 的展示顺序。每条样本只评审一次，并非每题双向评审。输出保留裁判理由或失败明细，每题保存检查点。统计胜率只在该指标明确选出 A/B 的记录中计算，失败或无法判定计入 `Error/Tie` 并单独显示。LLM 裁判不能替代人工质量验证。

## 检索评测与权重消融

```bash
python -m experiments.evaluate_retrieval --input examples/data/retrieval_demo.json
python -m experiments.rrf_ablation --input examples/data/retrieval_demo.json
```

这些命令会加载主系统检索资源。真实评测集使用非空 JSON 数组，每条至少包含字符串 `query` 和 `golden_doc_id`，可附 `id`、`subset`。合成示例只检查流程。

评测保留原实现的四路中英文检索、每路 20 个候选、RRF 分母 `60 + rank`（rank 从 0 开始）和两阶段重排；权重消融使用 `0、0.2、0.4、0.5、0.6、0.8、1`。整数 ID 的结尾 `.0` 被归一化，其他带小数点的字符串 ID 保持不变。

每条记录只指定一个相关文档，因此这些指标不是多相关文档或多跳证据覆盖指标。失败查询现在写入明细并按零分计入分母，重排失败不再悄悄冒充成功的两阶段结果。**这与历史代码静默丢弃失败记录的统计方式不同，不能直接与旧表格数字作无说明的比较。**

主问答图只使用中文混合检索；此处的评测器明确使用中英文检索器。这是保留英文索引代码的原因。

## 问答数据生成

```bash
python -m experiments.generate_qa_dataset --zh examples/data/tech_news_zh.csv --en examples/data/tech_news_en.csv --per-subset 1 --max-attempts 10 --seed 42
python -m experiments.generate_multihop_dataset --zh examples/data/tech_news_zh.csv --en examples/data/tech_news_en.csv --target 1 --max-attempts 10 --seed 42
```

两个生成器现在共用 CSV 读取器，兼容无表头五列以及带 `id/title/content/date/URL` 或对应中文列名的表头。空正文会被跳过。主服务自己的 CSV 加载仍按无表头五列读取，因此供服务使用的语料应遵循根 README 格式。

单文档生成器保留实体约束、语义概念的提示词与 `golden_doc_id` 字段。默认每个子集 250 题，最多分别尝试 2500 次。多文档生成器默认目标 200 题、最多 2000 次调用，并在每次抽样中保留原来的中英文随机选择方式。`--seed` 控制本地随机抽样，不保证远程模型输出完全可复现。

失败或重复回答仍受尝试上限约束；每新增一题保存，并在退出时保存已生成内容。未达到目标数返回非零退出码。多跳提示词使用截断文档，保存的是完整文档，问题与参考答案仍需人工核查。输出字段可供 RAGAS 数据准备使用，脚本并没有运行 RAGAS 或验证其版本兼容性。

## 流式性能手动测量

先启动服务，然后运行：

```bash
python -m experiments.measure_performance --message "演示系统如何结合关键词检索和向量检索？" --timeout 120
```

使用单调时钟记录完整 SSE 事件到达时间，分别给出首次事件延迟和首次答案文本延迟。只统计 `answer_chunk` / `answer` 中的文本，不把来源或错误事件计入答案字符数。无答案或只有一个文本块时，不能计算的指标为 `null`。

TPOT 使用“字符数 / 1.5”估算 token，并基于首尾文本块时间计算，**不是服务商 tokenizer 或逐 token 的精确计时**。请求有连接与读取超时；失败请求不会被报告为成功完成。

## 历史工具

`direct_results.py`、`picture.py` 已移至 `archive/analysis/`，两旧界面移至 `archive/frontend/`。它们不属于活跃实验链路。保留原因和已知问题见 [归档说明](ARCHIVE.md) 与 [archive/README.md](../archive/README.md)。
