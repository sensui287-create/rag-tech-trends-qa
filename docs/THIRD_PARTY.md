# 数据、模型和第三方资源

## 本仓库包含的内容

- 项目原创 Python 实现、网页演示和文档。
- `examples/data/` 内专为发布编写的合成文本，采用项目 MIT 许可证。
- 历史代码快照和研究脚本，按各自文档标记使用范围。

## 默认保留在本地的内容

`.gitignore` 排除 `backend/` 顶层的 CSV、JSON、PDF、SVG、TXT，以及模型目录、Chroma 数据库、BM25 缓存、ZIP 备份、日志和 `.env`。这会覆盖原目录中的新闻语料、问答数据集、人工评测表、实验结果及论文图表；不会删除这些本地文件。

原始新闻和派生数据尚未提供完整的来源、采集条款与再分发授权记录，因此不随公开源码包分发。需要公开研究数据时，应逐个数据集补充来源、版本、字段、处理方法、授权、隐私检查和下载位置，再有选择地调整忽略规则与发布检查器。不能把合成示例描述为原实验数据。

## 外部组件

| 组件 | 用途 | 获取/授权说明 |
| --- | --- | --- |
| [LangGraph](https://github.com/langchain-ai/langgraph) / [LangChain](https://github.com/langchain-ai/langchain) 及相关集成 | 图、提示词、检索和模型调用 | 通过依赖管理器安装，遵循各组件许可 |
| [Chroma](https://github.com/chroma-core/chroma) | 向量存储 | 运行时生成的数据库不入库 |
| [Ollama](https://ollama.com/) | 本地嵌入服务 | 需单独安装；模型遵循模型自身许可 |
| [BAAI/bge-reranker-large](https://huggingface.co/BAAI/bge-reranker-large) | 文档重排 | 从官方模型页下载并查看模型卡与许可；权重不随本仓库分发 |
| 通义模型 / DashScope | 路由、研究和生成 | 使用者自行开通并配置凭证，遵循服务条款 |
| Vue、Tailwind CSS、Marked、DOMPurify、Highlight.js、Phosphor Icons、Day.js | 浏览器界面 | 原有 HTML 从第三方 CDN 加载，保留各库许可与归属 |

本项目的 MIT 许可只适用于作者有权授权的项目内容，不替代外部模型、数据或服务的条款。研究脚本中的评审结果与模型输出不构成通用性能保证。
