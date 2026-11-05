# 公开演示数据

[English](README.en.md)

本目录的两个 CSV 为项目自编的合成文本，随项目采用 MIT 许可证。它们不摘录原有新闻语料，也不代表实验基准。`example.org` 链接是演示标识，不是可查询的新闻来源。

每个文件包含六条记录，UTF-8 编码，**无表头**，字段顺序为：

```text
id,title,content,date,URL
```

上面一行仅说明字段，不要把它作为 CSV 表头加入文件。含逗号、引号或换行的字段应按标准 CSV 规则转义。`content` 必须非空，`id` 应在同一语种内唯一。推荐问题：`演示系统如何结合关键词检索和向量检索？`

加载自己的语料时，通过根目录 `.env` 指定 `CSV_FILE_PATH_ZH`、`CSV_FILE_PATH_EN`，并同时更换两种 Chroma 和 BM25 路径，防止旧索引被复用。

`qa_demo.json` 提供一条带支持原文和参考答案的问答输入，可供 `experiments.agent_experiment` 后续生成三种方法的答案；`retrieval_demo.json` 提供两条带相关文档 ID 的检索输入。文档元数据 ID 按 `语种_原始id` 组合，例如 `zh_zh-demo-2`。这些 JSON 同样属于合成演示数据，不代表真实实验结果。
