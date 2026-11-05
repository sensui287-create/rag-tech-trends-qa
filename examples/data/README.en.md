# Public Demo Data

[中文文档](README.md)

The two CSV files in this directory contain synthetic text written for this project and are distributed under the project's MIT License. They do not quote the original news corpus and are not benchmark datasets. The `example.org` links are demonstration identifiers, not news sources to query.

Each file contains six records, uses UTF-8 encoding, and has **no header row**. The fields appear in this order:

```text
id,title,content,date,URL
```

The line above describes the fields; do not add it as a header to either CSV file. Escape fields containing commas, quotation marks, or newlines according to standard CSV rules. `content` must not be empty, and each `id` should be unique within its language. Suggested question: `演示系统如何结合关键词检索和向量检索？` (How does the demo combine keyword and vector retrieval?)

To load your own corpus, set `CSV_FILE_PATH_ZH` and `CSV_FILE_PATH_EN` in the root `.env`. Change the Chroma and BM25 paths for both languages at the same time so old indexes are not reused.

`qa_demo.json` contains one question-answer input with supporting passages and a reference answer. It can be used by `experiments.agent_experiment` to generate answers with the three methods. `retrieval_demo.json` contains two retrieval inputs with relevant document IDs. Document metadata IDs combine the language code and original ID, such as `zh_zh-demo-2`. These JSON files are also synthetic demo data, not actual experiment results.
