# 发布到 GitHub

本目录已按源码发布准备。当前工作只整理本地文件，不创建远程仓库、不推送代码。项目许可证为 MIT，署名使用 `LongChain contributors`；如需个人或机构署名，可在首次发布前修改 LICENSE 的版权行。

## 发布文件与本地资源

项目根目录包含 README、MIT LICENSE、依赖清单、配置示例、贡献/安全说明、issue/PR 模板及离线 CI。GitHub 会展示根目录 README；这些文件的组织依据可参见 [GitHub README 文档](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes) 和 [许可证文档](https://docs.github.com/en/communities/setting-up-your-project-for-healthy-contributions/adding-a-license-to-a-repository)。

上传时排除真实 `.env`、模型、向量库、缓存及原始实验数据。具体范围见 [.gitignore](../.gitignore) 与 [第三方资源说明](THIRD_PARTY.md)。源码中的历史硬编码 API 密钥已改为环境配置；真实 `.env` 保留在本地。若这些凭证曾经对外共享，应在提供商处轮换。

## 方式一：使用已经生成的源码包

源码包位置：`dist/LongChain-github.zip`。重新生成：

```bash
python scripts/check_repository.py
python scripts/prepare_release.py
```

脚本利用 Git 的忽略规则构建文件集，再检查语法、凭证模式和本地资源。未初始化 Git 的目录使用外部临时 Git 元数据进行检查，不改动当前目录的 Git 状态。已被追踪的忽略文件仍会被检查并阻断。

最终归档同时生成：

- `dist/LongChain-github.zip`：源码、文档、合成示例及历史参考代码。
- `dist/LongChain-github.manifest.json`：每个源码文件的相对路径、字节数和 SHA-256；ZIP 内还有同一份 `ARCHIVE_MANIFEST.json`。
- `dist/LongChain-github.zip.sha256`：完整 ZIP 的 SHA-256。

ZIP 内清单不把自身纳入逐文件哈希，以免自引用；完整 ZIP 的 sidecar 覆盖整个归档。Windows 可用 `Get-FileHash -Algorithm SHA256 dist/LongChain-github.zip` 与 sidecar 比较，macOS / Linux 可在 `dist` 中运行 `sha256sum -c LongChain-github.zip.sha256`。解压后运行离线检查与测试，确认包内环境独立完整。生成的 `ARCHIVE_MANIFEST.json` 被 Git 忽略，GitHub 源码保留的是 `docs/code_inventory.json` 代码审阅清单。

归档结构、逐文件职责和旧路径映射见 [ARCHIVE.md](ARCHIVE.md)。代码变更后应先审阅，再通过 `scripts/update_inventory.py` 刷新清单；仅刷新哈希不能代替审阅。

1. 解压 ZIP，进入其中的 `LongChain` 文件夹。
2. 在 GitHub 创建仓库，建议仓库名 `LongChain`；公开发布时选择 Public。
3. 上传解压后的**源码文件**并创建提交，保留 `.github`、`.gitignore`、`.env.example` 等点开头文件。不要把 ZIP 作为唯一的仓库内容。
4. 网页上传容易漏掉隐藏文件，推荐通过下方 Git 方式提交解压后的目录。

## 方式二：用 Git 推送

在项目根目录或解压后的源码目录中执行以下命令。请先安装 Git 并完成自己的 GitHub 认证。`YOUR_USERNAME`、`YOUR_REPOSITORY`、作者名和邮箱均为需要自行替换的占位符。

建议远程创建**空仓库**，不要让 GitHub 另外生成 README、LICENSE 或 .gitignore，以免第一次推送出现不相同的初始提交。

```bash
python scripts/check_repository.py
git init -b main
git config user.name "YOUR_NAME"
git config user.email "YOUR_EMAIL_OR_GITHUB_NOREPLY_EMAIL"
git add .
git diff --cached --stat
python scripts/check_repository.py
git commit -m "chore: prepare LongChain open-source release"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

已初始化的仓库不必重复初始化或添加相同 remote。如果远程已有提交，先正常拉取、核对并处理合并，不要使用强制推送覆盖。

检查器会扫描常见凭证模式，并匹配本地 `.env` 中的密钥值；它不是完整的安全审计。推送前仍应查看暂存文件清单和 diff。不要使用 `git add -f` 将被忽略的本地资源加入公开提交。

## GitHub 页面建议填写

**Description：**

```text
An agentic RAG research prototype built with LangGraph and FastAPI, featuring hybrid retrieval, research planning, reranking, and SSE streaming.
```

**Topics：**

```text
rag, agentic-rag, langgraph, fastapi, retrieval-augmented-generation, bm25, chroma, ollama, reranking
```

首次提交可先保留为默认分支代码；有实际验证记录后再创建 Release，说明模型、依赖环境、已知限制和变更。不要在缺乏实验环境与数据时宣传基准指标或复现保证。

发布后确认：README 正常显示，LICENSE 被识别，`.github/workflows/ci.yml` 的离线检查通过，密钥/权重/原始数据未出现于仓库。若接收安全报告，可在仓库设置中开启私密漏洞报告。
