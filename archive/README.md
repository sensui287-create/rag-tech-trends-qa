# 历史参考代码

本目录保存演进过程中的实现。当前服务只使用 `backend/server.py`、`backend/index.html`、`backend/static/` 与 `backend/rag_core/`；活跃实验使用 `experiments/`。归档代码不会被这些入口自动导入。

| 文件 | 原位置 | 保留原因及范围 |
| --- | --- | --- |
| `rag_core_monolithic.py` | `backend/rag_core.py` | 保留旧父子文档检索、检索主管与联网分支，供算法演进查阅；原快照主体不重写 |
| `frontend/new.html` | `backend/new.html` | 旧交互和可取消请求版本，固定后端地址，与当前页面实现有差异 |
| `frontend/liushi.html` | `backend/liushi.html` | 较早 SSE 页面；来源事件的处理不符合现行元数据协议 |
| `analysis/direct_results.py` | `backend/direct_results.py` | 历史答案按顺序比较，依赖未提供的三份外部输入；优先改用有证据和稳定 ID 的现行评审器 |
| `analysis/picture.py` | `backend/picture.py` | 保留手填分数的示意雷达图；分数未改，不从实验结果自动计算 |

## 使用边界

旧单文件中的 `setup_retrievers` 会删除并重建相对路径下的 Chroma 目录，初始化还带有历史代理设置。不要把它替换到当前服务或在现有研究索引目录中调用。它不属于当前可运行性承诺。

两个旧 HTML 按原内容保留，不由当前服务对外提供；仍有固定 `localhost` 地址、旧流式解析和历史渲染逻辑。不要据此部署新服务。

旧答案比较器按三份输入的最短长度截断，缺少按 ID 对齐和证据校验；不能用它替代现行 `experiments.agent_results`。其配置导入已适配新目录，但数据和方法局限仍保留并明确记录。

绘图脚本已去掉重复导入和旧注释实现，并加 `main` 保护，导入不会打开图窗。只有查阅固定示意数据时才显式运行 `python -m archive.analysis.picture`。它的成功执行不代表实验复现。

如确需独立研究历史代码，可安装 `requirements-archive.txt`，并自行核对旧接口、数据、路径和服务配置。归档整理仅进行静态检查；没有执行历史索引删除、联网检索、付费评审或绘图。
