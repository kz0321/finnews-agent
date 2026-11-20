# FinNews Agent 项目实现与交付报告

NLP 课程小组项目 | 交付版本 0.1.0 | 2026-11-20

Xu Kaizhi · Liu Hangxu · Zhang Yiqun · Fu Xuanyu · Guo Xinrui

公开仓库：https://github.com/kz0321/finnews-agent

## 1. 项目内容与完成度

FinNews Agent 基于 GPT Researcher，将公司新闻、财报发布稿等资料转化为带原文引用的金融研究简报。
系统支持 AAPL、MSFT、NVDA、AMZN 和 GOOGL，接受研究截止日、7/30 天窗口和研究重点。
课程技术重点是混合检索、证据追溯以及最多一次补充检索与修订，而非交易、预测或模型训练。

**软件原型和可复现验证已完成；正式金融研究实验尚未完成。** 已实现并测试界面、CLI、
数据存储、真实 MiniLM 检索、GPT Researcher 接口和 DeepSeek 兼容接入。
尚未使用真实 DeepSeek/Tavily 凭据运行付费服务，也未将模型评分或虚构数据冒充人工实验结果。

| 模块 | 已交付内容 | 验证状态 |
| --- | --- | --- |
| 研究基座 | GPT Researcher 0.17.0 的采集与外部上下文写作接口 | 本地 HTTP/SSE 集成测试通过；真实服务待配置 |
| 混合检索 | BM25、CPU MiniLM、RRF、上下文筛选、向量缓存 | 单元测试及真实模型加载通过 |
| 证据持久化 | SQLite 原文、哈希、日期、偏移、运行与调用记录 | 读写与完整性测试通过 |
| 核验恢复 | JSON 校验、引句/数字/币种/期间检查、有界修订 | 正常、失败、超时与越界路径通过 |
| 应用 | Streamlit 简报/证据/记录面板，批量 CLI | 页面交互与 60 次演示运行通过 |
| 评估工具 | 原文覆盖率、盲评导出、双人评分统计 | 工具测试通过；真实人工评分待完成 |
| 真实资料 | 30 篇官方原文、25 个问题、131 个候选证据区间 | 已下载到本地；日期和 gold 待人工复核 |
| 部署与交付 | 锁文件、配置样例、Docker 演示配置、文档、main/test 分支 | 本地环境已验证；Docker 未运行 |

当前版本可作为可演示、可继续实验的课程工程交付，不能据此声称 B2 已在真实金融任务上优于 B0/B1。

<!-- pagebreak -->

## 2. 技术路径与实现边界

工作流为：采集或导入原文 → 保存快照 → 分块与索引 → 检索筛选 → 结构化草稿 → 核验 →
必要时同池补检索与修订 → 复核 → 输出简报和信息缺口。

**基座复用。** GPT Researcher 负责原有搜索规划、网页采集及 `write_report()`。
FinNews 通过 `conduct_research()` 和 `get_research_sources()` 获取原文，以
`ext_context` 和 `custom_prompt` 提供筛选后的证据与结构化要求。没有复制或伪称原创其核心能力。

**检索。** 真实模式使用 MiniLM tokenizer，片段最多 180 token、约 24 token 重叠；标题和正文
共同检查 256 token 输入上限。模型 revision 固定为
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`，输出 384 维归一化向量。
BM25 和 dense 各取最多 20 个候选，以 RRF 常数 60 融合，最多选 8 个片段、每来源 3 个、背景 2 个。

**预算。** 证据预算包括 ID、标题、URL 和日期。默认按 UTF-8 字节上界保守控制，完全离线可用；
正式实验可一致启用 `FINNEWS_TOKEN_COUNTER=cl100k`，使用 2,000 个公共代理 token。
后者不是 DeepSeek 的精确 tokenizer，实际 token 用量和计费以 API 返回为准。
不同计数模式必须在开发集阶段固定，不可混入同次对照实验。

**时间与引用。** 未来来源被排除，早期来源标为背景，未知日期来源只保留展示。
chunk 保存原文字符偏移和稳定 ID，不能用搜索摘要替代原文。背景内容也必须成为普通带引用的
finding，经过同样的检查，不能通过独立背景段落绕过核验。

**有界恢复。** 正常路径最多四个逻辑模型阶段，整个运行额外共享一次 schema repair。
核验发现问题后，只能在原候选池里补充最多两个片段；已支持结论的证据固定保留。
修订不得改动受保护结论或引入新 ID，最终不能支持的结论转为信息缺口。
模型核验不等于事实真值，确定性检查也不能证明因果关系正确。

**模型与费用。** 使用一个可配置的 OpenAI 兼容模型，当前示例为 DeepSeek `deepseek-flash`，
显式关闭 thinking 以控制输出预算。每个生成阶段最多一次传输重试，并受阶段总时限限制。
GPT Researcher 的采集规划仍保留上游重试策略，另有总超时与采集审计文件，不能与生成阶段上限混为一谈。
单价、usage 或搜索费用未知时记录为空值，不把未知费用写成零。

<!-- pagebreak -->

## 3. 已执行的验证

验证在 Windows、Python 3.12.15 环境进行；依赖由 `uv.lock` 锁定。

| 验证项目 | 实际结果 | 能证明的范围 |
| --- | --- | --- |
| 完整依赖测试 | 52 passed | 检索、存储、状态机、接口、评估及页面行为 |
| 仅演示依赖测试 | 49 passed，3 skipped | 没有 live extra 仍可运行；3 项依赖集成测试明确跳过 |
| Ruff | 通过 | 当前源码和测试的静态规则检查 |
| 实际 GPT Researcher 接口 | 本地 HTTP/SSE 服务测试通过 | 验证参数、流式响应、token 记录和最多两次实际请求 |
| 实际 MiniLM | CPU 输出 2×384；语义检索及超长拒绝通过 | 真实模型可运行；测试文本为人工合成 |
| 批量演示 | 20 个测试问题 × 3 配置 = 60 次，无失败 | B0/B1 共 40 个未核验草稿，B2 共 20 个完成结果 |
| 盲评导出 | 60 份报告、180 条陈述、12 份双评分配 | 已生成空白评分表，未伪填人工分数 |
| 官方资料下载 | 30 篇、25 个问题、131 个候选 span | 已获取可读原文，不代表日期和 gold 已人工审核 |

测试覆盖错误引用、非原文引句、币种替换、季度/财年替换、正负号、保护结论越界修改、
schema repair 超额、超时保留未核验草稿，以及遗漏原子陈述、部分双评误计和开发测试窗口重叠。

真实资料位于 `data/private/real-corpus/`，不进入公开 Git。公开代码提供官方来源清单和
可恢复下载脚本。演示数据位于 `data/demo/`，明确标为虚构；其中数字不描述真实上市公司。

## 4. 正式实验仍需完成的部分

- 在本地 `.env` 填入有效的 DeepSeek key；联网采集另需 Tavily key。
- 人工复核原文日期和内容，为每个问题确定 2–4 个 gold facts 与原文 span。
- 在 5 个开发问题上冻结参数，再执行 20 个测试问题的 B0/B1/B2 真实模型实验。
- 两名组员独立评分，至少 12 份报告完整双评；拆分复合陈述，保留原始分歧和裁决。
- 单独执行 5 次实时采集演示，记录服务可用性、费用和错误案例。

正式指标应同时报告支持且正确引用比例、关键事实覆盖率、时间一致性、有效报告数、失败数、
延迟及费用。失败和缺失运行仍留在总体分母；未完成标注时不输出看似完整的最终质量分数。
当前实现三组独立生成初稿，采样差异也是实验限制。Docker 配置尚未在交付机器运行。

<!-- pagebreak -->

## 5. 本地部署与使用

项目目录：`D:\RA\finnews-agent`。该机器已建立 `.venv`，可直接执行：

```powershell
cd D:\RA\finnews-agent
.\.venv\Scripts\python.exe -m finnews.cli demo
.\.venv\Scripts\python.exe -m streamlit run app.py
```

浏览器访问 `http://localhost:8501`。选择“离线演示”即可无密钥体验；“已导入资料”读取
固定 JSON 快照；“联网研究”会调用 GPT Researcher 搜索采集后生成。

在其他机器从公开仓库安装：

```bash
git clone https://github.com/kz0321/finnews-agent.git
cd finnews-agent
uv sync --frozen --extra dev
uv run finnews demo
uv run streamlit run app.py
```

真实研究安装 `uv sync --frozen --extra live --extra dev`，复制 `.env.example` 为 `.env`，
填入以下设置。密钥只保存在本地，不需要发送到聊天或 GitHub：

```dotenv
LLM_API_KEY=your-deepseek-key
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-flash
TAVILY_API_KEY=your-tavily-key
```

已有原文时不需要 Tavily，可运行：

```bash
uv run --extra live finnews run --dataset data/private/real-corpus/dataset.json --variant B2
```

正式评测前须先审核资料和参考标注。实时搜索示例、盲评、评分、缓存与错误排查命令，
分别见 README、DEPLOYMENT.md、EVALUATION.md。完整源码包括 `src/finnews/`、`app.py`、
`tests/`、`scripts/`、数据模板和依赖锁文件。

<!-- pagebreak -->

## 6. 输出、维护和开源说明

每次运行生成 `brief.md`、`run.json` 和 `chunks.json`，保存在 `runtime/runs/<run-id>/`。
SQLite 存放证据与运行，向量缓存使用模型 revision 和文本指纹避免错误复用。
界面可查看每条结论的原文、发布日期、来源 URL、偏移和模型核验状态，并下载结果。

`main` 用于验证后的版本，`test` 用于后续更新。修改后先运行测试与静态检查，再提交到
test 分支，通过 Pull Request 合入 main。GitHub Actions 自动运行不需要密钥的验收流程。

```bash
git switch test
uv run pytest -q
uv run ruff check src tests app.py
git add <需要提交的源码或文档>
git commit -m "Describe the change"
git push origin test
```

公开仓库不包含 `.env`、API key、真实来源全文、SQLite 数据库或本地向量缓存。
保留五位组员署名、MIT 许可证及 GPT Researcher 等第三方归属说明。
本项目原始 proposal 中的个人学号不随公开源码重新发布。

## 7. 关键限制

当前网页快照不能证明历史时点的全部内容；原文日期和表格文本仍需人工审核。
短片段可能拆散必要的财务上下文，混合检索也可能遗漏精确数值。
同一模型生成与核验可能产生相关错误，不能把 verifier 的 supported 标签当成研究结论。
证据不足时简报可少于目标 250–400 词，系统记录长度，优先保留信息缺口。

演示和本地 HTTP 接口测试均不证明实际 DeepSeek/Tavily 服务可用；这些真实调用尚未执行。
本报告没有给出虚构的准确率、提升百分比、统计显著性或供应商费用。
系统不进行交易、不训练模型、不提供投资建议。

## 8. 主要来源

- GPT Researcher：https://github.com/assafelovic/gpt-researcher
- 固定软件版本：https://pypi.org/project/gpt-researcher/0.17.0/
- MiniLM 模型：https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- DeepSeek API：https://api-docs.deepseek.com/
- DeepSeek thinking 配置：https://api-docs.deepseek.com/guides/thinking_mode/
- 本地实现与评测规范：仓库 README、ARCHITECTURE.md、DEPLOYMENT.md、EVALUATION.md。
