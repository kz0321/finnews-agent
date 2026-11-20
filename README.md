# FinNews Agent

基于 [GPT Researcher](https://github.com/assafelovic/gpt-researcher) 的金融新闻研究课程项目。
输入公司、截止日期、7/30 天窗口和研究重点，输出附原文证据的简报。

**支持 DeepSeek API，不需要 OpenAI key。** 本地演示无需任何密钥。
真实研究使用 GPT Researcher 采集/写作、CPU MiniLM 检索，以及同一个托管模型进行核验和修订。

作者：Xu Kaizhi、Liu Hangxu、Zhang Yiqun、Fu Xuanyu、Guo Xinrui。

## 功能

- 五家公司：AAPL、MSFT、NVDA、AMZN、GOOGL；限定候选资料池与时间窗口。
- BM25、MiniLM 语义检索和 RRF 融合；最多 8 个片段、每来源 3 个、背景 2 个。
- SQLite 保存原文、日期、内容哈希、片段偏移、查询、状态转换和请求记录。
- Pydantic 结构化结论；检查引用、引句、数字和日期，最多一次本地补充检索与修订。
- Streamlit 证据面板、批量 CLI、盲评导出和人工评分统计。
- B0：关键词草稿；B1：混合检索草稿；B2：混合检索与有界核验。

> 演示语料是自行编写的虚构数据，使用哈希向量和确定性抽取，只验证软件流程。
> 不能据此声称真实金融研究准确率或 MiniLM 的实验效果。实际验证范围见项目报告。

## 快速开始

需要 Python 3.12 和 [uv](https://docs.astral.sh/uv/getting-started/installation/)。

```bash
git clone https://github.com/kz0321/finnews-agent.git
cd finnews-agent
uv sync --frozen --extra dev
uv run finnews demo
uv run streamlit run app.py
```

打开 `http://localhost:8501`，选择“离线演示”，点击“生成研究简报”。
结果位于 `runtime/runs/<run-id>/`，包含简报、运行记录和原文片段。
不用 uv 时可在 Python 3.12 虚拟环境执行 `python -m pip install -e ".[dev]"`；
完整传递依赖锁定请使用 `uv.lock`。

## DeepSeek 与真实资料

```bash
uv sync --frozen --extra live --extra dev
```

复制 `.env.example` 为 `.env`，仅在本地填写：

```dotenv
LLM_API_KEY=你的DeepSeek密钥
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-flash
TAVILY_API_KEY=你的Tavily密钥
```

`.env` 已被 Git 忽略。DeepSeek 用于生成和核验；Tavily 只用于联网搜索。
MiniLM 首次运行下载模型，之后复用缓存。其他 OpenAI 兼容服务可修改端点和模型名。

已有原文 JSON 快照时，无需 Tavily：

```bash
uv run --extra live finnews run --dataset data/private/corpus.json --query-id your-query-id --variant B2
```

先采集再研究：

```bash
uv run --extra live finnews collect --company AAPL --as-of 2025-11-01 --days 30 --output runtime/apple
uv run --extra live finnews run --dataset runtime/apple/collected.json --variant B2 --output runtime/apple
```

示例日期仅说明命令格式。历史研究须检查发布日期及原文快照；当前网页不能证明其
全部历史内容。日期未知的资料会存储展示，但不能建立窗口内事件。

## 测试与批量实验

```bash
uv run pytest -q
uv run ruff check src tests app.py
uv run finnews batch --demo --split test --output runtime/demo-batch
uv run finnews validate data/demo/dataset.json --strict
```

演示包含 5 个开发、20 个测试案例，批量命令输出 60 份对照结果。正式实验必须另行
准备真实资料并进行人工标注，模型核验标签不作为人工评分。

## 文档

- [项目内容、完成度和验证报告](docs/PROJECT_REPORT.md)
- [部署、使用与故障排查](docs/DEPLOYMENT.md)
- [架构与 proposal 对照](docs/ARCHITECTURE.md)
- [数据、盲评与实验指南](docs/EVALUATION.md)
- [第三方开源声明](THIRD_PARTY_NOTICES.md)

`src/finnews/` 包含模型、采集、检索、存储、核验、评测和 CLI；`app.py` 是界面；
`data/demo/` 是虚构测试语料；`data/evaluation/` 是正式实验材料；`tests/` 是测试；
`runtime/` 保存不入 Git 的运行数据。

`main` 保存验证后的提交；`test` 用于后续更新。两个分支均运行 GitHub Actions。
项目使用 MIT 许可证，原始网页保留其权利。系统不执行交易或价格预测。
