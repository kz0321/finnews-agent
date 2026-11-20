# 部署、使用与故障排查

## 环境与启动

Python 3.12；建议 uv 和 `uv.lock`。真实模式需要首次下载 MiniLM，CPU 可运行，
无需 GPU 或数据库服务。模型及完整依赖会占用数 GB 磁盘。

```bash
git clone https://github.com/kz0321/finnews-agent.git
cd finnews-agent
uv sync --frozen --extra dev
uv run finnews demo
uv run streamlit run app.py
```

打开 `http://localhost:8501`。Windows 复制配置命令是 `Copy-Item .env.example .env`；
macOS/Linux 是 `cp .env.example .env`。已填好配置时不要覆盖。
交付机器目录为 `D:\RA\finnews-agent`，也可直接使用已安装的虚拟环境：

```powershell
.\.venv\Scripts\python.exe -m finnews.cli demo
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## DeepSeek

真实模式安装 `uv sync --frozen --extra live --extra dev`，本地 `.env` 配置：

```dotenv
LLM_API_KEY=your-deepseek-key
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-flash
TAVILY_API_KEY=your-tavily-key
FINNEWS_DATA_DIR=runtime
```

OpenAI 兼容指请求协议，不要求 OpenAI 账号。所有生成角色用同一模型，向量在本地
计算。已有原文快照不需要 Tavily。价格未硬编码，可通过
`LLM_INPUT_PRICE_PER_MILLION` / `LLM_OUTPUT_PRICE_PER_MILLION` 配置当前每百万 token
美元单价。usage 或单价缺失时成本未知，不把未知当成免费。

## 操作

界面“离线演示”可直接运行；“已导入资料”上传合法 JSON 原文快照；“联网研究”由
GPT Researcher 采集原文并生成。命令行用法见 README，正式语料和评分见 EVALUATION.md。

输出位于 `runtime/`：`evidence.sqlite` 存证据和运行；`vectors/` 存向量缓存；
`runs/<id>/` 保存 `brief.md`、`run.json`、`chunks.json`。`batch-runs.json` 保存批量记录。
真实语料、密钥和缓存均不提交 Git。

## Docker 演示

运行 `docker compose up --build`，访问 `http://localhost:8501`。容器只安装演示依赖，
绑定本机端口并持久化 runtime。真实模型推荐先使用本地部署验证。
Docker 在交付机器上是否执行过，以项目报告为准。

## 排查

| 现象 | 处理 |
| --- | --- |
| No eligible dated evidence | 检查资料池和日期；不要用抓取日期冒充发布日期。 |
| 缺少 GPT Researcher 或 Sentence Transformers | 安装 live extra，确认使用同一个虚拟环境。 |
| 401、余额不足、模型不可用 | 检查服务商后台、密钥和模型名；不要将密钥贴到 issue。 |
| MiniLM 首次下载失败 | 检查 Hugging Face 连接，先使用演示模式。 |
| unverified | B0/B1 的预期状态；B2 超时也会明确返回未核验草稿。 |
| failed | 查看本地 trace；不能把失败当成成功研究。 |
| 少于 250 词 | 证据不足时不凑字数；日志记录长度是否达标。 |

## 后续更新

```bash
git switch test
# 修改代码
uv run pytest -q
uv run ruff check src tests app.py
git add <明确选择的源码或文档>
git commit -m "Describe the change"
git push origin test
```

验证后通过 PR 合入 main。不要提交 `.env`、账号凭据、受限新闻全文或本地数据库。
