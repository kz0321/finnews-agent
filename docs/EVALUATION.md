# 评估、数据准备与人工标注

本项目提供可运行的评估工具，以及两种用途不同的数据。`data/demo/dataset.json` 是团队自行编写的**虚构软件测试资料**，包含 35 篇文本、5 个开发问题和 20 个测试问题。文中的公司代码只是场景标签，所有金额和事件都不是实际披露。这个数据集可以验证数据流、检索约束、分支执行与文件导出，不能证明金融研究质量。

`data/evaluation/real_plan.json` 是真实公司评估计划：AAPL、MSFT、NVDA、AMZN、GOOGL 各有 1 个开发窗口和 4 个测试窗口，共 25 个问题、30 个官方 HTML 来源种子。发布时间覆盖 2024 年初至 2025 年初。第一组财报事件仅用于开发，后四组用于测试。种子文件只有 URL、日期、任务和来源关系，不含下载后的正文，也不表示已经完成人工评估。

## 1. 准备真实语料

在项目环境中运行：

```powershell
python scripts/build_real_corpus.py
```

默认在被 Git 忽略的 `data/private/real-corpus/` 下保存 `dataset.json` 和 `download-report.json`。后者逐项记录成功、失败、正文长度及内容哈希。下载器只访问固定的官方域名，不绕过 403/429、登录或其他访问限制；正文重复时不会增加文档数量。抓取失败时仍保存已经获得的资料，并以非零退出码报告不完整。

使用 `python scripts/build_real_corpus.py --resume` 可以校验并复用已有正文哈希，只补抓失败来源。首次部署验收已成功获取 30 篇原文并生成 25 个待审核问题；这并不表示人工 gold 或模型质量评估已完成。

可先检查少量来源：

```powershell
python scripts/build_real_corpus.py --limit 3 --output data/private/smoke-corpus
```

语料下载不使用模型 API。当前网页副本只能证明本次下载内容，不能证明页面在历史截止日时完全相同。种子日期仍须由组员对照原始页面确认；表格转换后的币种、列标题、会计期间也须人工检查。确认后的日期写在 `published_at`，原始文本固定后不要在标注期间重新清洗。

下载器给每个问题添加 `annotation_candidates`：从原文挑选带数值的候选段落及精确字符偏移。这些只是**待审核候选**，不是 gold facts，`gold_facts` 保持为空。仅在人工确认后，把 2–4 个关键事实写入 `gold_facts` 并把问题的 `annotation_status` 改为 `human_reviewed`：

```json
{
  "id": "unique-fact-id",
  "text": "人工核实后的简洁事实表述，保留数字、单位和期间",
  "evidence": [
    {"source_id": "实际来源 ID", "start": 100, "end": 245}
  ]
}
```

`start` 包含、`end` 不包含，均基于保存的 Unicode 文本字符位置。一个事实需要多段证据时，列出所有必需区间。不要把从模型输出倒推的答案当作预先标注的关键事实。真实语料不足 30 篇、测试问题不足 20 个、gold 未审核时，严格检查会失败；应修复来源或记录与 proposal 的偏差，不能用重复网页补足数字。

```powershell
finnews validate data/private/real-corpus/dataset.json --strict
```

`system_case()` 仅允许 `Query` 与 `Source` 的已定义字段进入运行流程。gold facts、候选标注、测试标签均不会进入 Agent 提示词。每个问题仅能检索其 `source_ids` 指定的最多 8 篇候选文档。

## 2. 冻结实验与运行三组配置

先在 5 个开发问题上确认切块、数字 token、日期过滤、费用和输出格式，然后固定来源快照、模型、提示词、检索设置和依赖锁文件。20 个测试问题不能继续用于调参。

| 配置 | 初次检索 | 生成后的处理 |
| --- | --- | --- |
| B0 | BM25 | 结构化初稿 |
| B1 | BM25 + MiniLM + RRF | 结构化初稿 |
| B2 | 与 B1 相同 | 核验、可选本地补检索、一次修订与复核 |

```powershell
# 60 次虚构场景的软件验收：不调用模型 API
finnews batch --demo --dataset data/demo/dataset.json --split test --output runtime/demo-evaluation

# 真实资料：需要先完成部署文档中的模型与密钥设置
finnews batch --dataset data/private/real-corpus/dataset.json --split test --output runtime/real-evaluation
```

保存文件包括 `batch-runs.json`、`batch-chunks.json` 和运行摘要。每个配置保持相同的候选池、日期规则、证据预算及生成模型。B1 相对 B0 衡量证据选择变化，B2 相对 B1 衡量整个核验与修订流程，不能把后者解释为单独某一步的效果。当前实现各配置独立生成初稿，真实模型采样差异应作为限制报告。

如果网页更新，重新下载会得到不同哈希。不要把新旧快照混入同一次对比。`test` Git 分支用于后续开发，并不意味着该分支上的输出已经通过盲评。

## 3. 计算参考证据覆盖率

```powershell
python -m finnews.evaluation retrieval --dataset data/demo/dataset.json --runs runtime/demo-evaluation/batch-runs.json --chunks runtime/demo-evaluation/batch-chunks.json --output runtime/demo-evaluation/retrieval-metrics.json
```

计算规则是：一个关键事实所需的全部原文区间，是否被选中 chunk 的区间并集覆盖。相邻或重叠 chunk 可以共同覆盖，区间中存在缺口则不算覆盖；来源不同的文字不能拼接成同一来源的证据。初次检索与最终检索分别计算，并输出按问题配对的结果与 `temporal_stress` 标记。失败或缺失的测试运行在总体分母中保留，覆盖率记为零。没有 gold spans 的问题不会产生看似完整的覆盖率。

覆盖率是有限参考事实的检索指标，不是全部相关信息的召回率，也不是报告事实正确率。四个时间压力案例单独分析，不代表实际业务中未来信息污染的发生频率。

## 4. 双人盲评

```powershell
python -m finnews.evaluation export-blind --runs runtime/demo-evaluation/batch-runs.json --chunks runtime/demo-evaluation/batch-chunks.json --output runtime/demo-evaluation/annotations
```

导出内容：

| 文件 | 用途 |
| --- | --- |
| `primary_claims.csv` | 第一轮逐项人工判断；默认评分栏为空 |
| `secondary_claims.csv` | 随机抽取至少 20% 报告，交另一名组员独立判断 |
| `reports.csv` | 研究日期、窗口、任务和是否有报告，保留空报告可见性 |
| `double_score_assignments.json` | 双评分配清单 |
| `PRIVATE_mapping.json` | 盲编号与运行 ID、B0/B1/B2 的对应关系；评分结束前仅由协调人保管 |

评分表显示报告陈述与实际引用的原文、日期和 URL，不显示配置名称或模型核验标签。若表中出现 `EVIDENCE NOT SUPPLIED`，必须先补齐对应 chunk，不能仅根据一个可打开的链接给满分。一个 finding 包含多个可独立判断的事实时，复制该行并拆成多个 `claim_id`，例如 `F1.1`、`F1.2`，保留相同 `finding_id`。

每行填写以下字段：

| 字段 | 允许值与判断方法 |
| --- | --- |
| `supported` | `1`：该原子陈述完整地被所引证据支持；缺引、部分支持、矛盾均为 `0` |
| `temporal_correct` | `1`：事件日期与会计期间准确；错误为 `0`，无法确定为 `unknown` |
| `covered_gold_fact_ids` | 该陈述正确且有证据覆盖的预标关键事实 ID，多个用分号分隔 |
| `annotator` | 评分人姓名或稳定匿名 ID；双评必须来自不同的人 |
| `comment` | 数字、单位、期间错误，或解释评分分歧 |

支持率判断必须检查数值、币种、单位与会计期间。核验器输出 `supported`、JSON 合法、URL 有效，都不能代替人工判断。未知时间不获得时间正确率分子信用。不要把未评分行填写为零假装已完成评价。

CSV 评分命令会拒绝任何未填写完整的原子陈述行，防止拆出的第二个陈述被遗漏后仍得到完整报告分数。双评报告数只统计所有第一评分人的原子陈述都由第二人独立评分的报告；仅评分其中一条不能算整份报告完成双评。开发集评分和未知运行 ID 不能进入测试集统计。

```powershell
python -m finnews.evaluation score --dataset data/demo/dataset.json --runs runtime/demo-evaluation/batch-runs.json --annotations runtime/demo-evaluation/annotations/primary_claims.csv --secondary runtime/demo-evaluation/annotations/secondary_claims.csv --mapping runtime/demo-evaluation/annotations/PRIVATE_mapping.json --output runtime/demo-evaluation/human-metrics.json
```

有争议的行可以另存为相同格式的 `adjudicated_claims.csv`，使用 `--adjudicated` 加入最终评分。原始独立评分应保留，原始一致率使用裁决前的两人标签计算。

## 5. 指标、费用与报告原则

最终报告同时展示有效报告数、失败数、未评分报告数和原子陈述数。支持率与时间正确率按已经人工评分的原子陈述计算；若还有未评分的成功报告，整体关键事实覆盖率返回空值。关键事实覆盖率以全部 20 个测试问题为分母，失败、缺失及空报告计零，防止删除困难事实后看起来更准确。

输出包含各问题的配对结果，不只报告平均值。人工双评的一致性采用原始支持标签的一致比例，并显示双评报告数、至少 20% 的要求与配对陈述数；样本很小，不能承诺统计显著性。

费用报告用 `known_llm_call_cost_usd` 表示已知模型调用费用，用 `llm_call_cost_usd` 表示费用完整时的模型调用合计；出现未计价调用或缺失运行时，后者为空值。由于还需要独立核对搜索与采集费用，`total_cost_usd` 始终保持空值，不能把模型调用小计称作项目总费用。网络获取费用、一次性索引、CPU 检索时间与模型调用费用应分开汇报；供应商账单仍需人工对账。`timings.total_seconds` 存在时输出中位延迟。

真实实验最终还需要：人工复核的原文日期和 gold、60 份真实模型报告、两名组员评分、至少 12 份报告双评，以及 5 次单独的实时采集演示。仓库包含完成这些工作的工具，未执行或未经人工完成的部分应在最终课程报告中明确标为待完成。虚构 fixture 的测试通过率和参考覆盖率只能作为实现验收结果。

## 6. 数据与复现边界

仓库内的虚构文本适用项目 MIT 许可。官方新闻稿与投资者关系正文权利属于原发布方，默认只保存在 `data/private/`，不随公共代码发布。公开复现资料使用来源清单、获取脚本、配置和内容哈希。分享真实运行报告前还应检查其中引用原文的长度与来源要求；不要把完整外部新闻稿放入公共仓库。
