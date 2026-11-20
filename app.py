"""Run with: streamlit run app.py"""
import asyncio
from datetime import date
from pathlib import Path

import streamlit as st

from finnews.models import COMPANIES, Query
from finnews.pipeline import render_brief
from finnews.service import DEMO_DATASET, data_dir, load_corpus, research

st.set_page_config(page_title="FinNews Agent", page_icon="📄", layout="wide")
st.title("FinNews Agent")
st.caption("金融新闻研究 · 原文证据 · 混合检索 · 有界核验")
root = data_dir()

with st.sidebar:
    st.header("研究设置")
    mode = st.radio("资料模式", ["离线演示", "已导入资料", "联网研究"])
    variant = st.selectbox("研究配置", ["B2", "B1", "B0"],
        format_func=lambda v: {"B0": "B0 · 关键词检索", "B1": "B1 · 混合检索", "B2": "B2 · 混合检索与核验"}[v])
    st.caption("B0/B1 输出未经核验的草稿。B2 的模型核验也不能替代人工判断。")

sources, query, dataset_kind = [], None, "real"
if mode == "离线演示":
    dataset, sources, queries = load_corpus(DEMO_DATASET)
    st.info("当前使用自行编写的虚构案例和确定性演示逻辑，不调用模型，也不代表真实市场情况。")
    choice = st.selectbox("演示案例", range(len(queries)),
                          format_func=lambda i: f"{queries[i].company} · {queries[i].as_of} · {queries[i].id}")
    query = queries[choice]
    dataset_kind = "synthetic_demo"
elif mode == "已导入资料":
    st.write("导入符合数据格式的 JSON 文件，在固定原文资料上运行 DeepSeek 和本地 MiniLM 检索。")
    uploaded = st.file_uploader("资料快照", type="json")
    if uploaded:
        try:
            # Save a fixed server-side name, never a client-supplied path.
            path = root / "imported-corpus.json"
            path.write_bytes(uploaded.getvalue())
            dataset, sources, queries = load_corpus(path)
            choice = st.selectbox("研究问题", range(len(queries)), format_func=lambda i: queries[i].id)
            query = queries[choice]
            dataset_kind = dataset.get("dataset_kind", "real")
        except (ValueError, KeyError, IndexError) as exc:
            st.error(f"资料格式不正确：{type(exc).__name__}")
else:
    st.write("由 GPT Researcher 搜索并采集原文，再使用本地检索和 DeepSeek 生成简报。")
    cols = st.columns(3)
    company = cols[0].selectbox("公司", list(COMPANIES), format_func=lambda c: f"{c} · {COMPANIES[c]}")
    as_of = cols[1].date_input("研究截止日期", value=date.today())
    days = cols[2].selectbox("回溯天数", [30, 7])
    focus = st.text_input("研究重点", value="earnings, material announcements and risks")
    query = Query(id=f"live-{company}-{as_of}", company=company, as_of=as_of, window_days=days, focus=focus)

if st.button("生成研究简报", type="primary", disabled=query is None):
    with st.spinner("正在采集或检索证据并生成简报…"):
        try:
            if mode == "联网研究":
                from finnews.collect import collect_sources
                sources = asyncio.run(collect_sources(query, audit_path=root / "collection-audit.json"))
                query.source_ids = [s.id for s in sources][:8]
            record, chunks, output = asyncio.run(research(query, sources, variant=variant,
                demo=mode == "离线演示", dataset_kind=dataset_kind, output_dir=root))
            st.session_state["result"] = (record, chunks, sources, str(output))
        except Exception as exc:
            st.error(f"运行未完成（{type(exc).__name__}）。请检查本地 .env、可用资料及安装的 live 依赖。")

if "result" in st.session_state:
    record, chunks, saved_sources, output = st.session_state["result"]
    metrics = st.columns(4)
    metrics[0].metric("状态", record.status)
    metrics[1].metric("证据片段", len(record.selected_ids))
    metrics[2].metric("生成阶段请求记录", len(record.calls))
    metrics[3].metric("耗时", f"{record.timings.get('total_seconds', 0):.2f} s")
    brief_tab, evidence_tab, trace_tab = st.tabs(["研究简报", "原文证据", "运行记录"])
    with brief_tab:
        markdown = render_brief(record, chunks)
        st.markdown(markdown)
        st.download_button("下载简报", markdown, "finnews-brief.md", "text/markdown")
        st.download_button("下载运行记录", record.model_dump_json(indent=2), "finnews-run.json", "application/json")
    with evidence_tab:
        checks = {c.finding_id: c for c in record.checks}
        shown_ids = set(record.selected_ids)
        if record.brief:
            shown_ids.update(cid for finding in record.brief.findings for cid in finding.evidence_ids)
        for chunk in chunks:
            if chunk.id in shown_ids:
                with st.expander(f"{chunk.title} · {chunk.published_at} · {chunk.id}"):
                    st.write(chunk.text)
                    st.markdown(f"[查看来源]({chunk.url})")
                    st.caption(f"原文偏移 [{chunk.start}, {chunk.end}) · 来源 ID {chunk.source_id}")
        undated = [s for s in saved_sources if s.published_at is None]
        if undated:
            st.warning(f"{len(undated)} 篇资料缺少可信发布日期，已保存但不能作为窗口内事件证据。")
        if checks:
            st.dataframe([c.model_dump() for c in checks.values()], use_container_width=True)
    with trace_tab:
        st.json({"query": record.query.model_dump(mode="json"), "encoder": record.encoder,
                 "dataset_kind": record.dataset_kind, "timings": record.timings, "trace": record.trace,
                 "calls": record.calls})
        st.caption(f"本地输出：{Path(output)}")
else:
    st.markdown("选择一个案例开始研究。每条结论附有证据片段，核验失败的结论会被修订或转为信息缺口。")
