"""Render the Chinese project report with embedded CJK fonts (optional reportlab)."""

from __future__ import annotations

import argparse
import html
import os
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import HRFlowable, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "docs/PROJECT_REPORT.md")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/FinNews-Project-Report.pdf")
    args = parser.parse_args()
    regular = Path(os.getenv("FINNEWS_REPORT_FONT", "C:/Windows/Fonts/msyh.ttc"))
    bold = Path(os.getenv("FINNEWS_REPORT_BOLD_FONT", "C:/Windows/Fonts/msyhbd.ttc"))
    if not regular.exists() or not bold.exists():
        raise SystemExit("Set FINNEWS_REPORT_FONT and FINNEWS_REPORT_BOLD_FONT to CJK TrueType font paths.")
    pdfmetrics.registerFont(TTFont("ReportCJK", str(regular), subfontIndex=0))
    pdfmetrics.registerFont(TTFont("ReportCJKBold", str(bold), subfontIndex=0))
    pdfmetrics.registerFontFamily(
        "ReportCJK", normal="ReportCJK", bold="ReportCJKBold", italic="ReportCJK", boldItalic="ReportCJKBold"
    )
    styles = getSampleStyleSheet()
    for name in ("BodyText", "Normal"):
        styles[name].fontName = "ReportCJK"
        styles[name].fontSize = 10
        styles[name].leading = 16
        styles[name].wordWrap = "CJK"
        styles[name].spaceAfter = 7
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            fontName="ReportCJKBold",
            fontSize=23,
            leading=31,
            textColor=colors.HexColor("#16324F"),
            spaceAfter=14,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportH2",
            fontName="ReportCJKBold",
            fontSize=14,
            leading=22,
            spaceBefore=12,
            spaceAfter=7,
            keepWithNext=True,
            wordWrap="CJK",
        )
    )
    styles.add(ParagraphStyle(name="ReportH3", parent=styles["ReportH2"], fontSize=11, leading=18))
    styles.add(
        ParagraphStyle(
            name="CodeText",
            parent=styles["BodyText"],
            fontSize=8,
            leading=12,
            backColor=colors.HexColor("#F1F5F9"),
            borderPadding=7,
        )
    )
    styles.add(ParagraphStyle(name="Cell", parent=styles["BodyText"], fontSize=8.5, leading=13, spaceAfter=0))
    styles.add(
        ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8, leading=12, alignment=TA_CENTER)
    )

    def rich(text):
        escaped = html.escape(text)
        escaped = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", escaped)
        escaped = re.sub(r"`([^`]+)`", r'<font color="#214D72">\1</font>', escaped)
        escaped = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", lambda m: m.group(1) + " (" + m.group(2) + ")", escaped)
        return escaped

    lines = args.input.read_text(encoding="utf-8").splitlines()
    story, i = [], 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line == "<!-- pagebreak -->":
            story.append(PageBreak())
            i += 1
            continue
        if line.startswith("```"):
            code = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                code.append(html.escape(lines[i]))
                i += 1
            story.append(Paragraph("<br/>".join(code), styles["CodeText"]))
            i += 1
            continue
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r"[-: ]+", c or "-") for c in cells):
                    rows.append([Paragraph(rich(c), styles["Cell"]) for c in cells])
                i += 1
            cols = len(rows[0])
            widths = [105, 244, 150] if cols == 3 else [499 / cols] * cols
            table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EFF5")),
                        ("LINEBELOW", (0, 0), (-1, 0), 0.7, colors.HexColor("#16324F")),
                        ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor("#CFD8E3")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 7),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                        ("TOPPADDING", (0, 0), (-1, -1), 6),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ]
                )
            )
            story += [table, Spacer(1, 8)]
            continue
        if line.startswith("# "):
            story += [
                Paragraph(rich(line[2:]), styles["ReportTitle"]),
                HRFlowable(width="100%", color=colors.HexColor("#214D72"), thickness=1.2),
                Spacer(1, 12),
            ]
        elif line.startswith("## "):
            story.append(Paragraph(rich(line[3:]), styles["ReportH2"]))
        elif line.startswith("### "):
            story.append(Paragraph(rich(line[4:]), styles["ReportH3"]))
        elif line.startswith("- "):
            story.append(Paragraph("• " + rich(line[2:]), styles["BodyText"]))
        else:
            paragraph = [line]
            while (
                i + 1 < len(lines)
                and lines[i + 1].strip()
                and not lines[i + 1].startswith(("#", "- ", "|", "```", "<!--"))
            ):
                i += 1
                paragraph.append(lines[i].strip())
            story.append(Paragraph(rich(" ".join(paragraph)), styles["BodyText"]))
        i += 1

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("ReportCJK", 8)
        canvas.setFillColor(colors.HexColor("#526477"))
        canvas.drawString(48, 25, "FinNews Agent | NLP Course Project | 2026-11-20")
        canvas.drawRightString(A4[0] - 48, 25, str(doc.page))
        canvas.restoreState()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(args.output),
        pagesize=A4,
        leftMargin=48,
        rightMargin=48,
        topMargin=40,
        bottomMargin=45,
        title="FinNews Agent 项目实现与交付报告",
        author="Xu Kaizhi, Liu Hangxu, Zhang Yiqun, Fu Xuanyu, Guo Xinrui",
    )
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    print(args.output)


if __name__ == "__main__":
    main()
