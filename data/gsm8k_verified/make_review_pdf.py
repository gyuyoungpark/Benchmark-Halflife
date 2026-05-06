"""GSM8K 30-item spot-check user-review PDF generator."""
import json
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, KeepTogether
)

# Register Korean font
pdfmetrics.registerFont(TTFont("Nanum", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"))
pdfmetrics.registerFont(TTFont("NanumBold", "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"))


styles = getSampleStyleSheet()
title_style = ParagraphStyle("title", parent=styles["Heading1"],
    fontName="NanumBold", fontSize=18, leading=22, spaceAfter=6, textColor=colors.HexColor("#1a3a6c"))
h2_style = ParagraphStyle("h2", parent=styles["Heading2"],
    fontName="NanumBold", fontSize=14, leading=18, spaceBefore=10, spaceAfter=4, textColor=colors.HexColor("#244f8c"))
h3_style = ParagraphStyle("h3", parent=styles["Heading3"],
    fontName="NanumBold", fontSize=12, leading=14, spaceBefore=8, spaceAfter=2, textColor=colors.HexColor("#333"))
body = ParagraphStyle("body", parent=styles["Normal"],
    fontName="Nanum", fontSize=10.5, leading=15, spaceAfter=4)
small = ParagraphStyle("small", parent=body, fontSize=9.5, leading=13, textColor=colors.HexColor("#444"))
mono = ParagraphStyle("mono", parent=body, fontName="Nanum", fontSize=9.5, leading=13,
    backColor=colors.HexColor("#f3f4f6"), borderPadding=4, leftIndent=4, rightIndent=4)
emph = ParagraphStyle("emph", parent=body, fontName="NanumBold", textColor=colors.HexColor("#b00"))
ok = ParagraphStyle("ok", parent=body, fontName="NanumBold", textColor=colors.HexColor("#0a6"))


def p(t, s=body):
    return Paragraph(t, s)


# Load items
from pathlib import Path
HERE = Path(__file__).parent
items = json.load(open(HERE / "spotcheck_30_items.json"))
target = {56: "1차에서 partial 판정 — 소수점 등장",
          47: "abstract 첫 문장 anecdote의 원본 — 정말 elementary arithmetic으로 풀리는지",
          57: "orig은 70+70 (덧셈), pert는 50×3 (곱셈) — 같은 op로 봐야 하나?",
          43: "8단계 — 모든 step이 정말 1:1 parallel한가?",
          102: "자연어 + 숫자 동시 변경 — 정말 같은 구조인가?"}

doc = SimpleDocTemplate(
    str(HERE / "USER_REVIEW.pdf"),
    pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=1.8*cm, bottomMargin=1.8*cm,
)

story = []

# ---- Cover ----
story.append(p("GSM8K 30-item Spot-Check", title_style))
story.append(p("사용자(2차 annotator) 검토 가이드", h2_style))
story.append(Spacer(1, 8))

story.append(p("<b>이 PDF는 무엇인가?</b>", h3_style))
story.append(p("논문 §4의 GSM8K \"125-item equivalence-verified subset\" 주장을 강화하기 위해, "
    "30개 item을 두 명의 annotator(=평가자)가 독립적으로 평가합니다. "
    "1차 평가자(Claude AI)는 이미 30개 모두 평가 완료. "
    "사용자께서 그 중 5개의 \"애매한\" item만 직접 평가하시면 됩니다."))

story.append(p("<b>왜 5개만?</b>", h3_style))
story.append(p("나머지 25개는 1차 평가자가 \"명백히 동등(equivalent)\"으로 판정해서 사용자 검토가 거의 의미 없습니다. "
    "5개는 reviewer(논문 심사자)가 공격할 가능성이 있는 borderline item들입니다. "
    "이 5개에 대한 사용자 의견이 §4 본문 한 문장으로 들어가서 reviewer 방어가 강화됩니다."))

story.append(p("<b>각 item에 대해 무엇을 결정하는가?</b>", h3_style))
story.append(p("orig 문제 ↔ pert 문제 두 개를 비교해서, 두 문제가 \"같은 어려움/같은 풀이 구조\"인지 한 단어로 답합니다. "
    "세 가지 단어 중 하나:"))

label_table = Table([
    ["라벨", "의미", "예시 판단 기준"],
    ["equivalent\n(동등)", "두 문제가 거의 같은 어려움 + 같은 풀이 구조",
     "같은 종류의 연산(+/-/×/÷)\n같은 단계 수\n비슷한 자릿수/난이도"],
    ["partial\n(부분 동등)", "구조는 같지만 한쪽이 미세하게 다름",
     "예: 한쪽은 정수, 다른 쪽은 소수 등장\n예: 한쪽은 carry 많고 다른 쪽은 적음"],
    ["non-equivalent\n(비동등)", "구조나 어려움이 분명히 다름",
     "예: 단계 수가 다름\n예: 한쪽이 훨씬 더 어려움"]
], colWidths=[3*cm, 5*cm, 8.5*cm])
label_table.setStyle(TableStyle([
    ("FONTNAME", (0, 0), (-1, -1), "Nanum"),
    ("FONTNAME", (0, 0), (-1, 0), "NanumBold"),
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#244f8c")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTSIZE", (0, 0), (-1, -1), 9.5),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.grey),
    ("BOX", (0, 0), (-1, -1), 0.5, colors.black),
    ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ("TOPPADDING", (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ("BACKGROUND", (0, 1), (0, 1), colors.HexColor("#e6f4ea")),
    ("BACKGROUND", (0, 2), (0, 2), colors.HexColor("#fff3cd")),
    ("BACKGROUND", (0, 3), (0, 3), colors.HexColor("#fdecea")),
]))
story.append(label_table)
story.append(Spacer(1, 10))

story.append(p("<b>5개 item을 다 보신 후 답변 양식</b>", h3_style))
story.append(p('한 줄로: "43=equivalent, 47=equivalent, 56=partial, 57=equivalent, 102=equivalent"', mono))
story.append(p("이 결과를 알려주시면 §4에 한 문장 추가하고 PDF 재컴파일합니다.", small))
story.append(Spacer(1, 6))
story.append(p("<b>1차 평가자(Claude) 의견 요약</b>", h3_style))
story.append(p("idx 43 = equivalent &nbsp;&nbsp;&nbsp; idx 47 = equivalent &nbsp;&nbsp;&nbsp; "
    "idx 56 = <font color='#b00'>partial</font> (소수점 1.5/40.5 등장) &nbsp;&nbsp;&nbsp; "
    "idx 57 = equivalent &nbsp;&nbsp;&nbsp; idx 102 = equivalent", small))
story.append(p("→ 사용자 의견이 1차 평가자와 같으면 그대로, 다르면 사용자 의견 우선.", small))

story.append(PageBreak())

# ---- Per-item sections ----
order = [43, 47, 56, 57, 102]
items_by_idx = {it["verified_idx"]: it for it in items}

def fmt_text(s):
    """Escape XML and turn newlines into <br/>."""
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace("\n", "<br/>"))

for n_, idx in enumerate(order, 1):
    it = items_by_idx[idx]
    note = target[idx]
    block = []
    block.append(p(f"Item {n_} of 5 &nbsp;·&nbsp; idx {idx}", h2_style))
    block.append(p(f"<b>왜 검토가 필요한가:</b> {note}", small))
    block.append(Spacer(1, 4))

    block.append(p("<b>ORIG (원본 문제)</b>", h3_style))
    block.append(p("<b>문제:</b> " + fmt_text(it["orig_question"]), body))
    block.append(p("<b>풀이/정답:</b><br/>" + fmt_text(it["orig_answer"]), mono))
    block.append(Spacer(1, 4))

    block.append(p("<b>PERT (변형 문제 — 숫자만 바뀜)</b>", h3_style))
    block.append(p("<b>문제:</b> " + fmt_text(it["pert_question"]), body))
    block.append(p("<b>풀이/정답:</b><br/>" + fmt_text(it["pert_answer"]), mono))
    block.append(Spacer(1, 6))

    block.append(p("<b>여기서 사용자 결정:</b>", h3_style))
    block.append(p("위 두 문제(ORIG vs PERT)를 비교했을 때:", body))
    block.append(p("&nbsp;&nbsp;• 같은 종류의 연산을 같은 순서로 사용하는가?", body))
    block.append(p("&nbsp;&nbsp;• 비슷한 자릿수/소수/받아올림 등 비슷한 어려움인가?", body))
    block.append(p("&nbsp;&nbsp;→ 한 단어로 답: <b>equivalent</b> / <b>partial</b> / <b>non-equivalent</b>", body))
    story.append(KeepTogether(block))
    story.append(Spacer(1, 14))

# ---- Final ----
story.append(PageBreak())
story.append(p("최종 답변 작성 예시", title_style))
story.append(Spacer(1, 8))
story.append(p("5개 item을 다 보셨다면, 다음 중 한 형태로 답변해주세요:", body))
story.append(Spacer(1, 4))
story.append(p("<b>형태 1 — 항목별 라벨</b>", h3_style))
story.append(p('43=equivalent<br/>47=equivalent<br/>56=partial<br/>57=equivalent<br/>102=equivalent', mono))
story.append(Spacer(1, 8))
story.append(p("<b>형태 2 — 1차 평가자와 다른 것만 짚기</b>", h3_style))
story.append(p('"56과 57에 동의 안 함. 56은 non-equivalent, 57은 partial. 나머지 3개 동의."', mono))
story.append(Spacer(1, 8))
story.append(p("<b>형태 3 — 신뢰 기반 빠른 결정</b>", h3_style))
story.append(p('"1차 평가자 결과 그대로 신뢰. 추가 변경 없음."', mono))
story.append(Spacer(1, 12))
story.append(p("이 결과를 받으면 다음을 진행합니다:", h3_style))
story.append(p("• Cohen's κ (annotator 일치도) 계산", body))
story.append(p("• §4 fidelity caveat에 한 문장 추가 (예: \"30 items 중 N개가 두 평가자 모두 equivalent 판정\")", body))
story.append(p("• PDF 재컴파일 및 최종 검증", body))

doc.build(story)
print(f"Wrote: {HERE / 'USER_REVIEW.pdf'}")
