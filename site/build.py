"""steps/ → docs/ (GitHub Pages)

    uv run python site/build.py

단계마다 README.md(설명), transformer.py(그 단계까지의 코드), demo.py(실행한 코드),
output.txt + charts/(실행 결과)를 읽어 정적 HTML을 만든다. 실행은 하지 않는다.
결과를 다시 만들려면 먼저 `uv run python tools/run_step.py all`.
"""

import difflib
import html
import json
import re
import shutil
from pathlib import Path

import markdown
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer
from pygments.token import Token

ROOT = Path(__file__).resolve().parents[1]
STEPS_DIR = ROOT / "steps"
OUT = ROOT / "docs"
REPO = "https://github.com/dev-jonghoonpark/annotated-transformer-step-by-step"
ORIGINAL_KO = "https://dev-jonghoonpark.github.io/annotated-transformer-modern/"

PARTS = [
    ("1부 · 모델 만들기", range(1, 11)),
    ("2부 · 학습시키기", range(11, 15)),
    ("3부 · 진짜 번역", range(15, 19)),
]

SUMMARY = {
    1: "softmax(QKᵀ/√d)V 한 줄",
    2: "미래와 패딩을 가리는 마스크",
    3: "어텐션 8개를 한 번에",
    4: "토큰마다 따로 도는 2층 MLP",
    5: "단어 id → 벡터, 순서 → 사인파",
    6: "잔차 연결과 LayerNorm",
    7: "같은 층을 6번 쌓은 인코더",
    8: "원문을 보는 어텐션이 더해진 디코더",
    9: "부품 다섯 개를 하나로",
    10: "한 토큰씩 고르는 생성",
    11: "정답을 한 칸 밀어 만든 배치",
    12: "label smoothing과 손실",
    13: "warmup 학습률과 학습 루프",
    14: "숫자열 따라 쓰기로 첫 학습",
    15: "spaCy 토큰화와 사전",
    16: "문장을 텐서 배치로",
    17: "Multi30k 독→영 8 epoch",
    18: "학습된 어텐션 들여다보기",
}


# ── 코드 하이라이트 (줄 단위) ─────────────────────────────────────────────────

_fmt = HtmlFormatter()


def _css_class(ttype):
    while ttype not in _fmt.ttype2class and ttype is not Token:
        ttype = ttype.parent
    return _fmt.ttype2class.get(ttype, "")


def highlight_lines(code):
    """여러 줄 문자열(docstring 등)이 있어도 줄마다 닫힌 HTML이 되도록 토큰을 줄 단위로 자른다."""
    lines = [[]]
    for ttype, value in PythonLexer().get_tokens(code):
        cls = _css_class(ttype)
        for i, part in enumerate(value.split("\n")):
            if i:
                lines.append([])
            if part:
                esc = html.escape(part)
                lines[-1].append(f'<span class="{cls}">{esc}</span>' if cls else esc)
    out = ["".join(l) for l in lines]
    n = code.count("\n") + (0 if code.endswith("\n") else 1)
    return (out + [""] * n)[:n]


# ── diff ─────────────────────────────────────────────────────────────────────

CONTEXT = 3


def row(kind, old_no, new_no, code_html):
    sign = {"add": "+", "del": "−", "ctx": ""}[kind]
    return (
        f'<tr class="{kind}"><td class="ln">{old_no or ""}</td><td class="ln">{new_no or ""}</td>'
        f'<td class="sg">{sign}</td><td class="cd">{code_html or " "}</td></tr>'
    )


def render_diff(old, new):
    a, b = old.splitlines(), new.splitlines()
    ha, hb = highlight_lines(old) if old else [], highlight_lines(new)
    ops = difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
    rows, added, removed = [], set(), 0
    for idx, (tag, i1, i2, j1, j2) in enumerate(ops):
        if tag == "equal":
            n = i2 - i1
            head = CONTEXT if idx > 0 else 0
            tail = CONTEXT if idx < len(ops) - 1 else 0
            if n <= head + tail + 2:
                rows += [row("ctx", i1 + k + 1, j1 + k + 1, hb[j1 + k]) for k in range(n)]
                continue
            rows += [row("ctx", i1 + k + 1, j1 + k + 1, hb[j1 + k]) for k in range(head)]
            hidden = [row("ctx", i1 + k + 1, j1 + k + 1, hb[j1 + k]) for k in range(head, n - tail)]
            rows.append(
                f'<tr class="fold"><td colspan="4"><button type="button">⋯ 그대로인 {len(hidden)}줄 펼치기</button></td></tr>'
                f'<tbody class="hidden">{"".join(hidden)}</tbody>'
            )
            rows += [row("ctx", i1 + k + 1, j1 + k + 1, hb[j1 + k]) for k in range(n - tail, n)]
        else:
            for k in range(i1, i2):
                rows.append(row("del", k + 1, None, ha[k]))
                removed += 1
            for k in range(j1, j2):
                rows.append(row("add", None, k + 1, hb[k]))
                added.add(k)
    diff_html = f'<table class="code diff"><tbody>{"".join(rows)}</tbody></table>'
    full_rows = "".join(
        f'<tr class="{"add" if k in added else "ctx"}"><td class="ln">{k + 1}</td><td class="cd">{hb[k] or " "}</td></tr>'
        for k in range(len(b))
    )
    full_html = f'<table class="code full"><tbody>{full_rows}</tbody></table>'
    return diff_html, full_html, len(added), removed, len(b)


def render_plain_code(code):
    rows = "".join(
        f'<tr class="ctx"><td class="ln">{k + 1}</td><td class="cd">{h or " "}</td></tr>'
        for k, h in enumerate(highlight_lines(code))
    )
    return f'<table class="code full"><tbody>{rows}</tbody></table>'


# ── 마크다운 (수식 보호) ─────────────────────────────────────────────────────


def render_md(text):
    stash = []

    def keep(m):
        stash.append(html.escape(m.group(0)))
        return f"\x00{len(stash) - 1}\x00"

    text = re.sub(r"\$\$.+?\$\$|\$[^$\n]+?\$", keep, text, flags=re.S)
    out = markdown.markdown(text, extensions=["tables", "fenced_code"])
    out = re.sub(r"\x00(\d+)\x00", lambda m: stash[int(m.group(1))], out)
    out = out.replace('<a href="http', '<a target="_blank" rel="noopener" href="http')
    return out


# ── 실행 결과 ────────────────────────────────────────────────────────────────


def render_output(step, text):
    parts = re.split(r"\[\[chart:([\w-]+)\]\]\n?", text)
    out = []
    for i, p in enumerate(parts):
        if i % 2 == 0:
            p = p.strip("\n")
            if p:
                out.append(f'<pre class="out">{html.escape(p)}</pre>')
        else:
            src = f"charts/{step['slug']}/{p}.json"
            out.append(f'<div class="chart" data-spec="{src}"><div class="chart-loading">차트 불러오는 중…</div></div>')
    return "\n".join(out)


# ── 페이지 틀 ────────────────────────────────────────────────────────────────

HEAD = """<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="{desc}">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable-dynamic-subset.min.css">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.22/dist/katex.min.css">
<link rel="stylesheet" href="style.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.22/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.22/dist/contrib/auto-render.min.js"></script>
{extra_head}
<script defer src="app.js"></script>
</head>
<body>
"""

VEGA = """<script defer src="https://cdn.jsdelivr.net/npm/vega@6"></script>
<script defer src="https://cdn.jsdelivr.net/npm/vega-lite@6"></script>
<script defer src="https://cdn.jsdelivr.net/npm/vega-embed@7"></script>"""


def sidebar(steps, current=None):
    items = []
    for part, rng in PARTS:
        lis = "".join(
            f'<li><a href="{s["slug"]}.html"{" aria-current=\"page\"" if current and s["num"] == current["num"] else ""}>'
            f'<span class="num">{s["num"]:02d}</span><span>{html.escape(s["title"])}</span></a></li>'
            for s in steps if s["num"] in rng
        )
        items.append(f'<div class="part">{part}</div><ol>{lis}</ol>')
    label = f'{current["num"]:02d} / {len(steps)} · {html.escape(current["title"])}' if current else "단계 목록"
    return f"""<nav class="side">
  <a class="brand" href="index.html">Annotated Transformer<br><b>한 단계씩</b></a>
  <details class="toc" open><summary>{label}</summary>{"".join(items)}</details>
</nav>"""


def step_page(steps, i):
    s = steps[i]
    prev = steps[i - 1] if i > 0 else None
    nxt = steps[i + 1] if i + 1 < len(steps) else None
    diff_html, full_html, n_add, n_del, n_lines = render_diff(prev["code"] if prev else "", s["code"])
    stat = f'<span class="plus">+{n_add}</span> <span class="minus">−{n_del}</span> · 파일 전체 {n_lines}줄'
    has_charts = "[[chart:" in s["output"]
    recorded = ""
    if s["recorded"]:
        recorded = f'<div class="note">{render_md(s["recorded"])}</div>'
    nav = '<nav class="pager">'
    nav += (f'<a class="prev" href="{prev["slug"]}.html"><small>← 이전 {prev["num"]:02d}</small>{html.escape(prev["title"])}</a>'
            if prev else '<a class="prev" href="index.html"><small>← 처음으로</small>소개</a>')
    nav += (f'<a class="next" href="{nxt["slug"]}.html"><small>다음 {nxt["num"]:02d} →</small>{html.escape(nxt["title"])}</a>'
            if nxt else '<a class="next" href="index.html"><small>끝 →</small>처음으로</a>')
    nav += "</nav>"
    progress = s["num"] / len(steps) * 100
    return (
        HEAD.format(title=f'{s["num"]:02d}. {html.escape(s["title"])} · Annotated Transformer 한 단계씩',
                    desc=html.escape(SUMMARY[s["num"]]), extra_head=VEGA if has_charts else "")
        + f"""<div class="progress"><span style="width:{progress:.1f}%"></span></div>
<div class="layout">
{sidebar(steps, s)}
<main data-prev="{prev["slug"] + ".html" if prev else "index.html"}" data-next="{nxt["slug"] + ".html" if nxt else "index.html"}">
<header class="step-head">
  <div class="kicker">STEP {s["num"]:02d} <span>/ {len(steps)}</span></div>
  <h1>{html.escape(s["title"])}</h1>
</header>
<article class="prose">{s["body_html"]}</article>

<section class="block">
  <div class="block-head">
    <h2>코드 변화</h2>
    <div class="stat">{stat}</div>
  </div>
  <p class="hint">이전 단계의 <code>transformer.py</code>와 비교했다. 초록 줄이 새로 생긴 줄, 빨간 줄이 사라진 줄이다.</p>
  <div class="tabs" role="tablist">
    <button role="tab" aria-selected="true" data-tab="diff">바뀐 부분</button>
    <button role="tab" aria-selected="false" data-tab="full">파일 전체</button>
    <a class="src" href="{REPO}/blob/main/steps/{s["slug"]}/transformer.py" target="_blank" rel="noopener">GitHub에서 보기 ↗</a>
  </div>
  <div class="tabpanel" data-panel="diff">{diff_html}</div>
  <div class="tabpanel" data-panel="full" hidden>{full_html}</div>
</section>

<section class="block">
  <div class="block-head"><h2>실행 결과</h2></div>
  {recorded}
  <details class="demo"><summary>실행한 코드 <code>demo.py</code> 보기</summary>{render_plain_code(s["demo"])}</details>
  <div class="outputs">{render_output(s, s["output"])}</div>
</section>
{nav}
<footer class="foot">코드: <a href="https://github.com/harvardnlp/annotated-transformer" target="_blank" rel="noopener">harvardnlp/annotated-transformer</a> (MIT) · 설명·구성: <a href="{REPO}" target="_blank" rel="noopener">annotated-transformer-step-by-step</a></footer>
</main>
</div>
</body>
</html>
"""
    )


def index_page(steps):
    max_lines = max(s["code"].count("\n") for s in steps)
    bars = "".join(
        f'<a class="bar" href="{s["slug"]}.html" style="--h:{s["code"].count(chr(10)) / max_lines * 100:.1f}%" '
        f'title="{s["num"]:02d}. {html.escape(s["title"])} — {s["code"].count(chr(10))}줄">'
        f'<span class="fill"></span><span class="lbl">{s["num"]}</span></a>'
        for s in steps
    )
    parts = []
    for part, rng in PARTS:
        cards = "".join(
            f'<li><a href="{s["slug"]}.html"><span class="num">{s["num"]:02d}</span>'
            f'<span class="t">{html.escape(s["title"])}</span><span class="d">{html.escape(SUMMARY[s["num"]])}</span>'
            f'<span class="lines">{s["code"].count(chr(10))}줄</span></a></li>'
            for s in steps if s["num"] in rng
        )
        parts.append(f'<section class="part-block"><h2>{part}</h2><ol class="cards">{cards}</ol></section>')
    first = steps[0]
    return (
        HEAD.format(title="Annotated Transformer 한 단계씩",
                    desc="The Annotated Transformer의 코드를 18단계에 걸쳐 한 조각씩 쌓아 가며 읽는 튜토리얼", extra_head="")
        + f"""<div class="layout">
{sidebar(steps)}
<main data-next="{first["slug"]}.html">
<header class="hero">
  <div class="kicker">THE ANNOTATED TRANSFORMER · STEP BY STEP</div>
  <h1>빈 파일에서 번역기까지,<br>Transformer 코드가 자라는 과정</h1>
  <p class="lede"><a href="https://nlp.seas.harvard.edu/annotated-transformer/" target="_blank" rel="noopener">The Annotated Transformer</a>는 논문을 코드 한 줄 한 줄로 옮긴 명작이지만, 노트북이 전체 구조(<code>EncoderDecoder</code>)부터 보여 줘서 처음 읽을 때 어디서부터 봐야 할지 막막하다. 이 페이지는 순서를 거꾸로 한다. <b>어텐션 함수 한 줄짜리 빈 파일</b>에서 시작해 부품을 하나씩 더하고, 단계마다 <b>바뀐 코드</b>와 <b>그 코드를 실제로 돌린 결과</b>를 나란히 보여 준다. 18단계를 넘기면 원본과 같은 코드가 완성되고, 그 코드로 독일어를 영어로 번역한다.</p>
  <div class="cta"><a class="btn" href="{first["slug"]}.html">01단계부터 시작 →</a><span class="kbd-hint">페이지에서 <kbd>←</kbd> <kbd>→</kbd>로 넘길 수 있다</span></div>
</header>

<section class="growth">
  <h2>단계별 <code>transformer.py</code> 줄 수</h2>
  <div class="bars">{bars}</div>
  <p class="hint">막대를 누르면 그 단계로 간다.</p>
</section>

<section class="howto prose">
  <h2>읽는 법</h2>
  <ul>
    <li><b>코드를 직접 칠 필요는 없다.</b> 단계마다 이전 단계와의 diff가 있고, 탭을 바꾸면 그 시점의 파일 전체를 볼 수 있다.</li>
    <li><b>실행 결과는 진짜다.</b> 각 단계의 <code>demo.py</code>를 그 단계의 코드로 실행한 출력이다. 17·18단계(GPU 학습)만 RTX 3070에서 실행한 기록을 가져왔다.</li>
    <li><b>코드는 원본 그대로다.</b> 순서만 바꿔 쌓았다. 원본과 다른 곳(torchtext 대체 등)은 해당 단계 설명에 적었다.</li>
    <li>원문(논문 + 주석)을 같이 보고 싶다면 단계마다 붙은 <a href="{ORIGINAL_KO}" target="_blank" rel="noopener">한국어 번역판</a> 링크를 따라가면 된다.</li>
  </ul>
  <figure><img src="images/ModalNet-21.png" alt="Transformer 구조 그림" width="360"><figcaption>논문 Figure 1. 1부(01~10단계)에서 이 그림의 부품을 아래에서부터 하나씩 만든다.</figcaption></figure>
</section>

{"".join(parts)}

<footer class="foot">코드: <a href="https://github.com/harvardnlp/annotated-transformer" target="_blank" rel="noopener">harvardnlp/annotated-transformer</a> (MIT, Sasha Rush 외) · 2026년 라이브러리 버전 수정: <a href="https://github.com/dev-jonghoonpark/annotated-transformer-modern" target="_blank" rel="noopener">annotated-transformer-modern</a> · 이 페이지의 소스: <a href="{REPO}" target="_blank" rel="noopener">annotated-transformer-step-by-step</a></footer>
</main>
</div>
</body>
</html>
"""
    )


def load_steps():
    steps = []
    for d in sorted(p for p in STEPS_DIR.iterdir() if p.is_dir()):
        readme = (d / "README.md").read_text()
        title, body = readme.split("\n", 1)
        rec = d / "RECORDED.md"
        steps.append({
            "slug": d.name,
            "num": int(d.name[:2]),
            "title": title.lstrip("# ").strip(),
            "body_html": render_md(body),
            "code": (d / "transformer.py").read_text(),
            "demo": (d / "demo.py").read_text(),
            "output": (d / "output.txt").read_text(),
            "charts": sorted((d / "charts").glob("*.json")) if (d / "charts").exists() else [],
            "recorded": rec.read_text() if rec.exists() else "",
        })
    return steps


def build():
    steps = load_steps()
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    shutil.copytree(ROOT / "images", OUT / "images")
    for name in ("style.css", "app.js"):
        shutil.copy(ROOT / "site" / name, OUT / name)
    for s in steps:
        for c in s["charts"]:
            dst = OUT / "charts" / s["slug"] / c.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(json.dumps(json.loads(c.read_text()), separators=(",", ":")))
    (OUT / "index.html").write_text(index_page(steps))
    for i, s in enumerate(steps):
        (OUT / f"{s['slug']}.html").write_text(step_page(steps, i))
    (OUT / ".nojekyll").write_text("")
    print(f"docs/: index + {len(steps)} steps")


if __name__ == "__main__":
    build()
