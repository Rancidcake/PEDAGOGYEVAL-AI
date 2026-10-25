"""EXAMINER'S LEDGER: every colour, font rule and HTML helper for the app lives here.

The palette is the single source of truth for CSS, Graphviz, Altair and matplotlib.
Streamlit is imported lazily so the notebook can reuse ``PALETTE`` / ``mpl_style()``
without Streamlit installed.
"""
from __future__ import annotations

import html
from pathlib import Path
from typing import Iterable, Mapping, Sequence

PALETTE = {
    "paper": "#F3EFE6",
    "paper-2": "#EAE4D7",
    "ink": "#1B1A17",
    "ink-soft": "#5E5A52",
    "rule": "#C9C1B0",
    "redpen": "#C8372D",
    "ochre": "#B7861B",
    "moss": "#3F6B34",
    "highlight": "#F2D64B",
}
FONT_STACK = "'JetBrains Mono', 'IBM Plex Mono', monospace"
GRAPH_FONT = "JetBrains Mono"
DOT_FONT = "Courier"  # graphviz layout metrics; CSS re-renders svg text in JetBrains Mono
FONT_DIR = Path(__file__).resolve().parent / "static" / "fonts"
STATUS = {
    "covered": ("✓ COVERED", "moss"),
    "partial": ("~ PARTIAL", "ochre"),
    "missing": ("✗ MISSING", "redpen"),
}
MODEL_COLORS = {"A_tfidf": PALETTE["ink-soft"], "B_embedding": PALETTE["ink"], "C_concept": PALETTE["redpen"]}

_VARS = "\n".join(f"  --{k}: {v};" for k, v in PALETTE.items())
_FACES = "\n".join(
    f"""@font-face {{ font-family: 'JetBrains Mono'; font-style: {style}; font-weight: {weight}; font-display: swap;
  src: local('JetBrains Mono {local}'), url('/app/static/fonts/JetBrainsMono-{file}.woff2') format('woff2'); }}"""
    for file, weight, style, local in (
        ("Regular", 400, "normal", "Regular"),
        ("Medium", 500, "normal", "Medium"),
        ("Bold", 700, "normal", "Bold"),
        ("Italic", 400, "italic", "Italic"),
    )
)

CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:ital,wght@0,400;0,500;0,700;1,400&family=IBM+Plex+Mono:wght@400;500;700&display=swap');
{_FACES}
:root {{
{_VARS}
  --mono: {FONT_STACK};
}}

/* ---------- canvas: warm paper, faint ruling, red margin ---------- */
html, body, .stApp {{ background: var(--paper) !important; color: var(--ink); }}
[data-testid="stMain"] {{
  background-color: var(--paper);
  background-image: repeating-linear-gradient(to bottom, transparent 0 31px, rgba(201,193,176,0.35) 31px 32px);
}}
.block-container {{
  max-width: 1200px !important; padding-top: 2.2rem !important; padding-bottom: 3rem !important;
  border-left: 1px solid rgba(200,55,45,0.28); padding-left: 2rem !important;
}}
@media (max-width: 760px) {{ .block-container {{ border-left: none; padding-left: 1rem !important; padding-right: 1rem !important; }} }}

/* ---------- typography ---------- */
.stApp, .stApp p, .stApp li, .stApp label, .stApp span, .stApp div, .stApp input, .stApp textarea,
.stApp button, .stApp select, .stApp code, .stApp pre, .stApp table, .stApp th, .stApp td,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {{ font-family: var(--mono) !important; }}
.stApp span[data-testid="stIconMaterial"], .stApp .material-icons, .stApp .material-symbols-rounded,
.stApp [data-testid="stFileUploaderDropzone"] span[data-testid="stIconMaterial"] {{ font-family: 'Material Symbols Rounded' !important; text-transform: none !important; letter-spacing: normal !important; }}
.stApp {{ font-size: 15px; line-height: 1.6; }}
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {{ text-transform: uppercase; letter-spacing: 0.08em; color: var(--ink); font-weight: 700; }}
.stApp h1 {{ font-size: 1.55rem; }} .stApp h2 {{ font-size: 1.2rem; }} .stApp h3 {{ font-size: 1.0rem; }}
.stApp a {{ color: var(--ink); text-decoration: underline; text-decoration-color: var(--redpen); }}
.stApp code {{ background: var(--paper-2); color: var(--ink); border: 1px solid var(--rule); padding: 0 .3em; }}

/* ---------- no rounded corners, no shadows anywhere ---------- */
.stApp *, .stApp *::before, .stApp *::after {{ border-radius: 0 !important; box-shadow: none !important; }}
.stApp :focus-visible {{ outline: 2px solid var(--redpen) !important; outline-offset: 1px; }}

/* ---------- chrome: hide streamlit header/footer/menu/deploy ---------- */
/* keep stToolbar itself: it hosts the sidebar re-open button. Hide only menu / deploy / status. */
#MainMenu, footer, [data-testid="stDecoration"], [data-testid="stStatusWidget"], [data-testid="stToolbarActions"],
[data-testid="stAppDeployButton"], .stDeployButton, [data-testid="stMainMenu"] {{ display: none !important; visibility: hidden !important; }}
[data-testid="stHeader"] {{ background: transparent !important; height: 3rem; }}
[data-testid="stToolbar"] {{ background: transparent !important; visibility: visible !important; display: flex !important; }}

/* ---------- sidebar open / close controls: always visible ---------- */
[data-testid="stSidebarCollapseButton"], [data-testid="stExpandSidebarButton"] {{
  visibility: visible !important; opacity: 1 !important; display: flex !important;
}}
[data-testid="stExpandSidebarButton"] {{
  border: 1px solid var(--ink) !important; background: var(--paper-2) !important; color: var(--ink) !important;
  padding: .25rem .7rem !important; height: auto !important; width: auto !important; gap: .4rem; cursor: pointer;
  z-index: 1000;
}}
[data-testid="stExpandSidebarButton"] span[data-testid="stIconMaterial"] {{ display: none !important; }}
[data-testid="stExpandSidebarButton"]::after {{
  content: "☰ MENU"; font-family: var(--mono); font-weight: 700; letter-spacing: .1em; font-size: .8rem; color: var(--ink);
}}
[data-testid="stExpandSidebarButton"]:hover {{ background: var(--ink) !important; }}
[data-testid="stExpandSidebarButton"]:hover::after {{ color: var(--paper); }}
[data-testid="stSidebarCollapseButton"] button {{ border: 1px solid var(--ink) !important; color: var(--ink) !important; }}
[data-testid="stSidebarCollapseButton"] button:hover {{ background: var(--ink) !important; }}
[data-testid="stSidebarCollapseButton"] button:hover * {{ color: var(--paper) !important; }}

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {{ background: var(--paper-2) !important; border-right: 1px solid var(--ink); }}
[data-testid="stSidebar"] * {{ color: var(--ink); }}
[data-testid="stSidebarNav"] a span {{ text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.85rem; }}
[data-testid="stSidebarNav"] a[aria-current="page"] {{ background: var(--paper) !important; border-left: 3px solid var(--redpen); }}

/* ---------- buttons: ink outline, transparent, uppercase ---------- */
.stButton > button, .stDownloadButton > button, [data-testid="stBaseButton-secondary"], [data-testid="stFormSubmitButton"] > button {{
  background: transparent !important; color: var(--ink) !important; border: 1px solid var(--ink) !important;
  text-transform: uppercase; letter-spacing: 0.08em; font-weight: 500; font-size: 0.82rem; min-height: 2.4rem;
  transition: background 80ms linear, color 80ms linear;
}}
.stButton > button:hover, .stDownloadButton > button:hover, [data-testid="stBaseButton-secondary"]:hover {{
  background: var(--ink) !important; color: var(--paper) !important;
}}
.stButton > button:hover *, .stDownloadButton > button:hover * {{ color: var(--paper) !important; }}
.stApp .stButton > button[data-testid="stBaseButton-primary"], .stApp button[kind="primary"] {{
  background: var(--redpen) !important; color: var(--paper) !important; border: 1px solid var(--redpen) !important;
  font-weight: 700; letter-spacing: 0.14em;
}}
.stApp button[data-testid="stBaseButton-primary"] * {{ color: var(--paper) !important; }}
.stApp .stButton > button[data-testid="stBaseButton-primary"]:hover {{ background: var(--ink) !important; border-color: var(--ink) !important; }}
.stApp .stButton > button:disabled {{ opacity: .45; background: transparent !important; color: var(--ink-soft) !important; border-color: var(--ink-soft) !important; }}
.stApp .stButton > button:disabled * {{ color: var(--ink-soft) !important; }}

/* ---------- inputs ---------- */
[data-baseweb="input"], [data-baseweb="textarea"], [data-baseweb="select"] > div, [data-baseweb="base-input"],
.stNumberInput div[data-baseweb="input"] {{
  background: var(--paper) !important; border: 1px solid var(--ink) !important;
}}
[data-baseweb="textarea"] textarea, [data-baseweb="input"] input {{ background: var(--paper) !important; color: var(--ink) !important; font-size: 15px; }}
[data-testid="stWidgetLabel"] p {{ text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.78rem; color: var(--ink-soft); }}
[data-testid="stFileUploaderDropzone"] {{ background: var(--paper) !important; border: 1px dashed var(--ink) !important; }}
[data-baseweb="tag"] {{ background: var(--paper-2) !important; border: 1px solid var(--ink); }}

/* ---------- tabs ---------- */
[data-baseweb="tab-list"] {{ gap: 0; border-bottom: 1px solid var(--ink); }}
[data-baseweb="tab"] {{ text-transform: uppercase; letter-spacing: 0.1em; padding: 0.4rem 1.1rem !important; border: 1px solid transparent; }}
[data-baseweb="tab"][aria-selected="true"] {{ border: 1px solid var(--ink); border-bottom: none; background: var(--paper-2); }}
[data-baseweb="tab-highlight"] {{ background: var(--redpen) !important; }}
[data-baseweb="tab-border"] {{ display: none; }}

/* ---------- alerts / expanders / dataframes ---------- */
[data-testid="stAlert"], [data-testid="stAlertContainer"] {{ background: var(--paper-2) !important; border: 1px solid var(--ink) !important; color: var(--ink) !important; }}
[data-testid="stExpander"] details {{ border: 1px solid var(--ink) !important; background: var(--paper); }}
[data-testid="stExpander"] summary p {{ text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.82rem; }}
[data-testid="stDataFrame"], [data-testid="stDataEditor"] {{ border: 1px solid var(--ink); }}
[data-testid="stGraphVizChart"] svg {{ max-width: 100%; height: auto; }}
/* layout uses Courier metrics (same 0.6em advance as JetBrains Mono); render in JetBrains Mono */
[data-testid="stGraphVizChart"] svg text {{ font-family: var(--mono) !important; }}
hr {{ border-color: var(--ink) !important; }}

/* ================= ledger components ================= */
.pe-section {{ display: flex; align-items: baseline; gap: .8rem; margin: 1.1rem 0 .55rem; border-bottom: 1px solid var(--ink); padding-bottom: .15rem; }}
.pe-section .num {{ color: var(--redpen); font-weight: 700; letter-spacing: .08em; }}
.pe-section .lbl {{ font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }}
.pe-section .hint {{ margin-left: auto; color: var(--ink-soft); font-size: .78rem; text-transform: uppercase; letter-spacing: .06em; }}

.pe-field {{ border: 1px solid var(--ink); background: var(--paper); padding: .55rem .75rem; margin-bottom: .6rem; }}
.pe-field .k {{ display: block; font-size: .72rem; letter-spacing: .1em; text-transform: uppercase; color: var(--ink-soft); }}
.pe-field .v {{ white-space: pre-wrap; }}

.pe-title {{ font-size: 15px; line-height: 1.35; margin: 0 0 .6rem; white-space: pre; overflow-x: auto; font-weight: 700; }}
.pe-sub {{ color: var(--ink-soft); letter-spacing: .04em; }}

.pe-score {{ display: flex; align-items: flex-end; gap: 1.2rem; flex-wrap: wrap; margin: .2rem 0 .6rem; }}
.pe-score .big {{ font-size: 96px; line-height: 1; font-weight: 700; border-bottom: 3px double var(--redpen); padding-bottom: .05em; letter-spacing: -0.02em; }}
.pe-score .big .den {{ color: var(--ink-soft); font-weight: 500; }}
.pe-score .meta {{ font-size: .8rem; color: var(--ink-soft); text-transform: uppercase; letter-spacing: .08em; padding-bottom: .5rem; }}
.pe-bar {{ height: 6px; border: 1px solid var(--ink); background: var(--paper); margin: .3rem 0 1rem; }}
.pe-bar > div {{ height: 100%; background: var(--ink); }}

.pe-response {{ border: 1px solid var(--ink); background: var(--paper); padding: .8rem 1rem; font-size: 1.02rem; line-height: 1.9; }}
.pe-response mark {{ background: rgba(242,214,75,0.45); color: var(--ink); padding: .05em .1em; }}
.pe-response del {{ color: var(--redpen); text-decoration: line-through; text-decoration-thickness: 2px; text-decoration-color: var(--redpen); }}
.pe-response sup {{ font-size: .62rem; color: var(--ink-soft); margin-left: 1px; }}
.pe-legend {{ font-size: .74rem; color: var(--ink-soft); margin-top: .3rem; letter-spacing: .04em; }}

table.pe-ledger {{ width: 100%; border-collapse: collapse; border: 1px solid var(--ink); background: var(--paper); font-size: .86rem; }}
table.pe-ledger th {{ background: var(--paper-2); text-transform: uppercase; letter-spacing: .08em; font-size: .72rem; text-align: left; padding: .45rem .6rem; border-bottom: 1px solid var(--ink); }}
table.pe-ledger td {{ padding: .42rem .6rem; border-bottom: 1px solid var(--rule); vertical-align: top; }}
table.pe-ledger td.num, table.pe-ledger th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
table.pe-ledger tr.best td {{ background: rgba(200,55,45,0.07); font-weight: 700; }}
table.pe-ledger tr.best td:first-child {{ border-left: 3px solid var(--redpen); }}
table.pe-ledger .ev {{ color: var(--ink-soft); font-style: italic; }}
.pe-scroll {{ overflow-x: auto; }}

.pe-tag {{ font-weight: 700; white-space: nowrap; letter-spacing: .03em; }}
.pe-tag.moss {{ color: var(--moss); }} .pe-tag.ochre {{ color: var(--ochre); }} .pe-tag.redpen {{ color: var(--redpen); }}

.pe-note {{ border-left: 3px solid var(--redpen); padding: .5rem .9rem; font-style: italic; background: transparent; color: var(--ink); margin: .6rem 0; }}
.pe-note .k {{ font-style: normal; font-size: .72rem; letter-spacing: .1em; color: var(--redpen); display: block; }}
.pe-review {{ border: 1px solid var(--ochre); color: var(--ochre); padding: .55rem .8rem; font-weight: 700; margin: .6rem 0; letter-spacing: .03em; }}
.pe-review .why {{ font-weight: 400; color: var(--ink); display: block; font-size: .85rem; margin-top: .2rem; }}
.pe-ok {{ border: 1px solid var(--moss); color: var(--moss); padding: .45rem .8rem; margin: .6rem 0; font-weight: 700; }}

.pe-strip {{ display: grid; grid-template-columns: repeat(3, 1fr); border: 1px solid var(--ink); margin: .6rem 0; }}
.pe-strip > div {{ padding: .5rem .7rem; border-right: 1px solid var(--ink); }}
.pe-strip > div:last-child {{ border-right: none; }}
.pe-strip .m {{ font-size: .72rem; text-transform: uppercase; letter-spacing: .08em; color: var(--ink-soft); }}
.pe-strip .s {{ font-size: 1.6rem; font-weight: 700; }}
.pe-strip .sel {{ background: var(--paper-2); }}
.pe-strip .sel .s {{ border-bottom: 3px double var(--redpen); }}

.pe-notice {{ border: 1px dashed var(--ink); padding: .9rem 1rem; background: var(--paper); white-space: pre-wrap; }}
.pe-notice.err {{ border: 1px solid var(--redpen); color: var(--redpen); }}
.pe-valid {{ font-size: .88rem; margin: .1rem 0; }}
.pe-valid.ok {{ color: var(--moss); }} .pe-valid.bad {{ color: var(--redpen); }}
.pe-stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); border: 1px solid var(--ink); margin: .5rem 0 1rem; }}
.pe-stats > div {{ padding: .55rem .8rem; border-right: 1px solid var(--rule); }}
.pe-stats .k {{ font-size: .7rem; text-transform: uppercase; letter-spacing: .1em; color: var(--ink-soft); }}
.pe-stats .v {{ font-size: 1.5rem; font-weight: 700; }}
.pe-foot {{ color: var(--ink-soft); font-size: .75rem; margin-top: 2rem; border-top: 1px solid var(--rule); padding-top: .4rem; }}
"""


# ------------------------------------------------------------------ streamlit plumbing
def _st():
    import streamlit as st

    return st


def apply(page_title: str = "PedagogyEval AI") -> None:
    """Call first on every page: page config + ledger CSS + Altair theme."""
    st = _st()
    st.set_page_config(page_title=f"{page_title} · PedagogyEval AI", page_icon="§", layout="wide", initial_sidebar_state="expanded")
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
    register_altair_theme()


def render(markup: str) -> None:
    _st().markdown(markup, unsafe_allow_html=True)


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def fmt_mark(x: float) -> str:
    x = float(x)
    return f"{x:.0f}" if abs(x - round(x)) < 1e-9 else f"{x:.1f}" if abs(x * 10 - round(x * 10)) < 1e-9 else f"{x:.2f}"


# ------------------------------------------------------------------ HTML helpers (return strings)
def section_label(num: str, label: str, hint: str = "") -> str:
    h = f'<span class="hint">{esc(hint)}</span>' if hint else ""
    return f'<div class="pe-section"><span class="num">§{esc(num)}</span><span class="lbl">{esc(label)}</span>{h}</div>'


def field(key: str, value) -> str:
    return f'<div class="pe-field"><span class="k">{esc(key)}</span><span class="v">{esc(value)}</span></div>'


def ascii_title(title: str, width: int = 34) -> str:
    inner = width - 2
    top = "┌" + "─" * inner + "┐"
    mid = "│ " + title.ljust(inner - 1) + "│"
    bot = "└" + "─" * inner + "┘"
    return f'<div class="pe-title">{esc(top)}\n{esc(mid)}\n{esc(bot)}</div>'


def status_tag(status: str) -> str:
    text, color = STATUS.get(status, (status.upper(), "ink"))
    return f'<span class="pe-tag {color}">[{text}]</span>'


def score_block(score: float, max_marks: float, caption: str = "") -> str:
    pct = 0 if not max_marks else max(0.0, min(1.0, float(score) / float(max_marks)))
    return (
        f'<div class="pe-score"><div class="big">{fmt_mark(score)}<span class="den">/{fmt_mark(max_marks)}</span></div>'
        f'<div class="meta">{esc(caption)}<br>{pct:.0%} of marks</div></div>'
        f'<div class="pe-bar"><div style="width:{pct * 100:.1f}%"></div></div>'
    )


def highlighted_response(text: str, rows: Sequence[Mapping]) -> str:
    """Response with evidence spans marked (yellow) and negated spans struck through (red).

    Each concept id is written once, as a superscript at the end of its span.
    """
    if not text:
        return '<div class="pe-response"><span style="color:var(--ink-soft)">(empty answer)</span></div>'
    kind = [None] * len(text)
    ends: dict[int, list[str]] = {}
    for r in rows:
        s, e = r.get("start"), r.get("end")
        k = "neg" if r.get("negated") else "ev" if r.get("coverage", 0) > 0 else None
        if s is None or e is None or k is None:
            continue
        s, e = max(0, s), min(len(text), e)
        for i in range(s, e):
            if kind[i] != "neg":
                kind[i] = k
        ends.setdefault(e, []).append(("−" if k == "neg" else "") + str(r.get("concept_id")))
    out, i = [], 0
    while i < len(text):
        j = i + 1
        while j < len(text) and kind[j] == kind[i] and j not in ends:
            j += 1
        seg = esc(text[i:j])
        out.append(f"<mark>{seg}</mark>" if kind[i] == "ev" else f"<del>{seg}</del>" if kind[i] == "neg" else seg)
        if j in ends:
            out.append(f"<sup>{esc(','.join(ends[j]))}</sup>")
        i = j
    legend = (
        '<div class="pe-legend"><mark style="background:rgba(242,214,75,.45)">evidence</mark> &nbsp; '
        '<del style="color:var(--redpen)">negated</del> &nbsp; superscript = concept id (− = negated)</div>'
    )
    return f'<div class="pe-response">{"".join(out)}</div>{legend}'


def concept_table(rows: Sequence[Mapping]) -> str:
    body = []
    for r in rows:
        sem = r.get("semantic")
        method = esc(r.get("method", ""))
        if sem is not None:
            method += f' <span style="color:var(--ink-soft)">cos={sem:.2f}</span>'
        ev = esc(r.get("evidence") or "—")
        body.append(
            f"<tr><td><b>{esc(r.get('concept_id'))}</b> {esc(r.get('name'))}</td>"
            f"<td class='num'>{fmt_mark(r.get('credit', 0))}/{fmt_mark(r.get('weight', 0))}</td>"
            f"<td>{status_tag(r.get('status', ''))}</td><td>{method}</td><td class='ev'>{ev}</td></tr>"
        )
    return (
        '<div class="pe-scroll"><table class="pe-ledger"><thead><tr><th>Concept</th><th class="num">Weight</th>'
        f'<th>Status</th><th>Method</th><th>Evidence</th></tr></thead><tbody>{"".join(body)}</tbody></table></div>'
    )


def ledger_table(df, highlight_index=None, num_cols: Iterable[str] = (), fmt: str = "{:.3f}") -> str:
    """Render a DataFrame as a ledger. ``highlight_index`` marks the best row in red pen."""
    num_cols = set(num_cols)
    head = "".join(f'<th class="{"num" if c in num_cols else ""}">{esc(c)}</th>' for c in df.columns)
    rows = []
    for idx, rec in df.iterrows():
        cells = []
        for c in df.columns:
            v = rec[c]
            if c in num_cols and v is not None and v == v:
                try:
                    cells.append(f'<td class="num">{fmt.format(float(v))}</td>')
                    continue
                except (TypeError, ValueError):
                    pass
            cells.append(f"<td>{esc('—' if v is None or v != v else v)}</td>")
        cls = ' class="best"' if highlight_index is not None and idx == highlight_index else ""
        rows.append(f"<tr{cls}>{''.join(cells)}</tr>")
    return f'<div class="pe-scroll"><table class="pe-ledger"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def feedback_note(text: str) -> str:
    return f'<div class="pe-note"><span class="k">EXAMINER\'S NOTE</span>{esc(text)}</div>'


def review_box(flag: bool, reasons: Sequence[str]) -> str:
    if not flag:
        return '<div class="pe-ok">[✓] NO REVIEW FLAG: evidence is unambiguous</div>'
    why = "".join(f"· {esc(r)}<br>" for r in reasons)
    return f'<div class="pe-review">⚠ NEEDS HUMAN REVIEW: {esc(reasons[0]) if reasons else ""}<span class="why">{why}</span></div>'


def model_strip(marks: Mapping[str, tuple[str, float | None, float]], selected: str) -> str:
    cells = []
    for key, (label, score, mx) in marks.items():
        s = "n/a" if score is None else f"{fmt_mark(score)}/{fmt_mark(mx)}"
        cls = ' class="sel"' if key == selected else ""
        cells.append(f'<div{cls}><div class="m">{esc(label)}</div><div class="s">{esc(s)}</div></div>')
    return f'<div class="pe-strip">{"".join(cells)}</div>'


def stats_row(items: Sequence[tuple[str, str]]) -> str:
    return '<div class="pe-stats">' + "".join(f'<div><div class="k">{esc(k)}</div><div class="v">{esc(v)}</div></div>' for k, v in items) + "</div>"


def notice(text: str, error: bool = False) -> str:
    return f'<div class="pe-notice{" err" if error else ""}">{esc(text)}</div>'


def validation_lines(problems: Sequence[str], ok_lines: Sequence[str]) -> str:
    out = [f'<div class="pe-valid ok">✓ {esc(x)}</div>' for x in ok_lines]
    out += [f'<div class="pe-valid bad">✗ {esc(x)}</div>' for x in problems]
    return "".join(out)


# ------------------------------------------------------------------ graphviz (DOT strings)
def _tint(hex_color: str, amount: float) -> str:
    """Mix ``hex_color`` with paper; amount=0 -> paper, 1 -> colour (flat tint, not a gradient)."""
    p = PALETTE["paper"]
    a = [int(p[i : i + 2], 16) for i in (1, 3, 5)]
    b = [int(hex_color[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * amount):02X}" for x, y in zip(a, b))


def _dq(s: str) -> str:
    return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def concept_graph_dot(concepts: Sequence[Mapping], edges: Sequence[Mapping], statuses: Mapping[str, str] | None = None, rankdir: str = "TB") -> str:
    P = PALETTE
    lines = [
        "digraph G {",
        f'  graph [rankdir={rankdir}, bgcolor="transparent", fontname="{DOT_FONT}", pad=0.2, nodesep=0.35, ranksep=0.45];',
        f'  node [shape=box, style="filled", fontname="{DOT_FONT}", fontsize=11, color="{P["ink"]}", fillcolor="{P["paper-2"]}", fontcolor="{P["ink"]}", penwidth=1.2, margin="0.18,0.08"];',
        f'  edge [color="{P["ink"]}", fontname="{DOT_FONT}", fontsize=9, fontcolor="{P["ink-soft"]}", arrowsize=0.7];',
    ]
    for c in concepts:
        cid = str(c.get("id", ""))
        st = (statuses or {}).get(cid)
        name = str(c.get("name", ""))
        w = c.get("weight", "")
        lab = f"{cid} · {name}\nw={fmt_mark(w) if w not in ('', None) else '?'}"
        if st:
            col = P[STATUS[st][1]]
            lab += f"\n[{STATUS[st][0]}]"
            lines.append(f'  {_dq(cid)} [label={_dq(lab)}, color="{col}", fillcolor="{_tint(col, 0.16)}", penwidth=2.2];')
        else:
            lines.append(f"  {_dq(cid)} [label={_dq(lab)}];")
    for e in edges:
        lines.append(f'  {_dq(e.get("source"))} -> {_dq(e.get("target"))} [label={_dq(" " + str(e.get("relation", "")) + " ")}];')
    lines.append("}")
    return "\n".join(lines)


def pipeline_dot() -> str:
    P = PALETTE
    steps = [
        ("in", "INPUT\\nquestion + rubric\\n+ response"),
        ("pre", "PREPROCESS\\nNFKC · whitespace\\nlowercase copy"),
        ("cg", "CONCEPT GRAPH\\nweighted concepts\\n+ relations"),
        ("lex", "LEXICAL\\nsynonym regex\\n+ fuzzy"),
        ("sem", "SEMANTIC\\nMiniLM cosine\\nt_low / t_high"),
        ("agg", "WEIGHTED\\nAGGREGATION\\nmin(M, Σ w·p)"),
        ("exp", "EXPLANATION\\nevidence · feedback\\nreview flag"),
        ("ev", "EVALUATION\\nMAE · RMSE · F1\\nconfusion"),
    ]
    lines = [
        "digraph P {",
        f'  graph [rankdir=LR, bgcolor="transparent", nodesep=0.25, ranksep=0.32, fontname="{DOT_FONT}"];',
        f'  node [shape=box, style="filled", fillcolor="{P["paper"]}", color="{P["ink"]}", fontname="{DOT_FONT}", fontsize=10, fontcolor="{P["ink"]}", penwidth=1.2, margin="0.12,0.07"];',
        f'  edge [color="{P["ink"]}", arrowsize=0.6, penwidth=1.0];',
    ]
    for k, lab in steps:
        extra = f', color="{P["redpen"]}", penwidth=2.0' if k == "agg" else ""
        lines.append(f'  {k} [label="{lab}"{extra}];')
    lines += ["  in -> pre -> cg;", "  cg -> lex; cg -> sem;", "  lex -> agg; sem -> agg;", "  agg -> exp -> ev;", '  { rank=same; lex; sem; }', "}"]
    return "\n".join(lines)


# ------------------------------------------------------------------ charts
SEQ_RANGE = [PALETTE["paper"], PALETTE["ink"]]


def register_altair_theme() -> None:
    import altair as alt

    P = PALETTE

    def ledger():
        return {
            "config": {
                "background": "transparent",
                "font": FONT_STACK.replace("'", ""),
                "view": {"stroke": P["ink"], "strokeWidth": 1},
                "axis": {
                    "domainColor": P["ink"], "tickColor": P["ink"], "gridColor": P["rule"], "gridOpacity": 0.7,
                    "labelColor": P["ink"], "titleColor": P["ink"], "labelFont": GRAPH_FONT, "titleFont": GRAPH_FONT,
                    "labelFontSize": 11, "titleFontSize": 11, "titleFontWeight": 700,
                },
                "legend": {"labelFont": GRAPH_FONT, "titleFont": GRAPH_FONT, "labelColor": P["ink"], "titleColor": P["ink"], "symbolType": "square"},
                "title": {"font": GRAPH_FONT, "color": P["ink"], "fontSize": 12, "anchor": "start", "fontWeight": 700},
                "header": {"labelFont": GRAPH_FONT, "titleFont": GRAPH_FONT, "labelColor": P["ink"], "titleColor": P["ink"]},
                "text": {"font": GRAPH_FONT, "color": P["ink"]},
                "bar": {"stroke": P["ink"], "strokeWidth": 1, "cornerRadius": 0},
                "rect": {"stroke": P["paper"], "strokeWidth": 1},
                "range": {
                    "category": [P["ink-soft"], P["ink"], P["redpen"], P["ochre"], P["moss"], P["rule"]],
                    "heatmap": SEQ_RANGE,
                    "ramp": SEQ_RANGE,
                },
            }
        }

    try:  # altair >= 5.5
        alt.theme.register("ledger", enable=True)(ledger)
    except AttributeError:  # pragma: no cover
        alt.themes.register("ledger", ledger)
        alt.themes.enable("ledger")


def mpl_style() -> None:
    """Apply the same palette + mono font to matplotlib (used by the notebook)."""
    import matplotlib as mpl
    from cycler import cycler
    from matplotlib import font_manager

    family = ["DejaVu Sans Mono"]
    if FONT_DIR.exists():
        for f in FONT_DIR.glob("JetBrainsMono-*.ttf"):
            if "-" in f.stem and f.stem.split("-")[1][0].isalpha():
                font_manager.fontManager.addfont(str(f))
        family = ["JetBrains Mono", "IBM Plex Mono", "DejaVu Sans Mono"]
    P = PALETTE
    mpl.rcParams.update({
        "font.family": "monospace",
        "font.monospace": family,
        "font.size": 10,
        "figure.facecolor": P["paper"],
        "savefig.facecolor": P["paper"],
        "axes.facecolor": P["paper"],
        "axes.edgecolor": P["ink"],
        "axes.labelcolor": P["ink"],
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": P["rule"],
        "grid.linewidth": 0.6,
        "xtick.color": P["ink"],
        "ytick.color": P["ink"],
        "text.color": P["ink"],
        "legend.frameon": True,
        "legend.edgecolor": P["ink"],
        "legend.facecolor": P["paper"],
        "legend.fancybox": False,
        "patch.edgecolor": P["ink"],
        "axes.prop_cycle": cycler(color=[P["ink-soft"], P["ink"], P["redpen"], P["ochre"], P["moss"]]),
        "image.cmap": "ledger",
    })
    from matplotlib.colors import LinearSegmentedColormap

    if "ledger" not in mpl.colormaps:
        mpl.colormaps.register(LinearSegmentedColormap.from_list("ledger", SEQ_RANGE))
