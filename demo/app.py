"""
ReviewRenew — Streamlit demo (replay-only brand dashboard).

Turn customer reviews into hero shots. Zero API calls, zero env vars —
every pixel and every prompt is read from disk.

    streamlit run demo/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import streamlit as st

# Make `src` importable when launched as `streamlit run demo/app.py` from the repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from src.models import ImagePromptBundle, ProductCatalog  # type: ignore
except Exception:  # pragma: no cover — pydantic optional at runtime
    ImagePromptBundle = None  # type: ignore
    ProductCatalog = None  # type: ignore


# =============================================================================
# Constants
# =============================================================================

DATA_DIR = REPO_ROOT / "data"
PRODUCTS_PATH = DATA_DIR / "products.json"
OG_IMAGES_DIR = DATA_DIR / "og-product-images"
OUTPUTS_ROOT = REPO_ROOT / "outputs"

OG_IMAGE_MAP = {
    "B0C4BFWLNC": "pet-bed",
    "B0BG9Q18ZZ": "cerave",
    "B0BXSQN4HV": "cap-clean",
}

IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp")

PROVIDER_LABEL = {
    "openai": "OpenAI DALL·E 3",
    "gemini": "Gemini",
}

ACCENT = "#00b4d8"


# =============================================================================
# Data loaders (cached)
# =============================================================================


@st.cache_data(show_spinner=False)
def load_products() -> list[dict[str, Any]]:
    if not PRODUCTS_PATH.exists():
        return []
    raw = json.loads(PRODUCTS_PATH.read_text(encoding="utf-8"))
    # Best-effort validation; fall through to raw dict if drift.
    if ProductCatalog is not None:
        try:
            ProductCatalog.model_validate(raw)
        except Exception:  # noqa: BLE001
            pass
    return raw.get("products", []) if isinstance(raw, dict) else []


def pick_run_root() -> Path | None:
    """Resolve the active outputs root.

    The pipeline writes to ``outputs/run_<timestamp>/``; users sometimes copy a
    run up to ``outputs/`` directly. Prefer ``outputs/`` when it has bundles,
    else the most-recent ``outputs/run_*/`` that contains any
    ``*_image_prompt_bundle.json``.
    """
    if not OUTPUTS_ROOT.exists():
        return None

    def _has_artifacts(p: Path) -> bool:
        if not p.is_dir():
            return False
        return (p / "run_log.json").exists() or any(p.glob("*_image_prompt_bundle.json"))

    if _has_artifacts(OUTPUTS_ROOT):
        return OUTPUTS_ROOT

    candidates = [c for c in OUTPUTS_ROOT.iterdir() if _has_artifacts(c)]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def load_bundle(asin: str) -> dict[str, Any] | None:
    rd = pick_run_root()
    if rd is None:
        return None
    path = rd / f"{asin}_image_prompt_bundle.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if ImagePromptBundle is not None and isinstance(data, dict):
        try:
            ImagePromptBundle.model_validate(data)
        except Exception:  # noqa: BLE001
            pass
    return data if isinstance(data, dict) else None


def load_hero_evals() -> dict[tuple[str, str], dict[str, Any]]:
    rd = pick_run_root()
    if rd is None:
        return {}
    path = rd / "hero_shot_evaluation.json"
    if not path.exists():
        return {}
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for r in rows if isinstance(rows, list) else []:
        asin = str(r.get("asin", "")).strip()
        model = str(r.get("model", "")).strip().lower()
        if asin and model:
            out[(asin, model)] = r
    return out


def list_images(dir_path: Path | None) -> list[Path]:
    if dir_path is None or not dir_path.exists() or not dir_path.is_dir():
        return []
    return sorted(p for p in dir_path.iterdir() if p.suffix.lower() in IMG_EXTS)


def og_image_for(asin: str) -> Path | None:
    stem = OG_IMAGE_MAP.get(asin)
    if not stem:
        return None
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        p = OG_IMAGES_DIR / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def review_stats(product: dict[str, Any]) -> tuple[int, float | None]:
    reviews = product.get("reviews") or []
    rated = [r.get("rating") for r in reviews if isinstance(r.get("rating"), (int, float))]
    avg = sum(rated) / len(rated) if rated else None
    return len(reviews), avg


def collect_generated(asin: str) -> list[dict[str, Any]]:
    """Return every generated image for a product as dicts.

    Shape: ``{"path": Path, "provider": "openai"|"gemini", "is_hero": bool}``.
    Images are sorted so hero shots float to the front within each provider.
    """
    rd = pick_run_root()
    if rd is None:
        return []
    out: list[dict[str, Any]] = []
    for provider in ("openai", "gemini"):
        for p in list_images(rd / asin / provider):
            out.append(
                {
                    "path": p,
                    "provider": provider,
                    "is_hero": "hero" in p.name.lower() or "_1_" in p.name,
                }
            )
    out.sort(key=lambda d: (d["provider"], not d["is_hero"], d["path"].name))
    return out


def attach_scores(items: list[dict[str, Any]], asin: str) -> None:
    evals = load_hero_evals()
    for item in items:
        if not item["is_hero"]:
            item["score"] = None
            item["finding"] = None
            continue
        rec = evals.get((asin, item["provider"]))
        if rec is None:
            item["score"] = None
            item["finding"] = None
        else:
            item["score"] = rec.get("overall_score")
            item["finding"] = rec.get("key_finding")


def pick_primary(items: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not items:
        return None
    scored = [i for i in items if isinstance(i.get("score"), (int, float))]
    if scored:
        return max(scored, key=lambda i: i["score"])
    heroes = [i for i in items if i.get("is_hero")]
    return heroes[0] if heroes else items[0]


# =============================================================================
# Page shell + CSS
# =============================================================================

st.set_page_config(
    page_title="ReviewRenew",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = f"""
<style>
/* ----- Hide Streamlit chrome ----- */
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
header {{ visibility: hidden; }}
[data-testid="stToolbar"] {{ display: none; }}
[data-testid="stDecoration"] {{ display: none; }}

/* ----- Base dark theme ----- */
html, body, [class*="st-"], .stApp {{
  font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, sans-serif;
  color: #e8e8e8;
}}
.stApp {{ background-color: #0b0b0c; }}
.block-container {{ padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1400px; }}

h1, h2, h3, h4, h5 {{ color: #ffffff; letter-spacing: -0.015em; }}
p, li, label, span {{ color: #d0d0d2; }}
hr {{ border-color: #1c1c1e; }}

/* ----- Sidebar ----- */
[data-testid="stSidebar"] {{
  background-color: #0f0f11;
  border-right: 1px solid #1a1a1c;
}}
[data-testid="stSidebar"] > div:first-child {{ padding: 2rem 1.2rem 1.5rem 1.2rem; }}

.rr-logo {{
  font-size: 1.9rem; font-weight: 800; color: #ffffff;
  letter-spacing: -0.035em; margin: 0; line-height: 1;
}}
.rr-logo .rr-mark {{ color: {ACCENT}; }}
.rr-tagline {{
  color: #7a7a80; font-size: 0.85rem; margin: 6px 0 1.8rem 0;
  letter-spacing: 0.01em; line-height: 1.4;
}}
.rr-sidebar-label {{
  font-size: 0.68rem; font-weight: 700; letter-spacing: 0.14em;
  color: #5a5a60; text-transform: uppercase; margin: 1.2rem 0 0.6rem 0;
}}
.rr-cat-badge {{
  display: inline-block; font-size: 0.62rem; font-weight: 700;
  letter-spacing: 0.12em; text-transform: uppercase;
  color: {ACCENT}; background: rgba(0, 180, 216, 0.08);
  padding: 3px 8px; border-radius: 4px;
  margin: 0 0 4px 0; border: 1px solid rgba(0, 180, 216, 0.18);
}}
.rr-replay-pill {{
  display: inline-flex; align-items: center; gap: 6px;
  background: rgba(0, 180, 216, 0.06);
  color: {ACCENT};
  border: 1px solid rgba(0, 180, 216, 0.18);
  padding: 6px 12px; border-radius: 999px;
  font-size: 0.72rem; font-weight: 500;
  margin-top: 1.5rem;
}}
.rr-replay-pill::before {{
  content: "●"; font-size: 0.55rem;
  color: {ACCENT}; filter: drop-shadow(0 0 4px {ACCENT});
}}

/* ----- Buttons (all) ----- */
.stButton > button {{
  background: #161618;
  color: #e4e4e6;
  border: 1px solid #242428;
  border-radius: 10px;
  text-align: left;
  padding: 12px 14px;
  font-weight: 500;
  font-size: 0.88rem;
  line-height: 1.3;
  white-space: normal;
  height: auto;
  transition: all 0.15s ease;
  box-shadow: none;
}}
.stButton > button:hover {{
  border-color: rgba(0, 180, 216, 0.55);
  background: #1b1b1f;
  color: #ffffff;
  transform: translateY(-1px);
}}
.stButton > button:focus {{
  box-shadow: 0 0 0 2px rgba(0, 180, 216, 0.25) !important;
  outline: none !important;
}}
.stButton > button[kind="primary"] {{
  background: linear-gradient(135deg, {ACCENT} 0%, #0077b6 100%);
  color: #ffffff;
  border-color: {ACCENT};
  font-weight: 600;
  box-shadow: 0 0 16px rgba(0, 180, 216, 0.25);
}}
.stButton > button[kind="primary"]:hover {{
  box-shadow: 0 0 22px rgba(0, 180, 216, 0.45);
  transform: translateY(-1px);
}}

/* ----- Product header ----- */
.rr-hero {{
  border-bottom: 1px solid #1a1a1c;
  padding-bottom: 1.5rem;
  margin-bottom: 1.5rem;
}}
.rr-product-title {{
  font-size: 1.95rem; font-weight: 700; color: #ffffff;
  letter-spacing: -0.025em; line-height: 1.18; margin: 0.25rem 0 0.7rem 0;
}}
.rr-meta-row {{
  display: flex; flex-wrap: wrap; gap: 1.1rem; align-items: center;
  color: #9a9a9f; font-size: 0.86rem; margin-top: 4px;
}}
.rr-meta-row .rr-chip {{
  background: #141416; border: 1px solid #242428;
  padding: 4px 10px; border-radius: 999px;
  font-size: 0.75rem; color: #b8b8bc; font-weight: 500;
}}
.rr-meta-row .rr-chip strong {{ color: #ffffff; font-weight: 600; }}
.rr-meta-row .rr-stars {{ color: #f4b400; letter-spacing: 0.05em; }}

/* Arrow hint panel */
.rr-hint {{
  background:
    radial-gradient(ellipse at center, rgba(0, 180, 216, 0.08) 0%, transparent 70%),
    #111113;
  border: 1px dashed rgba(0, 180, 216, 0.35);
  border-radius: 14px;
  padding: 1.4rem;
  height: 100%; min-height: 200px;
  display: flex; flex-direction: column;
  align-items: flex-start; justify-content: center;
  gap: 0.4rem;
}}
.rr-hint-kicker {{
  font-size: 0.7rem; font-weight: 700; letter-spacing: 0.16em;
  color: {ACCENT}; text-transform: uppercase;
}}
.rr-hint-headline {{ font-size: 1.35rem; font-weight: 700; color: #fff; line-height: 1.25; }}
.rr-hint-sub {{ color: #8a8a90; font-size: 0.9rem; line-height: 1.5; }}
.rr-hint-arrow {{
  display: inline-block; font-size: 2.2rem; color: {ACCENT};
  filter: drop-shadow(0 0 12px rgba(0, 180, 216, 0.5));
  margin-top: 0.3rem;
}}

/* ----- Section headers ----- */
.rr-section-title {{
  font-size: 1.5rem; font-weight: 700; color: #ffffff;
  letter-spacing: -0.015em; margin: 2.5rem 0 0.35rem 0;
}}
.rr-section-sub {{
  color: #7a7a80; font-size: 0.92rem; margin-bottom: 1.4rem;
}}

/* ----- Cards via st.container(border=True) — only style cards that actually
   have content so empty column slots don't render as ghost pills. ----- */
[data-testid="stVerticalBlockBorderWrapper"]:has(> div:not(:empty)) {{
  background: #111113 !important;
  border: 1px solid #1e1e22 !important;
  border-radius: 14px !important;
  padding: 14px !important;
  transition: box-shadow 0.25s ease, border-color 0.25s ease, transform 0.2s ease;
}}
[data-testid="stVerticalBlockBorderWrapper"]:has(> div:not(:empty)):hover {{
  border-color: rgba(0, 180, 216, 0.45) !important;
  box-shadow: 0 0 26px rgba(0, 180, 216, 0.18);
  transform: translateY(-2px);
}}
/* Truly empty border wrappers (auto-generated by columns) collapse away. */
[data-testid="stVerticalBlockBorderWrapper"]:not(:has(> div:not(:empty))) {{
  display: none !important;
}}

/* Image cards should sit flush to their border */
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stImage"] img {{
  border-radius: 10px;
}}

/* ----- Provider / score badges ----- */
.rr-badge-row {{ display: flex; align-items: center; gap: 8px; margin-top: 10px; }}
.rr-provider-badge {{
  display: inline-flex; align-items: center; gap: 6px;
  font-size: 0.68rem; font-weight: 700; letter-spacing: 0.1em;
  text-transform: uppercase; color: {ACCENT};
  background: rgba(0, 180, 216, 0.08);
  border: 1px solid rgba(0, 180, 216, 0.3);
  padding: 4px 9px; border-radius: 999px;
}}
.rr-provider-badge.gemini {{
  color: #c77dff; background: rgba(199, 125, 255, 0.08);
  border-color: rgba(199, 125, 255, 0.3);
}}
.rr-hero-tag {{
  font-size: 0.64rem; font-weight: 700; letter-spacing: 0.12em;
  text-transform: uppercase; color: #f4b400;
  background: rgba(244, 180, 0, 0.08); border: 1px solid rgba(244, 180, 0, 0.3);
  padding: 3px 7px; border-radius: 999px;
}}

.rr-score-wrap {{ margin-top: 10px; }}
.rr-score-head {{
  display: flex; justify-content: space-between; align-items: baseline;
  font-size: 0.7rem; color: #888; margin-bottom: 4px;
  gap: 8px; flex-wrap: nowrap; white-space: nowrap;
  text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600;
}}
.rr-score-head .rr-score-val {{ color: #ffffff; font-weight: 700; font-size: 0.78rem; letter-spacing: 0; }}
.rr-score-bar {{
  height: 5px; background: #1a1a1e; border-radius: 3px; overflow: hidden;
}}
.rr-score-fill {{
  height: 100%;
  background: linear-gradient(90deg, {ACCENT} 0%, #48cae4 100%);
  border-radius: 3px;
  box-shadow: 0 0 10px rgba(0, 180, 216, 0.45);
}}
.rr-score-finding {{
  color: #9a9a9f; font-size: 0.78rem; margin-top: 8px;
  line-height: 1.45; font-style: italic;
  display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical;
  overflow: hidden;
}}

/* ----- Before / After ----- */
.rr-ba-arrow {{
  display: flex; align-items: center; justify-content: center;
  height: 100%; font-size: 3.2rem; font-weight: 300;
  color: {ACCENT}; line-height: 1;
  text-shadow: 0 0 18px rgba(0, 180, 216, 0.55);
}}
.rr-ba-caption {{
  text-align: center; font-size: 0.7rem; font-weight: 700;
  letter-spacing: 0.16em; text-transform: uppercase;
  color: #7a7a80; margin-top: 8px;
}}
.rr-ba-caption.after {{ color: {ACCENT}; }}

/* ----- Insights + prompts block ----- */
.rr-insight-h {{
  font-size: 0.72rem; font-weight: 700; letter-spacing: 0.14em;
  color: {ACCENT}; text-transform: uppercase; margin: 0 0 10px 0;
}}
.rr-insight-list {{ list-style: none; padding: 0; margin: 0; }}
.rr-insight-list li {{
  background: #111113; border: 1px solid #1e1e22;
  border-left: 3px solid {ACCENT};
  border-radius: 8px; padding: 10px 12px; margin-bottom: 8px;
  color: #d0d0d4; font-size: 0.88rem; line-height: 1.5;
}}
.rr-insight-list li strong {{ color: #ffffff; font-weight: 600; }}
.rr-insight-list li .role {{
  display: inline-block;
  font-size: 0.64rem; font-weight: 700;
  letter-spacing: 0.1em; text-transform: uppercase;
  color: {ACCENT}; background: rgba(0, 180, 216, 0.08);
  padding: 2px 6px; border-radius: 4px; margin-right: 8px;
  border: 1px solid rgba(0, 180, 216, 0.2);
}}

/* Expander */
[data-testid="stExpander"] {{
  background: #0d0d0f !important;
  border: 1px solid #1a1a1c !important;
  border-radius: 12px !important;
}}
[data-testid="stExpander"] summary {{
  color: #e8e8e8 !important;
  font-weight: 600;
  font-size: 0.95rem;
}}
[data-testid="stExpander"] summary:hover {{ color: {ACCENT} !important; }}

/* Code block */
[data-testid="stCodeBlock"] {{
  background: #08080a !important;
  border: 1px solid #1a1a1c !important;
  border-radius: 10px !important;
}}
[data-testid="stCodeBlock"] pre {{
  font-size: 0.8rem !important;
  line-height: 1.5 !important;
  color: #c8c8cc !important;
}}

/* Empty-state card */
.rr-empty {{
  background: #111113; border: 1px dashed #2a2a2e;
  border-radius: 14px; padding: 2rem;
  text-align: center; color: #7a7a80;
}}
.rr-empty code {{
  background: #1a1a1e; padding: 3px 8px; border-radius: 5px;
  color: {ACCENT}; font-size: 0.88rem;
}}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =============================================================================
# Small rendering helpers
# =============================================================================


def render_stars(avg: float | None) -> str:
    if avg is None:
        return "—"
    full = int(round(avg))
    full = max(0, min(5, full))
    return "★" * full + "☆" * (5 - full)


def render_score_block(score: float | None, finding: str | None) -> None:
    if score is None:
        return
    try:
        s = float(score)
    except (TypeError, ValueError):
        return
    pct = max(0.0, min(1.0, s / 5.0)) * 100
    html = (
        '<div class="rr-score-wrap">'
        '<div class="rr-score-head">'
        '<span>VLM score</span>'
        f'<span class="rr-score-val">{s:.1f} / 5</span>'
        "</div>"
        '<div class="rr-score-bar">'
        f'<div class="rr-score-fill" style="width: {pct:.0f}%"></div>'
        "</div>"
    )
    if finding:
        html += f'<div class="rr-score-finding">“{finding}”</div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_image_card(item: dict[str, Any]) -> None:
    provider = item.get("provider", "openai")
    label = PROVIDER_LABEL.get(provider, provider.title())
    with st.container(border=True):
        st.image(str(item["path"]), use_container_width=True)
        badge_html = (
            f'<div class="rr-badge-row">'
            f'<span class="rr-provider-badge {provider}">{label}</span>'
        )
        if item.get("is_hero"):
            badge_html += '<span class="rr-hero-tag">Hero</span>'
        badge_html += "</div>"
        st.markdown(badge_html, unsafe_allow_html=True)
        render_score_block(item.get("score"), item.get("finding"))


# =============================================================================
# Sidebar (product selector)
# =============================================================================

products = load_products()

if products and "selected_asin" not in st.session_state:
    st.session_state.selected_asin = products[0]["asin"]

with st.sidebar:
    st.markdown(
        '<h1 class="rr-logo">Review<span class="rr-mark">Renew</span></h1>'
        '<p class="rr-tagline">Turn customer reviews into hero shots.</p>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="rr-sidebar-label">Catalog</div>', unsafe_allow_html=True)

    if not products:
        st.markdown(
            '<div class="rr-empty">Could not load <code>data/products.json</code>.</div>',
            unsafe_allow_html=True,
        )
    else:
        for product in products:
            asin = product.get("asin", "")
            title = product.get("title", "(untitled)")
            title_trunc = title if len(title) <= 40 else title[:39].rstrip() + "…"
            category = product.get("category", "—")
            is_selected = st.session_state.get("selected_asin") == asin

            st.markdown(
                f'<div class="rr-cat-badge">{category}</div>',
                unsafe_allow_html=True,
            )
            if st.button(
                title_trunc,
                key=f"rr_select_{asin}",
                use_container_width=True,
                type="primary" if is_selected else "secondary",
            ):
                st.session_state.selected_asin = asin
                st.rerun()

    st.markdown(
        '<div class="rr-replay-pill">Replay mode — all outputs pre-generated</div>',
        unsafe_allow_html=True,
    )


# =============================================================================
# Main showcase
# =============================================================================


def render_empty_state(asin: str) -> None:
    st.markdown(
        f'<div class="rr-empty">'
        f"<strong>Pipeline outputs not found for <code>{asin}</code>.</strong><br>"
        f"Run <code>python run_pipeline.py</code> from the repo root to generate "
        f"bundles and hero shots, then refresh this page."
        "</div>",
        unsafe_allow_html=True,
    )


def render_product(product: dict[str, Any]) -> None:
    asin = product.get("asin", "")
    title = product.get("title", "(untitled)")
    category = product.get("category", "—")
    n_reviews, avg_rating = review_stats(product)
    og_path = og_image_for(asin)
    items = collect_generated(asin)
    attach_scores(items, asin)
    bundle = load_bundle(asin)

    # ---- Section 1: Product header ----
    st.markdown(
        f'<div class="rr-cat-badge" style="margin-bottom:0.2rem;">{category}</div>'
        f'<div class="rr-hero">'
        f'<div class="rr-product-title">{title}</div>'
        f'<div class="rr-meta-row">'
        f'<span class="rr-chip"><strong>ASIN</strong> &nbsp; {asin}</span>'
        f'<span class="rr-chip"><strong>Reviews</strong> &nbsp; {n_reviews:,}</span>'
        f'<span class="rr-chip">'
        f'<span class="rr-stars">{render_stars(avg_rating)}</span>'
        f' &nbsp; {f"{avg_rating:.2f}" if avg_rating is not None else "—"}'
        f"</span>"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 1], gap="large")
    with left:
        if og_path:
            with st.container(border=True):
                st.image(str(og_path), caption="Amazon listing photo", use_container_width=True)
        else:
            st.markdown(
                '<div class="rr-empty">No original product image on file.</div>',
                unsafe_allow_html=True,
            )
    with right:
        n_generated = len(items)
        st.markdown(
            '<div class="rr-hint">'
            '<div class="rr-hint-kicker">↓ See what ReviewRenew generated</div>'
            f'<div class="rr-hint-headline">{n_generated or "0"} AI hero shots, '
            f"grounded in {n_reviews:,} customer reviews</div>"
            '<div class="rr-hint-sub">Each shot was planned from cross-review themes, '
            f"rendered by two models, then scored by a VLM rubric.</div>"
            '<div class="rr-hint-arrow">↓</div>'
            "</div>",
            unsafe_allow_html=True,
        )

    if not items:
        render_empty_state(asin)
        return

    # ---- Section 2: Hero shot gallery ----
    st.markdown(
        '<div class="rr-section-title">AI-Generated Hero Shots</div>'
        '<div class="rr-section-sub">All images below were produced by the replay '
        "pipeline — OpenAI Images and Gemini side-by-side, evaluated on a five-"
        "dimension rubric.</div>",
        unsafe_allow_html=True,
    )

    cols_per_row = 3
    for row_start in range(0, len(items), cols_per_row):
        row = items[row_start : row_start + cols_per_row]
        cols = st.columns(cols_per_row, gap="medium")
        for col, item in zip(cols, row):
            with col:
                render_image_card(item)

    # ---- Section 3: What the AI Saw ----
    if bundle:
        with st.expander("How ReviewRenew analyzed the reviews →", expanded=False):
            c_left, c_right = st.columns(2, gap="large")
            shots = sorted(
                bundle.get("planned_shots") or [],
                key=lambda s: s.get("shot_index", 0),
            )
            with c_left:
                st.markdown(
                    '<div class="rr-insight-h">Customer insights</div>',
                    unsafe_allow_html=True,
                )
                if shots:
                    items_html = "".join(
                        (
                            "<li>"
                            f'<span class="role">Shot {s.get("shot_index", "?")} · '
                            f'{s.get("role", "shot")}</span>'
                            f'{s.get("rationale_from_reviews", "") or ""}'
                            "</li>"
                        )
                        for s in shots
                    )
                    st.markdown(
                        f'<ul class="rr-insight-list">{items_html}</ul>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown("_No shot rationales in bundle._")

                cross = bundle.get("cross_review_benefits") or []
                if cross:
                    st.markdown(
                        '<div class="rr-insight-h" style="margin-top:1.2rem;">'
                        "Cross-review benefits</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        '<ul class="rr-insight-list">'
                        + "".join(f"<li>{c}</li>" for c in cross)
                        + "</ul>",
                        unsafe_allow_html=True,
                    )
            with c_right:
                st.markdown(
                    '<div class="rr-insight-h">Prompts generated</div>',
                    unsafe_allow_html=True,
                )
                if shots:
                    for s in shots:
                        role = s.get("role", "shot")
                        idx = s.get("shot_index", "?")
                        prompt = s.get("prompt") or ""
                        st.caption(f"Shot {idx} · {role}")
                        st.code(prompt, language="markdown")
                else:
                    st.markdown("_No prompts found._")

    # ---- Section 4: Before / After ----
    primary = pick_primary(items)
    if primary and og_path:
        st.markdown(
            '<div class="rr-section-title">Before → After</div>'
            '<div class="rr-section-sub">The Amazon listing photo versus '
            "ReviewRenew's best hero shot.</div>",
            unsafe_allow_html=True,
        )
        b_col, arrow_col, a_col = st.columns([5, 1, 5], gap="small")
        with b_col:
            with st.container(border=True):
                st.image(str(og_path), use_container_width=True)
                st.markdown(
                    '<div class="rr-ba-caption">Before · Amazon listing</div>',
                    unsafe_allow_html=True,
                )
        with arrow_col:
            st.markdown('<div class="rr-ba-arrow">→</div>', unsafe_allow_html=True)
        with a_col:
            with st.container(border=True):
                st.image(str(primary["path"]), use_container_width=True)
                after_label = PROVIDER_LABEL.get(primary["provider"], primary["provider"].title())
                score_suffix = ""
                if isinstance(primary.get("score"), (int, float)):
                    score_suffix = f" · {primary['score']:.1f}/5"
                st.markdown(
                    f'<div class="rr-ba-caption after">After · {after_label}{score_suffix}</div>',
                    unsafe_allow_html=True,
                )

        others = [i for i in items if i["path"] != primary["path"]]
        if others:
            st.markdown(
                '<div class="rr-section-sub" style="margin-top:1.5rem; margin-bottom:0.6rem;">'
                "Other generations</div>",
                unsafe_allow_html=True,
            )
            strip_cols = st.columns(min(len(others), 6), gap="small")
            for col, item in zip(strip_cols, others[: len(strip_cols)]):
                with col:
                    st.image(str(item["path"]), use_container_width=True)
                    st.caption(PROVIDER_LABEL.get(item["provider"], item["provider"]))


# ---- Dispatch ----
if not products:
    st.markdown(
        '<div class="rr-empty">Could not load any products. '
        "Check that <code>data/products.json</code> exists.</div>",
        unsafe_allow_html=True,
    )
else:
    selected_asin = st.session_state.get("selected_asin") or products[0]["asin"]
    selected_product = next((p for p in products if p.get("asin") == selected_asin), products[0])
    render_product(selected_product)
