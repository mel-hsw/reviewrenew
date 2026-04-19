from __future__ import annotations

import json
from typing import Any, Callable

from openai import OpenAI

from . import config
from .llm_harness import StructuredLLMHarness
from .models import (
    CreativePromptPack,
    ImagePromptBundle,
    Product,
    ReviewImageryBrief,
    merge_review_brief_with_creative,
)


def _client() -> OpenAI:
    return OpenAI(api_key=config.OPENAI_API_KEY)


# Hybrid retrieval: broad pillars first (concerns / benefits / expectation gap), then evidence-rich follow-ups.
# Keeps recall high for any category (pets, skin care, tools) without over-indexing on one sensory channel.
RAG_RETRIEVAL_QUERIES = [
    # --- Broad pillars ---
    "What concerns, complaints, risks, disappointments, defects, damage, or returns do customers mention?",
    "What benefits, relief, positive outcomes, loyalty, or praise patterns appear in reviews?",
    "What expectations versus reality show up: misleading photos, wrong size, wrong texture or type, or comparison to other products?",
    # --- Visual + material evidence (hero-shot grounding) ---
    "What do customers say about appearance, packaging, color, size, shape, texture, sheen, matte or glossy finish, and whether photos match the product?",
    "What materials, fabric, foam, plastic, liquid, fill, weave, stitching, or construction details are described?",
    "What defects, wear after washing, pilling, fading, or quality issues are mentioned?",
    # --- Outcomes + sensory (category-agnostic; not smell-only) ---
    "What outcomes do reviews describe: cleaning results, residue, efficacy, comfort, skin feel, sleep, fit, or how well it works?",
    "What sensory or physical qualities appear: softness, firmness, weight, warmth, coolness, rigidity, slipperiness, sound, temperature, or scent only if reviewers mention it?",
    "What mood, atmosphere, lifestyle, or setting do reviews imply?",
    # --- Map listing ↔ reviews (what to elevate in imagery) ---
    "Which specific features, claims, or bullets from the product listing do reviewers echo, praise, argue about, or say matter most—what should marketing imagery emphasize?",
    "Which benefits, outcomes, or feelings do multiple different reviewers mention or repeat—independent agreement worth highlighting?",
    "What do reviewers say about the brand name, packaging colors, bottle or box design, logo area, or overall look—anything that signals on-shelf or on-brand appearance?",
]


def _normalize_review_brief_json(data: dict[str, Any]) -> dict[str, Any]:
    """Coerce occasional schema drift from the text model before Pydantic validation."""
    data.pop("pipeline_meta", None)
    cnc = data.get("consumer_needs_and_concerns")
    if isinstance(cnc, str) and cnc.strip():
        data["consumer_needs_and_concerns"] = [s.strip() for s in cnc.split("\n") if s.strip()]
    elif isinstance(cnc, list):
        data["consumer_needs_and_concerns"] = [str(x).strip() for x in cnc if str(x).strip()]
    elev = data.get("listing_features_elevated_by_reviews")
    if isinstance(elev, str) and elev.strip():
        data["listing_features_elevated_by_reviews"] = [s.strip() for s in elev.split("\n") if s.strip()]
    elif isinstance(elev, list):
        data["listing_features_elevated_by_reviews"] = [str(x).strip() for x in elev if str(x).strip()]
    crb = data.get("cross_review_benefits")
    if isinstance(crb, str) and crb.strip():
        data["cross_review_benefits"] = [s.strip() for s in crb.split("\n") if s.strip()]
    elif isinstance(crb, list):
        data["cross_review_benefits"] = [str(x).strip() for x in crb if str(x).strip()]
    cvc = data.get("customer_visual_consensus")
    if isinstance(cvc, list):
        data["customer_visual_consensus"] = "\n".join(str(x).strip() for x in cvc if str(x).strip())
    vtn = data.get("visual_translation_notes")
    if isinstance(vtn, list):
        data["visual_translation_notes"] = "\n".join(str(x).strip() for x in vtn if str(x).strip())
    ps = data.get("product_summary")
    if isinstance(ps, list):
        data["product_summary"] = "\n".join(str(x).strip() for x in ps if str(x).strip())
    bvc = data.get("brand_visual_codes")
    if isinstance(bvc, str) and bvc.strip():
        data["brand_visual_codes"] = [s.strip() for s in bvc.split("\n") if s.strip()]
    elif isinstance(bvc, list):
        data["brand_visual_codes"] = [str(x).strip() for x in bvc if str(x).strip()]
    return data


def _normalize_creative_pack_json(data: dict[str, Any]) -> dict[str, Any]:
    data.pop("pipeline_meta", None)
    return data


def build_rag_context(retrieved_chunks: list[Any]) -> str:
    lines: list[str] = []
    for i, ch in enumerate(retrieved_chunks, start=1):
        src = getattr(ch, "source", "chunk")
        lines.append(f"[{i}] ({src}) {getattr(ch, 'text', '')}")
    return "\n".join(lines)


def _gather_unique_chunks(
    retrieve_fn: Callable[[str], list[Any]],
) -> list[Any]:
    all_chunks: list[Any] = []
    seen: set[str] = set()
    for q in RAG_RETRIEVAL_QUERIES:
        for ch in retrieve_fn(q):
            key = getattr(ch, "text", str(ch))
            if key in seen:
                continue
            seen.add(key)
            all_chunks.append(ch)
    return all_chunks


Q2_SYSTEM = """You are a senior e-commerce and consumer insights analyst for retail imagery strategy. \
Use ONLY the product description and retrieved review excerpts—ground every claim in that text. \
Your job is analysis and planning, not writing final text-to-image prompts (a separate creative step follows).

Analytic goals:
- **Reviews** identify which Official-description features deserve emphasis in imagery (what shoppers repeatedly care about, praise, or dispute) and **which benefits appear across many reviewers**—those must be prominent in your shot rationales and supporting fields.
- The **Official description** is the source of truth for what the SKU is: materials, construction, bundle contents, size/capacity class, certifications, care, design parts (rim, base, pump, clips), formulas as stated by the seller.
- Stay **on-brand for this product line**: infer **brand-appropriate palette, packaging finishes, and imagery mood** from the Title, Official description, and reviewer comments about packaging or appearance—do not invent an unrelated brand look.
- Write with rich, specific language: sight (color temperature, contrast, surface sheen), touch and material (plush, grainy, slick), atmosphere (warm daylight, soft bounce, crisp studio), and sensory dimensions reviewers actually mention.
- Do not invent ingredients or medical drug claims beyond the description and reviews.

**Information density:** Real retailer pages use **many** carousel images (hero, infographics, size charts, comparisons, lifestyle). Our **`planned_shots`** list will later become **one text-to-image prompt per shot**—plan **one clear storytelling job per shot** (hero truth vs in-use vs macro vs context), not one overcrowded scene.

**Brand-owned assets:** You have **no access** to official brand digital asset libraries or style guides. Visual cues must be inferred only from **this product’s** Title, Official description, and reviews."""


Q3_SYSTEM = """You are a creative director for premium e-commerce and editorial product photography. \
You receive a structured **Q2 analyst brief** (JSON) plus listing metadata. Your only job is to write **final English text-to-image prompts** for each planned shot—one scene, one primary idea per prompt.

Rules:
- Ground every visual claim in the Q2 brief and the Official description excerpt; do not invent features, ingredients, or certifications.
- **Prioritize** `listing_features_elevated_by_reviews`, **cross_review_benefits**, and **`brand_visual_codes`** in each prompt; weave in **visual_translation_notes** and **pitfalls_to_avoid** as guardrails.
- **Image-model limitations (critical):** Text-to-image models often **fail at tiny legible copy** (certification seals, barcodes, dense ingredient panels). **Avoid** readable micro-text on seals/logos, full nutrition or drug-facts panels, or exact multi-line label reproduction. **Prefer** product **silhouette**, **brand color blocking**, **material and finish**, pump/cap/clip geometry, and at most **one or two short** legible phrases from the **Title or Official description** (e.g. brand + product line). For certifications: **omit the seal**, show **soft blur**, or a **generic badge shape without readable association wording**. Avoid huge promotional banners with paragraphs of text.
- Avoid **physically implausible** staging. Photorealistic, plausible lighting; honest materials when reviews warn about misleading photos.
- **Brand-owned assets:** Do not instruct “official brand campaign” art you cannot verify—use only cues supported by the brief and listing.

Output **only** JSON matching the requested schema: `planned_shots` with `shot_index` and `prompt` for **every** required index exactly once."""


def _build_q2_user(product: Product, rag_block: str) -> str:
    return f"""Product metadata:
Title: {product.title}
Category: {product.category}
ASIN: {product.asin}

Official description (truncated only if extremely long; features and bullets matter):
{product.description[:8000]}

Retrieved excerpts (RAG; description and review chunks):
{rag_block}

Return a single JSON object with these keys. Follow the CHAIN OF THOUGHT below **in order** (the JSON keys mirror that order). Do **not** include full text-to-image prompts—`planned_shots` lists **roles and rationales only**.

CHAIN OF THOUGHT (must inform your writing; be explicit in category_imagery_norms, consumer_needs_and_concerns, listing_features_elevated_by_reviews, cross_review_benefits, brand_visual_codes, and shot_plan_rationale):
1) **Category norms:** Given Title and Category only, state what kinds of listing images typically persuade shoppers in this product type (hero truth, in-use, detail, context, etc.). Note how real listings reach high **information density** through **multiple** gallery frames—and that we approximate that by **splitting** jobs across separate planned shots.
2) **Consumer landscape:** From the excerpts, distill recurring **needs**, **benefits**, and **concerns** (risks, complaints, mismatches). Paraphrase; do not fabricate.
2b) **Listing ↔ review mapping:** Which **specific** Official-description features do reviewers **elevate** (praise, demand, or debate)?
2c) **Cross-review benefits:** Which benefits or outcomes appear **multiple times across different reviewers**?
2d) **Listing features inventory:** From the **Official description**, note photographable facts that support 2b–2c. Prioritize **elevated** features over a flat checklist.
2e) **Brand visual codes:** From **Title, Official description, and reviews**, distill **on-brand** cues for this SKU: colors, matte vs glossy packaging, silhouette, photography mood. Infer only what sources support.
3) **Shot plan:** Choose **3–5** shot **roles** that fit category norms, foreground elevated listing features and cross-review benefits, align with **`brand_visual_codes`**, and address major concerns honestly—**one clear job per shot**. The **first** entry in `planned_shots` must be the primary **hero** (shot_index 1). In `shot_plan_rationale`, explain order and which elevated features / benefits / brand codes each shot serves.
4) **Detail fields:** Fill brand codes, visual/sensory lists, translation notes, and pitfalls (include customer issues **and** downstream image-model risks such as unreadable seals, fake barcodes, banner hallucinations—so the creative step can avoid them).

GENERAL STYLE FOR ALL TEXT FIELDS:
- Prefer concrete, evocative adjectives and short phrases over generic praise.
- Tie claims to reviewers **and** to the Official description.

FIELDS (output ALL keys below):
- category_imagery_norms: 2–4 sentences from step 1.

- consumer_needs_and_concerns: array of **at least 3** short strings from step 2.

- listing_features_elevated_by_reviews: array of **at least 2** strings. Each line: one **specific** feature or claim from the **Official description** that reviews justify emphasizing in imagery, plus a short “because reviewers …”.

- cross_review_benefits: array of **at least 1** string. Benefits or outcomes that **multiple reviewers** independently stress.

- brand_visual_codes: array of **at least 2** short strings from step 2e (colors, finish, shape, mood, lighting bias)—grounded in sources.

- shot_plan_rationale: one paragraph from step 3.

- product_summary: 3–5 sentences blending what the product is with dominant sensory/emotional tone from reviews.

- visual_attributes: array of strings—vivid phrases about literal appearance.

- material_and_texture_cues: array of strings—tactile/material language if supported by sources.

- sensory_efficacy_and_ingredient_cues: array of strings—outcomes and sensory dimensions reviewers mention; scent only if reviews do.

- customer_visual_consensus: 2–4 sentences on what most people agree the experience feels and looks like.

- pitfalls_to_avoid: array of strings—**(a)** customer issues (misleading scale, wrong color, etc.) and **(b)** image-generation failure modes to avoid later (illegible seals, fake barcodes, dense label copy, impossible physics).

- visual_translation_notes: one rich paragraph on staging with light, props, environment; prefer shape/color blocking over dense on-image copy.

- planned_shots: array of **3 to 5** objects. Each object has **only**:
  - shot_index: consecutive integers starting at 1. **shot_index 1 = primary hero** for this product category.
  - role: short label (see SHOT PLAN); snake_case allowed.
  - rationale_from_reviews: 1–3 sentences tying THIS shot to review themes, elevated listing features, cross-review benefits, brand_visual_codes, and Official-description facts—**no `prompt` field**.

Prefer **5** distinct shots when the text supports them; **minimum 3**.

SHOT PLAN (adapt roles to the category; **shot 1 is always hero_product**; reorder 2–5 by relevance):
  1 — hero_product: Catalog hero; product alone; accurate silhouette, color, material truth; on-brand palette and packaging finish.
  2 — consumer_in_use: Realistic usage as reviewers describe.
  3 — ingredients_or_sensorial_or_formula: Texture/formula or mechanism staging as appropriate to category.
  4 — detail_macro: Texture, stitching, pump, clips—what reviewers praise or criticize.
  5 — context_lifestyle: Mood/room/context matching review atmosphere.

Safety: no invented medical or drug claims beyond the description and reviews.
"""


def _build_q3_user(product: Product, brief: ReviewImageryBrief) -> str:
    required = sorted(s.shot_index for s in brief.planned_shots)
    brief_json = json.dumps(brief.model_dump(), indent=2, ensure_ascii=False)
    req_str = ", ".join(str(i) for i in required)
    return f"""Listing metadata (for short verbatim phrases when needed):
Title: {product.title}
Category: {product.category}
ASIN: {product.asin}

Official description (excerpt; full detail for faithful prompts):
{product.description[:8000]}

**Required shot_index values — you must output exactly one `planned_shots` entry per index, no extras, no omissions:** {req_str}

Q2 analyst brief (JSON):
{brief_json}

Return a single JSON object with exactly this shape:
{{ "planned_shots": [ {{ "shot_index": <int>, "prompt": "<long English text-to-image prompt, 80–240 words>" }}, ... ] }}

Each `prompt`: **one scene, one primary idea**; camera, lighting, setting, materials, mood; prioritize elevated listing features and cross-review benefits from the brief; **weave in `brand_visual_codes`**. Include **at least three** paraphrased listing facts across the set of prompts where relevant (not necessarily every prompt). Follow IMAGE-MODEL CONSTRAINTS in the system message.
"""


def analyze_product_with_rag(
    product: Product,
    retrieve_fn: Callable[[str], list[Any]],
) -> tuple[ImagePromptBundle, dict[str, Any]]:
    """retrieve_fn(query) returns retrieved chunks for this product. Two LLM calls: Q2 analyst (RAG) + Q3 creative."""
    all_chunks = _gather_unique_chunks(retrieve_fn)
    rag_block = build_rag_context(all_chunks)

    harness_q2 = StructuredLLMHarness(
        "q2_analyst",
        config.OPENAI_TEXT_MODEL,
        temperature=0.35,
        max_attempts=2,
    )
    brief, tel_q2 = harness_q2.complete_json(
        system=Q2_SYSTEM,
        user=_build_q2_user(product, rag_block),
        model_cls=ReviewImageryBrief,
        pre_validate=_normalize_review_brief_json,
    )

    harness_q3 = StructuredLLMHarness(
        "q3_creative",
        config.OPENAI_Q3_TEXT_MODEL,
        temperature=0.4,
        max_attempts=2,
    )
    creative, tel_q3 = harness_q3.complete_json(
        system=Q3_SYSTEM,
        user=_build_q3_user(product, brief),
        model_cls=CreativePromptPack,
        pre_validate=_normalize_creative_pack_json,
    )

    bundle = merge_review_brief_with_creative(brief, creative)
    meta: dict[str, Any] = {
        "retrieval_queries": RAG_RETRIEVAL_QUERIES,
        "unique_chunks_used": len(all_chunks),
        "chunking": {
            "strategy": config.CHUNK_STRATEGY,
            "max_tokens": config.CHUNK_MAX_TOKENS,
            "overlap_tokens": config.CHUNK_OVERLAP_TOKENS,
        },
        "openai_text_model_q2": config.OPENAI_TEXT_MODEL,
        "openai_text_model_q3": config.OPENAI_Q3_TEXT_MODEL,
        "q2_harness": tel_q2,
        "q3_harness": tel_q3,
    }
    bundle = bundle.model_copy(update={"pipeline_meta": meta})
    return bundle, meta


def summarize_without_rag(product: Product) -> str:
    """Optional baseline: single-pass summary without retrieval (for report comparison)."""
    prompt = f"""Summarize this product in 2-3 sentences for a marketer. Then list 5 visual attributes.
Title: {product.title}
Description: {product.description[:6000]}
"""
    resp = _client().chat.completions.create(
        model=config.OPENAI_TEXT_MODEL,
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}],
    )
    return (resp.choices[0].message.content or "").strip()
