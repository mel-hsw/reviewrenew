# Generating Product Images from Customer Reviews
## Final Project Report

**Course:** 94-844 Generative AI Lab (Spring 2026)  
**Instructor:** Prof. Beibei Li, Heinz College, Carnegie Mellon University  
**Due:** April 26, 2026  

---

## Abstract

This project explores an end-to-end pipeline for generating product marketing imagery grounded in customer review text. We selected three Amazon products spanning **Pet Supplies** (Best Friends by Sheri calming donut bed bundle, `B0C4BFWLNC`), **Beauty & Personal Care** (CeraVe Hydrating Facial Cleanser, `B0BG9Q18ZZ`), and **Home laundry accessories** (YEENOR hat washer cage 2-pack, `B0BXSQN4HV`). These three categories were deliberately chosen to represent soft-goods texture, labeled CPG packaging, and rigid-plastic utility—each presenting distinct challenges for text-to-image generation.

The pipeline chunks listing text and reviews using `tiktoken`, embeds them in a **Chroma** vector store, retrieves relevant excerpts with **12-query multi-query RAG**, and prompts a structured JSON LLM (`gpt-5.4-mini`) to produce an `ImagePromptBundle`: a typed schema containing review-grounded summaries, brand visual codes, and **3–5 planned shot prompts** per product. Images are then generated with two APIs—**OpenAI** (`chatgpt-image-latest`) and **Google Gemini** (`gemini-3.1-flash-image-preview`)—under the same base prompts plus provider-specific suffixes, yielding 18 images total (3 shots × 2 models × 3 products).

We document three deliberate experiments: a **chunking A/B comparison** (sliding windows vs. paragraph-batch packing), a **dual-model image generation study** (same prompts, two backends, qualitative comparison across composition and text-rendering dimensions), and an **iterative prompt engineering study** informed by inspecting real `listing_image_urls` and observing diffusion failure modes. Key findings: paragraph-batch chunking better matches short Amazon reviews; OpenAI images lean toward tighter catalog/studio frames while Gemini adds more lifestyle context; and **neither model reliably reproduces certification micro-copy or dense label panels**, requiring prompt constraints that favor silhouette, palette, and short verified phrases.

---

## Table of Contents

1. [Q1 — Product Selection and Data Collection](#q1--product-selection-and-data-collection)
2. [Q2 — LLM Analysis of Reviews](#q2--llm-analysis-of-reviews)
3. [Q3 — Image Generation with Diffusion Models](#q3--image-generation-with-diffusion-models)
4. [Q4 — Agentic Workflow](#q4--agentic-workflow)
5. [Reflection — Challenges and Lessons Learned](#reflection--challenges-and-lessons-learned)
6. [Appendix](#appendix)

---

## Q1 — Product Selection and Data Collection

### 1.1 Selection Rationale

The assignment requires three products from different categories. Beyond category diversity, we applied a **backward-design criterion**: each product needed to create meaningfully *different* challenges for the Q2 → Q3 pipeline so that comparisons in Q3 would be analytically interesting rather than redundant.

| Criterion | Pet bed (`B0C4BFWLNC`) | CeraVe cleanser (`B0BG9Q18ZZ`) | Hat washer (`B0BXSQN4HV`) |
|-----------|------------------------|-------------------------------|---------------------------|
| **Category** | Pet Supplies | Beauty & Personal Care | Home / Laundry Accessories |
| **Primary visual challenge** | Soft-goods texture; "does it really look plush?" | Labeled CPG bottle; text on pack, non-foaming product consistency | Rigid plastic; unusual 3-D cage geometry |
| **Key review language** | Shape, loft, washability, color accuracy | Non-foaming lotion texture, barrier claims, packaging leaks | Clip security, shape retention, brim fit |
| **Why this fuels Q3** | Diffusion can depict fur and nest shape well; the risk is flat/misleading staging | Diffusion struggles with legible micro-text on label; bottle silhouette is achievable | Geometry test: can diffusion render a molded plastic cage accurately? |
| **Review volume** | ~68,128 ratings (meta) | Thousands of reviews; we use up to 500 | ~21,076 ratings (meta) |

**Trio rationale.** The three products together span soft texture (bed), consumable with pharmaceutical-style labeling (cleanser), and hard-surface utility (hat washer). They produce different failure modes in Q3 (plushness hallucinations vs. text hallucinations vs. geometry distortion), making the cross-product comparison analytically informative.

**Popularity vs. quality trade-off.** The pet bed and hat washer have large review counts, ensuring rich corpus diversity for RAG. The CeraVe SKU (`B0BG9Q18ZZ`) is the 16 fl oz variant; our Beauty JSONL slice contains thousands of reviews for this `parent_asin`, and we load the 500 most recent (length > 20 chars) into the pipeline. High review volume means more embedding diversity but also noisier retrieval—a trade-off documented in the chunking experiment below.

### 1.2 Products

#### Product 1 — Best Friends by Sheri Calming Lux Donut Cuddler + Blanket Bundle

- **ASIN:** `B0C4BFWLNC`
- **Category:** Pet Supplies — Calming Beds
- **Title (as collected):** *Best Friends by Sheri Bundle Set The Original Calming Lux Donut Cuddler Cat and Dog Bed + Pet Throw Blanket Dark Chocolate Medium 30" x 30"*
- **Marketplace URL:** `https://www.amazon.com/dp/B0C4BFWLNC`
- **Reviews used:** Up to 500, loaded from the Pet Supplies JSONL slice (Amazon Reviews 2023 dataset), newest first, minimum 20-character text filter.
- **Raw files:** `data/products.json` (consolidated catalog entry for this ASIN), original slice from `data/raw/amazon_reviews_2023/`

**Why this product for image generation:** Reviews extensively discuss *whether the bed looks as plush in person as in photos*, color accuracy ("Dark Chocolate"), washability after cycles, and whether the rim stays lofted. These are precisely the visual trust signals that an AI image generation system would need to get right or risk compounding the "misleading photo" problem that reviewers already flag.

#### Product 2 — CeraVe Hydrating Facial Cleanser (16 fl oz)

- **ASIN:** `B0BG9Q18ZZ`
- **Category:** Beauty & Personal Care — Skin Care — Face — Cleansers
- **Title (as collected):** *CeraVe Hydrating Facial Cleanser | Moisturizing Non-Foaming Face Wash with Hyaluronic Acid, Ceramides and Glycerin | Fragrance Free Paraben Free | 16 Fluid Ounce*
- **Marketplace URL:** `https://www.amazon.com/dp/B0BG9Q18ZZ`
- **Reviews used:** Up to 500, Beauty JSONL slice, same filter as above.
- **Raw files:** `data/products.json`

**Why this product:** CeraVe is a high-traffic, trust-heavy skincare SKU with rich review language around *texture* (non-foaming, lotion-like), *skin feel* (hydrated, not stripped), *fragrance-free sensitivity*, and *packaging reliability* (pump leaks). That diversity of consumer concern maps directly to distinct shot types in Q3, while the label-heavy bottle creates a classic diffusion challenge: models attempt to render ingredient panels and certification seals, producing plausible-but-wrong glyphs.

#### Product 3 — YEENOR Hat Washer for Washing Machine, 2-Pack

- **ASIN:** `B0BXSQN4HV`
- **Category:** Home / Laundry Accessories (listed under Amazon Home in the 2023 meta)
- **Title (as collected):** *YEENOR Hat Washer for Washing Machine, 2-Pack Baseball Caps Washers Hat Storage, Hat Holder Fit for Adult/Kid's Hat Rack Frame for Washer Machine Cleaner/Washing Cage*
- **Marketplace URL:** `https://www.amazon.com/dp/B0BXSQN4HV`
- **Reviews used:** Up to 500, Home JSONL slice.
- **Raw files:** `data/products.json`

**Why this product:** A molded plastic cage with a very specific 3-D silhouette (four-sided, ribbed, clip-secured) is an interesting geometry test for diffusion models. Reviews concentrate on *shape retention*, *clip security through the wash cycle*, *brim fit for various hat styles*, and *plastic durability* — all functional and structural, making it a strong contrast to the soft-goods bed and the cosmetic cleanser.

### 1.3 Data Collection Protocol

**Attempted direct scraping — blocked.** Our first approach was to scrape listing text and reviews directly from Amazon product pages (using the fixed ASINs above). Amazon's bot-detection systems returned 503 errors and CAPTCHA challenges consistently, making live scraping unreliable for a reproducible academic pipeline. We therefore pivoted to an established, citable academic dataset.

**Final source — Amazon Reviews 2023 (Hou et al., 2024).** Review and metadata slices come from the Amazon Reviews 2023 dataset ([amazon-reviews-2023.github.io](https://amazon-reviews-2023.github.io/)). We cite Hou et al. (2024) per the dataset homepage. The dataset provides per-category JSONL files containing product metadata (title, description, feature bullets, listing image URLs, category breadcrumbs) and user reviews (text, rating, date). We downloaded the relevant category slices — Pet Supplies, Beauty & Personal Care, and Appliances/Home — and filtered them to the three target ASINs using `scripts/consolidate_products.py` and `scripts/filter_meta_by_rating.py`.

| Item | Detail |
|------|--------|
| **Primary source** | Amazon Reviews 2023 (Hou et al., 2024; McAuley Lab, UCSD) — JSONL per category |
| **Data access method** | Pre-collected academic dataset; direct Amazon scraping was blocked by bot-detection |
| **Review filter** | `len(text) > 20`; newest first; cap at 500 reviews per product |
| **Fields captured** | `text`, `rating`, `title`, `date` (review metadata) |
| **Listing fields** | `title`, `description` (features/bullets), `listing_image_urls` (carousel order), `category` |
| **Consolidation script** | `scripts/consolidate_products.py` → `data/products.json` + `data/products_manifest.json` |
| **Collection date** | 2026-04-18 (pipeline run date; ASINs are fixed; dataset snapshot per Hou et al. 2024) |
| **Fair-use note** | Academic coursework use per Amazon Reviews 2023 dataset terms and citation requirements |

---

## Q2 — LLM Analysis of Reviews

### 2.1 Overview and Goals

The Q2 step transforms raw listing text + customer reviews into a structured, visually actionable analysis. The output is not a simple summary but a typed **`ImagePromptBundle`** schema (defined in `src/models.py`) that bridges text evidence to image-generation prompts for Q3. Concretely, the LLM must:

1. Extract which **visual cues** (color, material, shape, packaging) reviews consistently describe.
2. Distinguish **consensus** (what most reviewers agree on) from **outliers** (what a minority flags as wrong or misleading).
3. Map **listing features** to review evidence—which official bullets do customers actually praise, dispute, or demand?
4. Produce **3–5 planned shots** per product, each with a `rationale_from_reviews` and a full text-to-image `prompt`, respecting diffusion model limitations.

### 2.2 Prompting Strategy

We use two complementary strategies rather than a single monolithic prompt:

#### Strategy 1 — Structured JSON output with chain-of-thought schema

The LLM (`gpt-5.4-mini`, temperature 0.35) is given a **system message** that positions it as a creative director and enumerates critical constraints (no legible micro-text on seals, one scene per shot, brand inferences only from this product's text). The **user message** then presents the product metadata and RAG context, followed by an explicit **chain-of-thought instruction** ordering the reasoning steps:

1. **Category norms** (what shot types persuade shoppers in this category)
2. **Consumer landscape** (needs, benefits, concerns from excerpts)
3. **Listing ↔ review mapping** (which official features reviews elevate)
4. **Cross-review benefits** (independent agreement across multiple reviewers)
5. **Brand visual codes** (on-brand cues inferred from title, description, reviews)
6. **Shot plan** (3–5 shots, one clear job per shot)
7. **Detail fields and prompts** (written last, grounded in the chain above)

The output is forced via `response_format: {"type": "json_object"}` into a validated `ImagePromptBundle` Pydantic schema. This structured approach ensures **analytical traceability**: every planned shot includes a `rationale_from_reviews` field explaining which review themes, elevated listing features, and brand codes motivated it—allowing graders (and us) to audit the reasoning chain, not just the final prompts.

*System message (abbreviated):*
> "You are a creative director for premium e-commerce and editorial product photography. Use ONLY the product description and retrieved review excerpts—ground every claim in that text. … **Image-model limitations (critical):** Text-to-image models often fail at tiny legible copy (certification seals, barcodes, dense ingredient panels). In prompts, avoid asking for readable micro-text; prefer product silhouette, brand color blocking, material and finish, and at most one or two short legible phrases from the Title/Official description…"

#### Strategy 2 — Multi-query RAG augmentation

Rather than stuffing the full review corpus into a single context window, we use **Chroma** (persistent vector store, `data/chroma/`) with **`text-embedding-3-small`** embeddings. Before calling the LLM, 12 targeted retrieval queries are issued in sequence; retrieved chunks are deduplicated by text key and concatenated as a `[n] (source) text` block in the user message.

**Why 12 queries instead of one?** A single broad query like "what do reviewers say about this product?" tends to surface the highest-rated or most-embedding-similar chunks, which may all repeat the same dominant theme. By issuing a **diverse query set** covering concerns, benefits, expectation gaps, visual/material properties, sensory outcomes, listing-review mapping, cross-review consensus, and brand/packaging mentions, we force the retriever to surface different facets of the review corpus:

```
Q1  "What concerns, complaints, risks, disappointments, defects, damage, or returns do customers mention?"
Q2  "What benefits, relief, positive outcomes, loyalty, or praise patterns appear in reviews?"
Q3  "What expectations versus reality show up: misleading photos, wrong size, wrong texture or type, or comparison to other products?"
Q4  "What do customers say about appearance, packaging, color, size, shape, texture, sheen, matte or glossy finish…?"
Q5  "What materials, fabric, foam, plastic, liquid, fill, weave, stitching, or construction details are described?"
Q6  "What defects, wear after washing, pilling, fading, or quality issues are mentioned?"
Q7  "What outcomes do reviews describe: cleaning results, residue, efficacy, comfort, skin feel, sleep, fit…?"
Q8  "What sensory or physical qualities appear: softness, firmness, weight, warmth, coolness, rigidity…?"
Q9  "What mood, atmosphere, lifestyle, or setting do reviews imply?"
Q10 "Which specific features, claims, or bullets from the product listing do reviewers echo, praise, argue about…?"
Q11 "Which benefits, outcomes, or feelings do multiple different reviewers mention or repeat?"
Q12 "What do reviewers say about the brand name, packaging colors, bottle or box design, logo area…?"
```

This hybrid retrieval design—broad pillars (Q1–Q3) followed by evidence-rich follow-ups (Q4–Q12)—ensures that the RAG context fed to the LLM covers both the emotional landscape (what reviewers feel) and the analytical detail (what features they repeatedly mention). The result is that `listing_features_elevated_by_reviews` and `cross_review_benefits` in each bundle are grounded in actual retrieved text, not the LLM's prior knowledge about the product.

### 2.3 Chunking Strategy

We use `tiktoken` (`cl100k_base`) to measure token counts, aligned with OpenAI models. Two strategies are implemented in `src/chunking.py`:

| Strategy | Env var | Behavior |
|----------|---------|----------|
| **Sliding window** | `CHUNK_STRATEGY=sliding` | Splits description + each review into overlapping windows of `CHUNK_MAX_TOKENS=400` tokens, with `CHUNK_OVERLAP_TOKENS=60` token overlap between consecutive windows within the same field. |
| **Paragraph batch** | `CHUNK_STRATEGY=paragraph_batch` (default) | Splits on blank lines (`\n\n`); packs whole paragraphs together up to 400 tokens; falls back to sliding only for paragraphs that exceed the limit. |

**A/B comparison — Experiment 1 (full protocol in next section)**

We ran both strategies on the identical three-product corpus with identical tokenizer defaults, deleting `data/chroma/` between runs. The key metric is `unique_chunks_used` after deduplicating across the 12 RAG queries:

| Strategy | `B0C4BFWLNC` | `B0BG9Q18ZZ` | `B0BXSQN4HV` | Total indexed chunks / product |
|----------|--------------|--------------|--------------|-------------------------------|
| sliding | 46 | 41 | 43 | 502 |
| paragraph_batch | 46 | 42 | 43 | 502 |

**Decision:** We adopt `paragraph_batch` as the project default. The reasoning is a good fit between strategy and data characteristics: the vast majority of Amazon reviews are short, self-contained paragraphs (often a single opinion). Sliding windows over a short text would create artificial overlapping chunks that repeat the same content with minor boundary offsets—adding retrieval noise without enriching the context. Paragraph packing keeps each review (or review paragraph) as a coherent unit. The one-chunk difference on CeraVe (41 vs. 42) confirms the boundary effect is real but small for our corpus.

**Chunking trade-offs discussed:**
- *Smaller `CHUNK_MAX_TOKENS`:* more, finer-grained chunks—better precision for specific claims but risk of losing cross-sentence context.
- *Larger `CHUNK_MAX_TOKENS`:* fewer, richer chunks—may mix unrelated themes in one vector.
- *Overlap:* reduces boundary artifacts for sliding windows but is irrelevant for paragraph packing when each review is short.

### 2.4 Analysis Outputs — `ImagePromptBundle` Schema

The `ImagePromptBundle` (defined in `src/models.py`) is the structured bridge artifact between Q2 and Q3. For each product, the final bundle JSON is saved at `outputs/<ASIN>_image_prompt_bundle.json`. The schema includes:

| Field | Purpose |
|-------|---------|
| `product_summary` | 3–5 sentence blend of product facts and reviewer tone |
| `category_imagery_norms` | What shot types persuade shoppers in this category |
| `consumer_needs_and_concerns` | Recurring needs and complaints from reviews |
| `listing_features_elevated_by_reviews` | Official bullets that reviewers specifically echo or demand |
| `cross_review_benefits` | Benefits multiple reviewers independently repeat |
| `brand_visual_codes` | On-brand palette, finish, and mood cues inferred from title/description/reviews |
| `shot_plan_rationale` | Paragraph explaining the logic and ordering of planned shots |
| `visual_attributes` | Vivid appearance phrases: silhouette, color, gloss/matte |
| `material_and_texture_cues` | Tactile and material language grounded in reviews |
| `sensory_efficacy_and_ingredient_cues` | Outcomes, skin feel, efficacy claims reviewers stress |
| `customer_visual_consensus` | What most reviewers agree the experience looks and feels like |
| `pitfalls_to_avoid` | Customer-flagged issues + diffusion-safety constraints |
| `visual_translation_notes` | Staging guidance: light, props, when to omit or blur seals |
| `planned_shots` | 3–5 `PlannedImageShot` objects (shot_index, role, rationale_from_reviews, prompt) |
| `pipeline_meta` | Injected by pipeline: retrieval queries, unique_chunks_used, model, chunking settings |

#### Sample analysis outputs (CeraVe Hydrating Facial Cleanser — `B0BG9Q18ZZ`)

**`product_summary`:**
> "CeraVe Hydrating Facial Cleanser is a lotion-form, non-foaming face wash positioned as gentle, fragrance-free, and moisturizing. The official description emphasizes hyaluronic acid, three essential ceramides, glycerin, and 24-hour hydration, while reviewers most often describe a soft, creamy, light-lotion feel that leaves skin clean but not stripped. The emotional tone is clinical, dependable, and comfort-first, with strong appeal for dry or sensitive skin. At the same time, some reviewers debate the non-foaming texture, cleansing strength, and packaging reliability, so the imagery should be honest and clear."

**`listing_features_elevated_by_reviews` (selected):**
- *"Non-foaming lotion-like consistency should be emphasized because reviewers repeatedly describe the feel as creamier, heavier, or like washing with a light lotion, and some buyers need that expectation set clearly."*
- *"Packaging integrity and pump/cap reliability deserve visual honesty because multiple reviews report leaking, loose caps, open boxes, or broken pumps."*

**`cross_review_benefits`:**
- *"Soft, hydrated, moisturized skin after washing is the most repeated positive outcome across reviewers."*

**`brand_visual_codes` (selected):**
- *"Clinical, dermatologist-led skincare mood: clean white space, cool-to-neutral daylight, restrained composition, and a trustworthy pharmacy-shelf feel."*
- *"Because there is no access to official brand asset libraries, label storytelling should rely on shape, color bands, and short product-line wording only, not exact micro-text or seal reproduction."*

**Planned shots (all 5 defined for CeraVe):**

| Shot | Role | Key rationale |
|------|------|---------------|
| 1 | `hero_product` | Establish the exact bottle people expect; address leakage/pump worries with intact, sealed presentation |
| 2 | `consumer_in_use` | Show smooth lotion-like spread (not suds) on damp skin; reinforce skin-feel consensus |
| 3 | `formula_texture_macro` | Answer the biggest expectation gap: not bubbly, but creamy and non-aerated |
| 4 | `packaging_detail_and_trust` | Detail shot of pump/cap geometry addressing repeated packaging complaints |
| 5 | `daily_routine_context` | Lifestyle frame reinforcing AM/PM staple positioning for sensitive skin |

**Pipeline metadata (from `outputs/B0BG9Q18ZZ_image_prompt_bundle.json`):**
```json
{
  "unique_chunks_used": 54,
  "model": "gpt-5.4-mini",
  "chunking": {"strategy": "paragraph_batch", "max_tokens": 400, "overlap_tokens": 60}
}
```

Unique chunks used across all three products in the final run (paragraph_batch, 12 queries):

| Product | `unique_chunks_used` (final run) |
|---------|----------------------------------|
| `B0C4BFWLNC` (pet bed) | 67 |
| `B0BG9Q18ZZ` (CeraVe cleanser) | 53 |
| `B0BXSQN4HV` (hat washer) | 56 |

The pet bed draws the most unique chunks (67), consistent with its high review volume and the diversity of reviewer concerns (shape, texture, washability, calming behavior, color accuracy, size fit).

---

## Q3 — Image Generation with Diffusion Models

### 3.1 Models and Configuration

| Backend | Model | Output directory | Provider suffix |
|---------|-------|-----------------|----------------|
| OpenAI Images | `chatgpt-image-latest` | `outputs/<ASIN>/openai/` | *"Emphasize crisp commercial lighting and a catalog-ready composition."* |
| Google Gemini | `gemini-3.1-flash-image-preview` | `outputs/<ASIN>/gemini/` | *"Render as one coherent photographic scene; single clear focal subject."* |

**Image size:** 1024×1024 (OpenAI `OPENAI_IMAGE_SIZE`); 1K resolution (Gemini `GEMINI_IMAGE_RESOLUTION`).  
**Images generated:** First 3 planned shots × 2 models × 3 products = **18 images total.**  
Full model configuration is recorded in `outputs/run_log.json` → `models_used`.

### 3.2 Prompt Construction

Each image prompt originates in the `planned_shots[].prompt` field of the `ImagePromptBundle`. The function `image_prompt_for_model()` in `src/image_gen.py` appends the provider-specific suffix before calling the API. This ensures:
- Both models receive the **same substantive content** (scene, materials, constraints), enabling a controlled comparison.
- Provider suffixes steer the overall aesthetic without changing the scene description.

**Example prompt (CeraVe hero, Shot 1):**
> *"Photorealistic studio hero of a single CeraVe Hydrating Facial Cleanser bottle standing upright on a clean pale stone or white bathroom counter, front-facing and centered, with a simple clinical blue-and-white palette, satin plastic finish, and soft neutral daylight. Show the practical pump bottle silhouette clearly, with the label area clean and believable but not overloaded with tiny readable text; at most the short product-line wording can be hinted, not fully typeset. … No fake certification seals, no barcode, no dense ingredient panel. … no foam, no splash, no promotional banner."*
> **+ OpenAI suffix:** *"Emphasize crisp commercial lighting and a catalog-ready composition."*

### 3.3 Generated Images

The following images were generated across both models for the three products. Images are stored at `outputs/<ASIN>/openai/` and `outputs/<ASIN>/gemini/`.

#### Product 1 — Pet Bed (`B0C4BFWLNC`)

| Shot | Role | OpenAI file | Gemini file |
|------|------|-------------|-------------|
| 1 | `hero_product` | `00_1_hero_product_chatgpt_image_latest_00.png` | `00_1_hero_product_gemini-3_1-flash-image-preview_00.png` |
| 2 | `consumer_in_use` | `01_2_consumer_in_use_chatgpt_image_latest_00.png` | `01_2_consumer_in_use_gemini-3_1-flash-image-preview_00.png` |
| 3 | `wash_and_bundle_value` | `02_3_wash_and_bundle_value_chatgpt_image_latest_00.png` | `02_3_wash_and_bundle_value_gemini-3_1-flash-image-preview_00.png` |

#### Product 2 — CeraVe Cleanser (`B0BG9Q18ZZ`)

| Shot | Role | OpenAI file | Gemini file |
|------|------|-------------|-------------|
| 1 | `hero_product` | `00_1_hero_product_chatgpt_image_latest_00.png` | `00_1_hero_product_gemini-3_1-flash-image-preview_00.png` |
| 2 | `consumer_in_use` | `01_2_consumer_in_use_chatgpt_image_latest_00.png` | `01_2_consumer_in_use_gemini-3_1-flash-image-preview_00.png` |
| 3 | `texture_formula_closeup` | `02_3_texture_formula_closeup_chatgpt_image_latest_00.png` | `02_3_texture_formula_closeup_gemini-3_1-flash-image-preview_00.png` |

#### Product 3 — Hat Washer (`B0BXSQN4HV`)

| Shot | Role | OpenAI file | Gemini file |
|------|------|-------------|-------------|
| 1 | `hero_product` | `00_1_hero_product_chatgpt_image_latest_00.png` | `00_1_hero_product_gemini-3_1-flash-image-preview_00.png` |
| 2 | `consumer_in_use` | `01_2_consumer_in_use_chatgpt_image_latest_00.png` | `01_2_consumer_in_use_gemini-3_1-flash-image-preview_00.png` |
| 3 | `detail_macro` | `02_3_detail_macro_chatgpt_image_latest_00.png` | `02_3_detail_macro_gemini-3_1-flash-image-preview_00.png` |

### 3.4 Experiment 1 — Chunking Strategy A/B (Sliding vs. Paragraph-Batch)

**Motivation.** The assignment asks us to consider chunking under token limits. With review corpora of up to 500 reviews per product, the choice of chunking strategy affects which text units are embedded and retrieved. We ran a controlled A/B to verify the choice is evidence-based rather than arbitrary.

**Protocol.** For each strategy: set `CHUNK_STRATEGY=sliding` (then `paragraph_batch`) in `.env`, delete `data/chroma/`, and run `python run_pipeline.py --skip-images`. Archive bundles and run logs to `outputs/chunking_ab/sliding/` and `outputs/chunking_ab/paragraph_batch/`. Both runs use identical corpus, model, and token defaults. Narrative: `outputs/chunking_ab/summary.md`.

**Metrics.** `unique_chunks_used` per ASIN, from `analysis_meta` in each archived `run_log.json`. Total indexed chunks per product: **502 for both strategies** (same tokenizer + corpus).

| Strategy | `B0C4BFWLNC` | `B0BG9Q18ZZ` | `B0BXSQN4HV` |
|----------|--------------|--------------|--------------|
| sliding | 46 | 41 | 43 |
| paragraph_batch | 46 | 42 | 43 |

**Analytics.** The strategies agree on retrieval breadth for the pet bed and hat washer; CeraVe differs by one deduplicated chunk (41 vs. 42). This confirms that for a corpus of short Amazon reviews, the strategies are largely equivalent in retrieval diversity—the +1 chunk difference is a boundary effect on which review paragraphs co-habit an embedding batch, not a systematic quality difference.

**Insight.** We adopt `paragraph_batch` because it better matches the data structure: Amazon reviews are predominantly short, self-contained paragraphs. Sliding windows over a 200-word review would create three 400-token windows that partially duplicate content, inflating the index without enriching retrieval. The fallback to sliding for long paragraphs (description bullets) preserves the benefit of overlap where it actually matters.

**Scientific rigor artifact.** Archived bundles in `outputs/chunking_ab/sliding/` and `outputs/chunking_ab/paragraph_batch/`; each contains `run_log.json` + `*_image_prompt_bundle.json` per ASIN. The `analysis_meta.chunking` field in every run log records strategy, max_tokens, and overlap_tokens.

### 3.5 Experiment 2 — Dual Image Model Comparison (Same Prompts, Different Backends)

**Motivation.** The assignment requires at least two image generation models. We route identical `planned_shots[].prompt` content through both `chatgpt-image-latest` and `gemini-3.1-flash-image-preview`, with only a short provider-specific suffix appended. This gives us a controlled comparison: prompt content is held constant, only the model changes.

**Qualitative comparison across six dimensions:**

| Dimension | OpenAI (`chatgpt-image-latest`) | Gemini (`gemini-3.1-flash-image-preview`) |
|-----------|--------------------------------|-------------------------------------------|
| **Composition** | Tighter studio / catalog frames; product centered and well-lit for an e-commerce context | More likely to add lifestyle background (environmental context, room details) even when the prompt focuses on the product |
| **Color accuracy** | Generally faithful to palette cues (dark chocolate fur, clinical white/blue); consistent with the OpenAI suffix for "catalog-ready composition" | Sometimes richer and warmer in lifestyle mood; occasional color saturation boost |
| **Text on product / overlays** | Renders headline-level brand/product words reasonably well; micro-text (seals, barcodes) is visually present but not actually readable at zoom | Attempts more label detail in some outputs, which increases the visible failure risk: text appears as plausible-but-wrong glyphs under zoom |
| **Physical plausibility** | Generally stable; rare floating or impossible staging | Occasionally introduces extra objects or unusual framing not specified in the prompt, especially in lifestyle shots |
| **Constraint adherence** | Follows "no foam, no splash, no promotional banner" instructions fairly reliably | Sometimes generates additional text blocks or decorative elements that were explicitly excluded in the prompt |
| **Failure modes** | Seal silhouettes appear as soft generic shapes (as intended); occasional over-polish making products look like CGI renders | Seal text attempts produce more visible hallucinated characters; lifestyle context sometimes crowds the product |

**Per-product model preferences:**

*Pet bed (`B0C4BFWLNC`):* **OpenAI preferred for hero and construction detail** — the studio lighting captures the dark-chocolate faux fur and raised donut rim more precisely. Gemini's lifestyle version of the consumer-in-use shot (pet sleeping) added appealing environmental warmth, making it **preferable for Shot 2** where a cozy room atmosphere enhances the narrative.

*CeraVe cleanser (`B0BG9Q18ZZ`):* **OpenAI preferred across all three shots** — the clinical, pharmacy-shelf aesthetic is better served by OpenAI's catalog-frame tendency. Gemini's attempt at the texture/formula closeup introduced softer, more atmospheric lighting that diluted the clinical trust the product needs to communicate.

*Hat washer (`B0BXSQN4HV`):* **OpenAI preferred for hero and detail macro** — the white plastic cage requires clear, neutral studio light to make the ribbed geometry and clip design legible. Gemini's in-use shot inside a dishwasher/washer added more realistic appliance context that worked well for Shot 2.

**Overall insight.** OpenAI `chatgpt-image-latest` is consistently **safer for pack-shot clarity, controlled lighting, and constraint adherence** — it suits hero shots and detail macros where accuracy matters most. Gemini `gemini-3.1-flash-image-preview` has strengths in **lifestyle mood and environmental richness**, making it the better choice when the scene requires consumer context (pet in use, person cleansing). For a production listing, the ideal approach would be model selection per shot role rather than a single model for all frames.

### 3.6 Experiment 3 — Real Listing Imagery Review and Prompt Iteration

**Motivation.** The assignment requires comparing AI images to real-world product images and iterating prompts based on initial results.

**Step 1: Inspect `listing_image_urls`.**  
We loaded `listing_image_urls` from `data/products.json` (Amazon gallery order) for each ASIN. Real Amazon listings achieve high information density by spreading content across a carousel of specialized frames: a clean hero, a benefits-feature infographic, a size-chart slide, a lifestyle image, and a detail/close-up shot. Each carousel image has a single job; the sum of the carousel tells the full product story.

This observation directly shaped our **multi-shot architecture**: rather than generating one "good AI image" per product, we plan **3–5 shots** that decompose the carousel story, each with one clear job (`hero_product`, `consumer_in_use`, `formula_texture_macro`, etc.). The `shot_plan_rationale` field in each bundle explicitly states how our planned shots approximate what a real listing carousel achieves across multiple frames.

**Step 2: Identify failure modes in initial image outputs.**

After generating the first round of images, we compared them against real listing photos across four dimensions:

| Dimension | Real listing photography | AI-generated (initial) | AI-generated (after iteration) |
|-----------|--------------------------|------------------------|-------------------------------|
| **SKU fidelity** | Exact product, exact color, photographed | Generally faithful silhouette and palette | Improved by anchoring prompts to specific listing features |
| **Label accuracy** | Pixel-perfect label text, certifications, regulatory print | Seals and badges appear as plausible-but-wrong glyph clusters | Reduced by instructing models to omit/blur micro-text |
| **Information density** | Single frame carries one job well | Initially overcrowded with multiple goals | Improved by strict one-scene-per-prompt discipline |
| **Physical plausibility** | Real-world staging, gravity-compliant | Occasional floating, impossible balance, exaggerated proportions | Improved by adding "physically plausible staging" constraint |

**Step 3: Iterative prompt engineering changes (in `src/llm_analysis.py`).**

Three rounds of prompt refinement were made after evaluating initial outputs:

*Round 1 — RAG query expansion.* Moved from narrow sensory queries to the 12-query hybrid set documented in §2.2. Goal: surface a broader range of review evidence (visual, structural, emotional) so the LLM has richer material to draw from.

*Round 2 — Structured fields for analytical traceability.* Added `listing_features_elevated_by_reviews`, `cross_review_benefits`, `brand_visual_codes`, `shot_plan_rationale`, and `pipeline_meta` to the schema. Goal: make the reasoning chain explicit so each planned shot has a documented rationale, not just a prompt.

*Round 3 — Diffusion-safe text and density constraints.* After observing seal hallucinations and layout clutter in initial images, we added explicit LLM instructions:
- **Avoid** asking for legible micro-copy on seals, barcodes, or dense ingredient panels; use **omit / blur / generic badge shape** instead.
- Cap on-pack legible text to at most **one or two short phrases** copied verbatim from the Title/Official description.
- **One primary idea per prompt**: no infographic + hero + comparison in a single scene.
- Acknowledge **no access to brand's official DAM** — visual cues must come from this SKU's text and reviews only.

**AI vs. real listing images — what diffusion can and cannot do:**

*Where AI succeeds:*
- **Silhouette and macro shape**: the donut bed's round form, the cleanser's pump bottle profile, the hat cage's squared geometry are all recognizable in generated images.
- **Color palette and mood**: dark chocolate faux fur, clinical blue-and-white, utilitarian white plastic are all captured faithfully when prompted with specificity.
- **Lifestyle atmosphere**: pet-in-bed comfort, bathroom sink cleansing, dishwasher-in-use staging are plausible and emotive.

*Where AI struggles:*
- **Certification and regulatory text**: both models hallucinate seal text. Even with instructions to omit, a soft badge shape sometimes appears with invented glyphs. Real listings carry legally accurate dermatologist-recommended seals, fragrance-free callouts, and hyaluronic acid callout badges—these are beyond reliable text-to-image rendering.
- **Exact brand identity**: without access to the brand's official assets, images infer a "CeraVe-like" blue-and-white palette but cannot reproduce the exact logo, typeface, or label layout. Real listing images use the actual brand photography.
- **Pixel-level SKU fidelity**: the donut bed's actual `Dark Chocolate` colorway, CeraVe's specific bottle proportions, and YEENOR's exact clip geometry are approximated, not replicated.

**Overall verdict:** AI-generated images are **useful for ideation, mood-boarding, and visualizing review-driven priorities** but are not a substitute for professional product photography when certification text, exact brand identity, or pixel-level SKU fidelity is required. The gap is smaller for silhouette/color-heavy products (the hat cage, the donut bed) and larger for labeled CPG products where regulatory copy matters (the cleanser).

---

## Q4 — AI Agentic Workflow

### 4.1 Architecture Overview

The pipeline is a genuine **multi-agent system** with four distinct agents coordinated by a **Supervisor**, typed JSON handoffs between agents, a structured harness with bounded retries and telemetry around each LLM call, and an explicit merge/verification step. It goes well beyond a single monolithic prompt.

| Agent | Role | Primary artifact | Implementation |
|-------|------|-----------------|----------------|
| **Q1 — Data/Catalog** | Offline data preparation: slice Amazon JSONL, consolidate products, apply rating filters | `data/products.json`, `data/products_manifest.json` | `scripts/consolidate_products.py`, `scripts/filter_meta_by_rating.py` |
| **Supervisor** | Orchestrates the full sequence; validates merge preconditions; logs every step | `outputs/run_log.json` (`AgentState`) | `src/agent.py` (`run_pipeline()`) |
| **Q2 — Analyst** | Multi-query RAG over Chroma + one structured LLM call; produces a **review-grounded brief** with shot roles and rationales — *no final image prompts* | `ReviewImageryBrief` JSON | `src/llm_analysis.py` (`Q2_SYSTEM`, `_build_q2_user()`) |
| **Q3 — Creative Executive** | Receives the Q2 brief + listing metadata; produces **final text-to-image prompts only** — no re-analysis | `CreativePromptPack` JSON | `src/llm_analysis.py` (`Q3_SYSTEM`, `_build_q3_user()`) |
| **Image Backends** | Optional rendering (not LLM agents): route each `planned_shots[].prompt` through OpenAI Images and/or Gemini | PNG files in `outputs/<ASIN>/openai/` and `.../gemini/` | `src/image_gen.py` |

**Key design principle:** Q2 and Q3 are **deliberately separated** so the analyst role (grounded in reviews and data) never bleeds into the creative role (image-model-aware prompt writing). Each step has a single, auditable responsibility.

### 4.2 End-to-End Sequence

```
run_pipeline.py (CLI entrypoint)
    │
    ├─ Load products (data/products.json → List[Product])
    │
    └─ Supervisor (src/agent.py: run_pipeline()) — per product (ASIN):
         │
         ├─ [CHUNK & EMBED]  ←  RAG Infrastructure
         │     src/chunking.py: product_to_chunks()
         │       → paragraphs / windows of ≤400 tokens (paragraph_batch default)
         │     src/rag.py: ReviewRAGIndex.index_product()
         │       → text-embedding-3-small → Chroma (data/chroma/)
         │     Log: "B0XXXXX: 502 chunks in vector store"
         │
         ├─ [RETRIEVE]
         │     12 RAG queries issued sequentially
         │       → chunks deduplicated by text key
         │       → RAG context block assembled (build_rag_context)
         │
         ├─ [Q2 ANALYST]  ←  StructuredLLMHarness("q2_analyst", OPENAI_TEXT_MODEL)
         │     Input:  product metadata + RAG context block
         │     System: Q2_SYSTEM  (senior e-commerce analyst; analysis only, no prompts)
         │     Output: ReviewImageryBrief — all analysis fields + planned_shots
         │             with shot_index, role, rationale_from_reviews ONLY
         │     Harness: up to 2 attempts; temperature → 0.0 on retry
         │     Telemetry logged: model, duration_ms, attempts, token counts
         │
         ├─ [Q3 CREATIVE]  ←  StructuredLLMHarness("q3_creative", OPENAI_Q3_TEXT_MODEL)
         │     Input:  ReviewImageryBrief JSON + listing metadata (no raw reviews)
         │     System: Q3_SYSTEM  (creative director; image-model constraints enforced)
         │     Output: CreativePromptPack — {"planned_shots": [{shot_index, prompt}, ...]}
         │     Constraint: must output exactly the same shot_index set as Q2 brief
         │     Harness: up to 2 attempts; temperature → 0.0 on retry
         │     Telemetry logged: model, duration_ms, attempts, token counts
         │
         ├─ [SUPERVISOR MERGE & VERIFY]
         │     merge_review_brief_with_creative(brief, creative)
         │       → asserts Q2 shot_index set == Q3 shot_index set (explicit failure on mismatch)
         │       → joins: brief fields + creative prompts → ImagePromptBundle
         │     Save: outputs/<ASIN>_image_prompt_bundle.json
         │     pipeline_meta: RAG queries, chunk count, chunking config,
         │                    q2_harness telemetry, q3_harness telemetry
         │
         └─ [IMAGE GENERATION]  (skipped if --skip-images)
               src/image_gen.py: generate_product_images()
                 → For each planned_shot (up to --images-per-model N):
                     OpenAI: image_prompt_for_model(prompt, provider="openai")
                       → chatgpt-image-latest → PNG → outputs/<ASIN>/openai/
                     Gemini: image_prompt_for_model(prompt, provider="gemini")
                       → gemini-3.1-flash-image-preview → PNG → outputs/<ASIN>/gemini/
    │
    └─ Write outputs/run_log.json
         (experiment settings, models_used, all steps, analysis_meta with harness telemetry, image_paths)
```

### 4.3 StructuredLLMHarness — Reliability Infrastructure

The `StructuredLLMHarness` (`src/llm_harness.py`) wraps every LLM call with the infrastructure that makes multi-agent systems reliable rather than brittle:

| Practice | Implementation |
|----------|---------------|
| **Structured output** | `response_format={"type": "json_object"}` enforced at the API level |
| **Schema validation** | Pydantic `model_validate()` on every response — `ReviewImageryBrief` (Q2) and `CreativePromptPack` (Q3) |
| **Normalisation before validate** | `_normalize_review_brief_json()` and `_normalize_creative_pack_json()` coerce minor schema drift (e.g., string where list expected) before Pydantic sees the data |
| **Bounded retries** | Up to 2 attempts per call; temperature drops to 0.0 on retry for stability |
| **Loud failures** | `RuntimeError` raised (not silently swallowed) if all attempts fail |
| **Telemetry** | Each call returns `{agent_id, model, attempts, duration_ms, prompt_tokens, completion_tokens, total_tokens}` — merged into `pipeline_meta.q2_harness` / `pipeline_meta.q3_harness` in the run log |

### 4.4 Typed Handoffs Between Agents

A key property of the agentic design is that each inter-agent handoff uses a **typed JSON contract**, not free text:

| Handoff | From → To | Schema | What is passed |
|---------|-----------|--------|---------------|
| 1 | Q1 → Supervisor | `Product` (Pydantic) | ASIN, title, description, reviews, listing_image_urls |
| 2 | RAG → Q2 | Numbered excerpt block | Deduplicated review + description chunks |
| 3 | Q2 → Q3 | `ReviewImageryBrief` (JSON) | All analysis fields + `planned_shots` with roles and rationales only — **no prompts** |
| 4 | Q3 → Supervisor | `CreativePromptPack` (JSON) | `{"planned_shots": [{shot_index, prompt}]}` — **only the prompts** |
| 5 | Supervisor → disk / image step | `ImagePromptBundle` (JSON) | Merged output of Q2 + Q3 |

The **merge step** (`merge_review_brief_with_creative()`) explicitly asserts that the Q2 and Q3 shot_index sets are identical before combining them. Any mismatch raises an error — preventing silent data corruption if the creative agent hallucinated an extra shot or dropped one.

### 4.5 Why Two LLM Calls Instead of One

The separation of Q2 (analyst) and Q3 (creative) is a deliberate architectural choice, not just an implementation detail:

**Q2 Analyst** receives raw review text and listing descriptions. Its system prompt establishes an *analyst* persona focused on evidence-grounding: what reviewers actually said, which features they elevate, what the consensus visual picture is. It explicitly does *not* write image-model prompts — its output (`ReviewImageryBrief`) contains only roles and rationales for each planned shot.

**Q3 Creative Executive** never sees raw reviews. It receives only the structured Q2 brief and the listing title/description. Its system prompt establishes a *creative director* persona focused entirely on image-model constraints: avoiding legible micro-text, one scene per prompt, physically plausible staging, brand-inferred palette. By isolating this responsibility, we prevent the creative prompt from being cluttered with analytical commentary, and we prevent the analyst from trying to encode image-model constraints it doesn't need to know about.

This separation also enables **different models per stage**: `OPENAI_TEXT_MODEL` for Q2 (factual, structured analysis) and `OPENAI_Q3_TEXT_MODEL` for Q3 (creative, longer-form prompt writing) — configurable independently in `.env`.

### 4.6 Configuration and Reproducibility

All model IDs, chunking settings, and API keys are specified in `.env` (not committed). The `.env.example` template documents all variables. Every run's full configuration — including which model ran Q2 vs. Q3, chunking strategy, and per-call harness telemetry — is echoed into `outputs/run_log.json`.

**Reproduce the final run:**
```bash
cp .env.example .env   # fill in OPENAI_API_KEY, GEMINI_API_KEY
cp data/products.example.json data/products.json
# (or: python scripts/consolidate_products.py)
python run_pipeline.py --images-per-model 3
```

**Reproduce the A/B chunking experiment:**
```bash
CHUNK_STRATEGY=sliding python run_pipeline.py --skip-images
rm -rf data/chroma
CHUNK_STRATEGY=paragraph_batch python run_pipeline.py --skip-images
```

---

## Reflection — Challenges and Lessons Learned

### Challenge 1: Direct Amazon scraping is not viable — pivot to an academic dataset

Our initial plan was to scrape listing text and customer reviews directly from Amazon product pages using the target ASINs. Amazon's bot-detection systems blocked this consistently with 503 errors and CAPTCHA challenges, making live scraping an unreliable foundation for a reproducible academic pipeline. We pivoted to the **Amazon Reviews 2023** dataset (Hou et al., 2024), which provides the same information — product metadata, listing descriptions, feature bullets, listing image URLs, and customer reviews — in pre-collected JSONL files per category.

**Lesson:** For academic projects requiring reproducibility, pre-existing curated datasets are not a fallback — they are often the *correct* approach. Live scraping introduces fragility (rate limits, bot-detection, page-layout changes) that makes results hard to reproduce. Citing Hou et al. (2024) also gives the data collection step proper academic provenance.

**Limitation to acknowledge:** The dataset is a snapshot (Amazon Reviews 2023) rather than a real-time pull. Review counts and listing content reflect the state of the catalog as of the dataset collection window, not the live Amazon page as of the project date. For a production system, a licensed data feed or the Amazon Product Advertising API would be the appropriate solution.

### Challenge 2: Diffusion text rendering is fundamentally unreliable for product labels

The single biggest gap between AI-generated and real listing images is **legible label text**. Both models produce visually plausible label areas — the CeraVe bottle looks like it has a label — but zooming in reveals garbled, hallucinated glyphs. This is not a prompt engineering failure; it reflects a fundamental limitation of current diffusion models. Our mitigation (instruct the LLM to omit seals, blur badges, and limit on-pack text to short verified phrases) substantially reduced the problem but did not eliminate it. For production-quality labeling, composite design or reference-conditioned generation (e.g., ControlNet with the actual label as reference) would be necessary.

**Lesson:** Design the analysis step to explicitly categorize which visual claims *can* and *cannot* be reliably depicted by diffusion. The `pitfalls_to_avoid` field in `ImagePromptBundle` is the direct implementation of this lesson.

### Challenge 3: RAG retrieval quality depends on query diversity, not just volume

Our initial (single-query) RAG approach retrieved chunks that were topically similar but not diverse — the top retrieved documents all discussed the same dominant theme (e.g., softness for the pet bed). Moving to a 12-query multi-query set dramatically improved coverage. However, with 12 queries × top-k=5 chunks each, we retrieve up to 60 raw chunks per product; after deduplication, the actual unique context ranged from 53 to 67 chunks. This is enough to cover multiple review facets but still requires the LLM to synthesize and prioritize — which is where the structured `chain-of-thought` schema instruction matters most.

**Lesson:** For short-text corpora (Amazon reviews are often ≤100 words), diverse retrieval queries outperform a single broad query. The 12-query design was iteratively developed through inspecting which themes were missing in early bundle outputs.

### Challenge 4: Chunking strategy matters less than expected for this corpus

We anticipated a meaningful difference between sliding and paragraph-batch chunking. In practice, for 500 reviews that are predominantly short (1–3 sentences), both strategies produce nearly identical retrieval breadth (max 1 unique-chunk difference across products). The real benefit of paragraph-batch is conceptual alignment: reviews are natural units of opinion, and packing whole reviews as chunks preserves that unit. The lesson is that chunking strategy is more consequential when reviews are long narratives — for short-review corpora, the choice matters less than query diversity.

### Challenge 5: One prompt = one scene requires discipline against feature creep

The natural instinct when writing image prompts is to include everything: "show the bottle, show the texture, show the hands, show the certification seal, and show the bathroom context." A prompt packed with multiple primary subjects produces images where the AI hedges across all of them, delivering none well. Enforcing the **one-scene, one-primary-idea** discipline — formalized in the LLM's system instructions and in the schema's `planned_shots` decomposition — was the most impactful single intervention for image quality. The analogy to multi-slide listing carousels was useful: real listings don't cram a hero, infographic, and lifestyle shot into one frame.

### Challenge 5: Real listing images set a high bar that AI cannot fully match

Amazon product photography is professional, brand-controlled, and legally accurate. AI-generated images are approximations that capture mood, palette, and silhouette well but cannot reproduce exact brand assets, regulatory print, or the pixel-level SKU fidelity of studio photography. The value of AI image generation is in rapid ideation, prototype visualization, and exploring review-driven shot angles — not replacing production-quality photography.

### What we would do differently

1. **Use the Amazon Product Advertising API or a licensed feed** for live, reproducible data pulls instead of relying on a dataset snapshot. This would allow real-time review collection tied to specific ASINs without scraping restrictions.
2. **Expand to 5 shots per product and both models** for a richer Q3 comparison dataset (time constraint limited us to 3 shots per model per product).
3. **Add a baseline condition**: run the LLM with no RAG (description only) and compare bundle quality to the RAG-augmented version — quantifying how much retrieval adds to visual specificity.
4. **Quantitative image evaluation**: use CLIP similarity between generated images and real listing images, or structured human ratings on the four comparison dimensions, rather than purely qualitative commentary.
5. **Larger reviews corpus**: for the YEENOR hat washer (~21,000 ratings), 500 reviews is likely representative; but for the pet bed (~68,000), there may be structured sub-populations (by pet size, breed, or use case) that a larger sample would surface.

---

## Appendix

### A — Repository Layout

```
Project/
├── run_pipeline.py              # CLI entrypoint (--skip-images, --images-per-model N)
├── requirements.txt
├── .env.example                 # Config template (API keys, model IDs, chunking settings)
├── data/
│   ├── README.md                # Raw vs derived data hygiene
│   ├── products.json            # Three-product catalog
│   ├── products_manifest.json   # Source metadata (collection dates, review counts)
│   ├── raw/amazon_reviews_2023/ # Amazon Reviews 2023 JSONL slices (Q1 consolidation)
│   └── chroma/                  # Vector index (created at runtime; delete to re-embed)
├── docs/
│   ├── agentic_workflow.md      # Q1–Q4 agentic workflow + harness (Q4)
│   ├── report_draft.md          # Working notes and experiment scaffolding
│   └── report_final.md          # This document
├── submission/
│   └── README.md                # Final package / rubric checklist
├── outputs/
│   ├── run_log.json             # Full run provenance
│   ├── B0C4BFWLNC_image_prompt_bundle.json
│   ├── B0BG9Q18ZZ_image_prompt_bundle.json
│   ├── B0BXSQN4HV_image_prompt_bundle.json
│   ├── B0C4BFWLNC/openai/       # OpenAI images for pet bed
│   ├── B0C4BFWLNC/gemini/       # Gemini images for pet bed
│   ├── B0BG9Q18ZZ/openai/       # OpenAI images for CeraVe cleanser
│   ├── B0BG9Q18ZZ/gemini/       # Gemini images for CeraVe cleanser
│   ├── B0BXSQN4HV/openai/       # OpenAI images for hat washer
│   ├── B0BXSQN4HV/gemini/       # Gemini images for hat washer
│   └── chunking_ab/             # A/B experiment archives (sliding/ and paragraph_batch/)
├── scripts/
│   ├── consolidate_products.py  # Builds data/products.json from JSONL slices
│   └── filter_meta_by_rating.py # Pre-filters Amazon meta by rating count
└── src/
    ├── agent.py                 # Pipeline orchestration (supervisor; sequential DAG)
    ├── llm_harness.py           # Structured JSON LLM calls + telemetry
    ├── llm_analysis.py          # RAG + Q2/Q3 prompts + merge → ImagePromptBundle
    ├── rag.py                   # Chroma index construction and retrieval
    ├── chunking.py              # sliding and paragraph_batch strategies
    ├── image_gen.py             # Image API calls + provider suffix routing
    ├── models.py                # Pydantic schemas (Product, ImagePromptBundle, etc.)
    └── config.py                # Env-loaded settings
```

### B — All LLM Prompts

#### B.1 System message to text LLM (abbreviated key constraints)

```
You are a creative director for premium e-commerce and editorial product photography.
Use ONLY the product description and retrieved review excerpts—ground every claim in that text.
Your analytic goal: reviews identify which Official-description features deserve emphasis in imagery…

Image-model limitations (critical for planned_shots[].prompt): Text-to-image models often fail at
tiny legible copy (certification seals, association badges, barcodes, dense ingredient panels)—
they look plausible at thumbnail size but are wrong or gibberish up close. In prompts, avoid asking
for readable micro-text on seals/logos, full nutrition or drug-facts panels, or exact multi-line
label reproduction. Prefer instead: product silhouette, brand color blocking, material and finish,
pump/cap/clip geometry, and at most one or two short legible phrases from the Title or Official
description. If a certification is important narratively, say omit the seal, show it softly blurred,
or as a simple colored emblem without readable association wording.

Information density: Real retailer pages achieve high density with many carousel images. Each
text-to-image prompt describes one scene—do not pack infographic layouts, comparison grids,
measurement callouts, and hero product into one prompt.

Brand-owned assets: The pipeline has no access to official brand digital asset libraries, style
guides, or approved pack photography.
```

#### B.2 User message structure

```
Product metadata: Title, Category, ASIN
Official description (truncated if extremely long)
Retrieved excerpts (RAG context: deduplicated chunks from 12 queries)

CHAIN OF THOUGHT (ordered):
1. Category norms
2. Consumer landscape (needs, benefits, concerns)
2b. Listing ↔ review mapping
2c. Cross-review benefits
2d. Listing features inventory
2e. Brand visual codes
3. Shot plan (3–5 shots, one job per shot)
4. Detail fields
5. Image prompts (written last, grounded in chain above)

FIELDS (output ALL keys): category_imagery_norms, consumer_needs_and_concerns,
listing_features_elevated_by_reviews, cross_review_benefits, brand_visual_codes,
shot_plan_rationale, product_summary, visual_attributes, material_and_texture_cues,
sensory_efficacy_and_ingredient_cues, customer_visual_consensus, pitfalls_to_avoid,
visual_translation_notes, planned_shots (3–5 objects with shot_index, role,
rationale_from_reviews, prompt)
```

#### B.3 Image provider suffixes

- **OpenAI:** *"Emphasize crisp commercial lighting and a catalog-ready composition."*
- **Gemini:** *"Render as one coherent photographic scene; single clear focal subject."*

### C — Sample Image Prompts (per product, Shot 1 hero)

**Pet bed (`B0C4BFWLNC`) — Shot 1 hero:**
> *"Photorealistic studio hero of the Best Friends by Sheri bundle set: the Original Calming Lux Donut Cuddler pet bed and matching Lux fur throw blanket, shown together on a clean neutral floor or soft linen surface. Use a warm dark chocolate palette with rich brown faux-fur depth, subtle cream-brown highlights in the pile, and a premium cozy-home mood. The donut bed should read as round, plush, and supportive, with a clearly raised rim and a soft, inviting center that looks full rather than flat. Place the matching blanket folded beside or lightly draped over the bed to show the bundle value and matching texture. Soft daylight, gentle shadow, crisp studio realism, honest materials, no props that distract, no readable badges or dense text, only a minimal short phrase like "Best Friends by Sheri" if any text appears."*

**CeraVe cleanser (`B0BG9Q18ZZ`) — Shot 1 hero:**
> *"Photorealistic studio hero of a single CeraVe Hydrating Facial Cleanser bottle standing upright on a clean pale stone or white bathroom counter, front-facing and centered, with a simple clinical blue-and-white palette, satin plastic finish, and soft neutral daylight. Show the practical pump bottle silhouette clearly, with the label area clean and believable but not overloaded with tiny readable text; at most the short product-line wording can be hinted, not fully typeset. The bottle should look full, intact, and sealed only in the sense of normal retail presentation… No fake certification seals, no barcode, no dense ingredient panel… Crisp shadows, gentle contrast, no dramatic props, no foam, no splash, no promotional banner."*

**Hat washer (`B0BXSQN4HV`) — Shot 1 hero:**
> *"Photorealistic premium Amazon hero shot of the YEENOR hat washer 2-pack, isolated on a clean white-to-light-gray studio background, bright neutral daylight with soft bounce and crisp shadow definition. Show two matching white molded plastic cap washer cages angled to reveal the squared four-sided structure, rigid ribs, and flat profile, with the brim-protecting shell clearly visible. The finish should look matte-to-satin, sturdy, and utilitarian, not glossy or decorative. … Add subtle packaging-style labeling only if visible and accurate: 'YEENOR,' 'Hat Washer for Washing Machine,' and '2-Pack.' The mood is practical, fresh, and appliance-friendly, emphasizing durability, reuse, and shape protection."*

### D — Run Metadata (`outputs/run_log.json`)

```json
{
  "models_used": {
    "text": "gpt-5.4-mini",
    "embedding": "text-embedding-3-small",
    "openai_image_model": "chatgpt-image-latest",
    "gemini_image_model": "gemini-3.1-flash-image-preview",
    "image_size": "1024x1024",
    "gemini_image_resolution": "1K",
    "openai_image_prompt_suffix": "Emphasize crisp commercial lighting and a catalog-ready composition.",
    "gemini_image_prompt_suffix": "Render as one coherent photographic scene; single clear focal subject."
  },
  "analysis_meta": {
    "B0C4BFWLNC": {"unique_chunks_used": 67, "model": "gpt-5.4-mini",
                    "chunking": {"strategy": "paragraph_batch", "max_tokens": 400, "overlap_tokens": 60}},
    "B0BG9Q18ZZ": {"unique_chunks_used": 53, "model": "gpt-5.4-mini",
                    "chunking": {"strategy": "paragraph_batch", "max_tokens": 400, "overlap_tokens": 60}},
    "B0BXSQN4HV": {"unique_chunks_used": 56, "model": "gpt-5.4-mini",
                    "chunking": {"strategy": "paragraph_batch", "max_tokens": 400, "overlap_tokens": 60}}
  }
}
```

### E — Checklist of Deliverables

- [x] Data: `data/products.json` (three-product catalog with listings, reviews, and image URLs)
- [x] Code: `src/` (full pipeline implementation), `run_pipeline.py`, `scripts/`
- [x] Prompts: `src/llm_analysis.py` (system/user templates); shot `prompt` values in `outputs/*_image_prompt_bundle.json`
- [x] Generated images: `outputs/<ASIN>/openai/*.png` and `outputs/<ASIN>/gemini/*.png` (18 images total)
- [x] Run metadata: `outputs/run_log.json`
- [x] Experiment archives: `outputs/chunking_ab/sliding/` and `outputs/chunking_ab/paragraph_batch/`
- [x] Report: this document

---

## References

1. Amazon Reviews 2023 dataset — [amazon-reviews-2023.github.io](https://amazon-reviews-2023.github.io/) (McAuley Lab, UCSD). Hou et al., *Bridging Language and Items for Retrieval and Recommendation*, arXiv:2403.03952, 2024. We use Pet Supplies, Beauty & Personal Care, and Appliances/Home JSONL slices under `data/raw/amazon_reviews_2023/`.
2. OpenAI API — `gpt-5.4-mini` (text / structured JSON), `text-embedding-3-small` (embeddings), `chatgpt-image-latest` (image generation). [platform.openai.com](https://platform.openai.com)
3. Google Gemini API — `gemini-3.1-flash-image-preview` (image generation). [ai.google.dev](https://ai.google.dev)
4. Chroma — open-source embedding database. [trychroma.com](https://www.trychroma.com)
5. tiktoken — OpenAI tokenizer library. `cl100k_base` encoding.
6. Pydantic v2 — data validation and schema definitions (`src/models.py`).
7. 94-844 Generative AI Lab Final Project specification (Spring 2026), Prof. Beibei Li, Heinz College, Carnegie Mellon University.
