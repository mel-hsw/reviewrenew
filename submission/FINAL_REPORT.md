---
geometry: margin=1in
header-includes:
  - \usepackage{graphicx}
---
# Final Report: Generating Product Images from Customer Reviews

### Mohammad Taha Zakir, Mel Wong, Asli Gulcur, Smridhi Patwari, Vishnu Bala

## 1. Project Overview & Product Selection (Q1)

### 1.1 Selection Rationale
This project evaluates a generative AI pipeline using three distinct products. Beyond category diversity, we applied a **backward-design criterion**: each product needed to create meaningfully different challenges for the pipeline so that comparisons in Q3 would be analytically interesting rather than redundant. The three products span soft-goods texture, labeled CPG packaging, and rigid-plastic utility, each presenting a distinct failure mode for diffusion models.

*   **Pet Bed (Best Friends by Sheri Calming Lux Donut Cuddler + Blanket Bundle):** Evaluates the diffusion model's ability to render soft-goods texture (fur, plushness, loft). Reviews extensively discuss whether the bed looks as plush in-person as in photos, color accuracy ("Dark Chocolate"), washability, and rim loft. These are exactly the visual trust signals an AI image generation system must get right.
*   **CeraVe Hydrating Facial Cleanser 16 fl oz:** Tests a standard labeling challenge, requiring models to render a common CPG bottle profile without hallucinating illegible micro-text. The label-heavy bottle creates a classic diffusion challenge: models attempt to render ingredient panels and certification seals, producing plausible-but-wrong glyphs.
*   **YEENOR Hat Washer Cage 2-Pack:** Tests the model's capability to understand and accurately render a complex, 3-dimensional rigid plastic structure. A molded plastic cage with a specific 3D silhouette (four-sided, ribbed, clip-secured) is a strong geometry test for diffusion models.

### 1.2 Data Collection Protocol

**Attempted direct scraping (blocked).** Our first approach was to scrape listing text and reviews directly from Amazon product pages. Amazon's bot-detection systems returned 503 errors and CAPTCHA challenges consistently, making live scraping unreliable for a reproducible academic pipeline.

**Final source: Amazon Reviews 2023 (Hou et al., 2024).** We pivoted to the Amazon Reviews 2023 dataset, which provides per-category JSONL files containing product metadata (title, description, feature bullets, listing image URLs) and user reviews (text, rating, date). We downloaded the relevant category slices (Pet Supplies, Beauty & Personal Care, and Home) and filtered to the three target products.

| Item | Detail |
| :--- | :--- |
| **Primary source** | Amazon Reviews 2023 (Hou et al., 2024; McAuley Lab, UCSD) |
| **Data access** | Pre-collected academic dataset; direct scraping was blocked by bot-detection |
| **Review filter** | `len(text) > 20`; newest first; cap at 500 reviews per product |
| **Fields captured** | `text`, `rating`, `title`, `date` (reviews); `title`, `description`, `listing_image_urls`, `category` (listing) |
| **Review volume** | Pet Bed: ~68,128 ratings; CeraVe: thousands; Hat Washer: ~21,076 ratings |

**Lesson:** For academic projects requiring reproducibility, pre-existing curated datasets are often the correct approach. Live scraping introduces fragility (rate limits, bot-detection, layout changes) that makes results hard to reproduce. The dataset is a snapshot (Amazon Reviews 2023) rather than a real-time pull. For a production system, a licensed data feed or the Amazon Product Advertising API would be appropriate.

## 2. Text Analysis & Retrieval-Augmented Generation (Q2)

### 2.1 Overview
The Q2 step transforms raw listing text and customer reviews into a structured, visually actionable analysis. The output is a typed **`ImagePromptBundle`** schema that bridges text evidence to image-generation prompts. The LLM must: (1) extract which visual cues reviews consistently describe, (2) distinguish consensus from outliers, (3) map listing features to review evidence, and (4) produce 3–5 planned shots per product with rationales grounded in review text.

### 2.2 RAG Infrastructure

*   **Vector Database:** Product descriptions and customer reviews were chunked using paragraph batching to preserve semantic context, embedded using OpenAI's `text-embedding-3-small` model, and stored in a persistent ChromaDB instance (`data/chroma/`).
*   **Chunking Strategy:** We use `tiktoken` (`cl100k_base`) to measure token counts. Two strategies were evaluated in an A/B comparison:
    *   *Sliding window:* overlapping windows of 400 tokens with 60-token overlap.
    *   *Paragraph batch (default):* splits on blank lines, packs whole paragraphs up to 400 tokens. Better matches Amazon reviews, which are predominantly short, self-contained paragraphs. Sliding windows over a 200-word review would create overlapping chunks that duplicate content without enriching retrieval.
*   **Multi-Query Retrieval:** Rather than a single broad query, we issue **12 targeted retrieval queries** covering complaints, benefits, expectation gaps, visual/material properties, sensory outcomes, listing-review mapping, and brand mentions. This forces the retriever to surface different facets of the review corpus, yielding 53–67 unique deduplicated chunks per product.

### 2.3 LLM Analysis Agent

We deployed `gpt-5.4-mini` (temperature 0.35) as the Insights Analyst. Its system prompt positions it as a senior e-commerce analyst with explicit chain-of-thought instructions:

1.  **Category norms:** what shot types persuade shoppers in this category
2.  **Consumer landscape:** needs, benefits, concerns from retrieved excerpts
3.  **Listing vs. review mapping:** which official bullets reviewers elevate or dispute
4.  **Cross-review benefits:** independent agreement across multiple reviewers
5.  **Brand visual codes:** on-brand palette and mood cues inferred from text
6.  **Shot plan:** 3–5 shots, one clear job per shot
7.  **Prompts:** written last, fully grounded in the chain above

Output is forced via `response_format: {"type": "json_object"}` into a validated `ReviewImageryBrief` Pydantic schema. This ensures **analytical traceability**: every planned shot includes a `rationale_from_reviews` field that can be audited.

### 2.4 Sample Analysis Output: CeraVe Cleanser

**Product summary:** *"CeraVe Hydrating Facial Cleanser is a lotion-form, non-foaming face wash positioned as gentle, fragrance-free, and moisturizing. Reviewers most often describe a soft, creamy, light-lotion feel that leaves skin clean but not stripped. The emotional tone is clinical, dependable, and comfort-first."*

**Key review insights surfaced:**

*   *Non-foaming lotion-like consistency* should be emphasized. Reviewers repeatedly describe the feel as creamier than expected, and some buyers need that expectation set clearly.
*   *Packaging integrity* deserves visual honesty. Multiple reviews report leaking, loose caps, open boxes, or broken pumps.
*   *Soft, hydrated, moisturized skin* after washing is the single most repeated positive outcome.

**Planned shots for CeraVe:**

| Shot | Role | Key Rationale |
| :--- | :--- | :--- |
| 1 | Hero product | Establish the exact bottle; address leakage/pump worries with intact, sealed presentation |
| 2 | Consumer in-use | Show smooth lotion-like spread (not suds) on damp skin; reinforce skin-feel consensus |
| 3 | Formula texture macro | Answer the biggest expectation gap: not bubbly, but creamy and non-aerated |
| 4 | Packaging detail | Detail shot of pump/cap geometry addressing repeated packaging complaints |
| 5 | Daily routine context | Lifestyle frame reinforcing AM/PM staple positioning for sensitive skin |

**Unique chunks used across products (paragraph\_batch, 12 queries):**

| Product | Unique Chunks Used |
| :--- | :--- |
| Pet Bed | 67 |
| CeraVe Cleanser | 53 |
| Hat Washer | 56 |

The pet bed draws the most unique chunks (67), consistent with its high review volume and diversity of concerns (shape, texture, washability, color accuracy, size fit).

## 3. The Generative Vision Pipeline (Q3 & Q4)
The architecture utilizes an Autonomous Multi-Agent Workflow rather than a linear prompt mechanism, incorporating a self-correcting feedback loop to address known diffusion model limitations.

### 3.1 The Creative Director Agent
The pipeline passes the structured JSON brief to a secondary text LLM acting as the creative director. This agent ingests the extracted consumer features, identified pitfalls, and brand guidelines to construct 3 to 5 optimized text-to-image prompts tailored to specific shot roles (e.g., Hero Shot, In-Use Shot, Macro Detail Shot).

**Prompt Analysis:** The Creative Director produces prompts that function similarly to technical photography briefs, with three key strategies:

*   **Lighting & Staging:** Prescribes exact lighting conditions (e.g., "soft neutral daylight") and details the scene composition for each shot role.
*   **Material Textures:** Specifies surface descriptors (e.g., "rich brown faux-fur depth", "satin plastic finish") to guide diffusion model rendering.
*   **Negative Constraints:** Embeds explicit exclusions (e.g., "no fake certification seals, no barcode, no dense ingredient panel") to suppress hallucinated text and emphasize product silhouette.

### 3.2 The Critic Agent
Text-to-image models (`dall-e-3`, `gemini-3.1-flash-image`) frequently introduce artifacts, including unreadable text, anatomical errors, or floating elements. To mitigate these issues, we implemented a Vision-Language Model (VLM) using the `gpt-5-mini` Responses API to perform automated quality assurance, operating in three stages:

*   **Evaluation:** Upon generation, DALL-E 3 draft images were converted to base64 encoding and evaluated by the VLM against the intended text prompt.
*   **Constraint Criteria:** The VLM's system prompt was configured to prioritize standard e-commerce visual guidelines. It was instructed to reject images containing structural errors such as anatomical impossibilities or excessive hallucinated text.
*   **Self-Correction:** The VLM prioritized photorealism and structural integrity over strict textual literalism. When an image failed evaluation, it generated a revised prompt (`suggested_prompt_revision`), archived the invalid image to a `/rejected/` directory, and fed the revision back into the image generator.

## 4. Results & Comparative Analysis
*   **AI vs. Reality:** The pipeline successfully approximated silhouettes, colors, and the general lifestyle context (e.g., the hat washer's ribbed geometry and the pet bed's faux fur). However, the models cannot reliably achieve pixel-perfect SKU fidelity or accurate regulatory labeling, as they inherently alter precise micro-text and branding.
*   **Model Comparison:** We compared outputs from OpenAI (`chatgpt-image-latest`) and Gemini (`gemini-3.1-flash-image-preview`), observing distinct aesthetic tendencies:
    *   **OpenAI** performed better for tightly framed, catalog-style studio compositions (hero shots and macro details). It accurately rendered the clinical aesthetic of the CeraVe bottle and the geometric lines of the Hat Washer.
    *   **Gemini** consistently introduced richer environmental contexts. It was more suitable for "lifestyle" or "in-use" configurations, generating a warmer setting for the Pet Bed consumer-in-use shots compared to OpenAI's restricted framing.

    The hero-shot outputs for all three products are compared side-by-side in Figure 1.

```{=latex}
\begin{figure}[ht]
\centering
\small
\renewcommand{\arraystretch}{1.5}
\begin{tabular}{p{2.8cm} p{3.5cm} p{3.5cm} p{3.5cm}}
\hline
\textbf{Product} & \textbf{Original} & \textbf{OpenAI (dall-e-3)} & \textbf{Gemini (flash-img)} \\
\hline
\textbf{Pet Bed} &
  \includegraphics[width=3.3cm,height=3.3cm,keepaspectratio]{/Users/tahazakir/Documents/reviewrenew/data/og-product-images/pet-bed.jpg} &
  \includegraphics[width=3.3cm,height=3.3cm,keepaspectratio]{/Users/tahazakir/Documents/reviewrenew/outputs/run_20260421_000114/B0C4BFWLNC/openai/00_1_hero_product_try1_gpt_image_1_00.png} &
  \includegraphics[width=3.3cm,height=3.3cm,keepaspectratio]{/Users/tahazakir/Documents/reviewrenew/outputs/run_20260421_000114/B0C4BFWLNC/gemini/00_1_hero_product_try1_gemini-3_1-flash-image-preview_00.png} \\
\hline
\textbf{CeraVe Cleanser} &
  \includegraphics[width=3.3cm,height=3.3cm,keepaspectratio]{/Users/tahazakir/Documents/reviewrenew/data/og-product-images/cerave.jpg} &
  \includegraphics[width=3.3cm,height=3.3cm,keepaspectratio]{/Users/tahazakir/Documents/reviewrenew/outputs/run_20260421_000114/B0BG9Q18ZZ/openai/00_1_hero_product_try1_gpt_image_1_00.png} &
  \includegraphics[width=3.3cm,height=3.3cm,keepaspectratio]{/Users/tahazakir/Documents/reviewrenew/outputs/run_20260421_000114/B0BG9Q18ZZ/gemini/00_1_hero_product_try1_gemini-3_1-flash-image-preview_00.png} \\
\hline
\textbf{Hat Washer} &
  \includegraphics[width=3.3cm,height=3.3cm,keepaspectratio]{/Users/tahazakir/Documents/reviewrenew/data/og-product-images/cap-clean.jpg} &
  \includegraphics[width=3.3cm,height=3.3cm,keepaspectratio]{/Users/tahazakir/Documents/reviewrenew/outputs/run_20260421_000114/B0BXSQN4HV/openai/00_1_hero_product_try1_gpt_image_1_00.png} &
  \includegraphics[width=3.3cm,height=3.3cm,keepaspectratio]{/Users/tahazakir/Documents/reviewrenew/outputs/run_20260421_000114/B0BXSQN4HV/gemini/00_1_hero_product_try1_gemini-3_1-flash-image-preview_00.png} \\
\hline
\end{tabular}
\caption{Hero-shot comparison across all three products: original product photo vs.\ OpenAI (\texttt{dall-e-3}) vs.\ Gemini (\texttt{flash-img}).}
\label{fig:comparison}
\end{figure}
```
*   **Role of the Feedback Loop:** Comparing the images in the `/rejected/` folders with their accepted counterparts demonstrates the utility of the VLM feedback loop. Initial text prompts frequently resulted in physically implausible staging or dense, hallucinated product labels. The VLM identified these artifacts and revised the prompts to alter the staging, specifically enforcing "silhouette blocking" over legible text, which yielded higher quality final outputs.

### 4.1 Experiment 1: Chunking Strategy A/B (Sliding vs. Paragraph-Batch)

We ran a controlled A/B comparison of both chunking strategies on the identical three-product corpus, deleting `data/chroma/` between runs to ensure a clean slate. Both runs used identical corpus, model, and token defaults (400 max tokens, 60 overlap).

| Strategy | Pet Bed | CeraVe | Hat Washer | Total Indexed Chunks |
| :--- | :--- | :--- | :--- | :--- |
| Sliding window | 46 | 41 | 43 | 502 |
| Paragraph batch | 46 | 42 | 43 | 502 |

The strategies agree on retrieval breadth for two of three products; CeraVe differs by only one deduplicated chunk, a boundary effect rather than a systematic quality difference. We adopted **paragraph batch** as the default because Amazon reviews are predominantly short, self-contained paragraphs. Sliding windows over a 200-word review create overlapping chunks that inflate the index without enriching retrieval. The fallback to sliding for long paragraphs preserves the benefit of overlap only where it actually matters.

### 4.2 Experiment 2: Dual Image Model Comparison (Same Prompts, Two Backends)

Both `chatgpt-image-latest` and `gemini-3.1-flash-image-preview` received identical `planned_shots[].prompt` content with only a short provider-specific suffix appended, enabling a controlled comparison where prompt content is held constant and only the model changes. The hero-shot outputs are shown side-by-side in Figure 1.

| Dimension | OpenAI (`chatgpt-image-latest`) | Gemini (`gemini-3.1-flash-image-preview`) |
| :--- | :--- | :--- |
| **Composition** | Tighter studio/catalog frames; product well-lit for e-commerce | Adds lifestyle background and environmental context |
| **Color accuracy** | Faithful to palette cues; consistent with "catalog-ready" suffix | Richer, warmer in lifestyle mood; occasional saturation boost |
| **Text on product** | Headline-level words rendered reasonably; micro-text not readable | Attempts more label detail, increasing visible hallucination risk |
| **Constraint adherence** | Follows "no foam, no splash" instructions reliably | Sometimes generates excluded text blocks or decorative elements |
| **Failure modes** | Seal silhouettes appear as soft generic shapes (as intended) | Hallucinated label characters; lifestyle context sometimes crowds product |

**Per-product preferences:** OpenAI was preferred for hero and detail shots on all three products. The studio lighting better captured the CeraVe clinical aesthetic and the hat washer's ribbed geometry. Gemini was preferred for consumer-in-use shots where lifestyle warmth enhanced the narrative (e.g., the pet sleeping in the bed, the bathroom cleansing routine). **Overall verdict:** OpenAI is safer for pack-shot clarity and constraint adherence; Gemini excels at lifestyle mood and environmental richness. A production pipeline would select the model per shot role.

### 4.3 Experiment 3: Real Listing Imagery Review & Prompt Iteration

We compared AI outputs against real Amazon listing photos across four dimensions and iterated the prompt engineering in three rounds:

| Dimension | Real Listing Photography | AI (Initial) | AI (After Iteration) |
| :--- | :--- | :--- | :--- |
| **SKU fidelity** | Exact product, photographed | Generally faithful silhouette | Improved by anchoring to listing features |
| **Label accuracy** | Pixel-perfect text and certifications | Seals appear as plausible-but-wrong glyphs | Reduced by instructing models to omit/blur micro-text |
| **Information density** | One frame, one job | Initially overcrowded | Improved by strict one-scene-per-prompt discipline |
| **Physical plausibility** | Real-world, gravity-compliant staging | Occasional floating or impossible balance | Improved by adding "physically plausible staging" constraint |

*   **Round 1 (RAG query expansion):** Moved from narrow sensory queries to the 12-query hybrid set, surfacing a broader range of review evidence (visual, structural, emotional).
*   **Round 2 (Structured fields for traceability):** Added `listing_features_elevated_by_reviews`, `cross_review_benefits`, `brand_visual_codes`, and `shot_plan_rationale` to the schema so each shot has a documented rationale.
*   **Round 3 (Diffusion-safe constraints):** After observing seal hallucinations, added explicit LLM instructions to avoid legible micro-copy on seals and barcodes, cap on-pack text to at most one or two short phrases, and enforce one primary idea per prompt.

The above experiments demonstrate that the visual comparison table captures only a snapshot of a larger iterative process. The final outputs shown above reflect the converged state after all three rounds of refinement.

### 4.4 VLM Self-Correction in Action

The feedback loop was critical to achieving the final output quality. Beyond the prompt iteration rounds above, the VLM critic operated at inference time, evaluating each generated image and either accepting it or producing a revised prompt for re-generation. Figure 2 illustrates one such correction: the VLM detected an anatomical impossibility (an extra arm), rejected the draft, revised the prompt to remove ambiguous staging, and generated an anatomically correct replacement.

As shown in Figure 2, the left image was rejected due to an anatomical failure (an extra arm pumping the lotion), a common diffusion artifact. The VLM identified this error, revised the prompt to remove the ambiguous staging, and produced the anatomically correct accepted image on the right.

```{=latex}
\begin{center}
\begin{tabular}{cc}
\includegraphics[width=6.5cm,height=6cm,keepaspectratio]{/Users/tahazakir/Documents/reviewrenew/outputs/B0BG9Q18ZZ/gemini/rejected/lol_what.png} &
\includegraphics[width=6.5cm,height=6cm,keepaspectratio]{/Users/tahazakir/Documents/reviewrenew/outputs/run_20260421_000114/B0BG9Q18ZZ/gemini/accepted_improved_after_rejected.png} \\
\small\textit{Rejected: anatomical error (extra arm)} & \small\textit{Accepted: corrected after VLM revision} \\
\end{tabular}
\par\smallskip
\small\textbf{Figure 2:} VLM self-correction example (CeraVe, Gemini). Left: rejected draft with hallucinated extra arm. Right: accepted output after prompt revision.
\end{center}
```

### 4.5 Quantitative Hero Shot Evaluation

To move beyond qualitative comparison, we scored all 6 hero shots (3 products × 2 models) using a VLM evaluator (`gpt-5.4-mini`) against a 5-dimension rubric on a 1–5 integer scale. Scores were generated by `scripts/evaluate_hero_shots.py` and saved to `outputs/run_20260421_000114/hero_shot_evaluation.json`.

```{=latex}
\begin{table}[ht]
\centering
\small
\renewcommand{\arraystretch}{1.4}
\begin{tabular}{p{2.6cm} p{1.4cm} p{1.4cm} p{1.6cm} p{1.4cm} p{1.6cm} p{1.6cm} p{1.4cm}}
\hline
\textbf{Product} & \textbf{Model} & \textbf{Color} & \textbf{Silhouette} & \textbf{Realism} & \textbf{Label/Text} & \textbf{Staging} & \textbf{Overall} \\
\hline
Pet Bed       & OpenAI & 5 & 4 & 4 & 5 & 5 & 4.6 \\
Pet Bed       & Gemini & 5 & 4 & 5 & 5 & 4 & 4.6 \\
CeraVe        & OpenAI & 5 & 4 & 4 & 5 & 5 & 4.6 \\
CeraVe        & Gemini & 5 & 5 & 4 & 5 & 5 & 4.8 \\
Hat Washer    & OpenAI & 5 & 5 & 4 & 5 & 5 & 4.8 \\
Hat Washer    & Gemini & 5 & 5 & 4 & 5 & 5 & 4.8 \\
\hline
\textbf{Mean} & & \textbf{5.0} & \textbf{4.5} & \textbf{4.2} & \textbf{5.0} & \textbf{4.8} & \textbf{4.7} \\
\hline
\end{tabular}
\caption{Hero shot scores across 5 dimensions (1--5 scale). Evaluated by \texttt{gpt-5.4-mini} VLM against each product's generation prompt and pitfall list.}
\label{tab:scores}
\end{table}
```

**Analysis of findings.** All six images scored 4.6 or higher overall, confirming that the pipeline's prompt engineering (particularly the three rounds of iterative refinement and the VLM self-correction loop) successfully produced catalog-quality outputs. Color palette accuracy and label/text handling both achieved a perfect mean of 5.0, demonstrating that explicitly instructing the models to match brand colorways and omit dense micro-text was highly effective. The consistency across both models on these dimensions validates the shared prompt strategy rather than any model-specific capability.

The most meaningful differentiation appears in two dimensions. **Photorealism** is the lowest-scoring dimension (mean 4.2), and notably the only one where Gemini outperforms OpenAI on the Pet Bed (5 vs 4), consistent with Gemini's tendency toward richer environmental lighting that can read as more photographically natural for lifestyle products. Conversely, **Silhouette fidelity** shows OpenAI lagging on softer products (Pet Bed: 4, CeraVe: 4) while both models score 5 on the rigid Hat Washer, confirming that diffusion models render hard-edged geometry more reliably than organic soft-goods forms. **Staging and composition** is the dimension where the two models diverge most visibly. Gemini's looser environmental framing cost it a point on Pet Bed staging (4 vs 5), while its richer context helped CeraVe (5 vs 5, tied).

The near-identical overall scores (4.6–4.8) across models reflect the controlled prompt design: both models received the same substantive prompt, so differences are driven by each model's rendering aesthetic rather than by information asymmetry.

**Recommendations based on scores:**

*   **Photorealism (weakest dimension at 4.2):** For studio-style hero shots, request "subtle surface micro-texture, slight label wear, natural shadow variation" to avoid the AI over-polish effect that evaluators flag as CGI-like.
*   **Silhouette fidelity (second weakest at 4.5):** Add explicit shape-reference language to hero prompts for soft-goods products (e.g., "fully open donut rim clearly raised, inner nest visible, asymmetric natural fur flow") to push silhouette scores from 4 to 5.
*   **Model selection per shot role:** For soft-goods lifestyle shots (Pet Bed consumer-in-use), prefer Gemini's richer environmental lighting. For rigid-product hero shots (Hat Washer, CeraVe), both models are equivalent and OpenAI's tighter catalog framing is marginally safer.
*   **Label/Text and Color are solved problems:** The current prompt constraints (omit seals, one short verified phrase, explicit palette cues) are sufficient. No further refinement is needed on these dimensions.

## 5. Reflection: Challenges & Lessons Learned

Developing this multimodal workflow highlighted three fundamental challenges in transitioning from simple prompting to robust agentic engineering:

*   **VLM Over-Constraint:** Initial iterations of the VLM judge were too strict, frequently rejecting high-quality, photorealistic images for missing minor background details. We shifted our prompting strategy to explicitly tolerate deviations from textual literalism, provided the image demonstrated strong photorealism and accurate physics. Once recalibrated, the false-rejection rate plummeted and the pipeline retained visually striking assets (see also Figure 2).

*   **Networking & Infrastructure:** Building reliable AI pipelines proved to be a software engineering challenge as much as an LLM one. Passing uncompressed, base64-encoded PNG files (>4MB) in a JSON payload to the `gpt-5-mini` endpoint routinely triggered TLS/SSL MAC verification errors (`SSLV3_ALERT_BAD_RECORD_MAC`). We resolved this by intercepting images with the `PIL` library, applying 85% JPEG compression in-memory (reducing payload size by over 90% to ~300KB), and wrapping API calls in an exponential backoff retry mechanism to handle transient network drops.

*   **Prompt Information Density:** Packing multiple visual ideas into a single prompt caused catastrophic hallucinations, especially with label text. Current diffusion models cannot reliably render accurate regulatory micro-text. Adopting a "one scene per prompt" ideology and instructing models to obscure small text in favor of product silhouettes vastly improved consistency. Ultimately, decomposing the pipeline into independent, verifiable steps (Data Analyst → Creative Director → VLM Critic) proved far superior to any single monolithic prompt.
