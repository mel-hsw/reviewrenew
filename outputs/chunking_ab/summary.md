# Chunking A/B: sliding vs. paragraph_batch

**Date:** 2026-04-18 (three-product catalog in `data/products.json`)  
**Command:** `python run_pipeline.py --skip-images` after deleting `data/chroma/` each time (RAG + LLM bundles only; **no image APIs**).  
**Token settings:** `CHUNK_MAX_TOKENS=400`, `CHUNK_OVERLAP_TOKENS=60` (defaults).  
**Archives:** `outputs/chunking_ab/sliding/`, `outputs/chunking_ab/paragraph_batch/` (`run_log.json` + `*_image_prompt_bundle.json` per ASIN). Root `outputs/run_log.json` matches the **paragraph_batch** run (last executed).

## `unique_chunks_used` (from `analysis_meta` in each archived `run_log.json`)

| Strategy | B0C4BFWLNC | B0BG9Q18ZZ (CeraVe cleanser) | B0BXSQN4HV |
|----------|------------|--------------------------------|------------|
| sliding | 46 | 41 | 43 |
| paragraph_batch | 46 | 42 | 43 |

Chunking metadata in each log: `strategy` + `max_tokens` + `overlap_tokens` as configured.

## Indexed volume (from pipeline `steps`)

Per product, total chunks embedded in Chroma were **502 / 502 / 502** for **both** strategies (same tokenizer limits and cleaned review corpus).

## Findings (2–3 bullets)

- **Retrieval footprint:** Strategies agree on **B0C4BFWLNC** and **B0BXSQN4HV**; **B0BG9Q18ZZ** differs by **one** unique retrieved chunk (41 vs. 42) between sliding and paragraph_batch—small boundary effect on deduplicated RAG context.
- **Indexed volume:** Total chunks per product are identical across runs; differences are which paragraph boundaries win, not corpus size.
- **Bundles:** Spot-check `B0BG9Q18ZZ` JSON in each folder—both runs yield coherent, review-grounded `planned_shots`; pick one strategy for the final write-up for consistency.

## One-paragraph takeaway

For this catalog and defaults, **sliding** and **paragraph_batch** are close on RAG breadth; the only numeric spread is **+1** `unique_chunks_used` on CeraVe under `paragraph_batch`. **Project choice:** use **`paragraph_batch`** going forward—most cleaned reviews are **short** (not long narratives), so packing whole paragraphs/review bodies is a better fit than sliding overlap, which mainly helps when single reviews span many token windows. Set `CHUNK_STRATEGY=paragraph_batch` in `.env` and cite `analysis_meta` from `outputs/run_log.json`.
