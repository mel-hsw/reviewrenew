# Final submission package (checklist)

Aligned with the course **Final Project** deliverables and **grading rubric** (experiment design, analytics, insights, **scientific rigor**: documentation & reproducibility).

## Include in your submission

| Item | Suggested location |
|------|---------------------|
| **Code** | This repository (or a clean zip of the project root without `.venv/`, `data/chroma/`, `outputs/` if large). |
| **Data pointers** | Report + `data/README.md`: credit [**Amazon Reviews 2023**](https://amazon-reviews-2023.github.io/); where raw slices live (`data/raw/amazon_reviews_2023/`) or how you obtained them; ASINs and categories. |
| **Derived catalog** | `data/products.json` (+ `products_manifest.json` if produced by `consolidate_products.py`). |
| **Prompts** | `src/llm_analysis.py` (Q2/Q3 templates); per-run prompts also appear in `outputs/<ASIN>_image_prompt_bundle.json` under `planned_shots`. |
| **Generated images** | `outputs/<ASIN>/openai/`, `outputs/<ASIN>/gemini/` — zip **`outputs/`** or attach per course instructions. |
| **Run trace** | `outputs/run_log.json` (steps, models, `analysis_meta` with harness telemetry). |
| **Report** | PDF/Markdown final report + reflective discussion (challenges, lessons learned). |
| **Agentic workflow (Q4)** | Narrative + link: **`docs/agentic_workflow.md`**. |

## What graders should be able to do

1. Install deps (`requirements.txt`), copy `.env.example` → `.env`, add keys.  
2. Run `python run_pipeline.py --skip-images` (bundles only) or full pipeline with images.  
3. Trace Q2 → Q3 → merge from `run_log.json` and `pipeline_meta` inside each bundle.

## Large files

If the repo is too big for the upload portal, omit **`outputs/`** and **`data/chroma/`** from git and attach a **separate zip** of generated artifacts, as allowed by your instructor.
