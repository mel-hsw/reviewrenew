from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

from . import config
from .image_gen import generate_product_images, image_prompt_for_model
from .llm_analysis import analyze_product_with_rag
from .models import ImagePromptBundle, PipelineStepLog, Product
from .rag import ReviewRAGIndex
from .vlm_qa import evaluate_image


@dataclass
class AgentState:
    logs: list[PipelineStepLog] = field(default_factory=list)
    bundles: dict[str, ImagePromptBundle] = field(default_factory=dict)
    analysis_meta: dict[str, dict] = field(default_factory=dict)
    # ASIN -> provider ("openai" | "gemini") -> list of image paths
    image_paths: dict[str, dict[str, list[str]]] = field(default_factory=dict)


def _log(state: AgentState, step: str, detail: str = "") -> None:
    state.logs.append(PipelineStepLog(step=step, detail=detail))


def run_pipeline(
    products: list[Product],
    images_per_model: int = 3,
    chroma_dir: Path | None = None,
    *,
    skip_images: bool = False,
    enable_vlm_qa: bool = False,
) -> AgentState:
    """
    Agentic workflow (sequential with explicit steps):
      load → index (RAG) → Q2 analyst LLM (RAG + ReviewImageryBrief) → Q3 creative LLM (prompts) → merge bundle
      → (Optional) image gen (OpenAI + Gemini) with optional VLM QA reflection loop.

    If ``skip_images`` is True, indexing and LLM bundle generation still run; image generation is omitted
    (e.g. chunking A/B without calling image APIs).
    """
    state = AgentState()
    out_root = config.OUTPUTS_DIR
    out_root.mkdir(parents=True, exist_ok=True)
    rag = ReviewRAGIndex(persist_dir=str(chroma_dir) if chroma_dir else None)

    for p in products:
        _log(state, "index_chunks", f"{p.asin}: embedding listing + reviews")
        n = rag.index_product(p)
        _log(state, "rag_indexed", f"{p.asin}: {n} chunks in vector store")

        retrieve: Callable[[str], list] = lambda q, asin=p.asin: rag.retrieve(asin, q, k=8)

        bundle, meta = analyze_product_with_rag(p, retrieve)
        q2 = meta.get("q2_harness", {})
        q3 = meta.get("q3_harness", {})
        _log(
            state,
            "q2_analyst",
            f"{p.asin}: model={q2.get('model')} duration_ms={q2.get('duration_ms')}",
        )
        _log(
            state,
            "q3_creative",
            f"{p.asin}: model={q3.get('model')} duration_ms={q3.get('duration_ms')}",
        )
        state.bundles[p.asin] = bundle
        state.analysis_meta[p.asin] = meta

        bundle_path = out_root / f"{p.asin}_image_prompt_bundle.json"
        bundle_path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
        _log(state, "saved_bundle", str(bundle_path))

        if skip_images:
            state.image_paths[p.asin] = {}
            continue

        shots_sorted = sorted(bundle.planned_shots, key=lambda s: s.shot_index)
        if not shots_sorted:
            raise RuntimeError(f"No planned_shots for {p.asin}")

        product_dir = out_root / p.asin
        product_dir.mkdir(parents=True, exist_ok=True)

        image_jobs: list[tuple[Literal["openai", "gemini"], str, Path]] = [
            ("openai", config.OPENAI_IMAGE_MODEL, product_dir / "openai"),
        ]
        if config.GEMINI_IMAGE_MODEL:
            image_jobs.append(
                ("gemini", config.GEMINI_IMAGE_MODEL, product_dir / "gemini")
            )

        paths_asin: dict[str, list[str]] = {}
        for provider, model, out_sub in image_jobs:
            out_sub.mkdir(parents=True, exist_ok=True)
            for i in range(images_per_model):
                shot = shots_sorted[i] if i < len(shots_sorted) else shots_sorted[0]
                role_slug = "".join(
                    c if c.isalnum() or c in "-_" else "_" for c in shot.role.lower()
                ).strip("_")[:40] or "shot"
                label = f"{i:02d}_{shot.shot_index}_{role_slug}"
                pr = image_prompt_for_model(shot.prompt, model, provider=provider)
                _log(state, "image_gen", f"{p.asin} {provider} {model} {label}")
                
                tries = 0
                max_tries = config.VLM_MAX_RETRIES + 1 if enable_vlm_qa else 1
                current_pr = pr
                final_imgs = []
                
                while tries < max_tries:
                    tries += 1
                    imgs = generate_product_images(
                        current_pr,
                        model=model,
                        n=1,
                        out_dir=out_sub,
                        basename=f"{label}_try{tries}" if enable_vlm_qa else label,
                    )
                    
                    if not enable_vlm_qa or tries == max_tries:
                        final_imgs = imgs
                        break
                        
                    if not imgs:
                        break
                        
                    img_path = imgs[0]
                    qa = evaluate_image(img_path, current_pr, bundle.pitfalls_to_avoid)
                    _log(state, "vlm_qa", f"{p.asin} {provider} {model} {label} try {tries} passed={qa.passed} critique={qa.visual_critique}")
                    
                    if qa.passed:
                        final_imgs = imgs
                        break
                    else:
                        rejected_dir = out_sub / "rejected"
                        rejected_dir.mkdir(exist_ok=True)
                        dest_path = rejected_dir / img_path.name
                        img_path.rename(dest_path)
                        
                        # Save the rejection reason next to the image
                        reason_path = dest_path.with_suffix('.txt')
                        reason_text = f"CRITIQUE:\n{qa.visual_critique}\n\nSUGGESTED REVISION:\n{qa.suggested_prompt_revision}"
                        reason_path.write_text(reason_text, encoding="utf-8")
                        
                        _log(state, "vlm_qa", f"Rejected image saved to {dest_path}")
                        current_pr = image_prompt_for_model(qa.suggested_prompt_revision, model, provider=provider)
                        
                paths_asin.setdefault(provider, []).extend(str(x) for x in final_imgs)

        state.image_paths[p.asin] = paths_asin

    run_log = out_root / "run_log.json"
    payload = {
        "experiment": {"skip_images": skip_images, "enable_vlm_qa": enable_vlm_qa},
        "models_used": {
            "text": config.OPENAI_TEXT_MODEL,
            "text_q3_creative": config.OPENAI_Q3_TEXT_MODEL,
            "embedding": config.OPENAI_EMBEDDING_MODEL,
            "openai_image_model": config.OPENAI_IMAGE_MODEL,
            "gemini_image_model": config.GEMINI_IMAGE_MODEL or None,
            "image_size": config.OPENAI_IMAGE_SIZE,
            "image_quality": config.OPENAI_IMAGE_QUALITY or None,
            "gemini_image_resolution": config.GEMINI_IMAGE_RESOLUTION,
            "openai_image_prompt_suffix": config.OPENAI_IMAGE_PROMPT_SUFFIX or None,
            "gemini_image_prompt_suffix": config.GEMINI_IMAGE_PROMPT_SUFFIX or None,
        },
        "steps": [s.model_dump() for s in state.logs],
        "analysis_meta": state.analysis_meta,
        "image_paths": state.image_paths,
    }
    run_log.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _log(state, "complete", f"wrote {run_log}")
    return state
