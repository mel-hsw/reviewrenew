from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Review(BaseModel):
    rating: int | None = None
    title: str | None = None
    text: str = Field(..., min_length=1)
    date: str | None = None


class Product(BaseModel):
    asin: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    category: str = ""
    description: str = ""
    listing_image_urls: list[str] = Field(default_factory=list)
    reviews: list[Review] = Field(default_factory=list)


class ProductCatalog(BaseModel):
    products: list[Product]


class PlannedShotBrief(BaseModel):
    """Q2 analyst output: shot plan without full text-to-image prompts (filled by Q3 creative step)."""

    shot_index: int = Field(ge=1, le=5, description="1-based order in the shot plan.")
    role: str = Field(
        description=(
            "Shot type, e.g. hero_product | consumer_in_use | ingredients_sensorial | "
            "detail_macro | context_lifestyle — pick labels that fit the category."
        )
    )
    rationale_from_reviews: str = Field(
        description=(
            "Why this shot: review themes, Official-description features to showcase, and which "
            "**brand_visual_codes** (palette, finish, mood) this frame reinforces."
        )
    )


class PlannedImageShot(BaseModel):
    """One of 3–5 backward-designed images tied to review themes (assignment Q2 → Q3 bridge)."""

    shot_index: int = Field(ge=1, le=5, description="1-based order in the shot plan.")
    role: str = Field(
        description=(
            "Shot type, e.g. hero_product | consumer_in_use | ingredients_sensorial | "
            "detail_macro | context_lifestyle — pick labels that fit the category."
        )
    )
    rationale_from_reviews: str = Field(
        description=(
            "Why this shot: review themes, Official-description features to showcase, and which "
            "**brand_visual_codes** (palette, finish, mood) this frame reinforces."
        )
    )
    prompt: str = Field(
        description=(
            "Full text-to-image prompt: elevated listing features, cross-review benefits, and brand_visual_codes; "
            "on-pack or in-scene words allowed when they match authentic product labeling (see pipeline instructions)."
        )
    )


class ReviewImageryBrief(BaseModel):
    """Q2 (analyst + RAG): review-grounded imagery strategy and shot plan without final diffusion prompts."""

    category_imagery_norms: str = Field(
        description=(
            "2–4 sentences: typical e-commerce / hero-gallery expectations for THIS category "
            "(what kinds of shots persuade shoppers—before using review text)."
        ),
    )
    consumer_needs_and_concerns: list[str] = Field(
        min_length=3,
        description=(
            "Short bullets grounded in excerpts: recurring needs, hopes, and worries "
            "(benefits sought + risks or complaints). Not generic marketing copy."
        ),
    )
    listing_features_elevated_by_reviews: list[str] = Field(
        min_length=2,
        description=(
            "Each line: a concrete feature or claim from the **Official description** that reviews justify "
            "**elevating in imagery** (why shoppers care—tie to reviewer language). Not every bullet from "
            "the listing—only those the feedback singles out or stresses."
        ),
    )
    cross_review_benefits: list[str] = Field(
        min_length=1,
        description=(
            "Benefits, outcomes, or feelings that **multiple reviewers** independently emphasize—"
            "these deserve explicit highlight treatment in prompts or on-pack styling when photographable."
        ),
    )
    brand_visual_codes: list[str] = Field(
        min_length=2,
        description=(
            "On-brand visual cues for this SKU: palette, packaging colors/finishes, bottle or box shape language, "
            "typical listing photography mood—each bullet grounded in **Title, Official description, or reviews** "
            "(e.g. reviewer mentions of how the product looks on the shelf). Avoid generic stereotypes not tied to sources."
        ),
    )
    shot_plan_rationale: str = Field(
        description=(
            "Paragraph: how the planned shots connect category norms, consumer needs/concerns, and **brand_visual_codes**; "
            "which shot is the primary hero and why."
        ),
    )
    product_summary: str = Field(
        description="3–5 sentences; blend facts with dominant sensory/emotional tone from reviews."
    )
    visual_attributes: list[str] = Field(
        description="Vivid phrases: color palette, silhouette, gloss/matte, packaging—not single generic words."
    )
    material_and_texture_cues: list[str] = Field(
        default_factory=list,
        description="Tactile/material language: how surfaces look and feel if grounded in sources.",
    )
    sensory_efficacy_and_ingredient_cues: list[str] = Field(
        default_factory=list,
        description=(
            "Efficacy outcomes, comfort, formula or barrier claims when present; any sensory "
            "dimensions reviewers stress (touch, weight, temperature, sound, minimal scent, etc.). "
            "Omit categories not mentioned in sources.",
        ),
    )
    customer_visual_consensus: str = Field(
        description="What most reviewers agree on; include atmosphere if reviews share a vibe."
    )
    pitfalls_to_avoid: list[str] = Field(
        default_factory=list,
        description="Complaints: misleading photos, wear, defects, wrong expectations.",
    )
    visual_translation_notes: str = Field(
        default="",
        description=(
            "Paragraph: how to stage claims with light, props, environment; on-image text may match real packaging "
            "when faithful to the product—no invented claims."
        ),
    )
    planned_shots: list[PlannedShotBrief] = Field(
        min_length=3,
        max_length=5,
        description="3–5 planned shot roles and rationales; Q3 adds full text-to-image prompts.",
    )


class PlannedShotPromptOnly(BaseModel):
    """One shot’s diffusion prompt from the Q3 creative LLM call."""

    shot_index: int = Field(ge=1, le=5)
    prompt: str = Field(
        description="Full English text-to-image prompt for this shot_index (one scene, one primary idea).",
    )


class CreativePromptPack(BaseModel):
    """Q3 creative step: prompts keyed by shot_index; must match Q2 planned_shots indices exactly."""

    planned_shots: list[PlannedShotPromptOnly] = Field(
        min_length=3,
        max_length=5,
        description="Same shot_index set as Q2 ReviewImageryBrief.planned_shots, with prompt strings.",
    )


def merge_review_brief_with_creative(
    brief: ReviewImageryBrief,
    creative: CreativePromptPack,
) -> ImagePromptBundle:
    """Supervisor merge: Q2 brief + Q3 prompts → final ImagePromptBundle (pipeline_meta filled later)."""
    q2_indices = {s.shot_index for s in brief.planned_shots}
    by_idx = {p.shot_index: p.prompt for p in creative.planned_shots}
    q3_indices = set(by_idx.keys())
    if q2_indices != q3_indices:
        raise ValueError(
            "Q2 and Q3 shot_index sets differ: "
            f"Q2={sorted(q2_indices)}, Q3={sorted(q3_indices)}"
        )
    if len(by_idx) != len(creative.planned_shots):
        raise ValueError("Duplicate shot_index in Q3 creative.planned_shots")
    ordered = sorted(brief.planned_shots, key=lambda s: s.shot_index)
    merged_shots: list[PlannedImageShot] = []
    for pb in ordered:
        prompt = by_idx[pb.shot_index]
        merged_shots.append(
            PlannedImageShot(
                shot_index=pb.shot_index,
                role=pb.role,
                rationale_from_reviews=pb.rationale_from_reviews,
                prompt=prompt,
            )
        )
    return ImagePromptBundle(
        category_imagery_norms=brief.category_imagery_norms,
        consumer_needs_and_concerns=brief.consumer_needs_and_concerns,
        listing_features_elevated_by_reviews=brief.listing_features_elevated_by_reviews,
        cross_review_benefits=brief.cross_review_benefits,
        brand_visual_codes=brief.brand_visual_codes,
        shot_plan_rationale=brief.shot_plan_rationale,
        pipeline_meta={},
        product_summary=brief.product_summary,
        visual_attributes=brief.visual_attributes,
        material_and_texture_cues=brief.material_and_texture_cues,
        sensory_efficacy_and_ingredient_cues=brief.sensory_efficacy_and_ingredient_cues,
        customer_visual_consensus=brief.customer_visual_consensus,
        pitfalls_to_avoid=brief.pitfalls_to_avoid,
        visual_translation_notes=brief.visual_translation_notes,
        planned_shots=merged_shots,
    )


class ImagePromptBundle(BaseModel):
    """Structured output meant to be turned into diffusion / image API prompts."""

    category_imagery_norms: str = Field(
        description=(
            "2–4 sentences: typical e-commerce / hero-gallery expectations for THIS category "
            "(what kinds of shots persuade shoppers—before using review text)."
        ),
    )
    consumer_needs_and_concerns: list[str] = Field(
        min_length=3,
        description=(
            "Short bullets grounded in excerpts: recurring needs, hopes, and worries "
            "(benefits sought + risks or complaints). Not generic marketing copy."
        ),
    )
    listing_features_elevated_by_reviews: list[str] = Field(
        min_length=2,
        description=(
            "Each line: a concrete feature or claim from the **Official description** that reviews justify "
            "**elevating in imagery** (why shoppers care—tie to reviewer language). Not every bullet from "
            "the listing—only those the feedback singles out or stresses."
        ),
    )
    cross_review_benefits: list[str] = Field(
        min_length=1,
        description=(
            "Benefits, outcomes, or feelings that **multiple reviewers** independently emphasize—"
            "these deserve explicit highlight treatment in prompts or on-pack styling when photographable."
        ),
    )
    brand_visual_codes: list[str] = Field(
        min_length=2,
        description=(
            "On-brand visual cues for this SKU: palette, packaging colors/finishes, bottle or box shape language, "
            "typical listing photography mood—each bullet grounded in **Title, Official description, or reviews** "
            "(e.g. reviewer mentions of how the product looks on the shelf). Avoid generic stereotypes not tied to sources."
        ),
    )
    shot_plan_rationale: str = Field(
        description=(
            "Paragraph: how the planned shots connect category norms, consumer needs/concerns, and **brand_visual_codes**; "
            "which shot is the primary hero and why."
        ),
    )
    pipeline_meta: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Filled by the pipeline after validation—not from the text model. "
            "RAG query list, unique chunk count, text model id, chunking settings for report reproducibility."
        ),
    )
    product_summary: str = Field(
        description="3–5 sentences; blend facts with dominant sensory/emotional tone from reviews."
    )
    visual_attributes: list[str] = Field(
        description="Vivid phrases: color palette, silhouette, gloss/matte, packaging—not single generic words."
    )
    material_and_texture_cues: list[str] = Field(
        default_factory=list,
        description="Tactile/material language: how surfaces look and feel if grounded in sources.",
    )
    sensory_efficacy_and_ingredient_cues: list[str] = Field(
        default_factory=list,
        description=(
            "Efficacy outcomes, comfort, formula or barrier claims when present; any sensory "
            "dimensions reviewers stress (touch, weight, temperature, sound, minimal scent, etc.). "
            "Omit categories not mentioned in sources.",
        ),
    )
    customer_visual_consensus: str = Field(
        description="What most reviewers agree on; include atmosphere if reviews share a vibe."
    )
    pitfalls_to_avoid: list[str] = Field(
        default_factory=list,
        description="Complaints: misleading photos, wear, defects, wrong expectations.",
    )
    visual_translation_notes: str = Field(
        default="",
        description=(
            "Paragraph: how to stage claims with light, props, environment; on-image text may match real packaging "
            "when faithful to the product—no invented claims."
        ),
    )
    planned_shots: list[PlannedImageShot] = Field(
        min_length=3,
        max_length=5,
        description="3–5 backward-designed shots: hero, in-use, sensorial/formula, detail, context (subset if fewer).",
    )


class PipelineStepLog(BaseModel):
    step: str
    detail: str = ""


ChunkSource = Literal["description", "review"]


class TextChunk(BaseModel):
    asin: str
    source: ChunkSource
    review_index: int | None = None
    text: str
    chunk_index: int = 0
