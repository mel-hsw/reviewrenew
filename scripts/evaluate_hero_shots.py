"""
Evaluate hero-shot images for all 3 products x 2 models using a VLM scoring rubric.
Outputs a JSON results file and prints a Markdown summary table.

Usage:
    python scripts/evaluate_hero_shots.py
"""

from __future__ import annotations

import base64
import json
import sys
from io import BytesIO
from pathlib import Path

from PIL import Image
from openai import OpenAI

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import os
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
VLM_MODEL = "gpt-5.4-mini"  # supports vision via chat completions

RUN_DIR = ROOT / "outputs" / "run_20260421_000114"

PRODUCTS = [
    {
        "asin": "B0C4BFWLNC",
        "name": "Pet Bed",
        "bundle_json": RUN_DIR / "B0C4BFWLNC_image_prompt_bundle.json",
        "openai_hero": RUN_DIR / "B0C4BFWLNC" / "openai" / "00_1_hero_product_try1_gpt_image_1_00.png",
        "gemini_hero": RUN_DIR / "B0C4BFWLNC" / "gemini" / "00_1_hero_product_try1_gemini-3_1-flash-image-preview_00.png",
    },
    {
        "asin": "B0BG9Q18ZZ",
        "name": "CeraVe Cleanser",
        "bundle_json": RUN_DIR / "B0BG9Q18ZZ_image_prompt_bundle.json",
        "openai_hero": RUN_DIR / "B0BG9Q18ZZ" / "openai" / "00_1_hero_product_try1_gpt_image_1_00.png",
        "gemini_hero": RUN_DIR / "B0BG9Q18ZZ" / "gemini" / "00_1_hero_product_try1_gemini-3_1-flash-image-preview_00.png",
    },
    {
        "asin": "B0BXSQN4HV",
        "name": "Hat Washer",
        "bundle_json": RUN_DIR / "B0BXSQN4HV_image_prompt_bundle.json",
        "openai_hero": RUN_DIR / "B0BXSQN4HV" / "openai" / "00_1_hero_product_try1_gpt_image_1_00.png",
        "gemini_hero": RUN_DIR / "B0BXSQN4HV" / "gemini" / "00_1_hero_product_try1_gemini-3_1-flash-image-preview_00.png",
    },
]

SYSTEM_PROMPT = """You are a senior e-commerce art director evaluating AI-generated product hero shots.
Score the image on each of the following 5 dimensions using a 1–5 integer scale:

1. color_palette (1–5): How accurately does the image match the expected brand color palette and product colorway?
2. silhouette_fidelity (1–5): How recognizable and accurate is the product's 3D shape and silhouette?
3. photorealism (1–5): How photorealistic does the image look (5 = indistinguishable from a real photo)?
4. label_text_handling (1–5): Are labels/seals appropriately omitted or blurred rather than hallucinated as gibberish?
5. staging_composition (1–5): Is the scene well-lit, physically plausible, and catalog-appropriate?

Return ONLY a JSON object with these exact keys plus key_finding and recommendation:
{
  "color_palette": <int 1-5>,
  "silhouette_fidelity": <int 1-5>,
  "photorealism": <int 1-5>,
  "label_text_handling": <int 1-5>,
  "staging_composition": <int 1-5>,
  "key_finding": "<one sentence describing the most notable strength or weakness>",
  "recommendation": "<one sentence improvement suggestion>"
}
Return only the JSON object, no other text."""


def encode_image(path: Path) -> str:
    with Image.open(path) as img:
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode()


def score_image(client: OpenAI, image_path: Path, hero_prompt: str, pitfalls: list[str]) -> dict:
    b64 = encode_image(image_path)
    pitfall_text = "\n".join(f"- {p}" for p in pitfalls) if pitfalls else "None listed."
    user_text = f"Hero shot prompt used to generate this image:\n{hero_prompt}\n\nKnown pitfalls to avoid:\n{pitfall_text}\n\nScore this image."

    response = client.chat.completions.create(
        model=VLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}},
                ],
            },
        ],
        max_completion_tokens=512,
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    data = json.loads(raw.strip())
    dims = ["color_palette", "silhouette_fidelity", "photorealism", "label_text_handling", "staging_composition"]
    data["overall_score"] = round(sum(data[d] for d in dims) / len(dims), 1)
    return data


def main():
    client = OpenAI(api_key=OPENAI_API_KEY)
    results = []

    for product in PRODUCTS:
        bundle = json.loads(product["bundle_json"].read_text())
        hero_prompt = next(
            (s["prompt"] for s in bundle.get("planned_shots", []) if s.get("shot_index") == 1),
            bundle.get("planned_shots", [{}])[0].get("prompt", "")
        )
        pitfalls = bundle.get("pitfalls_to_avoid", [])

        for model_name, img_path in [("OpenAI", product["openai_hero"]), ("Gemini", product["gemini_hero"])]:
            print(f"  Scoring {product['name']} / {model_name}...", flush=True)
            scores = score_image(client, img_path, hero_prompt, pitfalls)
            results.append({
                "product": product["name"],
                "asin": product["asin"],
                "model": model_name,
                "image_path": str(img_path),
                **scores,
            })
            print(f"    -> overall {scores['overall_score']}/5.0")

    out_path = RUN_DIR / "hero_shot_evaluation.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}\n")

    # Print Markdown table
    dims = ["color_palette", "silhouette_fidelity", "photorealism", "label_text_handling", "staging_composition", "overall_score"]
    headers = ["Product", "Model", "Color", "Silhouette", "Realism", "Label/Text", "Staging", "Overall"]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in results:
        row = [r["product"], r["model"]] + [str(r[d]) for d in dims]
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
