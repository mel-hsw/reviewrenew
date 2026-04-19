from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Literal

from openai import OpenAI

from . import config


def _client() -> OpenAI:
    return OpenAI(api_key=config.OPENAI_API_KEY)


def _is_gemini_model(model: str) -> bool:
    return model.lower().strip().startswith("gemini-")


def image_prompt_for_model(
    base_prompt: str,
    model: str,
    *,
    provider: Literal["openai", "gemini"] | None = None,
) -> str:
    """
    Combine the shared bundle prompt with an optional provider-specific suffix from env.
    Empty suffixes leave the base prompt unchanged (fair cross-model comparison by default).
    Pass ``provider`` when routing is explicit (recommended); otherwise suffix follows model id.
    """
    if provider == "gemini":
        use_gemini_suffix = True
    elif provider == "openai":
        use_gemini_suffix = False
    else:
        use_gemini_suffix = _is_gemini_model(model)
    suffix = (
        config.GEMINI_IMAGE_PROMPT_SUFFIX
        if use_gemini_suffix
        else config.OPENAI_IMAGE_PROMPT_SUFFIX
    )
    if not suffix:
        return base_prompt
    base = base_prompt.rstrip()
    return f"{base}\n\n{suffix}"


def _openai_size_to_gemini_aspect(size: str) -> str:
    """Map OpenAI Images API size strings to Gemini ImageConfig aspect_ratio."""
    s = size.lower().replace(" ", "")
    return {
        "1024x1024": "1:1",
        "1024x1536": "2:3",
        "1536x1024": "3:2",
        "1792x1024": "16:9",
        "1024x1792": "9:16",
        "1536x672": "21:9",
    }.get(s, "1:1")


def _gemini_model_supports_resolution(model: str) -> bool:
    """Gemini 3.x native image models accept image_size; 2.5 Flash Image does not."""
    m = model.lower()
    return "2.5-flash-image" not in m


def _extract_gemini_image_bytes(response: Any) -> bytes | None:
    """Skip thought parts; use the last rendered image (final output in thinking models)."""
    parts = response.parts
    if not parts:
        return None
    out: bytes | None = None
    for part in parts:
        if getattr(part, "thought", None):
            continue
        inline = getattr(part, "inline_data", None)
        if inline is not None and inline.data:
            out = inline.data
    return out


def _is_gpt_image_model(model: str) -> bool:
    """GPT / ChatGPT image models use a different subset of parameters than DALL·E."""
    m = model.lower()
    return "gpt-image" in m or "chatgpt-image" in m


def _image_generate_kwargs(model: str, prompt: str, size: str) -> dict[str, Any]:
    """Build kwargs for images.generate; response_format is only for dall-e-2/3."""
    kw: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "n": 1,
        "size": size,
    }
    if _is_gpt_image_model(model):
        # GPT image models always return base64; response_format is unsupported.
        kw["output_format"] = "png"
        q = config.OPENAI_IMAGE_QUALITY
        if q in ("low", "medium", "high", "auto"):
            kw["quality"] = q
    else:
        kw["response_format"] = "b64_json"
        q = config.OPENAI_IMAGE_QUALITY
        if model.startswith("dall-e-3") and q in ("standard", "hd"):
            kw["quality"] = q
    return kw


def _generate_gemini_product_images(
    prompt: str,
    model: str,
    n: int,
    size: str | None = None,
    out_dir: Path | None = None,
    basename: str = "img",
) -> list[Path]:
    from google import genai
    from google.genai import types

    if not config.GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is required when image model id starts with 'gemini-' "
            "(set it in .env)."
        )
    size = size or config.OPENAI_IMAGE_SIZE
    aspect = _openai_size_to_gemini_aspect(size)
    out_dir = out_dir or config.OUTPUTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    client = genai.Client(api_key=config.GEMINI_API_KEY)
    paths: list[Path] = []
    for i in range(n):
        img_cfg: types.ImageConfig = types.ImageConfig(aspect_ratio=aspect)
        if _gemini_model_supports_resolution(model):
            img_cfg = types.ImageConfig(
                aspect_ratio=aspect,
                image_size=config.GEMINI_IMAGE_RESOLUTION,
            )
        response = client.models.generate_content(
            model=model,
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                image_config=img_cfg,
            ),
        )
        raw = _extract_gemini_image_bytes(response)
        if not raw:
            continue
        suffix = f"{i:02d}" if n > 1 else "00"
        safe_model = "".join(c if c.isalnum() or c in "-_" else "_" for c in model)
        fname = f"{basename}_{safe_model}_{suffix}.png"
        path = out_dir / fname
        path.write_bytes(raw)
        paths.append(path)
    return paths


def generate_product_images(
    prompt: str,
    model: str,
    n: int,
    size: str | None = None,
    out_dir: Path | None = None,
    basename: str = "img",
) -> list[Path]:
    """
    Generate n images; saves PNG files from b64_json in the response.
    DALL·E 3 only supports n=1 per request; we loop. GPT image models also use n=1 per call here.
    Models whose id starts with ``gemini-`` use the Gemini API (Nano Banana native image).
    """
    if _is_gemini_model(model):
        return _generate_gemini_product_images(
            prompt, model, n, size=size, out_dir=out_dir, basename=basename
        )
    size = size or config.OPENAI_IMAGE_SIZE
    out_dir = out_dir or config.OUTPUTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    client = _client()
    for i in range(n):
        result = client.images.generate(
            **_image_generate_kwargs(model, prompt, size),
        )
        item = result.data[0]
        b64 = getattr(item, "b64_json", None)
        if not b64:
            continue
        raw = base64.b64decode(b64)
        suffix = f"{i:02d}" if n > 1 else "00"
        fname = f"{basename}_{model.replace('-', '_')}_{suffix}.png"
        path = out_dir / fname
        path.write_bytes(raw)
        paths.append(path)
    return paths
