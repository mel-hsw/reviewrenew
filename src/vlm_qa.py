from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from io import BytesIO
from PIL import Image

from openai import OpenAI, APIConnectionError

from . import config
from .models import ImageQAFeedback


def _client() -> OpenAI:
    return OpenAI(api_key=config.OPENAI_API_KEY)


def encode_image(image_path: Path) -> str:
    """Read the PNG, convert to JPEG in memory, and base64 encode to shrink payload size."""
    with Image.open(image_path) as img:
        # Convert to RGB in case of RGBA
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


def evaluate_image(
    image_path: Path,
    original_prompt: str,
    pitfalls: list[str]
) -> ImageQAFeedback:
    """Uses the VLM to QA an image based on the prompt and any known pitfalls."""
    client = _client()
    base64_image = encode_image(image_path)
    # Get image format based on suffix. Usually png or jpg but we use png in image_gen.py
    img_fmt = "png"
    if image_path.suffix.lower() in [".jpeg", ".jpg"]:
        img_fmt = "jpeg"

    pitfalls_text = "\n".join(f"- {p}" for p in pitfalls) if pitfalls else "None"

    sys_instruction = """You are a strict Art Director evaluating a generated product image.
Your primary goal is to ensure the image looks like authentic, high-end, photorealistic e-commerce photography.

Check for the following CRITICAL failures:
1. AI Artifacts (Critical Fail): Are there weird hands, extra fingers, physically impossible human anatomy, merged objects, or clearly "fake AI" uncanny valley hallmarks? If yes, REJECT it immediately.
2. Physical Reality (Critical Fail): Are the objects physically impossible? Are they floating where they shouldn't be, or missing structural logic?
3. Gibberish Text: Is there excessive hallucinated, unreadable, or fake text dominating the image?

However, if the image is breathtakingly beautiful, highly photorealistic, and has flawless human anatomy/physics, do NOT reject it just because it missed a minor aesthetic constraint from the prompt. Prioritize visual excellence and realism over strict textual literalism.

Return a JSON object matching this schema:
{
    "passed": true or false,
    "visual_critique": "A brief explanation of what is right/wrong with the image.",
    "suggested_prompt_revision": "If passed is false, provide a revised, optimized text-to-image prompt to fix the issues. If passed is true, leave empty."
}
Only return the JSON object, nothing else.
"""

    user_text = f"""
Original Prompt:
{original_prompt}

Pitfalls to avoid:
{pitfalls_text}

Examine the attached image and return your JSON critique.
"""

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            response = client.responses.create(
                model=config.VLM_QA_MODEL,
                input=[
                    {
                        "role": "system",
                        "content": sys_instruction,
                    },
                    {
                        "role": "user",
                        "content": [
                            { "type": "input_text", "text": user_text },
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"
                            },
                        ],
                    }
                ],
            )
            break # Success, break out of retry loop
        except APIConnectionError as e:
            if attempt == max_retries:
                raise e
            print(f"SSL/Connection error uploading image to VLM. Retrying ({attempt}/{max_retries})...")
            time.sleep(2)
    
    raw_output = response.output_text.strip()
    # Handle possible markdown fences
    if raw_output.startswith("```json"):
        raw_output = raw_output[7:]
    if raw_output.endswith("```"):
        raw_output = raw_output[:-3]
    raw_output = raw_output.strip()

    data = json.loads(raw_output)
    return ImageQAFeedback.model_validate(data)
