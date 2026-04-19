from __future__ import annotations

import json
import time
from typing import Any, Callable, TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from . import config

T = TypeVar("T", bound=BaseModel)


def _client() -> OpenAI:
    return OpenAI(api_key=config.OPENAI_API_KEY)


class StructuredLLMHarness:
    """
    One LLM step with structured JSON output, schema validation, bounded retries, and telemetry.

    Aligns with common harness practice: explicit I/O contracts, no silent parse failures,
    and per-call metadata (agent_id, model, duration, attempts) for supervisor logs.
    """

    def __init__(
        self,
        agent_id: str,
        model: str,
        *,
        temperature: float = 0.35,
        max_attempts: int = 2,
    ) -> None:
        self.agent_id = agent_id
        self.model = model
        self.temperature = temperature
        self.max_attempts = max(1, max_attempts)

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        model_cls: type[T],
        pre_validate: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> tuple[T, dict[str, Any]]:
        """
        Run chat.completions with json_object response, validate to ``model_cls``.

        Retries (bounded) on JSON decode errors and Pydantic validation errors.
        Second and later attempts use temperature 0 for stability.
        """
        client = _client()
        last_err: BaseException | None = None
        temp = self.temperature
        for attempt in range(1, self.max_attempts + 1):
            t0 = time.perf_counter()
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    temperature=temp,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                raw = (resp.choices[0].message.content or "").strip()
                data = json.loads(raw)
                if not isinstance(data, dict):
                    raise TypeError("Expected JSON object from LLM")
                if pre_validate is not None:
                    data = pre_validate(data)
                validated = model_cls.model_validate(data)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                usage: dict[str, Any] = {}
                u = getattr(resp, "usage", None)
                if u is not None:
                    usage = {
                        k: getattr(u, k)
                        for k in ("prompt_tokens", "completion_tokens", "total_tokens")
                        if getattr(u, k, None) is not None
                    }
                telemetry: dict[str, Any] = {
                    "agent_id": self.agent_id,
                    "model": self.model,
                    "attempts": attempt,
                    "duration_ms": round(elapsed_ms, 2),
                    **usage,
                }
                return validated, telemetry
            except (json.JSONDecodeError, TypeError, ValidationError, ValueError) as e:
                last_err = e
                if attempt >= self.max_attempts:
                    break
                temp = 0.0
        msg = f"{self.agent_id}: structured LLM failed after {self.max_attempts} attempt(s)"
        raise RuntimeError(msg) from last_err
