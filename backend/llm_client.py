"""
llm_client.py
-------------
Thin async wrapper around the Crusoe Cloud LLM API (OpenAI-compatible endpoint).

All backend agents import `call_llm` from here — this is the single place
where the model name, base URL, auth header, and retry logic live.
"""

from __future__ import annotations

import json
import logging
import os
import asyncio
from typing import Any

import httpx
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration — override via .env
# ---------------------------------------------------------------------------

CRUSOE_API_KEY: str = os.getenv("CRUSOE_API_KEY", "")
CRUSOE_BASE_URL: str = os.getenv(
    "CRUSOE_BASE_URL",
    "https://hackeurope.crusoecloud.com/v1",
)
CRUSOE_MODEL: str = os.getenv("CRUSOE_MODEL", "NVFP4/Qwen3-235B-A22B-Instruct-2507-FP4")

_HEADERS: dict[str, str] = {
    "Authorization": f"Bearer {CRUSOE_API_KEY}",
    "Content-Type": "application/json",
}

# ---------------------------------------------------------------------------
# Core async caller
# ---------------------------------------------------------------------------


async def call_llm(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    response_format: str | None = "json_object",   # enforce JSON output
    retries: int = 2,
) -> dict[str, Any]:
    """
    Send a chat-completion request to the Crusoe API.

    Returns the parsed JSON dict from the assistant's response.
    Raises `RuntimeError` if all retries are exhausted.

    Parameters
    ----------
    system_prompt   : Instructions for the LLM role.
    user_prompt     : The actual content/question for this call.
    temperature     : Sampling temperature (keep low for clinical use).
    max_tokens      : Upper bound on completion length.
    response_format : "json_object" forces the model to return valid JSON.
                      Pass None to allow plain text.
    retries         : Number of additional attempts on transient errors.
    """
    payload: dict[str, Any] = {
        "model": CRUSOE_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format:
        payload["response_format"] = {"type": response_format}

    last_error: Exception | None = None
    async with httpx.AsyncClient(timeout=60.0) as client:
        for attempt in range(retries + 1):
            try:
                response = await client.post(
                    f"{CRUSOE_BASE_URL}/chat/completions",
                    headers=_HEADERS,
                    json=payload,
                )

                if not response.is_success:
                    body = response.text[:800]
                    logger.error(
                        "Crusoe API error %s: %s", response.status_code, body
                    )

                # --- 429 rate-limit: back off and retry ---
                if response.status_code == 429:
                    retry_after = float(
                        response.headers.get("Retry-After", 0) or 0
                    )
                    wait = retry_after if retry_after > 0 else (2.0 ** attempt)
                    logger.warning(
                        "Rate limited (429). Waiting %.1fs before retry %d/%d ...",
                        wait, attempt + 1, retries,
                    )
                    await asyncio.sleep(wait)
                    last_error = RuntimeError(f"429 Too Many Requests (attempt {attempt+1})")
                    continue   # retry without raising

                response.raise_for_status()
                data = response.json()
                raw_text: str = data["choices"][0]["message"]["content"]
                return json.loads(raw_text)

            except (httpx.HTTPStatusError, httpx.RequestError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt < retries:
                    await asyncio.sleep(2.0 ** attempt)

    raise RuntimeError(
        f"LLM call failed after {retries + 1} attempts: {last_error}"
    )


# ---------------------------------------------------------------------------
# Sync convenience wrapper (for use in non-async code / Streamlit callbacks)
# ---------------------------------------------------------------------------

def call_llm_sync(
    system_prompt: str,
    user_prompt: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Blocking version of `call_llm`. Runs the coroutine in a new event loop."""
    return asyncio.run(call_llm(system_prompt, user_prompt, **kwargs))
