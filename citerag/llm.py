"""One function to call any LLM: Anthropic, any OpenAI-compatible API (Gemini, Groq, OpenAI), or an offline mock.

Every call is logged to logs/calls.jsonl with latency, tokens, cost and attempts.
Swap providers with env vars only — no code changes.
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

LOG_PATH = Path(os.getenv("LOG_PATH", "logs/calls.jsonl"))

OPENAI_COMPAT_BASE_URLS = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "groq": "https://api.groq.com/openai/v1",
    "openai": None,
}
KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "openai": "OPENAI_API_KEY",
}
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


@dataclass
class LLMResult:
    text: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    cost_usd: float
    attempts: int = 1
    fallback_used: bool = False


def _cost(input_tokens: int, output_tokens: int) -> float:
    p_in = float(os.getenv("PRICE_IN_PER_MTOK", "0"))
    p_out = float(os.getenv("PRICE_OUT_PER_MTOK", "0"))
    return round(input_tokens / 1e6 * p_in + output_tokens / 1e6 * p_out, 6)


def _log(result: LLMResult, tag: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": time.time(), "tag": tag, **asdict(result)}) + "\n")


def _mock(system: str, prompt: str) -> str:
    """Offline extractive baseline: best-matching source sentence, cited. Abstains on weak matches."""
    from citerag.retrieve import tokenize

    qm = re.search(r"Question:\s*(.+?)\nAnswer:", prompt, re.S)
    q = set(tokenize(qm.group(1))) if qm else set()
    best, best_n, best_score = "", 0, 0
    for n, body in re.findall(r"\[(\d+)\] \([^)]*\)\n(.*?)(?=\n\n\[\d+\] \(|\n\nQuestion:)", prompt, re.S):
        for sent in re.split(r"(?<=[.!?])\s+|\n+", body):
            score = len(q & set(tokenize(sent)))
            if score > best_score and not sent.lstrip().startswith("#"):
                best, best_n, best_score = sent.strip(), int(n), score
    if best_score < 2:
        return "I don't know based on the provided documents."
    return f"{best} [{best_n}]"


def _is_retryable(e: Exception) -> bool:
    """Retry overloaded servers, rate limits and network blips. Never retry bad keys or bad requests."""
    status = getattr(e, "status_code", None)
    if status is not None:
        return status in RETRYABLE_STATUS
    return any(w in type(e).__name__ for w in ("Connection", "Timeout"))


def _call_once(provider, model, prompt, system, max_tokens, temperature):
    if provider == "mock":
        return _mock(system, prompt), len(prompt) // 4, 20, "mock"

    if provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic(max_retries=0, timeout=60)
        resp = client.messages.create(
            model=model or "claude-haiku-5-5",
            max_tokens=max_tokens,
            temperature=temperature,
            system=system or anthropic.NOT_GIVEN,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return text, resp.usage.input_tokens, resp.usage.output_tokens, resp.model

    if provider in OPENAI_COMPAT_BASE_URLS:
        from openai import OpenAI

        if not model:
            raise ValueError(f"Set MODEL in .env for provider '{provider}'.")
        client = OpenAI(api_key=os.environ[KEY_ENV[provider]], base_url=OPENAI_COMPAT_BASE_URLS[provider],
                        max_retries=0, timeout=60)
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        resp = client.chat.completions.create(model=model, messages=msgs, max_tokens=max_tokens, temperature=temperature)
        text = resp.choices[0].message.content or ""
        tin = getattr(resp.usage, "prompt_tokens", 0) or 0
        tout = getattr(resp.usage, "completion_tokens", 0) or 0
        return text, tin, tout, model

    raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Use: mock, anthropic, {', '.join(OPENAI_COMPAT_BASE_URLS)}")


def complete(prompt: str, system: str = "", max_tokens: int = 800, temperature: float = 0.0, tag: str = "") -> LLMResult:
    """Retries transient errors with exponential backoff + jitter, then falls back to FALLBACK_MODEL."""
    provider = os.getenv("LLM_PROVIDER", "mock").lower()
    models = [os.getenv("MODEL", "")] + ([os.getenv("FALLBACK_MODEL")] if os.getenv("FALLBACK_MODEL") else [])
    max_retries = int(os.getenv("MAX_RETRIES", "3"))
    t0, attempts, last_err = time.perf_counter(), 0, None

    for i, model in enumerate(models):
        for retry in range(max_retries + 1):
            attempts += 1
            try:
                text, tin, tout, used = _call_once(provider, model, prompt, system, max_tokens, temperature)
                result = LLMResult(text, provider, used, tin, tout, round(time.perf_counter() - t0, 3),
                                   _cost(tin, tout), attempts, fallback_used=i > 0)
                _log(result, tag)
                return result
            except Exception as e:  # noqa: BLE001
                if not _is_retryable(e):
                    raise
                last_err = e
                if retry < max_retries:
                    wait = min(2 ** retry, 16) + random.uniform(0, 0.5)
                    print(f"[llm] {model}: {type(e).__name__} ({getattr(e, 'status_code', '-')}), "
                          f"retry {retry + 1}/{max_retries} in {wait:.1f}s", file=sys.stderr)
                    time.sleep(wait)
        if i + 1 < len(models):
            print(f"[llm] {model} still failing, falling back to {models[i + 1]}", file=sys.stderr)

    raise RuntimeError(f"LLM failed after {attempts} attempts across {models}") from last_err