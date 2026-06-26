"""
inference.py

Backend initialization and inference helpers for the ERE LLM stage.

Supported backends:
  - groq: Groq chat completions
  - api: generic OpenAI-compatible chat completions endpoint
  - ollama: local Ollama chat/generate
  - hf: HuggingFace text-generation pipeline

For the generic API backend, set:
  ERE_LLM_API_BASE=https://the-endpoint/v1
  ERE_LLM_API_KEY=...
"""

from __future__ import annotations

import os
import re
import time

SYSTEM_PROMPT = (
    "You are an expert linguist performing event relation extraction. "
    "Follow the requested output format exactly and do not add unrelated commentary."
)

_backend: str | None = None
_client = None
_model: str | None = None
_hf_pipeline = None


def load_model(backend: str, model_name: str, device: str = "cpu") -> None:
    global _backend, _client, _model, _hf_pipeline

    _backend = backend.lower()
    _model = model_name

    if _backend == "groq":
        from groq import Groq

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY must be set for --llm_backend groq")
        _client = Groq(api_key=api_key, timeout=120.0)
        print(f"Connected to Groq API (model: {_model})")

    elif _backend == "api":
        from openai import OpenAI

        api_key = os.environ.get("ERE_LLM_API_KEY")
        base_url = os.environ.get("ERE_LLM_API_BASE")
        if not api_key or not base_url:
            raise RuntimeError(
                "ERE_LLM_API_KEY and ERE_LLM_API_BASE must both be set for --llm_backend api"
            )
        _client = OpenAI(api_key=api_key, base_url=base_url, timeout=120.0)
        print(f"Connected to OpenAI-compatible API at {base_url} (model: {_model})")

    elif _backend == "ollama":
        import ollama

        _client = ollama
        print(f"Connected to Ollama (model: {_model})")

    elif _backend == "hf":
        from transformers import pipeline

        device_id = 0 if device == "cuda" else -1
        _hf_pipeline = pipeline("text-generation", model=_model, device=device_id)
        print(f"Loaded HuggingFace text-generation pipeline (model: {_model})")

    else:
        raise ValueError(f"Unsupported backend: {backend}")


def is_loaded() -> bool:
    return _backend is not None


def run_inference(
    prompt: str,
    max_new_tokens: int = 1600,
    temperature: float = 0.0,
    system_prompt: str = SYSTEM_PROMPT,
) -> tuple[str, float]:
    if not is_loaded():
        raise RuntimeError("Backend not loaded. Call load_model() first.")

    if _backend == "groq":
        return _run_chat_completions(prompt, max_new_tokens, temperature, system_prompt, provider="groq")
    if _backend == "api":
        return _run_chat_completions(prompt, max_new_tokens, temperature, system_prompt, provider="api")
    if _backend == "ollama":
        return _run_ollama(prompt, max_new_tokens, temperature, system_prompt)
    if _backend == "hf":
        return _run_hf(prompt, max_new_tokens, temperature, system_prompt)

    raise RuntimeError(f"Unknown loaded backend: {_backend}")


def _parse_retry_delay(error_text: str, default: float = 5.0) -> float:
    match = re.search(r"[Pp]lease try again in ([0-9a-zA-Z.\s]+)", error_text)
    if not match:
        return default

    text = match.group(1).lower()

    minutes_seconds = re.search(r"(?:(\d+(?:\.\d+)?)m)?\s*(?:(\d+(?:\.\d+)?)s)?", text)
    if minutes_seconds and (minutes_seconds.group(1) or minutes_seconds.group(2)):
        minutes = float(minutes_seconds.group(1) or 0.0)
        seconds = float(minutes_seconds.group(2) or 0.0)
        return minutes * 60.0 + seconds

    milliseconds = re.search(r"(\d+(?:\.\d+)?)\s*ms", text)
    if milliseconds:
        return float(milliseconds.group(1)) / 1000.0

    seconds = re.search(r"(\d+(?:\.\d+)?)\s*s", text)
    if seconds:
        return float(seconds.group(1))

    return default


def _run_chat_completions(
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    system_prompt: str,
    provider: str,
) -> tuple[str, float]:
    start = time.time()
    max_retries = 8 if provider == "groq" else 6

    if provider == "groq":
        import groq

        rate_limit_error = groq.RateLimitError
        transient_errors = (groq.APITimeoutError, groq.APIConnectionError)
    else:
        import openai

        rate_limit_error = openai.RateLimitError
        transient_errors = (openai.APITimeoutError, openai.APIConnectionError)

    for attempt in range(max_retries):
        try:
            response = _client.chat.completions.create(
                model=_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_new_tokens,
            )
            elapsed = time.time() - start
            text = response.choices[0].message.content or ""
            return text.strip(), elapsed

        except rate_limit_error as exc:
            delay = _parse_retry_delay(str(exc)) + 1.0
            print(f"\n[{provider} rate limit] sleeping {delay:.2f}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(delay)

        except transient_errors as exc:
            print(f"\n[{provider} {type(exc).__name__}] retrying in 5s (attempt {attempt + 1}/{max_retries})")
            time.sleep(5.0)

    raise RuntimeError(f"Exceeded maximum retries for {provider} backend")


def _run_ollama(
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    system_prompt: str,
) -> tuple[str, float]:
    start = time.time()
    options = {"temperature": temperature, "num_predict": max_new_tokens}

    if hasattr(_client, "chat"):
        response = _client.chat(
            model=_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            options=options,
        )
        elapsed = time.time() - start
        return response["message"]["content"].strip(), elapsed

    response = _client.generate(
        model=_model,
        prompt=f"{system_prompt}\n\n{prompt}",
        options=options,
    )