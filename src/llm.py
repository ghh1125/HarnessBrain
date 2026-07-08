
import hashlib
import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Protocol

try:
    from litellm import completion as litellm_completion
    from litellm import completion_cost, token_counter
except ImportError:
    def litellm_completion(*args, **kwargs):
        raise RuntimeError("litellm is required for provider LLM calls")

    def completion_cost(*args, **kwargs):
        return 0.0

    def token_counter(*args, **kwargs):
        text = kwargs.get("text")
        if text is not None:
            return len(str(text).split())
        messages = kwargs.get("messages") or []
        return sum(len(str(message.get("content", "")).split()) for message in messages)

try:
    from openai_harmony import HarmonyEncodingName, Role, load_harmony_encoding
except ImportError:
    HarmonyEncodingName = Role = None

    def load_harmony_encoding(*args, **kwargs):
        raise RuntimeError("openai_harmony is required for GPT-OSS harmony parsing")
try:
    from tenacity import (
        retry,
        retry_if_exception,
        stop_after_attempt,
        wait_exponential,
        wait_random,
    )
except ImportError:
    def retry(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def retry_if_exception(*args, **kwargs):
        return None

    def stop_after_attempt(*args, **kwargs):
        return None

    class _NoWait:
        def __add__(self, other):
            return self

    def wait_exponential(*args, **kwargs):
        return _NoWait()

    def wait_random(*args, **kwargs):
        return _NoWait()

logger = logging.getLogger(__name__)

CACHE_DIR = Path(
    os.path.expanduser(
        os.environ.get(
            "TEXT_CLASSIFICATION_LLM_CACHE_DIR",
            "~/.cache/text-classification/litellm",
        )
    )
)
CACHE_VERSION = 1
KNOWN_PROVIDER_PREFIXES = (
    "anthropic/",
    "azure/",
    "bedrock/",
    "cohere/",
    "gemini/",
    "groq/",
    "ollama/",
    "openai/",
    "openrouter/",
    "together_ai/",
    "togethercomputer/",
    "vertex_ai/",
    "xai/",
)

MAX_PROMPT_CHARS = 224_000

_HARMONY_ENC = None


def _get_harmony_enc():
    global _HARMONY_ENC
    if _HARMONY_ENC is None:
        _HARMONY_ENC = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return _HARMONY_ENC


def parse_harmony_response(raw_content: str) -> str:
    enc = _get_harmony_enc()
    try:
        tokens = enc.encode(raw_content, allowed_special="all")
        parsed = enc.parse_messages_from_completion_tokens(
            tokens, role=Role.ASSISTANT, strict=False
        )

        for msg in parsed:
            if msg.channel == "final":
                return "".join(c.text for c in msg.content if hasattr(c, "text"))

        if parsed:
            return "".join(c.text for c in parsed[-1].content if hasattr(c, "text"))
    except Exception:
        pass

    return raw_content


def _is_retryable(exc: Exception) -> bool:
    status = (
        getattr(exc, "status_code", None)
        or getattr(exc, "code", None)
        or getattr(exc, "status", None)
    )
    if status == 429:
        return True
    text = str(exc).lower()
    return any(
        token in text
        for token in (
            "429",
            "rate limit",
            "too many requests",
            "retry in",
            "timed out",
            "timeout",
            "ssl",
            "eof occurred",
            "connecterror",
            "connection error",
            "connection reset",
            "broken pipe",
        )
    )


def _extract_content(response: Any) -> str:
    message = response.choices[0].message
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            elif hasattr(item, "text"):
                parts.append(item.text)
        return "".join(parts)
    return ""


class LLMCallable(Protocol):

    def __call__(self, prompt: str) -> str: ...


class ProviderLLM:

    def __init__(
        self,
        model: str,
        max_concurrent: int = 4,
        api_key: str | None = None,
        api_base: str | None = None,
    ):
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.max_concurrent = max_concurrent
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.provider_prompt_tokens = 0
        self.provider_completion_tokens = 0
        self.provider_total_tokens = 0
        self.estimated_prompt_tokens = 0
        self.estimated_completion_tokens = 0
        self.estimated_total_tokens = 0
        self.cached_prompt_tokens = 0
        self.cached_completion_tokens = 0
        self.cached_total_tokens = 0
        self.provider_calls = 0
        self.estimated_calls = 0
        self.cached_calls = 0
        self._last_usage: dict[str, Any] | None = None
        self._usage_lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def __exit__(self, *exc):
        self._executor.shutdown(wait=True)
        return False

    def _normalized_model(self) -> str:
        if (
            self.api_base
            and "dashscope" in self.api_base
            and self.model.startswith("qwen/")
        ):
            return f"openai/{self.model.split('/', 1)[1]}"
        if self.api_base and not self.model.startswith(KNOWN_PROVIDER_PREFIXES):
            return f"openai/{self.model}"
        return self.model

    def _cache_path(
        self, prompt: str, system_prompt: str | None, kwargs: dict[str, Any]
    ) -> Path:
        payload = {
            "version": CACHE_VERSION,
            "model": self._normalized_model(),
            "api_base": self.api_base,
            "system_prompt": system_prompt,
            "prompt": prompt,
            "kwargs": kwargs,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        return CACHE_DIR / f"{digest}.json"

    def _load_cache(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None

    def _save_cache(self, path: Path, payload: dict[str, Any]) -> None:
        tmp = path.with_suffix(".tmp")
        try:
            with self._cache_lock:
                tmp.write_text(json.dumps(payload))
                tmp.replace(path)
        except OSError:
            pass

    def _make_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        usage_source: str,
    ) -> dict[str, Any]:
        prompt_tokens = int(prompt_tokens or 0)
        completion_tokens = int(completion_tokens or 0)
        return {
            "model": self.model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "usage_source": usage_source,
        }

    def _record_usage(self, usage: dict[str, Any], cost: float = 0.0) -> None:
        source = usage.get("usage_source", "estimated")
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or prompt_tokens + completion_tokens)

        with self._usage_lock:
            self._last_usage = dict(usage)
            if source == "provider":
                self.provider_prompt_tokens += prompt_tokens
                self.provider_completion_tokens += completion_tokens
                self.provider_total_tokens += total_tokens
                self.provider_calls += 1
                self.total_input_tokens += prompt_tokens
                self.total_output_tokens += completion_tokens
                self.total_cost += cost
            elif source == "cached":
                self.cached_prompt_tokens += prompt_tokens
                self.cached_completion_tokens += completion_tokens
                self.cached_total_tokens += total_tokens
                self.cached_calls += 1
            else:
                self.estimated_prompt_tokens += prompt_tokens
                self.estimated_completion_tokens += completion_tokens
                self.estimated_total_tokens += total_tokens
                self.estimated_calls += 1
                self.total_input_tokens += prompt_tokens
                self.total_output_tokens += completion_tokens
                self.total_cost += cost

    def get_last_usage(self) -> dict[str, Any] | None:
        with self._usage_lock:
            return dict(self._last_usage) if self._last_usage else None

    @retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_exponential(multiplier=2, min=2, max=32) + wait_random(1, 5),
        stop=stop_after_attempt(6),
        reraise=True,
    )
    def _call_completion(
        self, prompt: str, system_prompt: str | None, kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        call_kwargs = dict(kwargs)
        call_kwargs["timeout"] = 600.0
        if self.api_base:
            call_kwargs["base_url"] = self.api_base
            call_kwargs.setdefault("api_key", self.api_key or "local")
        elif self.api_key is not None:
            call_kwargs["api_key"] = self.api_key

        model = self._normalized_model()
        response = litellm_completion(model=model, messages=messages, **call_kwargs)
        content = _extract_content(response)

        usage = getattr(response, "usage", None)
        provider_prompt_tokens = getattr(usage, "prompt_tokens", None)
        provider_completion_tokens = getattr(usage, "completion_tokens", None)
        if provider_prompt_tokens is not None and provider_completion_tokens is not None:
            prompt_tokens = int(provider_prompt_tokens or 0)
            completion_tokens = int(provider_completion_tokens or 0)
            usage_source = "provider"
        else:
            usage_source = "estimated"
            try:
                prompt_tokens = token_counter(model=model, messages=messages)
            except Exception:
                prompt_tokens = 0
            try:
                completion_tokens = token_counter(model=model, text=content)
            except Exception:
                completion_tokens = 0

        if self.api_base:
            cost = 0.0
        else:
            try:
                cost = float(completion_cost(completion_response=response) or 0.0)
            except Exception:
                cost = 0.0

        usage_metadata = self._make_usage(prompt_tokens, completion_tokens, usage_source)
        return {
            "content": content,
            "input_tokens": usage_metadata["prompt_tokens"],
            "output_tokens": usage_metadata["completion_tokens"],
            "prompt_tokens": usage_metadata["prompt_tokens"],
            "completion_tokens": usage_metadata["completion_tokens"],
            "total_tokens": usage_metadata["total_tokens"],
            "usage_source": usage_metadata["usage_source"],
            "cost": cost,
        }

    def _generate_one(
        self, prompt: str, system_prompt: str | None, kwargs: dict[str, Any]
    ) -> str:
        cache_path = self._cache_path(prompt, system_prompt, kwargs)
        cached = self._load_cache(cache_path)
        if cached is not None:
            prompt_tokens = cached.get("prompt_tokens", cached.get("input_tokens", 0))
            completion_tokens = cached.get(
                "completion_tokens", cached.get("output_tokens", 0)
            )
            self._record_usage(
                self._make_usage(prompt_tokens, completion_tokens, "cached"),
                cost=0.0,
            )
            return cached["content"]

        result = self._call_completion(prompt, system_prompt, kwargs)
        self._save_cache(cache_path, result)
        self._record_usage(
            self._make_usage(
                result["prompt_tokens"],
                result["completion_tokens"],
                result["usage_source"],
            ),
            cost=result["cost"],
        )

        return result["content"]

    def generate(self, prompts, system_prompt: str | None = None, **kwargs):
        if isinstance(prompts, str):
            return [[self._generate_one(prompts, system_prompt, kwargs)]]

        prompts = list(prompts)
        if not prompts:
            return []

        results = [None] * len(prompts)
        futures = {
            self._executor.submit(
                self._generate_one, prompt, system_prompt, kwargs
            ): idx
            for idx, prompt in enumerate(prompts)
        }
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()
        return [[content] for content in results]


class LLM:

    def __init__(
        self,
        model: str = "openrouter/openai/gpt-oss-120b",
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 16384,
        max_workers: int = 32,
    ):
        self.model = model
        self.is_gpt_oss = "gpt-oss" in model.lower()

        if temperature is not None:
            self.temperature = temperature
        elif "gpt-5" in model or self.is_gpt_oss:
            self.temperature = 1.0
        else:
            self.temperature = 0.0

        if api_base is not None and not self.is_gpt_oss and max_tokens > 4096:
            max_tokens = 4096
        self.max_tokens = max_tokens

        self._model_kwargs: dict[str, Any] = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        self._system_prompt = "Reasoning: medium" if self.is_gpt_oss else None

        if api_base is None and max_workers > 4:
            max_workers = 4
        self._provider = ProviderLLM(
            model=model,
            max_concurrent=max_workers,
            api_key=api_key,
            api_base=api_base,
        )

        self._usage_lock = threading.Lock()
        self.total_calls = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if hasattr(self._provider, "__exit__"):
            self._provider.__exit__(*exc)
        return False

    @property
    def total_input_tokens(self):
        return self._provider.total_input_tokens

    @property
    def total_output_tokens(self):
        return self._provider.total_output_tokens

    @property
    def total_cost(self):
        return self._provider.total_cost

    def get_last_usage(self) -> dict[str, Any] | None:
        return self._provider.get_last_usage()

    def _truncate(self, prompt: str) -> str:
        if len(prompt) > MAX_PROMPT_CHARS:
            original_len = len(prompt)
            half = MAX_PROMPT_CHARS // 2
            prompt = prompt[:half] + "\n\n... [TRUNCATED] ...\n\n" + prompt[-half:]
            logger.warning(
                "Prompt truncated: %d -> %d chars (limit %d)",
                original_len,
                len(prompt),
                MAX_PROMPT_CHARS,
            )
        return prompt

    def __call__(self, prompt: str) -> str:
        prompt = self._truncate(prompt)

        results = self._provider.generate(
            prompt, system_prompt=self._system_prompt, **self._model_kwargs
        )
        content = results[0][0]

        with self._usage_lock:
            self.total_calls += 1

        if self.is_gpt_oss:
            content = parse_harmony_response(content)
        return content

    def get_usage(self) -> dict[str, Any]:
        with self._usage_lock:
            calls = self.total_calls
        provider_calls = self._provider.provider_calls
        estimated_calls = self._provider.estimated_calls
        cached_calls = self._provider.cached_calls
        active_sources = sum(1 for value in (provider_calls, estimated_calls, cached_calls) if value)
        return {
            "model": self.model,
            "calls": calls,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "estimated_cost_usd": round(self.total_cost, 4),
            "provider_prompt_tokens": self._provider.provider_prompt_tokens,
            "provider_completion_tokens": self._provider.provider_completion_tokens,
            "provider_total_tokens": self._provider.provider_total_tokens,
            "estimated_prompt_tokens": self._provider.estimated_prompt_tokens,
            "estimated_completion_tokens": self._provider.estimated_completion_tokens,
            "estimated_total_tokens": self._provider.estimated_total_tokens,
            "cached_prompt_tokens": self._provider.cached_prompt_tokens,
            "cached_completion_tokens": self._provider.cached_completion_tokens,
            "cached_total_tokens": self._provider.cached_total_tokens,
            "provider_calls": provider_calls,
            "estimated_calls": estimated_calls,
            "cached_calls": cached_calls,
            "has_provider_usage": provider_calls > 0,
            "has_mixed_usage_sources": active_sources > 1,
        }

    def reset_usage(self):
        with self._usage_lock:
            self.total_calls = 0
        self._provider.total_input_tokens = 0
        self._provider.total_output_tokens = 0
        self._provider.total_cost = 0.0
        self._provider.provider_prompt_tokens = 0
        self._provider.provider_completion_tokens = 0
        self._provider.provider_total_tokens = 0
        self._provider.estimated_prompt_tokens = 0
        self._provider.estimated_completion_tokens = 0
        self._provider.estimated_total_tokens = 0
        self._provider.cached_prompt_tokens = 0
        self._provider.cached_completion_tokens = 0
        self._provider.cached_total_tokens = 0
        self._provider.provider_calls = 0
        self._provider.estimated_calls = 0
        self._provider.cached_calls = 0
        self._provider._last_usage = None

    def batch(self, prompts: list[str]) -> list[str]:
        if not prompts:
            return []

        prompts = [self._truncate(p) for p in prompts]
        results = self._provider.generate(
            prompts, system_prompt=self._system_prompt, **self._model_kwargs
        )
        contents = [r[0] for r in results]

        with self._usage_lock:
            self.total_calls += len(prompts)

        if self.is_gpt_oss:
            contents = [parse_harmony_response(c) for c in contents]
        return contents


def make_local_llm(
    model: str = "gpt-oss-120b",
    host: str = os.environ.get("LOCAL_LLM_HOST", "localhost"),
    port: int = int(os.environ.get("LOCAL_LLM_PORT", "30000")),
    max_tokens: int = 4096,
    max_workers: int = 16,
) -> LLM:
    return LLM(
        model=model,
        api_base=f"http://{host}:{port}/v1",
        max_tokens=max_tokens,
        max_workers=max_workers,
    )


def make_stub_llm(response: str = '{"reasoning": "stub", "final_answer": "stub"}'):
    return lambda prompt: response


def call_llm(messages: list, model: str | None = None) -> str:
    model = model or os.environ.get("PROPOSER_MODEL", "qwen/qwen3.6-plus")
    api_base = os.environ.get("PROPOSER_API_BASE")
    api_key = os.environ.get("PROPOSER_API_KEY")
    response = litellm_completion(
        model=model,
        messages=messages,
        api_base=api_base,
        api_key=api_key,
    )
    return response.choices[0].message.content
