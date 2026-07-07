"""Environment-backed model configuration helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def _find_env_path() -> Path:
    here = Path(__file__).parent
    for candidate in [here, here.parent, here.parent.parent]:
        p = candidate / ".env"
        if p.exists():
            return p
    return here.parent / ".env"


ENV_PATH = _find_env_path()


def load_env_file(path: Path = ENV_PATH) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        values[key.strip()] = value
    return values


def _merged_env() -> dict[str, str]:
    values = load_env_file()
    values.update({k: v for k, v in os.environ.items() if v is not None})
    return values


def _fallback_model(config: dict[str, Any] | None) -> dict[str, str | None]:
    if not config or not config.get("models"):
        return {"model": None, "api_base": None, "api_key": None}
    model_cfg = config["models"][0]
    return {
        "model": model_cfg.get("model"),
        "api_base": model_cfg.get("api_base"),
        "api_key": model_cfg.get("api_key"),
    }


def model_config(role: str, config: dict[str, Any] | None = None) -> dict[str, str | None]:
    role = role.upper()
    env = _merged_env()
    fallback = _fallback_model(config)
    model = env.get(f"{role}_MODEL") or fallback.get("model")
    if not model:
        raise ValueError(f"{role}_MODEL is not configured in .env or environment")
    return {
        "model": model,
        "api_base": env.get(f"{role}_API_BASE") or fallback.get("api_base"),
        "api_key": env.get(f"{role}_API_KEY") or fallback.get("api_key"),
    }


def proposer_config(config: dict[str, Any] | None = None) -> dict[str, str | None]:
    return model_config("PROPOSER", config)


def classifier_config(config: dict[str, Any] | None = None) -> dict[str, str | None]:
    return model_config("CLASSIFIER", config)
