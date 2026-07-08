
import hashlib
import json
import re
import threading
from abc import ABC, abstractmethod
from typing import Any

try:
    from .llm import LLMCallable
except ImportError:
    from llm import LLMCallable


def extract_json_field(text: str, field: str, default: str = "") -> str:
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return str(data.get(field, default))
    except json.JSONDecodeError:
        pass

    for match in re.finditer(r"```(?:json)?\s*([\s\S]*?)\s*```", text):
        try:
            data = json.loads(match.group(1))
            if isinstance(data, dict):
                return str(data.get(field, default))
        except json.JSONDecodeError:
            pass

    for start in range(len(text)):
        if text[start] != "{":
            continue
        depth, pos, in_str = 1, start + 1, False
        while pos < len(text) and depth > 0:
            c = text[pos]
            if c == '"' and (pos == 0 or text[pos - 1] != "\\"):
                in_str = not in_str
            elif not in_str:
                depth += 1 if c == "{" else (-1 if c == "}" else 0)
            pos += 1
        if depth == 0:
            candidate = text[start:pos]
            candidate = re.sub(r",\s*([\]}])", r"\1", candidate)
            try:
                data = json.loads(candidate)
                if isinstance(data, dict):
                    return str(data.get(field, default))
            except json.JSONDecodeError:
                pass

    match = re.findall(rf'"{field}"\s*:\s*"([^"]*)"', text)
    return match[-1] if match else default


class MemorySystem(ABC):

    def __init__(self, llm: LLMCallable):
        self._llm = llm
        self._prompt_local = threading.local()

    def call_llm(self, prompt: str) -> str:
        self._prompt_local.last_prompt_len = len(prompt)
        self._prompt_local.last_prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
        self._prompt_local.last_prompt_text = prompt
        return self._llm(prompt)

    def get_last_prompt_info(self) -> dict[str, Any]:
        return {
            "prompt_len": getattr(self._prompt_local, "last_prompt_len", None),
            "prompt_hash": getattr(self._prompt_local, "last_prompt_hash", None),
            "prompt_text": getattr(self._prompt_local, "last_prompt_text", None),
        }

    @abstractmethod
    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        pass

    @abstractmethod
    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        pass

    def get_context_length(self) -> int:
        return len(self.get_state())

    @abstractmethod
    def get_state(self) -> str:
        pass

    @abstractmethod
    def set_state(self, state: str) -> None:
        pass
