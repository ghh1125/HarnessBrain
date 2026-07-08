import copy
from typing import Any, Dict, List

from litellm import Message


def add_anthropic_caching(
    messages: List[Dict[str, Any] | Message], model_name: str
) -> List[Dict[str, Any] | Message]:

    if not ("anthropic" in model_name.lower() or "claude" in model_name.lower()):
        return messages


    cached_messages = copy.deepcopy(messages)


    for n in range(len(cached_messages)):
        if n >= len(cached_messages) - 3:
            msg = cached_messages[n]


            if isinstance(msg, dict):

                if isinstance(msg.get("content"), str):
                    msg["content"] = [
                        {
                            "type": "text",
                            "text": msg["content"],
                            "cache_control": {"type": "ephemeral"},
                        }
                    ]
                elif isinstance(msg.get("content"), list):

                    for content_item in msg["content"]:
                        if isinstance(content_item, dict) and "type" in content_item:
                            content_item["cache_control"] = {"type": "ephemeral"}
            elif hasattr(msg, "content"):
                if isinstance(msg.content, str):
                    msg.content = [
                        {
                            "type": "text",
                            "text": msg.content,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ]
                elif isinstance(msg.content, list):
                    for content_item in msg.content:
                        if isinstance(content_item, dict) and "type" in content_item:
                            content_item["cache_control"] = {"type": "ephemeral"}

    return cached_messages
