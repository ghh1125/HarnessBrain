from harbor.agents.terminus_2.terminus_2 import Terminus2


_STALE_OBS_MARKER = (
    "\n<stale_output>\n"
    "This earlier observation was very long and has been compacted because it "
    "is no longer recent. The middle bytes were elided to save context. "
    "If you still need details from this step, re-run a narrower command "
    "(e.g. head/tail/sed -n 'A,Bp' on a known line range, or a more "
    "selective grep / find pattern). Prefer producing short focused output "
    "in future commands so this does not happen again.\n"
    "</stale_output>\n"
)


class AgentHarness(Terminus2):
    # context_management: how many of the most recent user observations to keep
    # in full (uncompacted). Recent context the model is actively reasoning about.
    _KEEP_RECENT_OBS: int = 4
    # context_management: only compact observations larger than this byte budget;
    # below it the message is already small and compaction would be net-negative.
    _OLD_OBS_KEEP_BYTES: int = 1500
    # context_management: head / tail byte budget for compacted older observations.
    _OLD_OBS_HEAD_BYTES: int = 600
    _OLD_OBS_TAIL_BYTES: int = 600

    def __init__(self, *args, **kwargs):
        # ICL single-action format (in-lineage winner; iter2 lever kept).
        kwargs.setdefault("parser_name", "xml")
        kwargs.setdefault("max_turns", 200)
        # Summarization left ON as a fallback for extreme chains where compaction
        # alone isn't enough. Compaction triggers first and usually suffices.
        super().__init__(*args, **kwargs)

    # ── context_management: tail-decay observation compaction ────────────────
    def _compact_one_observation(self, content: str) -> str:
        """Shrink a stale user observation to head + marker + tail.

        Returns the original content unchanged when it is already smaller than
        ``_OLD_OBS_KEEP_BYTES`` (no-op for short observations).
        """
        raw = content.encode("utf-8")
        if len(raw) <= self._OLD_OBS_KEEP_BYTES:
            return content
        head = raw[: self._OLD_OBS_HEAD_BYTES].decode("utf-8", errors="ignore")
        tail = raw[-self._OLD_OBS_TAIL_BYTES :].decode("utf-8", errors="ignore")
        elided = len(raw) - len(head.encode("utf-8")) - len(tail.encode("utf-8"))
        return (
            f"{head}\n[... {elided} stale interior bytes elided ...]"
            f"{_STALE_OBS_MARKER}{tail}"
        )

    def _apply_tail_decay(self, chat) -> bool:
        """Compact stale older user observations in chat._messages, in place.

        Preserves:
          - messages[0] (initial prompt: task + system template + initial state).
          - All assistant messages (the agent's plans / analysis / working memory).
          - The last ``_KEEP_RECENT_OBS`` user observations (active context).

        Returns True if any message was actually mutated.
        """
        messages = chat.messages
        if len(messages) <= 2:
            return False

        # Indices of user messages, excluding the very first one (the task spec).
        user_idxs: list[int] = []
        for i, msg in enumerate(messages):
            if i == 0:
                continue  # never touch the task spec
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if not isinstance(content, str):
                continue  # only plain-text user messages (skip structured content)
            user_idxs.append(i)

        # Keep the last K observations uncompacted; compact everything older.
        if len(user_idxs) <= self._KEEP_RECENT_OBS:
            return False
        stale_idxs = user_idxs[: -self._KEEP_RECENT_OBS]

        mutated = False
        for idx in stale_idxs:
            old = messages[idx]["content"]
            new = self._compact_one_observation(old)
            if new is not old and new != old:
                messages[idx] = {**messages[idx], "content": new}
                mutated = True
        return mutated

    async def _query_llm(
        self,
        chat,
        prompt: str,
        logging_paths,
        original_instruction: str = "",
        session=None,
    ):
        try:
            if self._apply_tail_decay(chat):
                # chat._messages was mutated in place; invalidate the Responses
                # API cache so the next call doesn't reuse a stale response id.
                chat.reset_response_chain()
        except Exception as e:  # noqa: BLE001 — never block the LLM call on
            # a compaction bug; degrade gracefully to parent behaviour.
            self.logger.warning(f"[evolved_iter6] tail-decay skipped: {e}")
        return await super()._query_llm(
            chat,
            prompt,
            logging_paths,
            original_instruction=original_instruction,
            session=session,
        )
