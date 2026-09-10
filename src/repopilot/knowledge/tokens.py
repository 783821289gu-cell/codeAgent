from __future__ import annotations

import tiktoken
from tiktoken import Encoding


class TokenCounter:
    """Model-compatible token counting with an o200k fallback for new OpenAI models."""

    def __init__(self, model: str) -> None:
        self.model = model
        try:
            self.encoding: Encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            self.encoding = tiktoken.get_encoding("o200k_base")

    def count(self, text: str) -> int:
        return len(self.encoding.encode(text, disallowed_special=()))

    def truncate_prefix(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        tokens = self.encoding.encode(text, disallowed_special=())
        if len(tokens) <= max_tokens:
            return text
        marker = "\n... [token budget pruned]"
        marker_tokens = self.encoding.encode(marker, disallowed_special=())
        kept = tokens[: max(0, max_tokens - len(marker_tokens))]
        return self.encoding.decode(kept + marker_tokens)

    def truncate_middle(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        tokens = self.encoding.encode(text, disallowed_special=())
        if len(tokens) <= max_tokens:
            return text
        marker = "\n... [token budget pruned] ...\n"
        marker_tokens = self.encoding.encode(marker, disallowed_special=())
        available = max(0, max_tokens - len(marker_tokens))
        head = available // 2
        tail = available - head
        return self.encoding.decode(tokens[:head] + marker_tokens + tokens[-tail:])
