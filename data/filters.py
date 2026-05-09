from __future__ import annotations


def filter_length(text: str, min_chars: int = 1, max_chars: int | None = None) -> bool:
    if len(text) < min_chars:
        return False
    if max_chars is not None and len(text) > max_chars:
        return False
    return True


def deduplicate_exact(texts: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for text in texts:
        if text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output
