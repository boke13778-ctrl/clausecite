"""Split contracts into overlapping chunks that keep their character offsets into the original text."""

from collections.abc import Iterable

from .types import Chunk, Contract


def chunk_text(contract_id: str, text: str, size: int = 1000, overlap: int = 150) -> list[Chunk]:
    chunks: list[Chunk] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            end = _snap_end(text, start, end, min_len=size // 2)
        chunks.append(Chunk(f"{contract_id}:{len(chunks)}", contract_id, start, end, text[start:end]))
        if end >= len(text):
            break
        start = _snap_start(text, max(end - overlap, start + 1), end)
    return chunks


def chunk_contracts(contracts: Iterable[Contract], size: int = 1000, overlap: int = 150) -> list[Chunk]:
    return [chunk for c in contracts for chunk in chunk_text(c.id, c.text, size, overlap)]


def _snap_end(text: str, start: int, end: int, min_len: int) -> int:
    """Pull `end` back to the last paragraph, sentence, or word break, keeping at least `min_len` chars."""
    for sep in ("\n", ". ", "; ", " "):
        i = text.rfind(sep, start + min_len, end)
        if i != -1:
            return i + len(sep)
    return end


def _snap_start(text: str, start: int, limit: int) -> int:
    """Push `start` forward to the next word boundary so a chunk doesn't open mid-word."""
    i = text.find(" ", start, limit)
    return i + 1 if i != -1 else start
