"""Answer a question about one contract from retrieved excerpts, citing the exact contract text.

Each excerpt goes to Claude as its own document with citations enabled. Claude's citations point
at character ranges inside an excerpt; we shift them by the excerpt's offset so every citation
lands on a range in the full contract that the UI can highlight.
"""

from dataclasses import dataclass, field
from typing import Any

import anthropic

from .types import Chunk

MODEL = "claude-opus-5-5"

SYSTEM_PROMPT = """You answer questions about a commercial contract for a lawyer who will check your answer against the source.

You see excerpts from the contract, not the whole document. Answer only from those excerpts and cite the language that supports each claim. If the excerpts don't address the question, say so plainly instead of guessing; the clause may be elsewhere in the contract.

Keep answers short and in plain English. Write plain text without Markdown; the interface shows it verbatim."""


@dataclass
class Citation:
    chunk_id: str
    start: int  # offsets into the full contract text
    end: int
    quote: str


@dataclass
class Segment:
    text: str
    citations: list[Citation] = field(default_factory=list)


@dataclass
class Answer:
    segments: list[Segment]
    stop_reason: str
    model: str

    @property
    def text(self) -> str:
        return "".join(s.text for s in self.segments)


def build_content(question: str, chunks: list[Chunk]) -> list[dict[str, Any]]:
    documents = [
        {
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": chunk.text},
            "title": f"Excerpt {n} (contract characters {chunk.start}-{chunk.end})",
            "citations": {"enabled": True},
        }
        for n, chunk in enumerate(chunks, 1)
    ]
    return [*documents, {"type": "text", "text": question}]


def parse_segments(content: list[Any], chunks: list[Chunk]) -> list[Segment]:
    segments = []
    for block in content:
        if block.type != "text":
            continue
        citations = []
        for cite in block.citations or []:
            if cite.type != "char_location":
                continue
            chunk = chunks[cite.document_index]
            start, end = cite.start_char_index, cite.end_char_index
            if chunk.text[start:end] != cite.cited_text:
                # Fall back to locating the quote if the offsets don't line up with our text.
                found = chunk.text.find(cite.cited_text.strip())
                if found != -1:
                    start, end = found, found + len(cite.cited_text.strip())
            citations.append(Citation(chunk.id, chunk.start + start, chunk.start + end, cite.cited_text))
        segments.append(Segment(block.text, citations))
    return segments


def answer(question: str, chunks: list[Chunk], client: anthropic.Anthropic | None = None) -> Answer:
    client = client or anthropic.Anthropic()
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        output_config={"effort": "medium"},
        # If a safety classifier declines, the API retries on a fallback model in the same call.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_content(question, chunks)}],
    )
    if response.stop_reason == "refusal":
        return Answer([Segment("The model declined to answer this question.")], "refusal", response.model)
    return Answer(parse_segments(response.content, chunks), response.stop_reason or "", response.model)
