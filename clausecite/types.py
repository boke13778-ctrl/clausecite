from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass(frozen=True, slots=True)
class Span:
    """A character range in a contract's text. `end` is exclusive."""

    start: int
    end: int


@dataclass(frozen=True, slots=True)
class Contract:
    id: str
    title: str
    text: str


@dataclass(frozen=True, slots=True)
class Label:
    """A clause category that CUAD's lawyer annotators marked in one contract."""

    contract_id: str
    category: str
    question: str  # plain-language form, e.g. "Which state/country's law governs the contract?"
    raw_question: str  # the full CUAD prompt, "Highlight the parts (if any) of this contract..."
    spans: tuple[Span, ...]


@dataclass(frozen=True, slots=True)
class Chunk:
    id: str
    contract_id: str
    start: int
    end: int
    text: str


@dataclass(frozen=True, slots=True)
class Hit:
    chunk: Chunk
    score: float
