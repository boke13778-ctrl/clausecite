"""A small BM25 index over contract chunks.

IDF is computed across every chunk in the corpus, so a term that appears in all contracts
("agreement") counts for little even when we only search inside one contract.
"""

import heapq
import math
import re
from collections import Counter, defaultdict

from .types import Chunk, Hit

_TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an and any are as at be been by for from has have if in into is it its of on or such "
    "that the their them then there these this those to was were which who will with".split()
)
# Longest first, so "terminations" loses "ations" rather than just "s".
_SUFFIXES = ("ations", "ation", "ating", "ments", "ated", "ates", "ings", "ment", "ate", "ing", "ed", "es", "s")


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


def _stem(token: str) -> str:
    """Light suffix stripping: "terminate", "terminated", "termination" all become "termin"."""
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    return token


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.2, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        # term -> contract id -> [(chunk index, term frequency)]; grouping by contract keeps
        # single-contract search from scanning every posting in the corpus.
        self.postings: dict[str, dict[str, list[tuple[int, int]]]] = defaultdict(lambda: defaultdict(list))
        self.lengths: list[int] = []
        doc_freq: Counter[str] = Counter()
        for i, chunk in enumerate(chunks):
            tf = Counter(tokenize(chunk.text))
            self.lengths.append(sum(tf.values()))
            doc_freq.update(tf.keys())
            for term, count in tf.items():
                self.postings[term][chunk.contract_id].append((i, count))
        self.avg_len = sum(self.lengths) / len(chunks) if chunks else 0.0
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - df + 0.5) / (df + 0.5)) for t, df in doc_freq.items()}

    def search(self, query: str, k: int = 5, contract_id: str | None = None) -> list[Hit]:
        return self.search_terms(dict.fromkeys(tokenize(query), 1.0), k, contract_id)

    def search_terms(self, terms: dict[str, float], k: int = 5, contract_id: str | None = None) -> list[Hit]:
        """Search with already-tokenized terms, each scaled by its weight."""
        scores: dict[int, float] = defaultdict(float)
        for term, weight in terms.items():
            if term not in self.idf:
                continue
            by_contract = self.postings[term]
            groups = by_contract.values() if contract_id is None else [by_contract.get(contract_id, [])]
            for postings in groups:
                for i, tf in postings:
                    norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg_len)
                    scores[i] += weight * self.idf[term] * tf * (self.k1 + 1) / (tf + norm)
        top = heapq.nlargest(k, scores.items(), key=lambda item: item[1])
        return [Hit(self.chunks[i], score) for i, score in top]
