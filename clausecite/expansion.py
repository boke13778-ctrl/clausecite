"""Learn per-category query expansion terms from lawyer-labeled clauses.

Users ask "which law governs this?" but contracts say "construed in accordance with the laws of".
For each clause category we find the terms that are much more common inside labeled clauses
than in contracts overall (smoothed log-odds), and add them to the query.

    python -m clausecite.expansion   # learns from CUAD minus the test split, writes data/expansions.json
"""

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from .bm25 import tokenize
from .cuad import load_cuad
from .types import DATA_DIR, Contract, Label

DEFAULT_PATH = DATA_DIR / "expansions.json"


def learn_expansions(
    contracts: list[Contract],
    labels: list[Label],
    top_n: int = 12,
    min_support: int = 3,
    min_support_share: float = 0.05,
    alpha: float = 1.0,
) -> dict[str, list[str]]:
    """A term must appear in at least `min_support` labels and `min_support_share` of the category's
    labels; otherwise rare tokens like statute numbers win on log-odds alone. 0.05 was best on dev."""
    texts = {c.id: c.text for c in contracts}
    background: Counter[str] = Counter()
    for c in contracts:
        background.update(tokenize(c.text))

    in_spans: dict[str, Counter[str]] = defaultdict(Counter)
    support: dict[str, Counter[str]] = defaultdict(Counter)  # how many labels contain the term
    label_counts = Counter(label.category for label in labels)
    for label in labels:
        tokens = [t for s in label.spans for t in tokenize(texts[label.contract_id][s.start : s.end])]
        in_spans[label.category].update(tokens)
        support[label.category].update(set(tokens))

    vocab = len(background)
    bg_total = sum(background.values())
    expansions = {}
    for category, counts in in_spans.items():
        total = sum(counts.values())

        def log_odds(term: str) -> float:
            p_span = (counts[term] + alpha) / (total + alpha * vocab)
            p_bg = (background[term] + alpha) / (bg_total + alpha * vocab)
            return math.log(p_span / p_bg)

        needed = max(min_support, min_support_share * label_counts[category])
        candidates = [t for t in counts if support[category][t] >= needed]
        expansions[category] = sorted(candidates, key=log_odds, reverse=True)[:top_n]
    return expansions


def expanded_terms(question: str, expansion: list[str], weight: float) -> dict[str, float]:
    terms = dict.fromkeys(expansion, weight)
    terms.update(dict.fromkeys(tokenize(question), 1.0))  # the user's own words keep full weight
    return terms


def load_expansions(path: Path = DEFAULT_PATH) -> dict[str, list[str]]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main() -> None:
    test_contracts, _ = load_cuad(DATA_DIR / "test.json")
    test_titles = {c.title for c in test_contracts}
    contracts, labels = load_cuad(DATA_DIR / "CUADv1.json")
    keep = {c.id for c in contracts if c.title not in test_titles}
    expansions = learn_expansions(
        [c for c in contracts if c.id in keep], [l for l in labels if l.contract_id in keep]
    )
    DEFAULT_PATH.write_text(json.dumps(expansions, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {len(expansions)} categories to {DEFAULT_PATH.name}")
    for category in ("Governing Law", "Termination For Convenience", "Non-Compete"):
        print(f"  {category}: {', '.join(expansions.get(category, []))}")


if __name__ == "__main__":
    main()
