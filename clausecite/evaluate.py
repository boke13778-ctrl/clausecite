"""Retrieval eval: does search surface the clause a lawyer flagged?

For every (contract, clause category) pair that CUAD annotators labeled, we run a query against
that contract, take chunks until a fixed character budget is spent, and check each gold span.
A span counts as found when at least half of its characters fall inside the retrieved chunks.

The budget keeps chunk sizes comparable: bigger chunks trivially "find" more text, so every
configuration gets the same amount of contract to hand to the model.

Contracts outside the official CUAD test set are the dev set. Dev is split in half: expansion
terms are learned on one half and configs are compared on the other. For the test run,
expansions are learned on all of dev. Only the baseline and the chosen config touch test.

    python -m clausecite.evaluate
"""

import argparse
import statistics
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .bm25 import BM25Index, tokenize
from .chunking import chunk_contracts
from .cuad import load_cuad
from .expansion import expanded_terms, learn_expansions
from .types import DATA_DIR, Chunk, Contract, Label, Span

FOUND_THRESHOLD = 0.5

Expansions = dict[str, list[str]]
QUERY_STYLES: dict[str, Callable[[Label, Expansions], dict[str, float]]] = {
    "raw CUAD prompt": lambda label, _: dict.fromkeys(tokenize(label.raw_question), 1.0),
    "question only": lambda label, _: dict.fromkeys(tokenize(label.question), 1.0),
    "category + question": lambda label, _: dict.fromkeys(tokenize(f"{label.category}. {label.question}"), 1.0),
    "+ learned terms (w=0.5)": lambda label, exp: expanded_terms(
        f"{label.category}. {label.question}", exp.get(label.category, []), 0.5
    ),
    "+ learned terms (w=1.0)": lambda label, exp: expanded_terms(
        f"{label.category}. {label.question}", exp.get(label.category, []), 1.0
    ),
}


@dataclass(frozen=True)
class Config:
    chunk_size: int
    query_style: str


@dataclass
class Result:
    config: Config
    recall: float  # mean fraction of gold spans found, per label
    hit_rate: float  # share of labels where at least one gold span was found
    mrr: float  # reciprocal rank of the first chunk that finds a gold span
    context_share: float  # retrieved characters as a share of the contract
    per_category: dict[str, float]


def span_coverage(span: Span, chunks: Sequence[Chunk]) -> float:
    """Fraction of the span's characters that fall inside at least one chunk."""
    if span.end <= span.start:
        return 0.0
    ranges = sorted((max(c.start, span.start), min(c.end, span.end)) for c in chunks)
    covered, reach = 0, span.start
    for lo, hi in ranges:
        lo = max(lo, reach)
        if hi > lo:
            covered += hi - lo
            reach = hi
    return covered / (span.end - span.start)


def evaluate(
    contracts: list[Contract], labels: list[Label], config: Config, budget_chars: int, expansions: Expansions
) -> Result:
    index = BM25Index(chunk_contracts(contracts, size=config.chunk_size))
    lengths = {c.id: len(c.text) for c in contracts}
    to_query = QUERY_STYLES[config.query_style]
    k = max(1, budget_chars // config.chunk_size)

    recalls, hits, ranks, shares = [], [], [], []
    by_category: dict[str, list[float]] = defaultdict(list)
    for label in labels:
        retrieved = [h.chunk for h in index.search_terms(to_query(label, expansions), k=k, contract_id=label.contract_id)]
        found = [s for s in label.spans if span_coverage(s, retrieved) >= FOUND_THRESHOLD]
        recall = len(found) / len(label.spans)
        recalls.append(recall)
        hits.append(1.0 if found else 0.0)
        by_category[label.category].append(recall)
        first = next(
            (r for r, c in enumerate(retrieved, 1) if any(span_coverage(s, [c]) >= FOUND_THRESHOLD for s in label.spans)),
            None,
        )
        ranks.append(1 / first if first else 0.0)
        shares.append(sum(c.end - c.start for c in retrieved) / lengths[label.contract_id])

    return Result(
        config=config,
        recall=statistics.mean(recalls),
        hit_rate=statistics.mean(hits),
        mrr=statistics.mean(ranks),
        context_share=statistics.mean(shares),
        per_category={cat: statistics.mean(v) for cat, v in by_category.items()},
    )


def _row(r: Result) -> str:
    return (
        f"| {r.config.chunk_size} | {r.config.query_style} | {r.recall:.1%} | {r.hit_rate:.1%} "
        f"| {r.mrr:.3f} | {r.context_share:.1%} |"
    )


_HEADER = (
    "| Chunk size | Query | Span recall | Hit rate | MRR | Share of contract retrieved |\n"
    "|---:|---|---:|---:|---:|---:|"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", type=Path, default=DATA_DIR / "CUADv1.json", help="full CUAD (510 contracts)")
    parser.add_argument("--test", type=Path, default=DATA_DIR / "test.json", help="official CUAD test split")
    parser.add_argument("--budget", type=int, default=4000, help="characters retrieved per question")
    parser.add_argument("--out", type=Path, default=DATA_DIR.parent / "eval" / "results.md")
    args = parser.parse_args()

    test_contracts, test_labels = load_cuad(args.test)
    test_titles = {c.title for c in test_contracts}
    all_contracts, all_labels = load_cuad(args.all)
    dev = [c for c in all_contracts if c.title not in test_titles]
    learn_half, tune_half = dev[0::2], dev[1::2]

    def labels_for(contracts: list[Contract]) -> list[Label]:
        ids = {c.id for c in contracts}
        return [l for l in all_labels if l.contract_id in ids]

    tune_labels = labels_for(tune_half)
    print(f"dev: {len(dev)} contracts (tuning on {len(tune_half)}, {len(tune_labels)} labels) | "
          f"test: {len(test_contracts)} contracts, {len(test_labels)} labels | budget {args.budget} chars")

    tune_expansions = learn_expansions(learn_half, labels_for(learn_half))
    sweep = []
    for size in (500, 1000, 2000):
        for style in QUERY_STYLES:
            result = evaluate(tune_half, tune_labels, Config(size, style), args.budget, tune_expansions)
            print(_row(result), flush=True)
            sweep.append(result)
    best = max(sweep, key=lambda r: r.recall)
    baseline = Config(1000, "raw CUAD prompt")

    test_expansions = learn_expansions(dev, labels_for(dev))
    test_results = [
        evaluate(test_contracts, test_labels, cfg, args.budget, test_expansions) for cfg in (baseline, best.config)
    ]
    chosen = test_results[-1]
    worst = sorted(chosen.per_category.items(), key=lambda kv: kv[1])[:8]

    lines = [
        "# Retrieval eval (CUAD)",
        "",
        f"Each question retrieves {args.budget:,} characters from its contract. A gold span counts as found "
        f"when at least {FOUND_THRESHOLD:.0%} of it is inside the retrieved chunks. Generated by "
        "`python -m clausecite.evaluate`.",
        "",
        f"## Dev sweep ({len(tune_half)} contracts, {len(tune_labels)} labeled clauses; "
        f"expansions learned on {len(learn_half)} others)",
        "",
        _HEADER,
        *map(_row, sweep),
        "",
        f"## Test ({len(test_contracts)} contracts, {len(test_labels)} labeled clauses)",
        "",
        _HEADER,
        *map(_row, test_results),
        "",
        "First row is the baseline (raw CUAD prompt, 1,000-char chunks); second is the best dev config.",
        "",
        "## Hardest categories on test (span recall, best config)",
        "",
        "| Category | Span recall |",
        "|---|---:|",
        *(f"| {cat} | {recall:.1%} |" for cat, recall in worst),
        "",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[lines.index(f"## Test ({len(test_contracts)} contracts, {len(test_labels)} labeled clauses)"):]))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
