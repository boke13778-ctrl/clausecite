from clausecite.bm25 import BM25Index, tokenize
from clausecite.chunking import chunk_text
from clausecite.evaluate import span_coverage
from clausecite.expansion import expanded_terms, learn_expansions
from clausecite.types import Chunk, Contract, Label, Span

SAMPLE = (
    "1. Term. This Agreement starts on the Effective Date and lasts three years.\n"
    "2. Termination. Either party may terminate this Agreement for convenience on 30 days notice.\n"
    "3. Governing Law. This Agreement is governed by the laws of the State of New York.\n"
)


def test_chunks_keep_offsets_and_cover_the_text():
    text = SAMPLE * 20
    chunks = chunk_text("c0", text, size=300, overlap=50)
    assert chunks[0].start == 0 and chunks[-1].end == len(text)
    for chunk in chunks:
        assert text[chunk.start : chunk.end] == chunk.text
        assert len(chunk.text) <= 300
    for a, b in zip(chunks, chunks[1:]):
        assert b.start <= a.end, "consecutive chunks must overlap or touch"
        assert b.start > a.start


def test_chunks_break_on_line_boundaries_when_possible():
    chunks = chunk_text("c0", SAMPLE * 5, size=200, overlap=0)
    assert all(c.text.endswith("\n") for c in chunks[:-1])


def test_tokenize_stems_related_forms_together():
    assert tokenize("terminate terminated termination") == ["termin"] * 3
    assert tokenize("The Governing law of the state") == ["govern", "law", "state"]


def test_bm25_ranks_the_relevant_clause_first_and_filters_by_contract():
    chunks = [
        Chunk("a:0", "a", 0, 10, "Either party may terminate this agreement on notice."),
        Chunk("a:1", "a", 10, 20, "This agreement is governed by the laws of New York."),
        Chunk("b:0", "b", 0, 10, "Governed by the laws of Delaware."),
    ]
    index = BM25Index(chunks)
    assert index.search("which law governs", k=1, contract_id="a")[0].chunk.id == "a:1"
    assert {h.chunk.contract_id for h in index.search("laws governed", k=5, contract_id="b")} == {"b"}
    assert index.search("nothing matches this", contract_id="a") == []


def test_span_coverage_merges_overlapping_chunks():
    chunks = [Chunk("x", "c", 0, 60, ""), Chunk("y", "c", 50, 80, "")]
    assert span_coverage(Span(40, 100), chunks) == 40 / 60
    assert span_coverage(Span(90, 100), chunks) == 0.0
    assert span_coverage(Span(10, 20), chunks) == 1.0


def test_learned_expansions_pick_words_from_labeled_clauses():
    contracts = [Contract(f"c{i}", "t", SAMPLE + "Payment is due monthly. " * 5) for i in range(4)]
    start = SAMPLE.index("This Agreement is governed")
    labels = [
        Label(c.id, "Governing Law", "Which law governs?", "", (Span(start, start + 75),)) for c in contracts
    ]
    terms = learn_expansions(contracts, labels, top_n=5)["Governing Law"]
    assert "govern" in terms and "payment" not in terms
    weighted = expanded_terms("which law?", terms, 0.5)
    assert weighted["law"] == 1.0  # the user's own words keep full weight
    assert weighted["govern"] == 0.5
