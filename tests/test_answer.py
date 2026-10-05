import json
from types import SimpleNamespace

from clausecite.answer import build_content, parse_segments
from clausecite.cuad import load_cuad
from clausecite.types import Chunk

CONTRACT = "Preamble. " * 10 + "This Agreement is governed by the laws of New York. More text follows."
CHUNKS = [
    Chunk("c0:0", "c0", 0, 60, CONTRACT[0:60]),
    Chunk("c0:1", "c0", 50, len(CONTRACT), CONTRACT[50:]),
]


def _text_block(text, citations=None):
    return SimpleNamespace(type="text", text=text, citations=citations)


def _cite(doc, start, end, quote):
    return SimpleNamespace(
        type="char_location", document_index=doc, start_char_index=start, end_char_index=end, cited_text=quote
    )


def test_build_content_sends_each_excerpt_as_a_cited_document():
    content = build_content("Which law governs?", CHUNKS)
    assert [b["type"] for b in content] == ["document", "document", "text"]
    assert content[1]["source"]["data"] == CHUNKS[1].text
    assert all(b["citations"] == {"enabled": True} for b in content[:2])
    assert content[-1]["text"] == "Which law governs?"


def test_citations_map_back_to_contract_offsets():
    quote = "governed by the laws of New York"
    local = CHUNKS[1].text.index(quote)
    content = [
        SimpleNamespace(type="thinking", thinking=""),
        _text_block("New York law governs", [_cite(1, local, local + len(quote), quote)]),
        _text_block("."),
    ]
    segments = parse_segments(content, CHUNKS)
    assert [s.text for s in segments] == ["New York law governs", "."]
    cite = segments[0].citations[0]
    assert cite.chunk_id == "c0:1"
    assert CONTRACT[cite.start : cite.end] == quote


def test_misaligned_citation_offsets_fall_back_to_finding_the_quote():
    quote = "governed by the laws of New York"
    segments = parse_segments([_text_block("x", [_cite(1, 0, 5, quote)])], CHUNKS)
    cite = segments[0].citations[0]
    assert CONTRACT[cite.start : cite.end] == quote


def test_load_cuad_dedupes_spans_and_skips_unlabeled_questions(tmp_path):
    question = 'Highlight the parts (if any) of this contract related to "Governing Law" ... Details: Which law governs?'
    data = {"data": [{"title": "Acme Supply", "paragraphs": [{"context": CONTRACT, "qas": [
        {"id": "Acme Supply__Governing Law", "question": question,
         "answers": [{"text": "laws of New York", "answer_start": 126}] * 2},
        {"id": "Acme Supply__Non-Compete", "question": "Details: Is there a non-compete?", "answers": []},
    ]}]}]}
    path = tmp_path / "cuad.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    contracts, labels = load_cuad(path)
    assert contracts[0].title == "Acme Supply"
    assert len(labels) == 1
    assert labels[0].category == "Governing Law"
    assert labels[0].question == "Which law governs?"
    assert len(labels[0].spans) == 1
