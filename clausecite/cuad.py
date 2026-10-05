"""Load the CUAD dataset (SQuAD-format JSON) into contracts and lawyer-labeled clause spans."""

import json
from pathlib import Path

from .types import Contract, Label, Span


def load_cuad(path: str | Path) -> tuple[list[Contract], list[Label]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))["data"]
    contracts: list[Contract] = []
    labels: list[Label] = []
    for i, doc in enumerate(data):
        para = doc["paragraphs"][0]  # CUAD stores each contract as a single paragraph
        contract_id = f"c{i}"
        contracts.append(Contract(contract_id, doc["title"], para["context"]))
        for qa in para["qas"]:
            # Several annotators can mark the same span; keep each range once.
            ranges = sorted({(a["answer_start"], a["answer_start"] + len(a["text"])) for a in qa["answers"]})
            if not ranges:
                continue
            raw = qa["question"]
            labels.append(
                Label(
                    contract_id=contract_id,
                    category=qa["id"].split("__")[-1],
                    question=raw.split("Details:", 1)[-1].strip(),
                    raw_question=raw,
                    spans=tuple(Span(s, e) for s, e in ranges),
                )
            )
    return contracts, labels
