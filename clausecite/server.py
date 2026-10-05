"""FastAPI app: pick a contract, search it, and ask Claude questions with cited answers.

    uvicorn clausecite.server:app --reload
"""

import os
from dataclasses import asdict
from functools import cache
from pathlib import Path

import anthropic
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import answer as qa
from .bm25 import BM25Index
from .chunking import chunk_contracts
from .cuad import load_cuad
from .expansion import expanded_terms, load_expansions
from .types import DATA_DIR, Contract, Hit

# Best config on the dev split; see eval/results.md.
CHUNK_SIZE = 500
BUDGET_CHARS = 4000
EXPANSION_WEIGHT = 1.0

DATA_PATH = Path(os.environ.get("CUAD_PATH", DATA_DIR / "test.json"))
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="ClauseCite")


class Corpus:
    def __init__(self, path: Path):
        contracts, _ = load_cuad(path)
        self.contracts: dict[str, Contract] = {c.id: c for c in contracts}
        self.index = BM25Index(chunk_contracts(contracts, size=CHUNK_SIZE))
        self.expansions = load_expansions()

    def search(self, contract_id: str, question: str, category: str | None) -> list[Hit]:
        expansion = self.expansions.get(category, []) if category else []
        query = f"{category}. {question}" if category else question
        terms = expanded_terms(query, expansion, EXPANSION_WEIGHT)
        return self.index.search_terms(terms, k=BUDGET_CHARS // CHUNK_SIZE, contract_id=contract_id)


@cache
def corpus() -> Corpus:
    return Corpus(DATA_PATH)


class Query(BaseModel):
    contract_id: str
    question: str = Field(min_length=1, max_length=2000)
    category: str | None = None


def _contract(contract_id: str) -> Contract:
    contract = corpus().contracts.get(contract_id)
    if contract is None:
        raise HTTPException(404, f"Unknown contract {contract_id!r}")
    return contract


def _hits_json(hits: list[Hit]) -> list[dict]:
    return [
        {"chunk_id": h.chunk.id, "start": h.chunk.start, "end": h.chunk.end, "score": round(h.score, 3)}
        for h in hits
    ]


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/contracts")
def list_contracts() -> dict:
    contracts = sorted(corpus().contracts.values(), key=lambda c: c.title.lower())
    return {
        "contracts": [{"id": c.id, "title": c.title, "chars": len(c.text)} for c in contracts],
        "categories": sorted(corpus().expansions),
    }


@app.get("/api/contracts/{contract_id}")
def get_contract(contract_id: str) -> dict:
    contract = _contract(contract_id)
    return {"id": contract.id, "title": contract.title, "text": contract.text}


@app.post("/api/search")
def search(query: Query) -> dict:
    _contract(query.contract_id)
    return {"hits": _hits_json(corpus().search(query.contract_id, query.question, query.category))}


@app.post("/api/ask")
def ask(query: Query) -> dict:
    _contract(query.contract_id)
    hits = corpus().search(query.contract_id, query.question, query.category)
    if not hits:
        return {"answer": {"segments": [{"text": "No part of this contract matched the question.", "citations": []}]},
                "hits": []}
    try:
        result = qa.answer(query.question, [h.chunk for h in hits])
    except anthropic.AuthenticationError:
        raise HTTPException(401, "Claude rejected the API key. Check ANTHROPIC_API_KEY.")
    except anthropic.RateLimitError:
        raise HTTPException(429, "Rate limited by the Claude API. Try again shortly.")
    except anthropic.APIStatusError as e:
        raise HTTPException(502, f"Claude API error {e.status_code}: {e.message}")
    except anthropic.APIConnectionError:
        raise HTTPException(502, "Could not reach the Claude API.")
    except anthropic.AnthropicError as e:  # e.g. no credentials configured
        raise HTTPException(503, f"Answers need Claude API access ({e}). Search still works without it.")
    return {"answer": asdict(result), "hits": _hits_json(hits)}
