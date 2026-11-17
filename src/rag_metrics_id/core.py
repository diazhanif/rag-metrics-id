"""Small, deterministic retrieval-ranking metrics for RAG experiments."""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Iterable


METRIC_NAMES = (
    "precision_at_k",
    "recall_at_k",
    "mrr_at_k",
    "ndcg_at_k",
    "hit_rate_at_k",
)


def _validate_ids(value: object, field: str, query_index: int, allow_empty: bool) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"field {field} pada query {query_index} harus berupa array ID")
    if not allow_empty and not value:
        raise ValueError(f"field {field} pada query {query_index} tidak boleh kosong")
    ids: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str) or not item.strip() or item != item.strip():
            raise ValueError(f"field {field} pada query {query_index} berisi ID yang tidak valid")
        if item in seen:
            raise ValueError(f"field {field} pada query {query_index} berisi ID duplikat")
        seen.add(item)
        ids.append(item)
    return ids


def _query_metrics(relevant: list[str], retrieved: list[str], k: int) -> dict[str, object]:
    relevant_set = set(relevant)
    top = retrieved[:k]
    relevant_ranks = [rank for rank, doc_id in enumerate(top, start=1) if doc_id in relevant_set]
    hits = len(relevant_ranks)

    precision = hits / k
    recall = hits / len(relevant)
    mrr = 1 / relevant_ranks[0] if relevant_ranks else 0.0
    dcg = sum(1 / math.log2(rank + 1) for rank in relevant_ranks)
    ideal_relevant = min(len(relevant), k)
    idcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_relevant + 1))
    ndcg = dcg / idcg if idcg else 0.0

    return {
        "retrieved_count": len(retrieved),
        "relevant_count": len(relevant),
        "hits_at_k": hits,
        "first_relevant_rank": relevant_ranks[0] if relevant_ranks else None,
        "precision_at_k": precision,
        "recall_at_k": recall,
        "mrr_at_k": mrr,
        "ndcg_at_k": ndcg,
        "hit_rate_at_k": 1.0 if hits else 0.0,
    }


def evaluate(records: list[dict[str, object]], k: int = 5) -> dict[str, object]:
    """Evaluate ranked retrieval lists against non-empty relevant-ID sets.

    Precision@k divides by k even when fewer than k documents were retrieved.
    Output contains metrics and row numbers only, not queries or document IDs.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k harus integer positif")
    if not isinstance(records, list) or not records:
        raise ValueError("dataset harus berupa array query yang tidak kosong")

    per_query: list[dict[str, object]] = []
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"query {index} harus berupa objek")
        relevant = _validate_ids(record.get("relevant"), "relevant", index, allow_empty=False)
        retrieved = _validate_ids(record.get("retrieved"), "retrieved", index, allow_empty=True)
        metrics = _query_metrics(relevant, retrieved, k)
        per_query.append({"query_index": index, **metrics})

    macro = {
        name: round(sum(float(row[name]) for row in per_query) / len(per_query), 6)
        for name in METRIC_NAMES
    }
    for row in per_query:
        for name in METRIC_NAMES:
            row[name] = round(float(row[name]), 6)

    return {"k": k, "query_count": len(per_query), "macro_average": macro, "per_query": per_query}


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rag-metrics-id",
        description="Hitung metrik ranking retrieval untuk eksperimen RAG.",
    )
    parser.add_argument("--k", type=int, default=5, help="potongan ranking yang dievaluasi (default: 5)")
    args = parser.parse_args(argv)
    try:
        records = json.loads(sys.stdin.read())
        report = evaluate(records, k=args.k)
    except (json.JSONDecodeError, TypeError, ValueError):
        print("rag-metrics-id: input harus JSON valid berisi array relevant/retrieved, dengan k dan ID yang valid.", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
