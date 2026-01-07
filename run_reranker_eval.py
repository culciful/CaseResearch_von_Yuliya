import json
from typing import Dict, List, Any

from encoder_reranker_stub import EncoderReranker


def _is_gold(c: Dict[str, Any], g: Dict[str, Any]) -> bool:
    return (
        c.get("head") == g.get("head")
        and c.get("relation") == g.get("relation")
        and c.get("tail") == g.get("tail")
        and c.get("date") == g.get("date")
    )


def hit_at_k(ranked: List[Dict[str, Any]], gold: Dict[str, Any], k: int) -> int:
    return int(any(_is_gold(c, gold) for c in ranked[:k]))


def mrr_at_k(ranked: List[Dict[str, Any]], gold: Dict[str, Any], k: int) -> float:
    for i, c in enumerate(ranked[:k], 1):
        if _is_gold(c, gold):
            return 1.0 / i
    return 0.0


def evaluate(bundle_path: str, reranker: EncoderReranker, k: int = 10) -> Dict[str, float]:
    data = json.load(open(bundle_path, "r", encoding="utf-8"))
    n = 0
    h1 = h5 = h10 = 0
    mrr = 0.0

    for ex in data:
        q = ex["question"]
        gold = ex["gold"]
        candidates = ex["candidates"]

        ranked = reranker.rerank(q, candidates, top_k=k)

        h1 += hit_at_k(ranked, gold, 1)
        h5 += hit_at_k(ranked, gold, 5)
        h10 += hit_at_k(ranked, gold, 10)
        mrr += mrr_at_k(ranked, gold, k)
        n += 1

    return {
        "n": n,
        "Hit@1": h1 / n if n else 0.0,
        "Hit@5": h5 / n if n else 0.0,
        "Hit@10": h10 / n if n else 0.0,
        "MRR@10": mrr / n if n else 0.0,
    }


if __name__ == "__main__":
    bundle = "reranker_eval_bundle.json"
    rr = EncoderReranker(model_name="sentence-transformers/all-MiniLM-L6-v2")
    scores = evaluate(bundle, rr, k=10)
    print(scores)
