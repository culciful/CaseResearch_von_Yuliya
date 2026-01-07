from __future__ import annotations
from typing import Dict, List, Any


class EncoderReranker:

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name

        # TODO: load your encoder model here
        # You can use: sentence-transformers, transformers, etc But! Keep it deterministic.

    @staticmethod
    def triple_to_text(tr: Dict[str, Any]) -> str:
        # Keep this format fixed for fair comparison
        return f"{tr.get('head','')} | {tr.get('relation','')} | {tr.get('tail','')} | {tr.get('date','')}"

    def rerank(self, question: str, candidates: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Return candidates sorted by candidate['score'] (desc). Should return at least top_k items if available.

        IMPORTANT:
        - Do NOT change candidates structure.
        - Do NOT use gold labels.
        - Do NOT change 'question' text.
        """
        if not candidates:
            return []

        # TODO:
        # 1) convert candidates to texts: [self.triple_to_text(c) for c in candidates]
        # 2) compute similarity(question, candidate_text)
        # 3) set c['score'] = similarity
        # 4) return sorted(candidates, key=lambda x: x['score'], reverse=True)[:top_k]

        # Baseline fallback (random-ish, not competitive) — replace it.
        for c in candidates:
            c["score"] = float(c.get("retriever_score", 0.0))
        return sorted(candidates, key=lambda x: x["score"], reverse=True)[:top_k]
