
import torch
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer


class EncoderReranker:
    def __init__(self, model_name: str, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)

    def rerank(self, question: str, triples: List[Dict], top_k: int = 10) -> List[Dict]:
        if not triples:
            return []

        if len(triples) <= top_k:
            return triples

        texts = [self._triple_to_text(t) for t in triples]

        q_emb = self.model.encode(
            [question], convert_to_tensor=True, normalize_embeddings=True
        )
        t_emb = self.model.encode(
            texts, convert_to_tensor=True, normalize_embeddings=True
        )

        scores = torch.mm(q_emb, t_emb.T).squeeze(0).cpu().tolist()

        for triple, score in zip(triples, scores):
            triple["retriever_score"] = triple.get("score", 0.0)
            triple["score"] = float(score)

        return sorted(triples, key=lambda x: x["score"], reverse=True)[:top_k]

    @staticmethod
    def _triple_to_text(triple: Dict) -> str:
        head = triple.get("head", "")
        rel = triple.get("relation", "").replace("_", " ").lower()
        tail = triple.get("tail", "")
        date = triple.get("date", "")
        return f"{head} {rel} {tail} on {date}"
