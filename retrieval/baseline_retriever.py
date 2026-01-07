
import json
from typing import List, Dict, Set
from collections import defaultdict


class BaselineRetriever:
    def __init__(self, events_path: str, cap: int = 1000):
        self.cap = cap
        self.events: List[Dict] = []
        self.entity_index: Dict[str, Set[int]] = defaultdict(set)
        self.entity_index_lc: Dict[str, Set[int]] = defaultdict(set)

        # cache of known entity keys for capped substring fallback
        self._keys: List[str] = []
        self._keys_lc: List[str] = []

        self._load_and_index(events_path)

    def _load_and_index(self, path: str) -> None:
        if path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                self.events = json.load(f)
        else:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) >= 4:
                        self.events.append({
                            "head": parts[0],
                            "relation": parts[1],
                            "tail": parts[2],
                            "date": parts[3],
                        })

        for idx, event in enumerate(self.events):
            head = event.get("head")
            tail = event.get("tail")

            if head:
                self.entity_index[head].add(idx)
                self.entity_index_lc[head.lower()].add(idx)

            if tail:
                self.entity_index[tail].add(idx)
                self.entity_index_lc[tail.lower()].add(idx)

        #  materialize key lists once for fallback scanning
        # (use lc index keys; covers both head+tail)
        self._keys_lc = list(self.entity_index_lc.keys())
        # keep original-case keys too (not strictly required, but cheap)
        self._keys = list(self.entity_index.keys())

    def retrieve(self, entities: List[str], cap: int | None = None) -> List[Dict]:
        cap = cap or self.cap
        indices: Set[int] = set()

        # 1) Exact + lowercase lookup (unchanged behavior)
        for entity in entities:
            if not entity:
                continue
            indices.update(self.entity_index.get(entity, []))
            indices.update(self.entity_index_lc.get(entity.lower(), []))

        # 2) new conservative substring fallback ONLY if nothing found
        if not indices:
            MAX_KEY_HITS = 200  # cap to prevent explosion
            key_hits = 0

            for entity in entities:
                if not entity:
                    continue
                e = entity.strip().lower()
                if len(e) < 4:
                    continue

                # Scan keys (lowercased). Stop early when enough hits.
                for k_lc in self._keys_lc:
                    if e in k_lc or k_lc in e:
                        indices.update(self.entity_index_lc.get(k_lc, []))
                        key_hits += 1
                        if key_hits >= MAX_KEY_HITS:
                            break
                if key_hits >= MAX_KEY_HITS:
                    break

        candidates = [self.events[i] for i in indices][:cap]

        for c in candidates:
            c["score"] = 1.0

        return candidates
