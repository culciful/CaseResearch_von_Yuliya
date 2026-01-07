import json
from preprocess.extractor import Extractor

_extractor = Extractor(spacy_model="en_core_web_sm")

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def coverage(data):
    stats = {"entity": 0, "role": 0, "date": 0}
    n = len(data)

    for ex in data:
        q = ex["question_implicit"]
        out = _extractor.extract(q)

        entities = out.get("entities", [])
        dates = out.get("dates", [])

        if len(entities) > 0:
            stats["entity"] += 1
        if any(e.get("type") == "ICEWS_ROLE" for e in entities):
            stats["role"] += 1
        if len(dates) > 0:
            stats["date"] += 1

    return {k: (v, n, v / n if n else 0.0) for k, v in stats.items()}


if __name__ == "__main__":
    data = load_json("../official_QA_eval_set.json")
    # data = load_json("../mini_qa_devset.json")
    stats = coverage(data)
    for k, (v, n, p) in stats.items():
        print(f"{k:>6}: {v}/{n} ({p:.1%})")



