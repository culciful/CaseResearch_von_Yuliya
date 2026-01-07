import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple
from preprocess.extractor import Extractor

_extractor = Extractor(spacy_model="en_core_web_sm")

def run_extractor(question: str):
    return _extractor.extract(question)


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_get_question(ex: Dict[str, Any]) -> str:
    # Your datasets have "question_implicit"
    return ex.get("question_implicit") or ex.get("question") or ""


def role_entities(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [e for e in entities if e.get("type") == "ICEWS_ROLE"]


def entity_names(entities: List[Dict[str, Any]]) -> List[str]:
    out = []
    for e in entities:
        n = e.get("name")
        if isinstance(n, str) and n.strip():
            out.append(n.strip())
    return out


def compute_coverage(data: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Returns:
      summary: coverage metrics + distributions
      failures: list of failure examples (for jsonl)
    """
    n = len(data)
    counts = Counter()
    dist = defaultdict(Counter)
    failures: List[Dict[str, Any]] = []

    for idx, ex in enumerate(data):
        q = safe_get_question(ex)
        gold = ex.get("quadruple", {})

        out = extract_all_entities(q)
        ents = out.get("entities", []) or []
        dates = out.get("dates", []) or []

        ent_names = entity_names(ents)
        roles = role_entities(ents)

        has_any_entity = len(ent_names) > 0
        has_role = len(roles) > 0
        has_date = len(dates) > 0

        counts["total"] += 1
        counts["has_entity"] += int(has_any_entity)
        counts["has_role"] += int(has_role)
        counts["has_date"] += int(has_date)

        # Distributions (useful to diagnose brittleness)
        dist["num_entities"][len(ent_names)] += 1
        dist["num_roles"][len(roles)] += 1
        dist["num_dates"][len(dates)] += 1

        # Failure mining: cases where pipeline will likely struggle
        # 1) No entities at all
        # 2) Has entities but no role entity (when gold has parentheses-style role)
        # 3) Question contains explicit date tokens but extractor found no date
        q_lower = q.lower()
        looks_like_has_date_token = ("-" in q and any(ch.isdigit() for ch in q)) or (" in " in q_lower)

        failure_reasons = []
        if not has_any_entity:
            failure_reasons.append("no_entity")
        if has_any_entity and ("(" in q and ")" in q) and not has_role:
            failure_reasons.append("role_pattern_missed")
        if looks_like_has_date_token and not has_date:
            failure_reasons.append("date_missed")

        if failure_reasons:
            failures.append(
                {
                    "id": ex.get("id", idx),
                    "reasons": failure_reasons,
                    "question": q,
                    "gold_quadruple": gold,
                    "extracted_entities": ent_names,
                    "extracted_entity_types": [e.get("type") for e in ents],
                    "extracted_dates": dates,
                }
            )

        # Track entity type distribution
        for e in ents:
            et = e.get("type", "UNKNOWN")
            dist["entity_type"][et] += 1

    summary = {
        "n": n,
        "coverage": {
            "entity": counts["has_entity"] / n if n else 0.0,
            "role": counts["has_role"] / n if n else 0.0,
            "date": counts["has_date"] / n if n else 0.0,
        },
        "counts": dict(counts),
        "distributions": {
            "num_entities": dict(dist["num_entities"]),
            "num_roles": dict(dist["num_roles"]),
            "num_dates": dict(dist["num_dates"]),
            "entity_type": dict(dist["entity_type"]),
        },
    }
    return summary, failures


def write_csv(summary: Dict[str, Any], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cov = summary["coverage"]
    counts = summary["counts"]

    rows = [
        ("n", summary["n"]),
        ("entity_coverage", f"{cov['entity']:.4f}"),
        ("role_coverage", f"{cov['role']:.4f}"),
        ("date_coverage", f"{cov['date']:.4f}"),
        ("has_entity", counts.get("has_entity", 0)),
        ("has_role", counts.get("has_role", 0)),
        ("has_date", counts.get("has_date", 0)),
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerows(rows)


def write_jsonl(failures: List[Dict[str, Any]], out_path: str, limit: int = 200) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Keep it readable: prioritize "no_entity" first, then others
    def score(fr: Dict[str, Any]) -> Tuple[int, int]:
        reasons = fr.get("reasons", [])
        return (0 if "no_entity" in reasons else 1, len(reasons))

    failures_sorted = sorted(failures, key=score)
    failures_sorted = failures_sorted[:limit]

    with open(out_path, "w", encoding="utf-8") as f:
        for ex in failures_sorted:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def write_md(summary: Dict[str, Any], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cov = summary["coverage"]
    dist = summary["distributions"]

    def fmt_dist(d: Dict[str, Any], topn: int = 10) -> str:
        items = sorted(d.items(), key=lambda x: (-int(x[1]), str(x[0])))
        return "\n".join([f"- {k}: {v}" for k, v in items[:topn]])

    content = f"""# Extractor report

## Coverage
- n: {summary['n']}
- entity coverage: {cov['entity']:.2%}
- role coverage: {cov['role']:.2%}
- date coverage: {cov['date']:.2%}

## Distributions
### num_entities
{fmt_dist(dist.get('num_entities', {}), topn=15)}

### num_roles
{fmt_dist(dist.get('num_roles', {}), topn=15)}

### num_dates
{fmt_dist(dist.get('num_dates', {}), topn=15)}

### entity_type (top)
{fmt_dist(dist.get('entity_type', {}), topn=20)}
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Path to QA json (e.g., official_QA_eval_set.json)")
    ap.add_argument("--out_dir", default="reports", help="Output directory")
    ap.add_argument("--fail_limit", type=int, default=200, help="Max failure examples to write")
    args = ap.parse_args()

    data = load_json(args.dataset)
    summary, failures = compute_coverage(data)

    base = os.path.splitext(os.path.basename(args.dataset))[0]
    csv_path = os.path.join(args.out_dir, f"extractor_coverage_{base}.csv")
    jsonl_path = os.path.join(args.out_dir, f"extractor_failures_{base}.jsonl")
    md_path = os.path.join(args.out_dir, f"extractor_summary_{base}.md")

    write_csv(summary, csv_path)
    write_jsonl(failures, jsonl_path, limit=args.fail_limit)
    write_md(summary, md_path)

    # Minimal console output
    cov = summary["coverage"]
    print(f"n={summary['n']} entity={cov['entity']:.2%} role={cov['role']:.2%} date={cov['date']:.2%}")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {jsonl_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
