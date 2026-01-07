import json
import re
from typing import Dict, List, Optional
import os, sys
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os; print(os.getcwd())

from pipeline import TKGQAPipeline
from preprocess.entity_extract import extract
from eval.utils import compute_hit_mrr, write_table, format_table



def quadruple_equal(candidate: Dict, gold: Dict) -> bool:
    return (
        candidate.get("head") == gold.get("s")
        and candidate.get("relation") == gold.get("p")
        and candidate.get("tail") == gold.get("o")
        and candidate.get("date") == gold.get("t")
    )


def _load_devset(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_single_config(
    pipeline: TKGQAPipeline,
    data: List[Dict],
    top_k: int = 10,
    use_implicit: bool = True,
    use_time_filter: bool = True,
    use_reranker: bool = True,
    verbose: bool = False,
) -> Dict[str, float]:
    ranks: List[Optional[int]] = []

    for idx, item in enumerate(data, start=1):
        question = item["question_implicit"]
        gold = item["quadruple"]

        results = pipeline.process(
            question=question,
            use_implicit=use_implicit,
            use_time_filter=use_time_filter,
            use_reranker=use_reranker,
        )

        candidates = results.get("final_triples", [])
        found_rank: Optional[int] = None

        for rank, cand in enumerate(candidates[:top_k], start=1):
            if quadruple_equal(cand, gold):
                found_rank = rank
                break

        ranks.append(found_rank)

        if verbose:
            status = "✓" if found_rank else "✗"
            expansion_added = results.get("expansion_added", [])
            extra = f" expanded={expansion_added}" if expansion_added else ""
            print(f"[{idx}/{len(data)}] {status}{extra}")

    return compute_hit_mrr(ranks, k=top_k)


def run_ablation_study(
    pipeline: TKGQAPipeline,
    dev_path: str,
    top_k: int = 10,
    verbose: bool = False,
    out_dir: str = "results",
) -> List[Dict]:
    data = _load_devset(dev_path)

    configs = [
        ("Full pipeline", dict(use_implicit=True, use_time_filter=True, use_reranker=True)),
        ("No implicit expansion", dict(use_implicit=False, use_time_filter=True, use_reranker=True)),
        ("No time filter", dict(use_implicit=True, use_time_filter=False, use_reranker=True)),
        ("No reranker", dict(use_implicit=True, use_time_filter=True, use_reranker=False)),
    ]

    rows: List[Dict] = []
    for name, cfg in configs:
        metrics = evaluate_single_config(
            pipeline=pipeline,
            data=data,
            top_k=top_k,
            verbose=verbose,
            **cfg,
        )
        rows.append(
            {
                "Setting": name,
                f"Hit@{top_k}": metrics[f"hit@{top_k}"],
                "MRR": metrics["mrr"],
                "Hits": f"{metrics['hits']}/{metrics['total']}",
            }
        )

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(dev_path))[0]
    write_table(
        rows,
        csv_path=os.path.join(out_dir, f"ablation_{base}_k{top_k}.csv"),
        json_path=os.path.join(out_dir, f"ablation_{base}_k{top_k}.json"),
    )

    print(f"\nAblation study on {dev_path} (n={len(data)})")
    print(format_table(rows, float_cols=(f"Hit@{top_k}", "MRR")))
    return rows


def run_incremental_ablation(
    pipeline: TKGQAPipeline,
    dev_path: str,
    top_k: int = 10,
    verbose: bool = False,
    out_dir: str = "results",
) -> List[Dict]:
    data = _load_devset(dev_path)

    configs = [
        ("Baseline (retrieval only)", dict(use_implicit=False, use_time_filter=False, use_reranker=False)),
        ("+ implicit expansion", dict(use_implicit=True, use_time_filter=False, use_reranker=False)),
        ("+ time filter", dict(use_implicit=True, use_time_filter=True, use_reranker=False)),
        ("+ reranker (full)", dict(use_implicit=True, use_time_filter=True, use_reranker=True)),
    ]

    rows: List[Dict] = []
    for name, cfg in configs:
        metrics = evaluate_single_config(
            pipeline=pipeline,
            data=data,
            top_k=top_k,
            verbose=verbose,
            **cfg,
        )
        rows.append(
            {
                "Setting": name,
                f"Hit@{top_k}": metrics[f"hit@{top_k}"],
                "MRR": metrics["mrr"],
                "Hits": f"{metrics['hits']}/{metrics['total']}",
            }
        )

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(dev_path))[0]
    write_table(
        rows,
        csv_path=os.path.join(out_dir, f"incremental_{base}_k{top_k}.csv"),
        json_path=os.path.join(out_dir, f"incremental_{base}_k{top_k}.json"),
    )

    print(f"\nIncremental ablation on {dev_path} (n={len(data)})")
    print(format_table(rows, float_cols=(f"Hit@{top_k}", "MRR")))
    return rows


def debug_expansion(dev_path: str, limit: int = 10) -> None:
    data = _load_devset(dev_path)
    pattern = r"^.+\(([^)]+)\)$"

    print(f"\nExpansion debug on {dev_path} (showing {min(limit, len(data))}/{len(data)})")
    for idx, item in enumerate(data[:limit], start=1):
        question = item["question_implicit"]
        gold_s = item["quadruple"]["s"]

        extraction = extract(question)
        entity_names = [e["name"] for e in extraction.get("entities", [])]

        expansion_adds = []
        for name in entity_names:
            match = re.match(pattern, name)
            if match:
                country = match.group(1)
                if country not in entity_names:
                    expansion_adds.append(f"{name} -> {country}")

        print(f"\n[{idx}] {question}")
        print(f"  Gold subject: {gold_s}")
        print(f"  Extracted: {entity_names}")
        print(f"  Expansion adds: {expansion_adds or 'nothing'}")


def main():
    RUN_DEBUG_EXPANSION = False

    pipeline = TKGQAPipeline(
        implicit_graph_path="implicit_relation_graph.json",
        icews_path="icews_2014_train.txt",
        encoder_model_name="BAAI/bge-large-en-v1.5",
        time_tolerance_days=30,
        device="cpu",
    )

    if RUN_DEBUG_EXPANSION:
        debug_expansion("mini_qa_devset.json", limit=10)

    run_ablation_study(pipeline, dev_path="mini_qa_devset.json", top_k=10, verbose=False)
    run_incremental_ablation(pipeline, dev_path="mini_qa_devset.json", top_k=10, verbose=False)


if __name__ == "__main__":
    main()
