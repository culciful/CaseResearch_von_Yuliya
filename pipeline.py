"""
Components:
1. extractor.py        → Extract entities + dates from question
2. implicit expansion  → Pattern-based: "Role (Country)" → add "Country"
3. baseline_retriever  → Retrieve candidates (entity index lookup)
4. time_filter.py      → Filter by temporal constraints
5. encoder_reranker.py → Rerank by semantic similarity
"""

from typing import Dict
from retrieval.baseline_retriever import BaselineRetriever
from retrieval.time_filter import TimeFilter
from retrieval.encoder_reranker import EncoderReranker
from preprocess.entity_extract import extract, wikipedia_candidates
from preprocess.expansion import (load_implicit_graph, expand_entities_pattern_based)

# main pipeline class
class TKGQAPipeline:

    def __init__(
        self,
        implicit_graph_path: str,
        icews_path: str,
        encoder_model_name: str = "BAAI/bge-large-en-v1.5",
        time_tolerance_days: int = 30,
        device: str = "cpu",
        retriever_cap: int = 1000,
    ):
        self.implicit_lookup = load_implicit_graph(implicit_graph_path)
        self.retriever = BaselineRetriever(events_path=icews_path, cap=retriever_cap)
        self.time_filter = TimeFilter(tolerance_days=time_tolerance_days)
        self.encoder = EncoderReranker(model_name=encoder_model_name, device=device)

    def process(
        self,
        question: str,
        encoder_top_k: int = 10,
        rerank_cap: int = 200,
        # ABLATION FLAGS
        use_implicit: bool = True,
        use_time_filter: bool = True,
        use_reranker: bool = True
    ) -> Dict:

        results = {"question": question}

        results["config"] = {
            "use_implicit": use_implicit,
            "use_time_filter": use_time_filter,
            "use_reranker": use_reranker
        }

        # Step 1: Extract entities + dates
        extraction = extract(question)
        entities = extraction["entities"]
        dates = extraction["dates"]
        results["extracted_entities"] = [e["name"] for e in entities]
        results["extracted_dates"] = dates

        # Step 2: Expand entities (PATTERN-BASED expansion)
        if use_implicit:
            expanded = expand_entities_pattern_based(entities, self.implicit_lookup)
        else:
            expanded = [e["name"] for e in entities]
        results["expanded_entities"] = expanded
        
        # Debug: show what expansion added
        original_names = set(e["name"] for e in entities)
        expansion_added = [e for e in expanded if e not in original_names]
        results["expansion_added"] = expansion_added

        # Step 3: Baseline retrieval
        candidates = self.retriever.retrieve(expanded)
        results["retrieved_candidates"] = len(candidates)

        # Step 4: Time filter
        if use_time_filter and dates:
            filtered = self.time_filter.filter(candidates, dates)
        else:
            filtered = candidates
        results["after_time_filter"] = len(filtered)

        # Cap before reranking
        filtered = filtered[:rerank_cap]
        results["rerank_input_capped"] = len(filtered)

        # Step 5: Encoder rerank
        if use_reranker and filtered:
            top_triples = self.encoder.rerank(question, filtered, top_k=encoder_top_k)
        else:
            top_triples = filtered[:encoder_top_k]
        
        results["final_triples"] = top_triples
        results["final_count"] = len(top_triples)

        # Step 6: Wikipedia candidates
        results["wikipedia_candidates"] = wikipedia_candidates(entities)
        return results

