from typing import Dict, Any, List
from .extractor import Extractor


_extractor = Extractor(spacy_model="en_core_web_sm")


def extract(question: str) -> Dict[str, Any]:
    """
    Returns a dict with:
      - entities: list of {name, type}
      - dates: list of {date, format}
    """
    return _extractor.extract(question)


def wikipedia_candidates(entities: List[Dict[str, Any]]) -> List[str]:
    """
    Placeholder: we currently do NOT use real Wikipedia retrieval.
    Keep function to satisfy pipeline imports.
    """
    return []
