"""
time_filter.py - Filter triples by temporal constraints
This module filters TransE-retrieved triples based on dates extracted from the question.
It is a SEPARATE stage from date extraction (which happens in extractor.py).
"""
from datetime import datetime
from typing import List, Dict


class TimeFilter:
    def __init__(self, tolerance_days: int = 30):
        self.tolerance_days = tolerance_days

    def filter(self, triples: List[Dict], dates: List[Dict]) -> List[Dict]:
        if not dates:
            return triples

        filtered = []
        for triple in triples:
            triple_date = triple.get("date", "")
            for date_info in dates:
                if self._matches(triple_date, date_info):
                    filtered.append(triple)
                    break

        return filtered if filtered else triples

    def _matches(self, triple_date: str, date_info: Dict) -> bool:
        query_date = date_info.get("date")
        date_format = date_info.get("format")

        if not triple_date or not query_date:
            return False

        if date_format == "iso":
            return self._within_tolerance(triple_date, query_date)

        if date_format in {"month_year", "year"}:
            return triple_date.startswith(query_date)

        return False

    def _within_tolerance(self, d1: str, d2: str) -> bool:
        try:
            date1 = datetime.strptime(d1, "%Y-%m-%d")
            date2 = datetime.strptime(d2, "%Y-%m-%d")
            return abs((date1 - date2).days) <= self.tolerance_days
        except ValueError:
            return False

