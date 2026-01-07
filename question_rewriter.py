
"""
YOUR MAIN CONTRIBUTION: Rule-based temporal normalization
Converts implicit → explicit questions WITHOUT LLM
"""

import re
from typing import Dict, List, Optional, Tuple


TEMPORAL_PATTERNS_ORDERED = [
    ("at_the_time", r"\b[Aa]t\s+the\s+time\s+(?:when\s+)?(.+?)(?=,|\s+who|\s+which|\s+what)"),
    ("following",   r"\b[Ff]ollowing\s+(?:the\s+)?(.+?)(?=,|\s+who|\s+which|\s+what)"),
    ("after",       r"\b[Aa]fter\s+(?:the\s+)?(.+?)(?=,|\s+who|\s+which|\s+what)"),
    ("before",      r"\b[Bb]efore\s+(?:the\s+)?(.+?)(?=,|\s+who|\s+which|\s+what)"),
    ("during",      r"\b[Dd]uring\s+(?:the\s+)?(.+?)(?=,|\s+who|\s+which|\s+what)"),
    ("once",        r"\b[Oo]nce\s+(.+?)(?=,|\s+who|\s+which|\s+what)"),
    ("when",        r"\b[Ww]hen\s+(.+?)(?=,|\s+who|\s+which|\s+what)"),
]

# Convenience dict if you want it elsewhere (not used for iteration)
TEMPORAL_PATTERNS = {k: v for k, v in TEMPORAL_PATTERNS_ORDERED}


class QuestionRewriter:

    def __init__(self, retriever):

        self.retriever = retriever

    def detect_temporal_signal(self, question: str) -> Tuple[Optional[str], Optional[str]]:

        for signal_type, pattern in TEMPORAL_PATTERNS_ORDERED:
            match = re.search(pattern, question)
            if match:
                anchor = match.group(1).strip()
                return signal_type, anchor
        return None, None

    def extract_entities_from_anchor(self, anchor: str) -> List[str]:
        entities: List[str] = []

        role_pattern = r"([A-Z][a-zA-Z\s]+?)\s*\(([^)]+)\)"
        role_matches = re.findall(role_pattern, anchor)

        role_spans = []
        has_full_name = False

        for m in re.finditer(role_pattern, anchor):
            role_spans.append((m.start(), m.end()))

        for role, country in role_matches:
            role_ent = f"{role.strip()} ({country})"
            if role_ent not in entities:
                entities.append(role_ent)
            if country not in entities:
                entities.append(country)

        ROLE_STOP = {"Member", "Judiciary", "Government", "President", "Minister", "Police", "Army"}

        #name_pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"
        name_pattern = r"\b([A-Z][a-z]+(?:\s+(?:[A-Z]\.)?[-']?[A-Za-z]+)*(?:\s+(?:al|el|bin|ibn|van|von|de|da|di|le|la)\s+[A-Z][a-z]+)*)\b"

        for m in re.finditer(name_pattern, anchor):
            name = m.group(1)
            if name in ROLE_STOP:
                continue
            inside_role = any(s <= m.start() < e for s, e in role_spans)
            if inside_role:
                continue
            # prefer multi-token person names; avoid single-token fragments
            if len(name) <= 2:
                continue

            if " " in name or "-" in name:  # full name / hyphenated
                if name not in entities:
                    entities.append(name)
                has_full_name = True
            else:
                # only keep single-token names if we didn't find any full names
                if not has_full_name and name not in entities:
                    entities.append(name)

        #simple normalization variants
        norm = []
        for e in entities:
            e2 = e.replace("'s", "").strip()
            if e2 != e:
                norm.append(e2)
        for x in norm:
            if x not in entities:
                entities.append(x)

        return entities

    def find_anchor_timestamp(
            self,
            anchor_entities: List[str],
            anchor_phrase: str,  # <-- change: pass anchor text (event description)
    ) -> Optional[str]:
        """
        Find timestamp of anchor event from TKG.
        Returns date string (e.g., "2014-11-18") or None.
        """
        if not anchor_entities:
            return None

        candidates = self.retriever.retrieve(anchor_entities)
        if not candidates:
            return None

        # Use ANCHOR text (not whole question) for disambiguation
        ctx = (anchor_phrase or "").lower()

        # Small synonym/normalization map to reduce lexical mismatch with ICEWS relations
        SYN = {
            "praise": ["praise", "praised", "offer praise", "offered praise", "commend", "laud", "hail"],
            "endorse": ["endorse", "endorsed", "back", "support"],
            "consult": ["consult", "consulted", "consultation"],
            "appeal": ["appeal", "appealed", "request", "requested", "call for", "urge"],
            "threaten": ["threaten", "threatened", "threat", "warn", "warning"],
            "reject": ["reject", "rejected", "deny", "denied", "refuse", "refused"],
            "visit": ["visit", "visited", "travel", "traveled", "trip"],
            "meet": ["meet", "met", "meeting", "talk", "talks"],
            "criticize": ["criticize", "criticised", "criticized", "condemn", "condemned"],
            "negotiate": ["negotiate", "negotiated", "negotiations", "bargain"],
            "host": ["host", "hosted"],
        }

        # Flatten to trigger phrases and which canonical relation they map to
        trigger_to_canon = []
        for canon, triggers in SYN.items():
            for t in triggers:
                trigger_to_canon.append((t, canon))

        best_fact = None
        best_score = -1

        for fact in candidates[:50]:
            score = 0
            relation = (fact.get("relation", "") or "").lower()
            head = (fact.get("head", "") or "").lower()
            tail = (fact.get("tail", "") or "").lower()

            # (A) anchor-text ⇄ relation match (with synonyms)
            for trig, canon in trigger_to_canon:
                if trig in ctx:
                    if canon in relation or trig in relation:
                        score += 3  # stronger than before

            # (B) entity alignment: anchor entities appearing in head/tail
            for ent in anchor_entities:
                ent_l = ent.lower()
                if ent_l in head:
                    score += 1
                if ent_l in tail:
                    score += 1

            # Keep best
            if score > best_score:
                best_score = score
                best_fact = fact

        # Only accept if we got some positive evidence
        if best_fact and best_score > 0:
            return best_fact.get("date")

        # Conservative fallback: if disambiguation failed, return None (safer than random date)
        # This typically improves quality even if rewrite rate stays similar.
        return None

    def rewrite(self, question: str) -> Dict:

        result = {
            "original": question,
            "rewritten": question,
            "signal_type": None,
            "anchor_phrase": None,
            "anchor_entities": [],
            "anchor_timestamp": None,
            "was_rewritten": False,
        }

        # Step 1: Detect temporal signal
        signal_type, anchor_phrase = self.detect_temporal_signal(question)
        if not signal_type or not anchor_phrase:
            return result

        result["signal_type"] = signal_type
        result["anchor_phrase"] = anchor_phrase

        # Step 2: Extract entities from anchor
        anchor_entities = self.extract_entities_from_anchor(anchor_phrase)
        result["anchor_entities"] = anchor_entities
        if not anchor_entities:
            return result

        # Step 3: Find anchor timestamp
        timestamp = self.find_anchor_timestamp(anchor_entities, anchor_phrase)
        if not timestamp:
            return result

        result["anchor_timestamp"] = timestamp

        # Step 4: Rewrite
        rewritten = self._apply_template(question, signal_type, timestamp)
        result["rewritten"] = rewritten
        result["was_rewritten"] = (rewritten != question)
        return result

    def _apply_template(self, question: str, signal_type: str, timestamp: str) -> str:
        """Rewrite the temporal introducer into an explicit timestamp, without breaking grammar."""
        replacements = {
            "after":       f"After {timestamp}",
            "before":      f"Before {timestamp}",
            "following":   f"After {timestamp}",
            "when":        f"On {timestamp}",
            "during":      f"On {timestamp}",
            "at_the_time": f"On {timestamp}",
            "once":        f"After {timestamp}",
        }
        replacement_base = replacements.get(signal_type, f"On {timestamp}")

        # Find the exact match span for this signal type (do NOT consume who/which/what/comma)
        pattern = TEMPORAL_PATTERNS.get(signal_type)
        if not pattern:
            return question

        m = re.search(pattern, question)
        if not m:
            return question

        start, end = m.start(), m.end()
        suffix = question[end:]  # begins with comma or space + (who/which/what) due to lookahead

        # Add a comma if the suffix starts with a question word but there's no comma already
        suffix_l = suffix.lstrip()
        needs_comma = suffix and (not suffix.startswith(",")) and suffix_l.startswith(("who", "which", "what"))
        replacement = replacement_base + ("," if needs_comma else "")

        # Ensure we don't glue words together (rare, but safe)
        if suffix and not suffix.startswith((" ", ",", "?", ".")):
            replacement += " "

        return question[:start] + replacement + suffix
