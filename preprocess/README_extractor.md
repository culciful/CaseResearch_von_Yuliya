"""
What the coverage script is for? tells where the bottleneck is and whether extractor changes improve it.
It answers 3 questions quickly:
Entity coverage: “In how many questions do we extract at least 1 entity?”
Role coverage: “In how many questions do we extract at least 1 Role (Country) entity?”
Date coverage: “In how many questions do we extract at least 1 date expression?”

If entity coverage is low → retrieval will fail regardless of reranker/time filter.
If role coverage is low →  implicit expansion contribution collapses.
If date coverage is low → time filtering can’t help.
"""

"""
Yao's work: 
1. Improve extractor.py role parsing robustness
2. Reduce false positives without killing recall
3. Validate improvements using a repeatable measurement

-> The coverage script is simply the validation tool to quantify improvement.

step by step what to do: 

Step A - Run baseline coverage on official eval set
* Run coverage script on current extractor.py
Save results as “baseline”

Step B — Identify failure patterns (sample 20–50 failures)
Questions where:
    - no entities extracted
    - role entity expected but not extracted
    - date mentioned but not extracted
    
Step C — Fix extractor rules
Typical hardening tasks:

    - Role head normalization (case, punctuation, slashes)
    - Handling roles not in the keyword list
    - Better parentheses matching
    - Avoid extracting junk role patterns

Step D — Re-run coverage and compare

    - Coverage should go up (or stay same with fewer false positives)
    - Save results in a report

Step E — Deliver

    - updated extractor.py
    - coverage report
"""
"""Your task is to harden extractor.py (especially Role (Country) parsing
 and validate improvements using a simple coverage script. Please run coverage on
  official_QA_eval_set.json before and after changes and write a short report: entity coverage, 
role coverage, date coverage, plus top failure patterns with examples."""