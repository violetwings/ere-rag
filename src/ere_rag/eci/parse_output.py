"""
parse_output.py

Parsing for the ECI (binary causal-relatedness) pipeline. Every config --
whole-document or per-query, with or without reasoning -- produces the
same shape of output:

    [optional REASONING: ... free text ...]
    EXTRACTION RESULTS
    RELATED: <X> -> <Y>, <A> -> <B>, ... (or none)
    [optional RATIONALES: ...]

There is no label to resolve (CAUSE vs PRECONDITION) -- a pair either
appears under RELATED (parsed as related) or it doesn't (implicitly
NOREL). One parser handles both whole-document ids (<e0>, <e1>, ...) and
per-query ids (<q>, <e0>, <e1>, ...) -- the caller supplies the set of
labels that are valid for that call, and anything else is dropped as a
hallucinated id rather than silently accepted.
"""

from __future__ import annotations

import re

EXTRACTION_RESULTS_RE = re.compile(r"^\s*EXTRACTION RESULTS\s*$", re.IGNORECASE)
PAIR_RE = re.compile(r"<\s*([A-Za-z0-9]+)\s*>\s*(?:->|-->|→)\s*<\s*([A-Za-z0-9]+)\s*>")
RELATED_LINE_RE = re.compile(r"^\s*RELATED\s*:\s*(.*)$", re.IGNORECASE)


def split_reasoning_and_results(llm_output: str) -> tuple[str, str]:
    """Everything before a line that is exactly "EXTRACTION RESULTS" is
    reasoning (kept only for logging, never scanned for pairs); everything
    from that line onward is the parsed section. If no such line exists,
    the whole response is treated as the parsed section.
    """
    lines = llm_output.splitlines()
    for i, line in enumerate(lines):
        if EXTRACTION_RESULTS_RE.match(line):
            return "\n".join(lines[:i]).strip(), "\n".join(lines[i:])
    return "", llm_output


def extract_reasoning_preamble(llm_output: str) -> str:
    reasoning, _ = split_reasoning_and_results(llm_output)
    return reasoning


def _is_none_payload(text: str) -> bool:
    cleaned = text.strip().lower()
    return cleaned in {"none", "n/a", "na", "no", "no relation", "norel", "-", ""}


def parse_pairs(llm_output: str, valid_labels: set[str]) -> list[dict]:
    """Return every distinct {"head", "tail", "rationale"} dict listed
    after a RELATED: line, restricted to labels in valid_labels (drops
    anything referencing an id that wasn't actually offered to the model).
    A later RATIONALES-style line can attach a short justification to a
    pair already found under RELATED, but never introduces a new pair on
    its own.
    """
    _, results_text = split_reasoning_and_results(llm_output)
    pairs: dict[tuple[str, str], dict] = {}

    for raw_line in results_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        m = RELATED_LINE_RE.match(line)
        if m:
            payload = m.group(1)
            if not _is_none_payload(payload):
                for pm in PAIR_RE.finditer(payload):
                    h, t = pm.group(1), pm.group(2)
                    if h in valid_labels and t in valid_labels and h != t:
                        pairs.setdefault((h, t), {"head": h, "tail": t, "rationale": ""})
            continue

        pm = PAIR_RE.search(line)
        if pm and (":" in line or " - " in line):
            h, t = pm.group(1), pm.group(2)
            if (h, t) in pairs:
                after = line.split(":", 1)[1].strip() if ":" in line else line
                rationale = after.split(" - ", 1)[1].strip() if " - " in after else after
                pairs[(h, t)]["rationale"] = rationale

    return list(pairs.values())