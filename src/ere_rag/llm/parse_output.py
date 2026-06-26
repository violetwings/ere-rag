"""
parse_output.py

Parses LLM output into structured relation predictions.

Preferred format, produced by the rewritten prompt.py, is LLMERE-style
relation buckets:

    EXTRACTION RESULTS
    TEMPORAL:
    BEFORE: <e0>, <e3>
    OVERLAP: none
    CAUSAL:
    CAUSE: <e2>
    PRECONDITION: none
    RATIONALES:
    <e0>: BEFORE - reason; <e2>: CAUSE - reason

For compatibility, the parser also accepts the old line format:

    e0_temporal: BEFORE | reason
    e0_causal: CAUSE | reason
"""

from __future__ import annotations

import re
from collections import defaultdict

TEMPORAL_LABELS = [
    "SIMULTANEOUS",
    "ENDS-ON",
    "BEGINS-ON",
    "OVERLAP",
    "CONTAINS",
    "BEFORE",
    "NOREL",
]

CAUSAL_LABELS = ["CAUSE", "PRECONDITION", "NOREL"]

LABELS_BY_CATEGORY = {
    "temporal": TEMPORAL_LABELS,
    "causal": CAUSAL_LABELS,
}

LABEL_MAPS = {
    category: {label.lower(): label for label in labels}
    for category, labels in LABELS_BY_CATEGORY.items()
}

LABEL_TO_CATEGORY = {
    label: category
    for category, labels in LABELS_BY_CATEGORY.items()
    for label in labels
    if label != "NOREL"
}

LEGACY_LINE_RE = re.compile(
    r"e(\d+)_(temporal|causal)\**\s*:\s*\**([A-Za-z\-]+)\**[.,]?\s*\|?\s*(.*)",
    re.IGNORECASE,
)

BUCKET_LINE_RE = re.compile(r"^\s*(?:[-*]\s*)?([A-Za-z][A-Za-z\-]*)\s*:\s*(.*?)\s*$")

EVENT_ID_RE = re.compile(
    r"<\s*e(\d+)(?:\s+[^>]*)?\s*>"
    r"|(?<![A-Za-z0-9_])e(\d+)(?![A-Za-z0-9_])"
    r"|(?:candidate|event)\s*(\d+)",
    re.IGNORECASE,
)

RATIONALE_START_RE = re.compile(r"^\s*RATIONAL(?:E|ES)\s*:?\s*$", re.IGNORECASE)


def _expected_categories(relation_type: str) -> list[str]:
    if relation_type == "both":
        return ["temporal", "causal"]
    if relation_type not in LABELS_BY_CATEGORY:
        raise ValueError(f"Unsupported relation_type: {relation_type}")
    return [relation_type]


def _resolve_label(raw: str, category: str) -> tuple[str, str]:
    cleaned = raw.strip().strip("*_`.,;:").lower()
    label_map = LABEL_MAPS[category]

    if cleaned in label_map:
        return label_map[cleaned], "ok"

    for lower, canonical in label_map.items():
        if cleaned.startswith(lower):
            return canonical, "ok"

    return "NOREL", "fallback"


def _resolve_bucket_label(raw: str, expected: list[str]) -> tuple[str | None, str | None]:
    for category in expected:
        label, status = _resolve_label(raw, category)
        if status == "ok" and label != "NOREL":
            return label, category
    return None, None


def _event_indices(text: str, n_candidates: int) -> list[int]:
    ids = []
    for match in EVENT_ID_RE.finditer(text):
        idx_text = match.group(1) or match.group(2) or match.group(3)
        idx = int(idx_text)
        if 0 <= idx < n_candidates and idx not in ids:
            ids.append(idx)
    return ids


def _is_none_payload(text: str) -> bool:
    cleaned = text.strip().lower()
    return cleaned in {"none", "n/a", "na", "no", "no relation", "norel", "-"}


def _put_prediction(
    parsed: dict[int, dict[str, tuple[str, str, str]]],
    idx: int,
    category: str,
    label: str,
    rationale: str = "",
    status: str = "ok",
) -> None:
    if category not in parsed[idx] or parsed[idx][category][0] == "NOREL":
        parsed[idx][category] = (label, rationale.strip(), status)
    elif rationale and not parsed[idx][category][1]:
        old_label, _, old_status = parsed[idx][category]
        parsed[idx][category] = (old_label, rationale.strip(), old_status)


def _parse_legacy_line(
    line: str,
    parsed: dict[int, dict[str, tuple[str, str, str]]],
    n_candidates: int,
    expected: list[str],
) -> bool:
    match = LEGACY_LINE_RE.search(line)
    if not match:
        return False

    idx = int(match.group(1))
    category = match.group(2).lower()
    if idx >= n_candidates or category not in expected:
        return True

    label, status = _resolve_label(match.group(3), category)
    rationale = (match.group(4) or "").strip()
    _put_prediction(parsed, idx, category, label, rationale, status)
    return True


def _parse_bucket_line(
    line: str,
    parsed: dict[int, dict[str, tuple[str, str, str]]],
    n_candidates: int,
    expected: list[str],
) -> bool:
    match = BUCKET_LINE_RE.match(line)
    if not match:
        return False

    label, category = _resolve_bucket_label(match.group(1), expected)
    if not label or not category:
        return False

    payload = match.group(2)
    if _is_none_payload(payload):
        return True

    ids = _event_indices(payload, n_candidates)
    if not ids:
        return True

    rationale = ""
    if " - " in payload:
        rationale = payload.split(" - ", 1)[1].strip()
    elif " because " in payload.lower():
        # Keep the original casing while finding a case-insensitive boundary.
        boundary = payload.lower().find(" because ")
        rationale = payload[boundary + 1 :].strip()

    for idx in ids:
        _put_prediction(parsed, idx, category, label, rationale, "ok")
    return True


def _candidate_rationale_items(llm_output: str) -> list[str]:
    """Return semicolon/newline-separated rationale items from the rationale
    section.  If the model omitted an explicit RATIONALES heading, this still
    finds candidate-prefixed items anywhere in the output.
    """
    lines = llm_output.splitlines()
    in_rationales = False
    collected = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if RATIONALE_START_RE.match(stripped):
            in_rationales = True
            continue
        if in_rationales:
            collected.append(stripped)
        elif EVENT_ID_RE.search(stripped) and re.search(r":| - |\|", stripped):
            collected.append(stripped)

    items = []
    for line in collected:
        items.extend(part.strip() for part in line.split(";") if part.strip())
    return items


def _parse_rationales(
    llm_output: str,
    parsed: dict[int, dict[str, tuple[str, str, str]]],
    n_candidates: int,
    expected: list[str],
) -> None:
    for item in _candidate_rationale_items(llm_output):
        ids = _event_indices(item, n_candidates)
        if not ids:
            continue

        # Handles: <e0>: temporal=BEFORE - reason | causal=NOREL - reason
        for category in expected:
            cat_match = re.search(
                rf"{category}\s*=\s*([A-Za-z\-]+)\s*(?:[-|:]\s*)?(.*)",
                item,
                flags=re.IGNORECASE,
            )
            if cat_match:
                label, status = _resolve_label(cat_match.group(1), category)
                rationale = cat_match.group(2).strip()
                for idx in ids:
                    _put_prediction(parsed, idx, category, label, rationale, status)

        # Handles: <e0>: BEFORE - reason
        after_colon = item.split(":", 1)[1].strip() if ":" in item else item
        label_match = re.match(r"([A-Za-z\-]+)\s*(?:[-|:]\s*)?(.*)$", after_colon)
        if not label_match:
            continue

        raw_label = label_match.group(1)
        rationale = label_match.group(2).strip()

        if raw_label.upper() == "NOREL" and len(expected) == 1:
            category = expected[0]
            for idx in ids:
                _put_prediction(parsed, idx, category, "NOREL", rationale, "ok")
            continue

        category = LABEL_TO_CATEGORY.get(raw_label.upper())
        if category in expected:
            for idx in ids:
                existing = parsed[idx].get(category, ("NOREL", "", "missing"))
                _put_prediction(parsed, idx, category, existing[0], rationale, existing[2])


def parse_llm_output(
    llm_output: str,
    n_candidates: int,
    candidate_events: list[dict],
    relation_type: str = "both",
) -> list[dict]:
    """Parse model output into one result per candidate/relation family."""
    expected = _expected_categories(relation_type)
    parsed: dict[int, dict[str, tuple[str, str, str]]] = defaultdict(dict)
    saw_bucket_format = False

    for raw_line in llm_output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _parse_legacy_line(line, parsed, n_candidates, expected):
            continue
        if _parse_bucket_line(line, parsed, n_candidates, expected):
            saw_bucket_format = True
            
    _parse_rationales(llm_output, parsed, n_candidates, expected)

    default_status = "implicit_norel" if saw_bucket_format else "missing"
    
    results = []
    for i, cand in enumerate(candidate_events):
        cand_id = cand.get("id") or cand.get("head_mention_id") or cand.get("mention_id") or f"e{i}"
        for category in expected:
            label, rationale, status = parsed[i].get(category, ("NOREL", "", default_status))
            results.append(
                {
                    "candidate_idx": i,
                    "candidate_id": cand_id,
                    "relation": label,
                    "relation_category": category,
                    "rationale": rationale,
                    "parse_status": status,
                }
            )
    return results


def predictions_to_pairs(query_event: dict, predictions: list[dict], doc_id: str) -> list[dict]:
    query_id = query_event.get("id") or query_event.get("head_mention_id") or query_event.get("mention_id") or ""
    return [
        {
            "doc_id": doc_id,
            "head_mention_id": query_id,
            "tail_mention_id": pred["candidate_id"],
            "predicted_label": pred["relation"],
            "relation_category": pred["relation_category"],
            "rationale": pred["rationale"],
            "parse_status": pred["parse_status"],
        }
        for pred in predictions
    ]
