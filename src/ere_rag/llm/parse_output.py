"""
parse_output.py

Parses LLM output into structured relation predictions.
"""

import re
from ere_rag.llm.prompt import TEMPORAL_LABELS, CAUSAL_LABELS

# ── Valid label set ────────────────────────────────────────────────────────

VALID_LABELS: set[str] = set(TEMPORAL_LABELS) | set(CAUSAL_LABELS)
LABEL_MAP: dict[str, str] = {lbl.lower(): lbl for lbl in VALID_LABELS}

# ── Parser ─────────────────────────────────────────────────────────────────

def parse_llm_output(
    llm_output:       str,
    n_candidates:     int,
    candidate_events: list[dict],
) -> list[dict]:
    # Handles both formats:
    #   Single: e0: BEFORE | rationale
    #   Dual:   e0_temporal: BEFORE | rationale
    #           e0_causal: NOREL | rationale
    
    dual_pattern = re.compile(
        r"e(\d+)\**\s*_\s*(temporal|causal)\s*:\s*\**([A-Za-z\-]+)\**\s*\|?\s*(.*)",
        re.IGNORECASE,
    )
    single_pattern = re.compile(
        r"e(\d+)\**\s*:\s*\**([A-Za-z\-]+)\**\s*\|?\s*(.*)",
        re.IGNORECASE,
    )

    # parsed[idx] = {"temporal": (label, rationale, status), "causal": (...)}
    parsed: dict[int, dict] = {}

    for line in llm_output.splitlines():
        line = line.strip()
        if not line:
            continue

        m = dual_pattern.search(line)
        if m:
            idx      = int(m.group(1))
            cat      = m.group(2).lower()   # "temporal" or "causal"
            label_raw = m.group(3)
            rationale = (m.group(4) or "").strip()
            if idx >= n_candidates:
                continue
            label, status = _resolve_label(label_raw)
            if idx not in parsed:
                parsed[idx] = {}
            if cat not in parsed[idx]:   # first occurrence wins
                parsed[idx][cat] = (label, rationale, status)
            continue

        m = single_pattern.search(line)
        if m:
            idx       = int(m.group(1))
            label_raw = m.group(2)
            rationale = (m.group(3) or "").strip()
            if idx >= n_candidates:
                continue
            label, status = _resolve_label(label_raw)
            if idx not in parsed:
                parsed[idx] = {}
            # Single format: decide category from label
            if label in {"CAUSE", "PRECONDITION"}:
                cat = "causal"
            else:
                cat = "temporal"
            if cat not in parsed[idx]:
                parsed[idx][cat] = (label, rationale, status)

    # Build output — one entry per (candidate, category) combination
    results = []
    for i, cand in enumerate(candidate_events):
        cand_id = cand.get("id") or cand.get("head_mention_id", f"unk_{i}")
        entry   = parsed.get(i, {})

        for cat in ("temporal", "causal"):
            label, rationale, status = entry.get(cat, ("NOREL", "", "missing"))
            results.append({
                "candidate_idx":      i,
                "candidate_id":       cand_id,
                "relation":           label,
                "relation_category":  cat,
                "rationale":          rationale,
                "parse_status":       status,
            })

    return results

# ── Prediction flattener ───────────────────────────────────────────────────

def predictions_to_pairs(query_event, predictions, doc_id):
    query_id = query_event.get("id") or query_event.get("head_mention_id", "")
    return [
        {
            "doc_id":            doc_id,
            "head_mention_id":   query_id,
            "tail_mention_id":   pred["candidate_id"],
            "predicted_label":   pred["relation"],
            "relation_category": pred["relation_category"],
            "rationale":         pred["rationale"],
            "parse_status":      pred["parse_status"],
        }
        for pred in predictions
    ]

def _resolve_label(raw: str) -> tuple[str, str]:
    cleaned = raw.strip().strip("*_`").lower()

    # Exact match (case-insensitive)
    if cleaned in LABEL_MAP:
        return LABEL_MAP[cleaned], "ok"

    # Prefix match — handles trailing punctuation like "BEFORE." or "CAUSE,"
    for lbl_lower, lbl_canonical in LABEL_MAP.items():
        if cleaned.startswith(lbl_lower):
            return lbl_canonical, "ok"

    # Nothing matched
    return "NOREL", "fallback"
