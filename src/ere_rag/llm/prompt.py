"""
prompt.py

Prompt builder for the LLM stage of the RAG ERE pipeline.

Instead, candidates are marked in the context and the model returns relation buckets, following this style:

    TEMPORAL:
    BEFORE: <e2>, <e5>
    OVERLAP: none

    CAUSAL:
    CAUSE: <e1>
    PRECONDITION: none
"""

from __future__ import annotations

import re
from typing import Iterable

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


TEMPLATE_CONFIG = {
    
    "llmere_compact": {"strict": False, "rationales": False, "examples": False, "descriptions": True},
    "llmere_rationale": {"strict": True, "rationales": True, "examples": False, "descriptions": True},
    "llmere_few_shot": {"strict": True, "rationales": True, "examples": True, "descriptions": True},
    "llmere_few_shot_no_rationale": {"strict": True, "rationales": False, "examples": True, "descriptions": True},
    
    "llmere_no_desc": {"strict": False, "rationales": False, "examples": False, "descriptions": False},
    "llmere_no_desc_rationale": {"strict": False, "rationales": True, "examples": False, "descriptions": False},
    "llmere_no_desc_few_shot": {"strict": False, "rationales": True, "examples": True, "descriptions": False},
    "llmere_no_desc_few_shot_no_rationale": {"strict": False, "rationales": False, "examples": True, "descriptions": False},
    
    "v1_baseline": {"strict": False, "rationales": False, "examples": False, "descriptions": True},
    "v2_detailed_rationale": {"strict": True, "rationales": True, "examples": False, "descriptions": True},
    "v3_few_shot": {"strict": False, "rationales": False, "examples": True, "descriptions": True},
    "v4_combined": {"strict": True, "rationales": True, "examples": True, "descriptions": True},
    "v5_strict": {"strict": True, "rationales": False, "examples": False, "descriptions": True},
}


TEMPORAL_TASK_DESCRIPTION = """The current task is an event temporal relation extraction task, which aims to identify temporal relations among events in texts. The temporal relation between events refers to the chronological order in which they occur, involving six subtypes: SIMULTANEOUS, ENDS-ON, BEGINS-ON, OVERLAP, CONTAINS, and BEFORE. In the provided context, event trigger words are annotated within angle brackets with event ids, for example <e0 storm>. The desired outcome is a list of candidate events that have temporal relations with the given query event."""

CAUSAL_TASK_DESCRIPTION = """The current task is an event causal relation extraction task, which aims to identify causal relations among events in texts. The causal relation between events denotes that the occurrence of the first event precipitates or enables the happening of the second event, involving two subtypes: CAUSE and PRECONDITION. In the provided context, event trigger words are annotated within angle brackets with event ids, for example <e0 flooding>. The desired outcome is a list of candidate events that have causal relations with the given query event."""


TEMPORAL_DEFINITIONS = """Temporal labels are interpreted from the query event to the candidate event:
SIMULTANEOUS: the query and candidate refer to the same occurrence or happen at the same time.
ENDS-ON: the query and candidate end at the same time, although they may start differently.
BEGINS-ON: the query and candidate start at the same time, although they may end differently.
OVERLAP: the query and candidate time spans intersect, but neither fully contains the other.
CONTAINS: the candidate occurs entirely within the duration of the query event.
BEFORE: the query event finishes before the candidate event begins."""

STRICT_TEMPORAL_DEFINITIONS = """Temporal labels are interpreted from the query event to the candidate event:
SIMULTANEOUS: use only when both triggers describe the same event occurrence or truly concurrent actions.
ENDS-ON: use only when both events share the same ending boundary.
BEGINS-ON: use only when both events share the same starting boundary.
OVERLAP: use when the events share part of a time span, but neither fully contains the other.
CONTAINS: use when the query is a longer event/state/period and the candidate occurs inside it.
BEFORE: use when the query is completed before the candidate begins.
Do not infer a temporal label merely because two events appear in the same sentence."""

CAUSAL_DEFINITIONS = """Causal labels are interpreted from the query event to the candidate event:
CAUSE: the query event directly causes, triggers, produces, or brings about the candidate event.
PRECONDITION: the query event is a necessary setup condition for the candidate event, but does not directly trigger it."""

STRICT_CAUSAL_DEFINITIONS = """Causal labels are interpreted from the query event to the candidate event:
CAUSE: use only for a direct causal trigger in this document context, not for loose topical association.
PRECONDITION: use when the query is required for the candidate to occur, but the candidate is not directly caused by the query.
Do not mark calendar dates, section titles, or background facts as causal unless the text explicitly makes them causal."""


FEW_SHOT_EXAMPLE = """Example:
Query event: <q storm>
Candidate ids: <e0>, <e1>, <e2>
Context:
The rains from <q storm> contributed to <e0 flooding> that caused $700 million in <e1 damage>. The city later began <e2 repairs>.
Expected output:
EXTRACTION RESULTS
CAUSAL:
CAUSE: <e0>, <e1>
PRECONDITION: none
RATIONALES:
CAUSAL:
<e0>: CAUSE - the storm's rains are stated to contribute to flooding.
<e1>: CAUSE - the flooding chain caused damage, so the storm is part of the causal chain to damage.
<e2>: NOREL - repairs happen later but are not stated as caused by the storm in this context."""


MARKED_MENTION_RE = re.compile(r"<m>\s*(.*?)\s*</m>", re.IGNORECASE | re.DOTALL)


def _clean_space(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _event_trigger(event: dict, fallback: str = "event") -> str:
    return _clean_space(
        event.get("trigger")
        or event.get("trigger_word")
        or event.get("text")
        or event.get("mention")
        or fallback
    )


def _mark_wrapped_sentence(sentence: str, marker_id: str, trigger: str) -> str:
    """Convert the retriever's <m>trigger</m> marker into
    angle-bracket markers such as <e0 trigger> or <q trigger>.
    """
    sentence = _clean_space(sentence)
    trigger = _clean_space(trigger)

    def replace(match: re.Match) -> str:
        mention = _clean_space(match.group(1)) or trigger
        return f"<{marker_id} {mention}>"

    if MARKED_MENTION_RE.search(sentence):
        return MARKED_MENTION_RE.sub(replace, sentence, count=1)

    if trigger and trigger in sentence:
        return sentence.replace(trigger, f"<{marker_id} {trigger}>", 1)

    return f"<{marker_id} {trigger}> {sentence}".strip()


def _dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        key = _clean_space(item).lower()
        if key and key not in seen:
            out.append(_clean_space(item))
            seen.add(key)
    return out


def _format_context(
    query_event: dict,
    candidates: list[dict],
    document_context: str | None = None,
) -> str:
    query_trigger = _event_trigger(query_event, "query")
    query_sentence = query_event.get("wrapped_sentence") or query_event.get("sentence") or ""
    query_line = _mark_wrapped_sentence(query_sentence, "q", query_trigger)

    candidate_lines = []
    for i, cand in enumerate(candidates):
        trigger = _event_trigger(cand, f"candidate_{i}")
        sentence = cand.get("wrapped_sentence") or cand.get("sentence") or ""
        candidate_lines.append(_mark_wrapped_sentence(sentence, f"e{i}", trigger))

    if document_context:
        context = _clean_space(document_context)
        # Keep the full context if the caller has it.  The query/candidate
        # snippets below still ensure all target ids are visible.
        return "\n".join(
            [
                "Document/context:",
                context,
                "",
                "Marked target snippets:",
                *[f"- {line}" for line in _dedupe_preserve_order([query_line, *candidate_lines])],
            ]
        )

    return "\n".join(
        [
            "Partition context:",
            *[f"- {line}" for line in _dedupe_preserve_order([query_line, *candidate_lines])],
        ]
    )


def _candidate_id_line(candidates: list[dict]) -> str:
    return "Candidate event ids: " + ", ".join(f"<e{i}>" for i in range(len(candidates)))


def _task_descriptions(relation_type: str) -> str:
    parts = []
    if relation_type in ("temporal", "both"):
        parts.append(TEMPORAL_TASK_DESCRIPTION)
    if relation_type in ("causal", "both"):
        parts.append(CAUSAL_TASK_DESCRIPTION)
    return "\n\n".join(parts)


def _definitions(relation_type: str, strict: bool) -> str:
    parts = []
    if relation_type in ("temporal", "both"):
        parts.append(STRICT_TEMPORAL_DEFINITIONS if strict else TEMPORAL_DEFINITIONS)
    if relation_type in ("causal", "both"):
        parts.append(STRICT_CAUSAL_DEFINITIONS if strict else CAUSAL_DEFINITIONS)
    return "\n\n".join(parts)


def _label_names_only(relation_type: str) -> str:
    parts = []
    if relation_type in ("temporal", "both"):
        parts.append("Temporal relation subtypes: " + ", ".join(TEMPORAL_LABELS[:-1]))
    if relation_type in ("causal", "both"):
        parts.append("Causal relation subtypes: " + ", ".join(CAUSAL_LABELS[:-1]))
    return "\n".join(parts)


def _output_schema(relation_type: str, include_rationales: bool) -> str:
    blocks = ["EXTRACTION RESULTS"]
    if relation_type in ("temporal", "both"):
        blocks.extend(
            [
                "TEMPORAL:",
                "SIMULTANEOUS: <event ids or none>",
                "ENDS-ON: <event ids or none>",
                "BEGINS-ON: <event ids or none>",
                "OVERLAP: <event ids or none>",
                "CONTAINS: <event ids or none>",
                "BEFORE: <event ids or none>",
            ]
        )
    if relation_type in ("causal", "both"):
        blocks.extend(
            [
                "CAUSAL:",
                "CAUSE: <event ids or none>",
                "PRECONDITION: <event ids or none>",
            ]
        )
    if include_rationales:
        blocks.extend(
            [
                "RATIONALES:",
                "For each candidate id, write the predicted label and one text-specific reason.",
                "Use this compact form: <e0>: LABEL - reason; <e1>: NOREL - reason",
            ]
        )
    return "\n".join(blocks)


def _rules(relation_type: str, include_rationales: bool) -> str:
    label_lines = []
    if relation_type in ("temporal", "both"):
        label_lines.append("Temporal labels: " + ", ".join(TEMPORAL_LABELS[:-1]))
    if relation_type in ("causal", "both"):
        label_lines.append("Causal labels: " + ", ".join(CAUSAL_LABELS[:-1]))

    rationale_rule = (
        "\n- Rationales must cite concrete evidence from this context; do not use generic filler."
        if include_rationales
        else ""
    )

    return (
        "Instructions:\n"
        "- The query event is marked as <q ...>. Predict relations from the query event to each candidate event.\n"
        "- Only classify candidate ids listed below. Ignore other unmarked or non-candidate events.\n"
        "- Put a candidate under at most one subtype per relation family. Candidates not listed under a subtype are treated as NOREL for that family.\n"
        "- If no candidate has a subtype, write that subtype as none.\n"
        "- Do not invent candidate ids.\n"
        f"- {'; '.join(label_lines)}."
        f"{rationale_rule}"
    )


def build_prompt(
    version: str,
    query_event: dict,
    candidates: list[dict],
    relation_type: str = "both",
    document_context: str | None = None,
) -> str:
    """Build a compact prompt.

    Args:
        version: Prompt template name.  Unknown values fall back to
            ``llmere_rationale``.
        query_event: Dict with at least trigger_word/trigger and a wrapped
            sentence if available.
        candidates: Candidate event dicts from the retrieval cache.
        relation_type: ``temporal``, ``causal``, or ``both``.
        document_context: Optional full document text.  If absent, the prompt
            uses the query and candidate snippets from the retrieval cache.
    """
    if relation_type not in {"temporal", "causal", "both"}:
        raise ValueError(f"Unsupported relation_type: {relation_type}")

    cfg = TEMPLATE_CONFIG.get(version, TEMPLATE_CONFIG["llmere_rationale"])

    sections = [_task_descriptions(relation_type)]

    if cfg["descriptions"]:
        sections.append(_definitions(relation_type, strict=cfg["strict"]))
    else:
        sections.append(_label_names_only(relation_type))

    sections.append(_rules(relation_type, include_rationales=cfg["rationales"]))

    if cfg["examples"]:
        sections.append(FEW_SHOT_EXAMPLE)

    sections.extend(
        [
            "INPUT",
            f"Query event: <q {_event_trigger(query_event, 'query')}>",
            _candidate_id_line(candidates),
            _format_context(query_event, candidates, document_context=document_context),
            "OUTPUT FORMAT",
            _output_schema(relation_type, include_rationales=cfg["rationales"]),
            "Return only the output. Do not include analysis outside the requested format.",
        ]
    )

    return "\n\n".join(section for section in sections if section)
