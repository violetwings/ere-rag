"""
document_text.py

Single source of truth for reconstructing marked event text from the raw
MAVEN-ERE data. Every other module gets event text through this module --
nothing reads a cached/stored "wrapped_sentence" field for display purposes.
This is the fix for the earlier bug where a sentence containing more than
one event mention got emitted once per mention instead of once with all
its markers merged together.

The retriever's cache is still used (in pipeline.py), but ONLY to rank
candidates for the retriever-filtered per-query mode -- never as a source
of display text.

Usage:
    load_raw_maven_ere(Path("data/raw/MAVEN_ERE/valid.jsonl"))
    ids = event_ids_for_doc(doc_id)                       # reading order
    text = build_marked_text(doc_id, {"m1": "q", "m2": "e0"})
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

# doc_id -> raw document dict (needs "tokens": list[list[str]])
_RAW_DOC_INDEX: dict[str, dict] = {}
# doc_id -> {mention_id: (sent_id, (start, end), trigger_word)}
_MENTION_INDEX: dict[str, dict[str, tuple[int, tuple[int, int], str]]] = {}


def load_raw_maven_ere(path: Path) -> None:
    """Index every event mention of every document by mention_id. Call once
    at pipeline startup, before any other function in this module is used.

    Only doc["events"] (coreference chains of event mentions) are indexed --
    TIMEX expressions are deliberately excluded, since causal relations in
    MAVEN-ERE never reference TIMEX ids (temporal-only), matching how the
    causal Document class elsewhere in this project already treats them.
    """
    global _RAW_DOC_INDEX, _MENTION_INDEX
    _RAW_DOC_INDEX = {}
    _MENTION_INDEX = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            _RAW_DOC_INDEX[doc["id"]] = doc
            mentions: dict[str, tuple[int, tuple[int, int], str]] = {}
            for chain in doc.get("events", []):
                for m in chain.get("mention", chain.get("mentions", [])):
                    mentions[m["id"]] = (m["sent_id"], tuple(m["offset"]), m["trigger_word"])
            _MENTION_INDEX[doc["id"]] = mentions


def is_loaded() -> bool:
    return bool(_RAW_DOC_INDEX)


def event_ids_for_doc(doc_id: str) -> list[str]:
    """Every event mention id known for this document, in document reading
    order (by sentence, then by token offset within the sentence). This is
    the full candidate universe for whole-document mode and for the
    no-retriever per-query mode.
    """
    mentions = _MENTION_INDEX.get(doc_id, {})
    return sorted(mentions.keys(), key=lambda mid: (mentions[mid][0], mentions[mid][1][0]))


def sort_by_reading_order(doc_id: str, mention_ids: list[str]) -> list[str]:
    """Sort a list of mention ids into document reading order (by sentence,
    then by token offset within the sentence). Use this after any
    similarity-ranked selection (e.g. retriever candidates) so ids get
    displayed as e0, e1, e2... in the order they actually appear in the
    text -- retrieval rank should decide WHICH events are included, not
    what order they're labeled in. Ids not found in the index are pushed
    to the end rather than raising.
    """
    mentions = _MENTION_INDEX.get(doc_id, {})

    def key(mid: str) -> tuple[int, int]:
        if mid not in mentions:
            return (10**9, 10**9)
        sent_id, offset, _trigger = mentions[mid]
        return (sent_id, offset[0])

    return sorted(mention_ids, key=key)


def build_marked_text(
    doc_id: str,
    marker_for_mention: dict[str, str],
    full_document: bool,
) -> str | None:
    """Render text with each mention in marker_for_mention wrapped as
    <LABEL trigger words>, e.g. <q collapsed> or <e3 resigned>.

    full_document=True renders every sentence in the document, in order
    (whole-document mode). full_document=False renders only sentences that
    contain at least one marked mention, in reading order, skipping
    everything else -- keeps a per-query call's context short instead of
    sending the whole document for one query event's candidates.

    Every sentence is rendered at most once, with ALL of its markers
    inserted together in a single pass -- this is what prevents a sentence
    with multiple event mentions from being duplicated once per mention.

    Returns None if the document isn't in the index (raw data not loaded
    yet, or this doc_id has no raw record).
    """
    doc = _RAW_DOC_INDEX.get(doc_id)
    mention_index = _MENTION_INDEX.get(doc_id)
    if doc is None or mention_index is None:
        return None

    by_sentence: dict[int, list[tuple[tuple[int, int], str]]] = defaultdict(list)
    for mention_id, label in marker_for_mention.items():
        if mention_id not in mention_index:
            continue
        sent_id, offset, _trigger = mention_index[mention_id]
        by_sentence[sent_id].append((offset, label))

    sentences_out = []
    for sent_id, tokens in enumerate(doc["tokens"]):
        markers = sorted(by_sentence.get(sent_id, []), key=lambda x: x[0][0])
        if not markers:
            if full_document:
                sentences_out.append(" ".join(tokens))
            continue
        pieces: list[str] = []
        cursor = 0
        for (start, end), label in markers:
            if start < cursor:
                continue  # overlapping mention span already covered
            pieces.append(" ".join(tokens[cursor:start]))
            pieces.append(f"<{label} {' '.join(tokens[start:end])}>")
            cursor = end
        pieces.append(" ".join(tokens[cursor:]))
        sentences_out.append(" ".join(p for p in pieces if p))

    return " ".join(sentences_out)