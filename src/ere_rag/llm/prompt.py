"""
prompt.py

Builds the prompt for batch LLM relation classification.

Design matches LLMERE's approach:
  - One prompt per (query_event, partition_of_candidates) pair
  - All candidates in a partition classified in a single LLM call
  - Rationale generation requested for every prediction (RQ3)

Relation label sets follow MAVEN-ERE exactly:
  Temporal: BEFORE, OVERLAP, CONTAINS, SIMULTANEOUS, ENDS-ON, BEGINS-ON, NOREL
  Causal:   CAUSE, PRECONDITION, NOREL
"""

# ── Label sets ─────────────────────────────────────────────────────────────

TEMPORAL_LABELS = ["BEFORE", "OVERLAP", "CONTAINS", "SIMULTANEOUS",
                   "ENDS-ON", "BEGINS-ON", "NOREL"]

CAUSAL_LABELS   = ["CAUSE", "PRECONDITION", "NOREL"]

ALL_LABELS      = TEMPORAL_LABELS[:-1] + CAUSAL_LABELS  # deduplicate NOREL at end
LABEL_SET_STR   = ", ".join(ALL_LABELS)


# ── Sentence reconstruction helper ─────────────────────────────────────────

def get_sentence(document_data: dict, sent_id: int) -> str:
    """Reconstruct sentence from tokens[] list."""
    tokens = document_data.get("tokens", [])
    if sent_id < len(tokens):
        return " ".join(tokens[sent_id])
    return ""


# ── Prompt builder ─────────────────────────────────────────────────────────

def build_prompt(
    query_event:      dict,
    candidate_events: list[dict],
    document_data:    dict,
) -> str:
    """
    Build a single prompt that classifies all candidates against the query
    event in one LLM call (batch classification) using an Expert Persona.
    """
    def get_trigger(event: dict) -> str:
        return event.get("trigger_word") or event.get("mention", "")

    def event_sentence(event: dict) -> str:
        sent = get_sentence(document_data, event.get("sent_id", 0))
        trigger = get_trigger(event)
        # Wrap the exact trigger in <m> tags for the LLM to focus on
        if trigger and trigger in sent:
            sent = sent.replace(trigger, f"<m>{trigger}</m>", 1)
        else:
            sent = f"{sent} <m>{trigger}</m>"
        return sent

    # Build document excerpt — sentences containing query or any candidate
    relevant_sent_ids = {query_event.get("sent_id", 0)}
    for c in candidate_events:
        relevant_sent_ids.add(c.get("sent_id", 0))

    # Include one sentence of context on each side of each relevant sentence
    context_ids: set[int] = set()
    tokens = document_data.get("tokens", [])
    n_sents = len(tokens)
    for sid in relevant_sent_ids:
        for offset in [-1, 0, 1]:
            if 0 <= sid + offset < n_sents:
                context_ids.add(sid + offset)

    context_lines = []
    for sid in sorted(context_ids):
        sent = " ".join(tokens[sid]) if sid < len(tokens) else ""
        context_lines.append(f"[S{sid}] {sent}")
    context_text = "\n".join(context_lines)

    # Query event line
    query_trigger = get_trigger(query_event)
    query_sent    = event_sentence(query_event)

    # Candidate lines
    candidates_text = ""
    for i, cand in enumerate(candidate_events):
        cand_trigger = get_trigger(cand)
        cand_sent    = event_sentence(cand)
        candidates_text += (
            f"e{i}: {cand_trigger}\n"
            f"    Sentence: {cand_sent}\n"
        )

    # Build dynamic output formatting instructions based on partition size
    format_lines = "\n".join([f"  e{i}: LABEL | reason" for i in range(len(candidate_events))])

    prompt = f"""You are an expert linguist performing Event Relation Extraction (ERE). 
Your task is to classify the temporal and causal relationship between a "Query" event and several "Candidate" events using the provided Document Context.

CRITICAL RULES:
1. MARKER TAGS: The exact event triggers are enclosed in <m> and </m> tags. You MUST evaluate the relationship between these specific marked words, even if they share the exact same sentence. Ignore other unmarked events.
2. TIE-BREAKER: If a relationship is both temporal and causal, Causal usually takes precedence (e.g., if A causes B, label it CAUSE rather than BEFORE).

RELATION DEFINITIONS:
- CAUSE: The query event directly causes or results in the candidate event.
- PRECONDITION: The query event is required for the candidate to happen, but doesn't directly cause it.
- BEFORE: The query event finishes completely before the candidate event starts.
- OVERLAP: The events share some time, but one doesn't completely contain the other.
- CONTAINS: The query event's timeframe completely surrounds the candidate event.
- SIMULTANEOUS: The events occur at the exact same time.
- BEGINS-ON: The query event starts at the exact same time as the candidate event.
- ENDS-ON: The query event finishes at the exact same time as the candidate event.
- NOREL: There is no direct temporal or causal relationship, or they are unrelated background events.

--- DOCUMENT CONTEXT ---
{context_text}

--- TARGET EVENTS ---
Query event (head): <m>{query_trigger}</m>
Query sentence: {query_sent}

Candidate events (tails) to classify:
{candidates_text.rstrip()}

--- TASK ---
Format your response exactly as:
{format_lines}
"""

    return prompt


# ── Prompt partitioner ─────────────────────────────────────────────────────

def partition_candidates(
    candidates:     list[dict],
    partition_size: int = 5,
) -> list[list[dict]]:
    """
    Split a candidate list into partitions of at most partition_size.
    """
    return [
        candidates[i : i + partition_size]
        for i in range(0, len(candidates), partition_size)
    ]