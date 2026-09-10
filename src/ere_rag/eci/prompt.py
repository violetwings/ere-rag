"""
prompt.py

Prompt construction for binary Event Causality Identification (ECI): a
directed pair <head> -> <tail> is RELATED if the head has any causal
bearing on the tail, or NOREL otherwise. No CAUSE/PRECONDITION subtype is
requested anywhere in this pipeline.

Exactly four configs exist, matching the four experiment conditions:

  whole_doc, reasoning=True   -- one call per document, asks for a
                                  REASONING section before the answer.
  whole_doc, reasoning=False  -- same call shape, no REASONING section.
                                  Paired with the above for RQ3 (the ONLY
                                  difference between the two is the
                                  REASONING instruction/section -- output
                                  schema, few-shot pair content, and
                                  everything else is held constant).
  per_query                   -- one call per query event. Never asks for
                                  reasoning (kept short and cheap on
                                  purpose). Used for both the no-retriever
                                  and retriever-filtered conditions -- the
                                  two conditions differ only in which
                                  candidate ids get sent to this function,
                                  not in the prompt itself.

There is no dispatch table and no legacy prompt_version string anymore --
each config is a plain function.
"""

from __future__ import annotations

import ere_rag.eci.fewshot_examples as fewshot_examples

TASK_DESCRIPTION = (
    "This is an event causality identification (ECI) task. You will judge whether a directed causal "
    "relationship exists between events -- one event playing any causal role in bringing about, enabling, or "
    "being a necessary condition for another -- without needing to name which specific causal subtype applies. "
    "A pair is RELATED if the head event has any causal bearing on the tail event; otherwise it is NOREL."
)

CAUSAL_DEFINITION = (
    "A directed pair <head> -> <tail> is RELATED if either of the following holds:\n"
    "- the head event directly produces the tail event as a guaranteed consequence, or\n"
    "- the tail event would not have happened, as described, if the head event had not happened -- even if "
    "the head alone doesn't guarantee the tail and other actors or steps are also involved.\n"
    "Two events appearing in the same sentence, or one following another in the narrative, is not by itself "
    "evidence of RELATED -- the text must support one of the two tests above. A pair where the head and tail "
    "are two mentions of the same real-world event (coreferent mentions) is NOREL -- an event has no causal "
    "relation with itself. Do not mark calendar dates, section titles, or purely background facts as RELATED "
    "unless the text explicitly gives them a causal role."
)

RELATION_NAMING_NOTE = (
    "Some mentions merely name a causal relationship without being one of its two arguments -- a trigger like "
    "\"caused\", \"triggered\", or \"led to\" often describes the relation between two OTHER events rather than "
    "being a cause or effect in its own right. Do not mark such a mention as RELATED to another candidate on "
    "its own; instead find the two actual events it connects, and judge those if they are also present as "
    "separate candidates."
)

WHOLE_DOC_REASONING_INSTRUCTION = (
    "Before giving your final answer, write a section headed \"REASONING:\". There is no single query event -- "
    "work through the WHOLE document once, checking each plausible pair against the RELATED test above. Name "
    "each pair you find support for and give the evidence in ONE or TWO sentences -- no full paragraph per "
    "pair, and never debate how to interpret these instructions; make the single most reasonable call and "
    "move on. Group events with no causal role into one short closing sentence instead of a note each. After "
    "your reasoning, write the exact heading \"EXTRACTION RESULTS\" on its own line, then the structured output "
    "below. Only text from \"EXTRACTION RESULTS\" onward is parsed, so every pair from your reasoning must also "
    "appear there, and you must always reach that heading."
)

WHOLE_DOC_PLAIN_INSTRUCTION = (
    "There is no single query event -- check every plausible directed pair in the WHOLE document against the "
    "RELATED test above. Go straight to the structured output below with no preamble and no explanation."
)

PER_QUERY_INSTRUCTION = (
    "The query event is marked <q ...>. For each candidate event listed below, decide whether the query event "
    "has any causal bearing on that candidate event -- that is, whether <q> -> <candidate> holds. Only report "
    "this one direction. Do not report or consider the reverse (<candidate> -> <q>): every event in the "
    "document is used as its own query in a separate call, so a case where a candidate causes the query is "
    "captured in that candidate's own call, not this one. Go straight to the structured output below with no "
    "preamble and no explanation."
)


def _output_schema(with_rationales: bool) -> str:
    lines = ["EXTRACTION RESULTS", "RELATED: <event id> -> <event id>, <event id> -> <event id>, ... (or none)"]
    if with_rationales:
        lines += [
            "RATIONALES:",
            "For each pair listed above, ONE short clause (max ~15 words) citing the specific textual evidence.",
            "Use this compact form: <e0> -> <e3> - reason",
        ]
    return "\n".join(lines)


def _per_query_output_schema() -> str:
    return "EXTRACTION RESULTS\nRELATED: <q> -> <event id>, <q> -> <event id>, ... (or none)"


def build_whole_document_messages(marked_document: str, reasoning: bool) -> list[dict]:
    instruction = WHOLE_DOC_REASONING_INSTRUCTION if reasoning else WHOLE_DOC_PLAIN_INSTRUCTION
    system = "\n\n".join([
        TASK_DESCRIPTION,
        CAUSAL_DEFINITION,
        RELATION_NAMING_NOTE,
        instruction,
        "OUTPUT FORMAT",
        _output_schema(with_rationales=True),
    ])
    messages = [{"role": "system", "content": system}]
    examples = fewshot_examples.WHOLE_DOC_EXAMPLES_REASONING if reasoning else fewshot_examples.WHOLE_DOC_EXAMPLES_PLAIN
    for ex in examples:
        messages.append({"role": "user", "content": ex["user"]})
        messages.append({"role": "assistant", "content": ex["assistant"]})
    messages.append({"role": "user", "content": f"Document:\n\n{marked_document}"})
    return messages


def build_per_query_messages(snippet_text: str, candidate_labels: list[str]) -> list[dict]:
    system = "\n\n".join([
        TASK_DESCRIPTION,
        CAUSAL_DEFINITION,
        RELATION_NAMING_NOTE,
        PER_QUERY_INSTRUCTION,
        "OUTPUT FORMAT",
        _per_query_output_schema(),
    ])
    messages = [{"role": "system", "content": system}]
    for ex in fewshot_examples.PER_QUERY_EXAMPLES:
        messages.append({"role": "user", "content": ex["user"]})
        messages.append({"role": "assistant", "content": ex["assistant"]})
    user_content = (
        "Candidate event ids: " + ", ".join(f"<{c}>" for c in candidate_labels) + "\n\n"
        "Context:\n" + snippet_text
    )
    messages.append({"role": "user", "content": user_content})
    return messages