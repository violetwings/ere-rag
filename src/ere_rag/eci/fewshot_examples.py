"""
fewshot_examples.py

Hand-written worked examples used as in-context few-shot turns. Pure data,
no logic -- prompt.py imports this and assembles messages from it.

Deliberately NOT generated through an LLM self-correction pipeline (the
project's earlier build_causal_fewshot_example.py approach): that pipeline
forced the model to match MAVEN-ERE's exact gold labels even when a pair's
direction reads as backwards on its own terms, which taught a confusing
pattern rather than a clean one. These examples are written directly from
real MAVEN-ERE document text, with relation calls chosen for how clearly
the text itself supports them.

Two whole-document examples are provided (WHOLE_DOC_EXAMPLES_REASONING /
WHOLE_DOC_EXAMPLES_PLAIN), built from the same underlying pairs so the
only difference between the reasoning and no-reasoning prompt configs is
the presence of the REASONING section -- this keeps the RQ3 comparison to
exactly one changed variable. Together the two documents demonstrate:
repeated trigger words for one underlying event (Zagreb), a real
precondition-style "characterization requires the event to have already
happened" pattern (Zagreb), coreferent mentions of the same real-world
event (Castor), an explicit negation not being evidence of a relation
(Castor), and narrative sequence ("later") not being evidence of causation
(Castor).

One per-query example (PER_QUERY_EXAMPLE) is provided, with no reasoning
section, matching the no-reasoning per-query prompt configs exactly.
"""

from __future__ import annotations


def _extraction_block(related_pairs: list[tuple[str, str]], rationales: dict[tuple[str, str], str] | None = None) -> str:
    lines = ["EXTRACTION RESULTS"]
    if related_pairs:
        lines.append("RELATED: " + ", ".join(f"<{h}> -> <{t}>" for h, t in related_pairs))
    else:
        lines.append("RELATED: none")
    if rationales:
        lines.append("RATIONALES:")
        for (h, t), reason in rationales.items():
            lines.append(f"<{h}> -> <{t}> - {reason}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Whole-document example 1: Zagreb rocket attacks
# ---------------------------------------------------------------------------

_ZAGREB_USER = (
    "Document:\n\n"
    "The Zagreb rocket <e0 attacks> were a series of two rocket attacks <e1 conducted> by the Army of the "
    "Republic of Serbian Krajina that <e2 used> multiple rocket launchers to <e3 strike> the Croatian capital "
    "of Zagreb during the Croatian War of Independence . The attack <e4 killed> seven and <e5 wounded> over "
    "200 Croatian civilians and was <e6 carried out> on 2 May and 3 May 1995 as <e7 retaliation> for the "
    "Croatian army 's offensive in Operation Flash . The rocket attacks deliberately <e8 targeted> civilian "
    "locations . Zagreb was the largest of several cities <e9 hit> by the attack . It was the only instance "
    "in the entire war in Croatia that cluster bombs were <e10 used> in combat . The International Criminal "
    "Tribunal for the former Yugoslavia ( ICTY ) <e11 characterized> the attack as a crime against humanity "
    "and <e12 convicted> Croatian Serb leader Milan Martić of <e13 ordering> the attack ."
)

_ZAGREB_PAIRS: list[tuple[str, str]] = [
    ("e3", "e4"), ("e1", "e5"), ("e6", "e5"), ("e1", "e4"), ("e0", "e9"), ("e6", "e4"), ("e3", "e5"),
    ("e13", "e12"), ("e1", "e11"), ("e6", "e12"), ("e1", "e12"), ("e6", "e11"), ("e6", "e7"), ("e3", "e12"), ("e3", "e11"),
]

_ZAGREB_RATIONALES: dict[tuple[str, str], str] = {
    ("e3", "e4"): "the strike is stated to have killed seven.",
    ("e1", "e5"): "the attack being conducted is linked to over 200 wounded in the same sentence.",
    ("e6", "e5"): "the attack being carried out is directly followed by the wounding.",
    ("e1", "e4"): "the attack being conducted is linked to the killing of seven in the same sentence.",
    ("e0", "e9"): "the attacks are described as resulting in Zagreb being the largest city hit.",
    ("e6", "e4"): "the attack being carried out is directly followed by the killing.",
    ("e3", "e5"): "the strike is directly followed by the wounding of over 200.",
    ("e13", "e12"): "ordering the attack is what Martić was specifically convicted of.",
    ("e1", "e11"): "the attack had to be conducted before it could be characterized as a crime against humanity.",
    ("e6", "e12"): "the attack had to be carried out before a conviction for it was possible.",
    ("e1", "e12"): "the attack had to be conducted before a conviction for it was possible.",
    ("e6", "e11"): "the attack had to be carried out before it could be characterized as a crime against humanity.",
    ("e6", "e7"): "the attack had to be carried out before it could be characterized as retaliation.",
    ("e3", "e12"): "the strike had to occur before a conviction for it was possible.",
    ("e3", "e11"): "the strike had to occur before it could be characterized as a crime against humanity.",
}

_ZAGREB_REASONING = (
    "REASONING:\n"
    "Several trigger words -- attacks (e0), conducted (e1), strike (e3), carried out (e6) -- all refer to "
    "essentially the same underlying event of the attack occurring, but are marked as distinct mentions, so "
    "each is checked independently against downstream events.\n\n"
    "Direct results: the strike (e3), the attack being conducted (e1), and the attack being carried out (e6) "
    "are each linked in the text to both the killing (e4) and the wounding (e5) -- RELATED in each case. The "
    "attacks (e0) are described as what resulted in Zagreb being hit (e9), the largest of the cities struck.\n\n"
    "The attack's occurrence (via e1, e6, e3) is also RELATED to things that could only happen once the attack "
    "existed: the ICTY's characterization of it as a crime against humanity (e11), its characterization as "
    "retaliation (e7), and Martić's conviction (e12) for ordering it. Martić's act of ordering (e13) is "
    "likewise RELATED to his conviction (e12), since the conviction is specifically for having ordered it.\n\n"
    "The remaining mentions -- multiple launchers used (e2), deliberate targeting (e8), cluster bombs (e10) -- "
    "add detail about how the attack was carried out but aren't stated to cause or require any other marked "
    "event, so they take no causal role."
)

# ---------------------------------------------------------------------------
# Whole-document example 2: Castor / Carysfort frigate action
# ---------------------------------------------------------------------------

_CASTOR_USER = (
    "Document:\n\n"
    "The frigate action of 29 May 1794 was a minor naval <e0 engagement> of the French Revolutionary Wars "
    "between a Royal Navy frigate and a French Navy frigate . `` Castor '' had previously been a British ship , "
    "<e1 seized> on 19 May by a French battle squadron in the Bay of Biscay and <e2 converted> to French "
    "service while still at sea . On 29 May `` Castor '' <e3 encountered> the smaller independently cruising "
    "British frigate HMS `` Carysfort '' . Captain Francis Laforey on `` Carysfort '' immediately <e4 attacked> "
    "the larger ship and in an engagement lasting an hour and fifteen minutes successfully forced its captain "
    "to <e5 surrender> , <e6 discovering> a number of British prisoners of war below decks . `` Castor '' was "
    "subsequently <e7 taken> back to Britain and an extended legal case <e8 ensued> between the Admiralty and "
    "Captain Laforey over the amount of prize money that should be awarded for the victory . The <e9 lawsuit> "
    "did not <e10 harm> Laforey 's career and he later <e11 served> at the Battle of Trafalgar and "
    "<e12 became> a prominent admiral ."
)

_CASTOR_PAIRS: list[tuple[str, str]] = [
    ("e1", "e2"), ("e3", "e4"), ("e4", "e5"), ("e5", "e6"), ("e5", "e7"), ("e7", "e8"),
]

_CASTOR_RATIONALES: dict[tuple[str, str], str] = {
    ("e1", "e2"): "the seizure is what put the ship in French hands, enabling its conversion.",
    ("e3", "e4"): "the encounter is directly followed by Laforey attacking.",
    ("e4", "e5"): "the text says the attack forced the surrender.",
    ("e5", "e6"): "the prisoners were discovered after the surrender, as a direct result of it.",
    ("e5", "e7"): "the surrender is why the ship was taken as a prize.",
    ("e7", "e8"): "being taken back to Britain is what the legal case over prize money followed from.",
}

_CASTOR_REASONING = (
    "REASONING:\n"
    "A clean chain runs through most of the document: the seizure (e1) put the ship in French hands, enabling "
    "its conversion (e2) to French service -- RELATED. The encounter with Carysfort (e3) is directly followed "
    "by the attack (e4) -- RELATED. The attack forced the surrender (e5) -- RELATED. The surrender directly "
    "led to both the discovery of prisoners (e6) and the ship being taken to Britain (e7) -- both RELATED. "
    "Being taken to Britain is what the legal case over prize money (e8) followed from -- RELATED.\n\n"
    "Two mentions, e8 (\"legal case ensued\") and e9 (\"the lawsuit\"), both refer to the same real-world legal "
    "dispute -- they are the same event described twice, not two events, so no pair is formed between them, "
    "and e9 takes no further causal role of its own since e8 already covers it. The text explicitly states the "
    "lawsuit did NOT harm Laforey's career (e10) -- an explicit negation is not evidence of a positive "
    "relation, so e10 is NOREL. His later service at Trafalgar (e11) and becoming an admiral (e12) are "
    "narrated as later achievements with no stated causal link back to the lawsuit's outcome -- narrative "
    "sequence alone ('later') is not causation, so both are NOREL. The overall engagement (e0) is a "
    "categorical label for the event rather than something with its own separate cause or effect, so it takes "
    "no role either."
)

WHOLE_DOC_EXAMPLES_REASONING = [
    {"user": _ZAGREB_USER, "assistant": _ZAGREB_REASONING + "\n\n" + _extraction_block(_ZAGREB_PAIRS, _ZAGREB_RATIONALES)},
    {"user": _CASTOR_USER, "assistant": _CASTOR_REASONING + "\n\n" + _extraction_block(_CASTOR_PAIRS, _CASTOR_RATIONALES)},
]

WHOLE_DOC_EXAMPLES_PLAIN = [
    {"user": _ZAGREB_USER, "assistant": _extraction_block(_ZAGREB_PAIRS, _ZAGREB_RATIONALES)},
    {"user": _CASTOR_USER, "assistant": _extraction_block(_CASTOR_PAIRS, _CASTOR_RATIONALES)},
]

# ---------------------------------------------------------------------------
# Per-query examples (no reasoning -- matches the per-query prompt exactly).
# Both are strictly <q> -> <candidate> -- the query is always the potential
# cause, never the potential effect, matching what the retriever itself is
# trained to surface for a given query.
# ---------------------------------------------------------------------------

_PER_QUERY_USER_1 = (
    "Candidate event ids: <e0>, <e1>, <e2>\n\n"
    "Context:\n"
    "The attack <q carried out> on 2 May and 3 May 1995 killed <e0 seven> and wounded over 200 Croatian "
    "civilians . The International Criminal Tribunal for the former Yugoslavia ( ICTY ) later "
    "<e1 characterized> the attack as a crime against humanity . A separate war crimes trial was "
    "<e2 scheduled> for the following autumn to address unrelated allegations ."
)
_PER_QUERY_ASSISTANT_1 = _extraction_block([("q", "e0"), ("q", "e1")])

_PER_QUERY_USER_2 = (
    "Candidate event ids: <e0>, <e1>, <e2>\n\n"
    "Context:\n"
    "Weeks of sustained <e0 protests> outside the ministry had built pressure for the resignation . The "
    "transportation minister <q resigned> the following week , citing the crisis as her reason . The prime "
    "minister then <e1 called> an early election in response . A local election commission had separately "
    "<e2 rejected> an unrelated redistricting proposal earlier that year ."
)
# e0 (protests) is a real cause of the resignation, but in the WRONG direction
# for this call -- e0 -> q, not q -> e0 -- so it is correctly left out here;
# it will be captured when e0 itself is used as the query in its own call.
_PER_QUERY_ASSISTANT_2 = _extraction_block([("q", "e1")])

PER_QUERY_EXAMPLES = [
    {"user": _PER_QUERY_USER_1, "assistant": _PER_QUERY_ASSISTANT_1},
    {"user": _PER_QUERY_USER_2, "assistant": _PER_QUERY_ASSISTANT_2},
]