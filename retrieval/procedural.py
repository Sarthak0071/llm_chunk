"""Production's procedural-decision penalty — and why it is a specific risk to chunks.

Hammurabi multiplies a result's score by 0.4 when its text looks like a procedural
ruling rather than a decision on the merits. The intent is sound: nobody searching
for case law wants "the hearing was adjourned" ranked above a judgment.

WHY THIS THREATENS CHUNKS

The test is a substring match over the text of a result. For a whole decision that
is a reasonable proxy: a 20,000-character judgment that happens to contain
"dilekçenin reddine" in its procedural history is still mostly merits, and the
phrase is diluted by everything around it.

A CHUNK has no such dilution. A `conclusion` chunk may be a single operative
sentence, and if that sentence is the procedural line, the whole chunk is judged
procedural and loses 60% of its score. The same text, in the same corpus, is
penalised differently purely because of how it is stored.

So this is not a detail to replicate for completeness — it is a mechanism that can
make chunks look worse for a reason unrelated to retrieval quality. Measuring it is
experiment E7, and the measurement has to happen before we interpret any score.
"""

import re

import config as cfg

PROCEDURAL_COMPILED = [re.compile(p, re.IGNORECASE) for p in cfg.PROCEDURAL_PATTERNS]

# hybrid_search_engine.py checks jurisdictional rulings too -- a decision that only
# says "this court has no jurisdiction" answers nothing about the law.
JURISDICTIONAL_PATTERNS = [
    r"görevsizlik kararı",
    r"yetkisizlik kararı",
    r"görev uyuşmazlığı",
]
JURISDICTIONAL_COMPILED = [re.compile(p, re.IGNORECASE)
                           for p in JURISDICTIONAL_PATTERNS]


def is_procedural(text):
    """(is_procedural, reason). Matches production: a substring hit anywhere in the
    text is enough, with no weighting by how much of the text it represents."""
    if not text:
        return False, ""
    for pattern in JURISDICTIONAL_COMPILED:
        if pattern.search(text):
            return True, "görevsizlik (jurisdictional)"
    for pattern in PROCEDURAL_COMPILED:
        if pattern.search(text):
            return True, "prosedürel red (procedural rejection)"
    return False, ""


def apply_penalty(hits, penalty=cfg.PROCEDURAL_PENALTY):
    """Rescale in place, as production does, recording what happened to each hit."""
    for h in hits:
        flagged, reason = is_procedural(h.get("text") or "")
        h["is_procedural"] = flagged
        h["procedural_reason"] = reason
        h["original_score"] = h.get("score")
        if flagged:
            h["score"] = h.get("score", 0.0) * penalty
    return hits


def density(text):
    """What fraction of the text is the matched phrase?

    This is the number production never computes, and the reason the penalty
    behaves differently for chunks: a phrase that is 0.1% of a judgment is 40% of a
    one-line conclusion chunk, yet both are penalised identically.
    """
    if not text:
        return 0.0
    for pattern in JURISDICTIONAL_COMPILED + PROCEDURAL_COMPILED:
        m = pattern.search(text)
        if m:
            return len(m.group(0)) / len(text)
    return 0.0
