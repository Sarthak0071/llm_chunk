"""
bm25.py -- Turkish tokenizer + Okapi BM25, written fresh for llm_chunk.

BM25 stands in for Meilisearch here. It is a deliberate under-estimate: a real
Meilisearch index applies a Turkish analyser, this does not. See LIMITATIONS.

LIMITATIONS (stated up front, never tuned against results)
  1. NO STEMMING. Turkish is agglutinative, so "yargilanmayla" does not match
     "yargilanma" and "defterleri" does not match "defterlerini". Every number
     produced with this scorer is therefore PESSIMISTIC relative to production
     search. A weak result here is not by itself evidence of bad chunking.
  2. Tokenisation is character-class based, not morphological.
  3. Do NOT tune k1, b, the stopword list or the character class by looking at
     retrieval results on a 10-document sample. That is overfitting, and it
     would quietly convert this harness from a measurement into a rubber stamp.
"""

import math
import re
import unicodedata
from collections import Counter

K1 = 1.5
B = 0.75

# Turkish stopwords. Kept small and boring on purpose -- an aggressive list
# tuned against results would be another way to rig the score.
TR_STOPWORDS = {
    "ve", "bir", "bu", "ile", "için", "olan", "olarak", "da", "de", "mı", "mi",
    "mu", "mü", "ki", "gibi", "ise", "ancak", "ya", "veya", "çok", "daha", "en",
    "her", "tüm", "üzere", "göre", "kadar", "sonra", "önce", "dair", "ilişkin",
    "olduğu", "olduğunu", "şu", "o", "ne", "mıdır", "midir", "mu?", "ya da",
}

# NOTE the circumflex vowels. A character class of [a-zçğıöşü0-9] silently
# destroys them: "hâlâ" tokenises to "h" + "l" (both dropped by the length
# filter) and "mahkûm" to "mahk". Both words appear in real Turkish legal text,
# so query terms an author thought were load-bearing would simply vanish.
TOKEN_RE = re.compile(r"[a-zçğıöşüâîûêô0-9]+")


def tr_lower(s):
    """Turkish-correct lowercase. Python's str.lower() maps 'İ' to 'i' plus a
    COMBINING DOT ABOVE (U+0307), which then survives into tokens as noise."""
    return (s or "").replace("İ", "i").replace("I", "ı").lower()


def tokenize(text):
    """Lowercase, strip combining marks left by Unicode decomposition, split on
    the Turkish-aware character class, drop stopwords and single characters."""
    lowered = tr_lower(text)
    # Remove any stray combining dot-above that arrived in the source text.
    lowered = "".join(c for c in unicodedata.normalize("NFC", lowered)
                      if not unicodedata.combining(c))
    return [t for t in TOKEN_RE.findall(lowered)
            if len(t) > 1 and t not in TR_STOPWORDS]


def query_terms(text):
    """Tokens for scoring a QUERY. De-duplicated: scoring a plain list lets a
    word repeated in the question count twice, which silently weights whichever
    term the author happened to say twice."""
    seen, out = set(), []
    for t in tokenize(text):
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


class BM25:
    """Textbook Okapi BM25 over a fixed list of pre-tokenised documents."""

    def __init__(self, docs_tokens, k1=K1, b=B):
        self.k1, self.b = k1, b
        self.N = len(docs_tokens)
        self.doc_lens = [len(d) for d in docs_tokens]
        self.avgdl = (sum(self.doc_lens) / self.N) if self.N else 0.0
        self.tf = [Counter(d) for d in docs_tokens]
        df = Counter()
        for d in docs_tokens:
            df.update(set(d))
        self.idf = {
            term: math.log(1 + (self.N - n + 0.5) / (n + 0.5))
            for term, n in df.items()
        }

    def score(self, terms, i):
        if not self.avgdl:
            return 0.0
        tf, dl, s = self.tf[i], self.doc_lens[i], 0.0
        for term in terms:
            f = tf.get(term, 0)
            if not f:
                continue
            denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            s += self.idf.get(term, 0.0) * (f * (self.k1 + 1)) / denom
        return s

    def rank(self, terms, top_k=None):
        """Returns [(index, score), ...] best first. Ties break on index, which
        is corpus order -- worth remembering when a pool is small and many
        documents score identically."""
        scored = [(i, self.score(terms, i)) for i in range(self.N)]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored if top_k is None else scored[:top_k]


def rank_of(ranked, accept_indices):
    """1-based rank of the first acceptable index, or None if absent.
    `accept_indices` is a set because one paragraph can legitimately live in
    several chunks once the 2000-char cap splits a segment."""
    for pos, (i, _score) in enumerate(ranked, 1):
        if i in accept_indices:
            return pos
    return None
