"""BAAI/bge-m3 embeddings, matching production exactly.

WHY ONNX AND NOT sentence-transformers
Production runs `SentenceTransformer("BAAI/bge-m3")`. We cannot: the installed
torch is 2.1.0 compiled against NumPy 1.x while this environment has NumPy 2.2.6,
so torch cannot even import, and transformers refuses it (needs >= 2.5).

fastembed was the plan, but it does not carry bge-m3 at all — checked across its
dense, sparse and late-interaction model lists. Using a different model would have
broken the one thing that makes the comparison valid.

The official BAAI/bge-m3 repo ships an ONNX export (`onnx/model.onnx`), and
onnxruntime is already present. So we run the REAL production weights with no deep
learning framework installed.

MATCHING PRODUCTION'S EMBEDDING EXACTLY
Three things have to line up or our vectors are not comparable with theirs:

  1. RAW TEXT. Production's `embed_query` passes the string straight in — no
     prefix, no instruction template, no title or metadata prepended. We do the
     same. Prepending anything moves our points to a different region of the space.
  2. CLS POOLING. bge-m3's SentenceTransformer config is
     [Transformer, Pooling(cls), Normalize], so the dense vector is the CLS token,
     not a mean over tokens.
  3. L2 NORMALISE. Included in that same config. Production never passes
     `normalize_embeddings=True`, but it does not need to — the module pipeline
     does it. Cosine distance assumes it.

CACHING
Vectors are cached to disk keyed by sha256(text) + model, so re-indexing and
re-running experiments costs nothing after the first pass. Embedding is free but
slow on CPU, and the corpus is ~1,000 documents.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / "report" / "vector_cache"

MODEL = "BAAI/bge-m3"
DIM = 1024
MAX_TOKENS = 8192          # production's ceiling; keep it, truncation is the finding

# XLM-RoBERTa-large: 24 layers, 16 heads. Only the head count matters for the
# attention-memory estimate that drives batching.
ATTENTION_HEADS = 16

# How much RAM one batch's attention may claim. Deliberately conservative: this
# machine has ~16 GB with ~8 GB free, and a single 8,192-token sequence alone needs
# 16 x 8192^2 x 4 = 4 GB. Anything larger gets split rather than swapped.
DEFAULT_BATCH_BYTES = 1.5 * 1024 ** 3


class Embedder:
    """Lazy-loading bge-m3 via onnxruntime, with a disk cache and safe batching."""

    def __init__(self, model=MODEL, cache=True, max_tokens=MAX_TOKENS,
                 batch_bytes=DEFAULT_BATCH_BYTES, max_batch=32):
        self.model_name = model
        self.max_tokens = max_tokens
        self.batch_bytes = batch_bytes
        self.max_batch = max_batch
        self.cache = cache
        self._session = None
        self._tok = None
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # -- lazy load ---------------------------------------------------------
    def _load(self):
        if self._session is not None:
            return
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer

        print(f"  loading {self.model_name} (ONNX)…", file=sys.stderr)
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        path = hf_hub_download(self.model_name, "onnx/model.onnx")
        # The weights live in a sibling file that onnxruntime resolves by name.
        try:
            hf_hub_download(self.model_name, "onnx/model.onnx_data")
        except Exception:
            pass
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(path, opts,
                                             providers=["CPUExecutionProvider"])
        self._inputs = {i.name for i in self._session.get_inputs()}
        print(f"  ready. inputs={sorted(self._inputs)}", file=sys.stderr)

    # -- cache -------------------------------------------------------------
    def _key(self, text, max_tokens):
        # max_tokens is part of the key: the SAME text truncated at 2048 and at
        # 8192 gives different vectors, so leaving it out would silently serve one
        # where the other was asked for -- and the two arms would stop being
        # comparable for reasons nothing in the output would reveal.
        stamp = f"{self.model_name}\x00{max_tokens}\x00{text}"
        return CACHE_DIR / f"{hashlib.sha256(stamp.encode('utf-8')).hexdigest()}.npy"

    def _cached(self, text, max_tokens):
        if not self.cache:
            return None
        p = self._key(text, max_tokens)
        return np.load(p) if p.is_file() else None

    def _store(self, text, vec, max_tokens):
        if self.cache:
            np.save(self._key(text, max_tokens), vec)

    # -- the actual embedding ---------------------------------------------
    def _forward(self, texts, max_tokens=None):
        self._load()
        enc = self._tok(texts, padding=True, truncation=True,
                        max_length=max_tokens or self.max_tokens, return_tensors="np")
        feed = {k: v.astype(np.int64) for k, v in enc.items() if k in self._inputs}
        out = self._session.run(None, feed)[0]          # (batch, seq, hidden)
        cls = out[:, 0, :]                              # CLS pooling, per bge-m3
        norm = np.linalg.norm(cls, axis=1, keepdims=True)
        return (cls / np.clip(norm, 1e-12, None)).astype(np.float32)

    def _plan_batches(self, idx_lens):
        """Group indices into batches that fit in memory.

        THIS EXISTS BECAUSE A FIXED BATCH SIZE FROZE A LAPTOP. Transformer attention
        costs `heads x batch x seq^2 x 4` bytes, and `padding=True` pads every text
        in a batch up to the LONGEST one in it. A batch of 8 mixing one 8,192-token
        decision with seven short ones padded all eight to 8,192 and asked for

            16 heads x 8 x 8192^2 x 4 bytes = 34 GB

        which is the exact number onnxruntime reported before the machine started
        swapping. Cost is driven by the longest item, not the average, so batching
        must be planned from real token counts.

        Two rules: sort by length so each batch is uniform (padding then wastes
        nothing), and choose the batch size from the memory budget rather than a
        constant. Long sequences end up alone; short ones batch freely.
        """
        order = sorted(range(len(idx_lens)), key=lambda i: idx_lens[i])
        budget = self.batch_bytes / (ATTENTION_HEADS * 4)      # in seq^2 * batch
        batches, cur = [], []
        for i in order:
            longest = max(idx_lens[i], max((idx_lens[j] for j in cur), default=0))
            allowed = max(1, int(budget // max(longest * longest, 1)))
            allowed = min(allowed, self.max_batch)
            if cur and len(cur) + 1 > allowed:
                batches.append(cur)
                cur = [i]
            else:
                cur.append(i)
        if cur:
            batches.append(cur)
        return batches

    def embed(self, texts, batch_size=None, progress=False, max_tokens=None):
        """Embed a list of strings. Returns (n, 1024) float32, L2-normalised.

        `max_tokens` overrides the instance ceiling for this call. Used to embed
        the documents under test at production's full 8,192 while capping the
        distractor haystack lower -- the distractors are byte-identical in both
        arms, so a lower cap there cannot tilt the comparison, and it turns a
        4.5-hour job into a 50-minute one.

        `batch_size` is accepted for compatibility but ignored: batching is planned
        from token counts and the memory budget. See _plan_batches.
        """
        if isinstance(texts, str):
            texts = [texts]
        cap = max_tokens or self.max_tokens
        out = [None] * len(texts)
        todo = []
        for i, t in enumerate(texts):
            c = self._cached(t, cap)
            if c is not None:
                out[i] = c
            else:
                todo.append(i)
        if not todo:
            return np.vstack(out)

        self._load()
        lens = [min(len(self._tok.encode(texts[i], add_special_tokens=True,
                                         truncation=True, max_length=cap)), cap)
                for i in todo]
        plan = self._plan_batches(lens)
        if progress:
            print(f"    {len(todo)} to embed in {len(plan)} batches "
                  f"({len(texts) - len(todo)} already cached); "
                  f"longest {max(lens)} tokens", file=sys.stderr)

        done = 0
        for bi, group in enumerate(plan):
            gi = [todo[i] for i in group]
            longest = max(lens[i] for i in group)
            peak = ATTENTION_HEADS * len(group) * longest * longest * 4 / 1024 ** 3
            vecs = self._forward([texts[i] for i in gi], max_tokens=cap)
            for i, v in zip(gi, vecs):
                out[i] = v
                self._store(texts[i], v, cap)
            done += len(gi)
            if progress and (bi % 20 == 0 or done == len(todo)):
                print(f"    {done}/{len(todo)}  (batch {len(group)} x "
                      f"{longest} tok, peak ~{peak:.2f} GB)", file=sys.stderr)

        return np.vstack(out)

    def token_count(self, text):
        self._load()
        return len(self._tok.encode(text, add_special_tokens=False, truncation=False))


def self_test():
    """Prove the embedder is sane before anything is indexed with it."""
    e = Embedder()
    a = "Yargıtay kararında kamulaştırma bedelinin tespiti değerlendirilmiştir."
    b = "Kamulaştırma bedeli tespiti hakkında Yargıtay değerlendirmesi yapılmıştır."
    c = "Kişisel verilerin korunması hakkında idari para cezası verilmiştir."

    v = e.embed([a, b, c])
    print(f"shape            : {v.shape}   (expect (3, {DIM}))")
    print(f"L2 norms         : {np.linalg.norm(v, axis=1).round(4)}   (expect ~1.0)")

    sim_ab = float(v[0] @ v[1])
    sim_ac = float(v[0] @ v[2])
    print(f"sim(related)     : {sim_ab:.4f}")
    print(f"sim(unrelated)   : {sim_ac:.4f}")

    again = e.embed([a])
    identical = bool(np.allclose(again[0], v[0]))
    print(f"deterministic    : {identical}   (same text -> same vector)")

    ok = (v.shape == (3, DIM)
          and np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-3)
          and sim_ab > sim_ac
          and identical)
    print("\nSELF-TEST", "PASS" if ok else "FAIL")
    if not ok:
        print("  related pair must score above the unrelated pair, vectors must be")
        print("  unit length, and the same text must always give the same vector.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(self_test())
