"""
test_chunk.py -- every check for chunker.py, in one place. Free by default;
anything that calls the API needs --live.

    python test_chunk.py --schema                 # closed schema rejects illegal values; worked examples validate
    python test_chunk.py --extraction [--source X] # do our paragraphs carry the whole record? all of data/
    python test_chunk.py --verify output/chunk     # every paragraph stored once? values legal? nulls where allowed?
    python test_chunk.py --replay output/chunk     # re-run repair/assemble/check on stored model answers, no API
    python test_chunk.py --live --audit output/chunk   # second read over stored output, applies fixes in place
    python test_chunk.py --live --smoke            # 2 documents (1963/18 norm review + one kvkk), then --verify
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import chunker as C
import prompts

SMOKE = {"aym": ["b376aa8a-7f7a-43c3-6e76-2932be9f34a6"],
         "kvkk": ["c3ce8681-4774-5eb9-bcff-3fc0bd188a7d"]}


def raw_records(sources):
    out = {}
    for s in sources:
        p = C.DATA_DIR / f"{s}.json"
        if p.is_file():
            for r in json.loads(p.read_text(encoding="utf-8")):
                out[(s, str(r.get("doc_id")))] = r
    return out


def usable(source):
    recs = json.loads((C.DATA_DIR / f"{source}.json").read_text(encoding="utf-8"))
    return [r for r in recs if r.get("status") in C.USABLE_STATUSES
            and ((r.get("content_text") or "").strip() or (r.get("html_content") or "").strip())]


# ---------------------------------------------------------------------------

def test_schema():
    ok = True
    for source in ["aym", "bam", "danistay", "first_degree", "kvkk", "yargitay"]:
        kinds = {"aym": ["aym_individual_application", "aym_norm_review"],
                 "yargitay": ["yargitay_hukuk", "yargitay_ceza"]}.get(source, [source])
        for kind in kinds:
            cands = ["mülkiyet_hakkı"] if kind == "aym_individual_application" else []
            m = C.response_model_for(source, kind, cands)
            bad = {"segments": [], "capsules": [{"outcome": "karsi_oy", "opinion_type": "dissent",
                   "subject_id": cands[0] if cands else "x_y", "conclusion_sentence": "a" * 30,
                   "reasoning_summary": "b" * 90, "supporting_local_ids": ["seg_1"]}]}
            try:
                m.model_validate(bad)
                print(f"  FAIL {kind}: accepted outcome 'karsi_oy'")
                ok = False
            except Exception:
                pass
            vocab = {"outcomes": C.OUTCOME_BY_KIND[kind], "gloss": C.OUTCOME_GLOSS,
                     "opinion_kinds": C.OPINION_KINDS if source == "kvkk" else C.OPINION_KINDS[:3],
                     "legislation_types": C.RESPONSE_LEGISLATION_TYPES}
            few = prompts.build_fewshot(source, kind, "2020/1", cands, vocab["outcomes"])
            status = "no example"
            if few:
                try:
                    m.model_validate(few)
                    status = "example validates"
                except Exception as e:
                    status = f"EXAMPLE INVALID: {str(e)[:160]}"
                    ok = False
            # keywords Gemini's response_schema does not accept next to enums
            for label, model_ in (("response", m), ("audit", C.audit_model_for(source, kind))):
                js = json.dumps(model_.model_json_schema())
                bad_kw = [k for k in ("maxItems", "maxLength", "pattern", "minimum", "maximum") if f'"{k}"' in js]
                if bad_kw:
                    print(f"  FAIL {kind} {label} schema uses {bad_kw} -- Gemini rejects these")
                    ok = False
            si = prompts.build_system_instruction(source, kind, "2020/1", cands, [], vocab)
            print(f"  {kind:28} schema ok | prompt {len(si):6} chars | {status}")
    print("SCHEMA:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def test_extraction(sources):
    rows = []
    for s in sources:
        for r in usable(s):
            paras, cov, _ = C.paragraphs(r)
            rows.append((s, str(r["doc_id"]), len(paras), cov, sum(len(p) for p in paras)))
    print(f"{'source':13} {'docs':>5} {'median paras':>12} {'median cov':>10} {'<0.85':>6}")
    for s in sources:
        rs = [x for x in rows if x[0] == s]
        if not rs:
            continue
        covs = sorted(x[3] for x in rs)
        ps = sorted(x[2] for x in rs)
        print(f"{s:13} {len(rs):>5} {ps[len(ps) // 2]:>12} {covs[len(covs) // 2]:>10.3f} "
              f"{sum(1 for c in covs if c < C.MIN_COVERAGE):>6}")
    low = [x for x in rows if x[3] < C.MIN_COVERAGE]
    for x in sorted(low, key=lambda x: x[3])[:20]:
        print(f"   LOW {x[0]:12} {x[1][:20]:20} cov {x[3]:.3f} paras {x[2]} chars {x[4]:,}")
    out = C.ROOT / "output" / "extraction_report.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps([dict(zip(("source", "doc_id", "paragraphs", "coverage", "chars"), x))
                               for x in rows], ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 1 if low else 0


ALLOWED_NULL = {"decision_date", "case_no", "rights", "firac_role", "law_no", "law_short", "law_name",
                "article_no", "paragraph_no", "law_date", "verbatim_mention", "canonical_id"}


def test_verify(out_dir):
    base = Path(out_dir)
    base = base if base.is_absolute() else C.ROOT / base
    raw = raw_records(C.SOURCES)
    rows, hard_total, values = [], 0, Counter()
    for path in sorted(base.glob("*.json")):
        source = path.stem
        if source not in C.SOURCES:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        by_doc = {}
        for ch in data["chunks"]:
            by_doc.setdefault(C.doc_of(ch), {"chunks": [], "capsules": []})["chunks"].append(ch)
        cdoc = {ch["chunk_id"]: C.doc_of(ch) for ch in data["chunks"]}
        for cap in data["reasoning_capsules"]:
            for d in {cdoc.get(i) for i in cap["supporting_chunk_ids"]} - {None}:
                by_doc[d]["capsules"].append(cap)
        for doc_id, d in by_doc.items():
            rec = raw.get((source, doc_id))
            if rec is None:
                rows.append((source, doc_id, "?", 0, 0, 0, 0, ["no_raw_record"], []))
                continue
            paras, cov, _ = C.paragraphs(rec)
            hard, soft = C.verify_document(d["chunks"], d["capsules"], paras)
            # value audit: closed fields, nulls only where allowed
            for ch in d["chunks"]:
                values[("role", ch["role"])] += 1
                for k, v in ch.items():
                    if v is None and k not in ALLOWED_NULL:
                        hard.append(f"null_field:{k}")
                    if k == "rights" and v is not None and source != "aym":
                        hard.append("rights_set_outside_aym")
                for cit in ch["cited_legislations"]:
                    values[("legislation_type", cit["legislation_type"])] += 1
                    if cit["legislation_type"] not in C.LEGISLATION_TYPES:
                        hard.append(f"legislation_type_illegal:{cit['legislation_type']}")
            for cap in d["capsules"]:
                values[("outcome", cap["outcome"])] += 1
                values[("opinion", cap["opinion_type"].split(":")[0])] += 1
                if cap["outcome"] not in C.OUTCOME_GLOSS:
                    hard.append(f"outcome_illegal:{cap['outcome']}")
            hard_total += bool(hard)
            n = len(paras)
            stored = sum(1 for i in range(1, n + 1) if any(f"p{i}" in ch["source_paragraph_ids"] for ch in d["chunks"]))
            rows.append((source, doc_id, C.kind_of(rec, source), n, stored, len(d["chunks"]),
                         len(d["capsules"]), hard, soft))
    print(f"{'source':12} {'doc':14} {'kind':27} {'paras':>5} {'stored':>6} {'chunks':>6} {'caps':>4}  verdict")
    for r in rows:
        v = "ISSUE" if r[7] else ("PASS" + (f" ({len(r[8])} note{'s' if len(r[8]) > 1 else ''})" if r[8] else ""))
        print(f"{r[0]:12} {r[1][:14]:14} {str(r[2]):27} {r[3]:>5} {r[4]:>6} {r[5]:>6} {r[6]:>4}  {v}"
              + (f"  {'; '.join(x[:70] for x in r[7][:3])}" if r[7] else ""))
    print(f"\ndocuments {len(rows)} | with issues {hard_total} | paragraphs in {sum(r[3] for r in rows)} "
          f"stored {sum(r[4] for r in rows)} missing {sum(r[3] - r[4] for r in rows)}")
    kinds = Counter(x.split(':')[0] for r in rows for x in r[8])
    if kinds:
        print("soft notes:", dict(kinds.most_common()))
    print("values:", {f"{k[0]}={k[1]}": v for k, v in sorted(values.items())})
    report = [dict(zip(("source", "doc_id", "kind", "paragraphs", "stored", "chunks", "capsules", "hard", "soft"), r))
              for r in rows]
    (base / "verify_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return 1 if hard_total else 0


def test_replay(out_dir):
    """Stored raw model answers -> repair -> assemble -> check, no API call."""
    base = Path(out_dir)
    base = base if base.is_absolute() else C.ROOT / base
    raw = raw_records(C.SOURCES)
    n = fails = 0
    for path in sorted(base.glob("*_review.json")) + sorted(base.glob("*_rejected.json")):
        source = path.stem.replace("_review", "").replace("_rejected", "")
        for r in json.loads(path.read_text(encoding="utf-8")):
            if not r.get("raw_response"):
                continue
            rec = raw.get((source, str(r["doc_id"])))
            if rec is None:
                continue
            paras, cov, _ = C.paragraphs(rec)
            kind = C.kind_of(rec, source)
            cands = C.candidate_rights(rec) if kind == "aym_individual_application" else []
            try:
                parsed = C.response_model_for(source, kind, cands).model_validate(r["raw_response"])
            except Exception as e:
                print(f"  {source} {r['doc_id'][:12]} raw answer no longer validates: {str(e)[:100]}")
                fails += 1
                n += 1
                continue
            segs, alias, log = C.normalise_segments(parsed.segments, paras)
            chunks, id_map = C.build_chunks(segs, alias, source, str(r["doc_id"]), r.get("case_no"), paras, None, log)
            caps = C.build_capsules(list(parsed.capsules), id_map, source, r.get("case_no"), None, "x", log)
            hard, soft = C.verify_document(chunks, caps, paras)
            n += 1
            fails += bool(hard)
            if hard:
                print(f"  {source} {r['doc_id'][:12]} ISSUES: {'; '.join(h[:80] for h in hard[:3])}")
    print(f"replayed {n} stored answers | with issues: {fails}")
    return 1 if fails else 0


def live_audit(out_dir):
    from types import SimpleNamespace
    from dotenv import load_dotenv
    load_dotenv(C.ROOT / ".env")
    client, model = C.build_client(), os.getenv("MODEL", "gemini-2.5-flash-lite")
    base = Path(out_dir)
    base = base if base.is_absolute() else C.ROOT / base
    raw = raw_records(C.SOURCES)
    stats, report = Counter(), []
    for path in sorted(base.glob("*.json")):
        source = path.stem
        if source not in C.SOURCES:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        chunks, caps = data["chunks"], data["reasoning_capsules"]
        by_doc = {}
        for ch in chunks:
            by_doc.setdefault(C.doc_of(ch), []).append(ch)
        cdoc = {ch["chunk_id"]: C.doc_of(ch) for ch in chunks}
        changed = 0
        for doc_id, dchunks in by_doc.items():
            rec = raw.get((source, doc_id))
            if rec is None:
                continue
            paras, _, _ = C.paragraphs(rec)
            kind = C.kind_of(rec, source)
            groups = {}
            for ch in dchunks:
                base_label = ch["chunk_label"].rsplit("_p", 1)[0] if "_p" in ch["chunk_label"].split("-p", 1)[1] else ch["chunk_label"]
                groups.setdefault(base_label, {"refs": ch["source_paragraph_ids"], "role": ch["role"], "chunks": []})["chunks"].append(ch)
            segs = [SimpleNamespace(local_id=k, paragraph_refs=g["refs"], role=g["role"]) for k, g in groups.items()]
            dcaps = [c for c in caps if any(cdoc.get(i) == doc_id for i in c["supporting_chunk_ids"])]
            cap_objs = [SimpleNamespace(**{k: c[k] for k in ("opinion_type", "outcome", "subject_id", "conclusion_sentence")}) for c in dcaps]
            try:
                verdict = C.audit(client, model, paras, segs, cap_objs, source, kind, rec, stats)
            except Exception as e:
                print(f"  [{source}] {doc_id} audit failed: {str(e)[:100]}")
                continue
            fixes = []
            rf = C.SOURCES[source]["role_field"]
            for fix in verdict.role_fixes:
                for ch in groups.get(fix.local_id, {}).get("chunks", []):
                    if ch["role"] != fix.role:
                        fixes.append(f"role {ch['chunk_label']} {ch['role']}->{fix.role}")
                        ch["role"] = fix.role
                        ch["firac_role" if rf == "firac_role" else rf] = fix.role
                        ch["content_type"], ch["reasoning_stage"] = C.derive(fix.role)
            for fix in verdict.outcome_fixes:
                if 0 <= fix.capsule_index < len(dcaps) and dcaps[fix.capsule_index]["outcome"] != fix.outcome:
                    fixes.append(f"outcome [{fix.capsule_index}] {dcaps[fix.capsule_index]['outcome']}->{fix.outcome}")
                    dcaps[fix.capsule_index]["outcome"] = fix.outcome
            fixes += [f"missing capsule {m.outcome} {C.slug(m.subject)}" for m in verdict.missing_capsules]
            changed += bool(fixes)
            report.append({"source": source, "doc_id": doc_id, "fixes": fixes})
            print(f"  [{source}] {doc_id}  {'; '.join(fixes[:3]) if fixes else 'audit agreed'}")
        path.write_text(json.dumps({"chunks": chunks, "reasoning_capsules": caps}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"=== {source} === {len(by_doc)} audited, {changed} changed | tokens in {stats['input_tokens']:,} out {stats['output_tokens']:,}")
    (base / "audit_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


KIND_SOURCE = {"aym_individual_application": "aym", "aym_norm_review": "aym", "bam": "bam",
               "danistay": "danistay", "first_degree": "first_degree",
               "yargitay_hukuk": "yargitay", "yargitay_ceza": "yargitay", "kvkk": "kvkk"}


def pick(n, out_name, seed=11, kinds=None, exclude=None):
    """A seeded, stratified run set: n documents per kind, spread across years,
    the largest document of each kind included once. Written to output/<out_name>."""
    import random
    exclude = set(exclude or [])
    rng = random.Random(seed)
    chosen = {}
    for kind in kinds or KIND_SOURCE:
        source = KIND_SOURCE[kind]
        recs = [r for r in usable(source) if C.kind_of(r, source) == kind
                and (source, str(r["doc_id"])) not in C.FEWSHOT_DOC_IDS and str(r["doc_id"]) not in exclude]
        if not recs:
            continue
        by_year = {}
        for r in recs:
            by_year.setdefault(r.get("karar_year") or r.get("esas_year") or 0, []).append(r)
        years = sorted(by_year)
        rng.shuffle(years)
        take = []
        biggest = max(recs, key=lambda r: len(r.get("content_text") or "") + len(r.get("html_content") or ""))
        take.append(biggest)
        while len(take) < min(n, len(recs)):
            for y in years:
                pool = [r for r in by_year[y] if r not in take]
                if pool:
                    take.append(rng.choice(pool))
                if len(take) >= min(n, len(recs)):
                    break
        chosen.setdefault(source, []).extend(str(r["doc_id"]) for r in take)
        print(f"  {kind:28} {len(take)} docs")
    out = C.ROOT / "output" / out_name
    out.write_text(json.dumps(chosen, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {out} ({sum(len(v) for v in chosen.values())} documents)")
    return 0


def report(out_dir):
    """One compact line per document: role sequence, capsules, notes. What to
    read after a run instead of dumping chunks by hand."""
    base = Path(out_dir)
    base = base if base.is_absolute() else C.ROOT / base
    raw = raw_records(C.SOURCES)
    for path in sorted(base.glob("*_review.json")):
        source = path.stem.replace("_review", "")
        data = json.loads((base / f"{source}.json").read_text(encoding="utf-8"))
        by_doc = {}
        for ch in data["chunks"]:
            by_doc.setdefault(C.doc_of(ch), []).append(ch)
        cdoc = {ch["chunk_id"]: C.doc_of(ch) for ch in data["chunks"]}
        caps = {}
        for cap in data["reasoning_capsules"]:
            for d in {cdoc.get(i) for i in cap["supporting_chunk_ids"]} - {None}:
                caps.setdefault(d, []).append(cap)
        roles_total = Counter()
        print(f"\n=== {source}")
        for r in json.loads(path.read_text(encoding="utf-8")):
            doc = r["doc_id"]
            if "paragraphs" not in r:
                print(f"  {doc[:12]:12} FAILED: {r.get('reason')}: {str(r.get('error', ''))[:90]}")
                continue
            chs = sorted(by_doc.get(doc, []), key=lambda c: int(c["source_paragraph_ids"][0][1:]))
            seq, last = [], None
            for c in chs:
                roles_total[c["role"]] += 1
                if c["role"] == last:
                    seq[-1] = (last, seq[-1][1] + 1)
                else:
                    seq.append((c["role"], 1))
                    last = c["role"]
            seq_s = ",".join(f"{a}×{k}" if k > 1 else a for a, k in seq)
            outs = "; ".join(f"{c['opinion_type'].split(':')[0][:5]}={c['outcome']}" for c in caps.get(doc, []))
            notes = [x.split(":")[0] for x in r.get("log", []) if x.startswith(("audited_", "audit_missing", "audit_fix_refused", "uncovered_filled", "dangling", "audit_failed"))]
            notes = ",".join(f"{k}×{v}" if v > 1 else k for k, v in Counter(notes).items())
            print(f"  {doc[:12]:12} {r.get('kind', ''):27} p{r['paragraphs']:<4} {seq_s}")
            print(f"  {'':12} {'':27} caps: {outs}" + (f"  | {notes}" if notes else "") + (f"  | ISSUES: {'; '.join(r['issues'])[:120]}" if r.get("issues") else ""))
        print(f"  roles: {dict(roles_total.most_common())}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", action="store_true")
    ap.add_argument("--extraction", action="store_true")
    ap.add_argument("--source")
    ap.add_argument("--verify", metavar="DIR")
    ap.add_argument("--replay", metavar="DIR")
    ap.add_argument("--audit", metavar="DIR")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--live", action="store_true", help="allow API calls (--audit, --smoke, --schema-online)")
    ap.add_argument("--schema-online", action="store_true",
                    help="PAID (under a cent): ask the API to accept every kind's response and audit schema")
    ap.add_argument("--pick", type=int, metavar="N", help="write a seeded stratified run set: N docs per kind")
    ap.add_argument("--out", default="eval_set.json", help="file name under output/ for --pick")
    ap.add_argument("--exclude", metavar="FILE", help="run-set JSON whose doc_ids --pick must avoid")
    ap.add_argument("--report", metavar="DIR", help="compact per-document report of stored output")
    a = ap.parse_args()
    if a.pick:
        excl = []
        if a.exclude:
            excl = [d for v in json.loads(Path(a.exclude).read_text(encoding="utf-8")).values() for d in v]
        return pick(a.pick, a.out, exclude=excl)
    if a.report:
        return report(a.report)
    if a.schema:
        return test_schema()
    if a.schema_online:
        if not a.live:
            print("this needs --live (it calls the API)")
            return 2
        from dotenv import load_dotenv
        load_dotenv(C.ROOT / ".env")
        pairs = [(KIND_SOURCE[k], k) for k in KIND_SOURCE]
        problems = C.schema_preflight(C.build_client(), os.getenv("MODEL", "gemini-2.5-flash-lite"), pairs)
        for pr in problems:
            print("  REJECTED:", pr)
        print("schema online:", "PASS" if not problems else f"FAIL ({len(problems)})")
        return 1 if problems else 0
    if a.extraction:
        srcs = [a.source] if a.source else [s for s in C.SOURCES if s not in C.TEXT_PENDING_SOURCES]
        return test_extraction(srcs)
    if a.verify:
        return test_verify(a.verify)
    if a.replay:
        return test_replay(a.replay)
    if a.audit or a.smoke:
        if not a.live:
            print("this needs --live (it calls the API)")
            return 2
        if a.audit:
            return live_audit(a.audit)
        C.run(["aym", "kvkk"], 0, SMOKE, "output/smoke")
        return test_verify("output/smoke")
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
