"""
prompts.py -- what the model is told. Generation prompt, per-kind notes, allowed
values with Turkish glosses, worked examples from fewshot/, and the audit prompt.
No imports from chunker.py: the vocabularies are passed in.
"""

import json
import re
from pathlib import Path

FEWSHOT_DIR = Path(__file__).resolve().parents[2] / "fewshot"

# Role field and allowed values per source (docs 14.1). aym uses FIRAC.
ROLE_VOCAB = {
    "aym": ("firac_role", ["facts", "issue", "rule", "application", "conclusion", "dissent", "unknown"]),
    "bam": ("court_reasoning_role", ["facts", "issue", "rule_application", "conclusion", "dissent", "other"]),
    "danistay": ("court_reasoning_role", ["facts", "issue", "rule_application", "conclusion", "dissent", "other"]),
    "first_degree": ("court_reasoning_role", ["facts", "issue", "rule_application", "conclusion", "dissent", "other"]),
    "yargitay": ("court_reasoning_role", ["facts", "issue", "rule_application", "conclusion", "dissent", "other"]),
    "uyusmazlik": ("court_reasoning_role", ["facts", "issue", "rule_application", "conclusion", "dissent", "other"]),
    "kvkk": ("regulatory_role", ["background", "analysis", "outcome"]),
    "rekabet": ("regulatory_role", ["background", "analysis", "outcome"]),
}
CATCHALL = {"aym": "unknown", "kvkk": "background", "rekabet": "background"}

KIND_NOTES = {
    # Written from an inventory of the raw records (headings counted over every
    # usable document), not from assumptions.
    "aym_individual_application": (
        "AYM individual application (bireysel başvuru). Layout: header lines TÜRKİYE CUMHURİYETİ / "
        "ANAYASA MAHKEMESİ / BİRİNCİ or İKİNCİ BÖLÜM or GENEL KURUL / KARAR / <applicant> BAŞVURUSU / "
        "(Başvuru Numarası: …) / Karar Tarihi / R.G. Tarih ve Sayı, then the panel list (Başkan, "
        "Üyeler, Raportör, Başvurucu, Vekili) -> ALL `unknown`, one segment. Then numbered paragraphs "
        "under Roman-numeral sections: I. BAŞVURUNUN KONUSU -> issue; II. BAŞVURU SÜRECİ -> facts; "
        "III. OLAY VE OLGULAR -> facts; IV. İLGİLİ HUKUK (quoted statutes, Constitution, ECHR case "
        "law) -> rule; V. İNCELEME VE GEREKÇE with A. Kabul Edilebilirlik and B. Esas / DEĞERLENDİRME "
        "-> application (a quoted provision inside it -> rule); VI. GİDERİM -> application; VII. HÜKÜM "
        "-> conclusion. KARŞIOY GEREKÇESİ -> dissent (opinion_type dissent); FARKLI GEREKÇE / EK "
        "GEREKÇE / DEĞİŞİK GEREKÇE -> dissent role with opinion_type concurring. Signature block "
        "(Başkan / Üye / names) -> `unknown`. Lone ':' lines are part of the header. '...' is a "
        "redaction, never a value. Scope `rights` per paragraph from the candidate list; one capsule "
        "per right the court decided."),
    "aym_norm_review": (
        "AYM norm review (norm denetimi): no applicant, no violated right -- a court or deputies ask "
        "whether a STATUTE conforms to the Constitution. Header: ANAYASA MAHKEMESİ KARARI / Esas "
        "Sayısı / Karar Sayısı / Karar Günü (/ Resmi Gazete) -> `unknown`, one segment. Then İTİRAZ "
        "YOLUNA BAŞVURAN / İPTAL DAVASINI AÇAN -> facts; İTİRAZIN KONUSU / İPTALİ İSTENEN -> issue; "
        "OLAY -> facts; İPTALİ İSTENEN KANUN HÜKÜMLERİ / DAYANILAN ANAYASA KURALLARI / YASA METİNLERİ "
        "(quoted text) -> rule; İLK İNCELEME -> issue; ESASIN İNCELENMESİ (per-provision merits "
        "sections 'A. …', 'B. …') -> application; SONUÇ / HÜKÜM -> conclusion; KARŞIOY / MUHALEFET "
        "ŞERHİ / AYRIŞIK OY -> dissent; signature block -> `unknown`. Decisions from the 1960s-80s "
        "carry few headings: judge by function. `rights` is null everywhere. `subject_id` names the "
        "reviewed provision as a Turkish slug (e.g. 5393_belediye_kanunu_m23); one capsule per "
        "provision decided. Outcome is about the NORM: annulled or denied."),
    "bam": (
        "BAM (bölge adliye mahkemesi, regional appellate court). Header block T.C. / <İL> / BÖLGE "
        "ADLİYE MAHKEMESİ / n. HUKUK DAİRESİ / DOSYA NO / KARAR NO / T Ü R K  M İ L L E T İ  A D I N A "
        "/ İ S T İ N A F  K A R A R I (letter-spaced) and the İNCELENEN KARARIN block (MAHKEMESİ, "
        "TARİHİ, NUMARASI, DAVANIN KONUSU, KARAR TARİHİ, KARAR YAZIM TARİHİ) -> `other`, one segment. "
        "DAVA / CEVAP / İLK DERECE MAHKEMESİ KARARININ ÖZETİ -> facts; İSTİNAF NEDENLERİ -> issue; "
        "GEREKÇE / GEREĞİ DÜŞÜNÜLDÜ / DELİLLERİN DEĞERLENDİRİLMESİ -> rule_application (rule and "
        "application are fused); HÜKÜM -> conclusion; BAŞKAN / ÜYE / KATİP signature -> `other`; "
        "MUHALEFET ŞERHİ -> dissent. Letter-spaced lines are headings: read them without the spaces."),
    "danistay": (
        "Danıştay (Council of State). Header: T.C. / D A N I Ş T A Y / n. DAİRE / Esas No / Karar No / "
        "TEMYİZ EDEN, KARŞI TARAF, VEKİLİ lines / TÜRK MİLLETİ ADINA -> `other`, one segment. İSTEMİN "
        "KONUSU -> issue; YARGILAMA SÜRECİ / MADDİ OLAY / DAVA KONUSU İSTEM (often embeds the whole "
        "lower ruling: segment it) -> facts; TEMYİZ EDENİN İDDİALARI / KARŞI TARAFIN SAVUNMASI -> "
        "issue; DANIŞTAY TETKİK HAKİMİ … DÜŞÜNCESİ / SAVCI DÜŞÜNCESİ -> other; İLGİLİ MEVZUAT (quoted "
        "law) -> rule_application; HUKUKİ DEĞERLENDİRME / İNCELEME VE GEREKÇE -> rule_application; "
        "KARAR SONUCU -> conclusion; KARŞI OY -> dissent. Newer records have no markers at all: judge "
        "by function. Some records begin with scraper noise ('Karar İçeriği', 'ee3', '0', "
        "'2023/1999999'); it is removed by code before you see it."),
    "first_degree": (
        "First-instance court (asliye ticaret / asliye hukuk). Header: T.C. / <İL> n. ASLİYE … "
        "MAHKEMESİ / ESAS NO / KARAR NO / DAVA TARİHİ / KARAR TARİHİ / GEREKÇELİ KARARIN YAZILDIĞI TARİH "
        "/ HAKİM / KATİP / DAVACI / DAVALI / VEKİLİ lines -> `other`, one segment. DAVA / CEVAP / "
        "DELİLLER -> facts; DELİLLERİN DEĞERLENDİRİLMESİ VE GEREKÇE / GEREĞİ DÜŞÜNÜLDÜ (one long "
        "reasoning block; use confidence low where nothing marks the role) -> rule_application; "
        "H Ü K Ü M / HÜKÜM -> conclusion; closing HAKİM / KATİP signature -> `other`."),
    "yargitay_hukuk": (
        "Yargıtay civil chamber: reviews a lower ruling on points of law, does not retry facts. Short, "
        "no fixed layout. Roles by function:\n"
        "  - header lines (chamber + E./K. numbers, 'İçtihat Metni', MAHKEMESİ :, SAYISI :, DAVA TÜRÜ :) "
        "and letter-spaced titles ('K A R A R', 'Y A R G I T A Y  K A R A R I') -> other\n"
        "  - what the courts below decided, the claim, the parties' history ('Taraflar arasındaki ... "
        "davasından dolayı ... hükmün ... temyiz edilmesi üzerine') -> facts\n"
        "  - procedural intro lines ('temyiz edilmekle evrak okunarak', 'Gereği görüşülüp düşünüldü', "
        "'dosya incelendi') -> other\n"
        "  - the appellant's arguments / the question examined -> issue\n"
        "  - the chamber's own reasoning, including numbered items 1), 2), a), b) after 'Gereği "
        "görüşülüp düşünüldü' -> rule_application\n"
        "  - the disposition paragraph -- ONANMASINA, BOZULMASINA, DÜZELTİLEREK ONANMASINA, REDDİNE, "
        "TEVDİİNE, GERİ ÇEVRİLMESİNE, İADESİNE, DÜŞMESİNE -- is ALWAYS `conclusion`, even when the same "
        "paragraph also reasons ('Dosyadaki yazılara ... göre ... ONANMASINA'). Never give it another role.\n"
        "  - a few chambers use a Roman-numeral layout: I. DAVA -> facts, II. CEVAP -> facts, III. İLK "
        "DERECE MAHKEMESİ KARARI -> facts, IV. İSTİNAF -> facts/issue, V. TEMYİZ -> issue, VI. KARAR -> "
        "conclusion (its GEREKÇE part -> rule_application).\n"
        "KABULÜNE/REDDİNE earlier in the text describe the lower court, not this chamber. Party names "
        "are redacted as '...' -- that is the source, not a value."),
    "yargitay_ceza": (
        "Yargıtay criminal chamber. Same rules as the civil chambers, plus: SUÇ : (the offence) and "
        "HÜKÜM : (the lower court's judgment) are content -> facts, and the offence is the subject_id "
        "(e.g. nitelikli_hirsizlik, sahte_belge_duzenleme). Numbered findings '1) ... 2) ...' that "
        "explain what the lower court got wrong are the chamber's reasoning -> rule_application; "
        "'Bozmayı gerektirmiş ... BOZULMASINA' is the disposition -> conclusion. When one decision "
        "disposes of several counts differently (one count affirmed, another reversed), emit one "
        "capsule per count."),
    "kvkk": (
        "KVKK (data protection board) decision summary. STAGES: the title line and the Karar Tarihi / "
        "Karar No / Konu Özeti rows -> background; the complaint (\"şikâyette özetle;\" and its list) "
        "and the data controller's defence (\"cevapta özetle;\" and its list) -> background; the "
        "Board's own evaluations -- the list after \"Kurulunun ... Kararı ile;\" stating what it found "
        "-> analysis; ONLY the disposition lines -> outcome (\"idari para cezası uygulanmasına\", "
        "\"talimatlandırılmasına\", \"yapılacak bir işlem olmadığına\", \"hatırlatılmasına\", "
        "\"sorumlular hakkında işlem yapılmasına\" and the closing \"karar verilmiştir.\"). A Board "
        "decision often carries SEVERAL dispositions: emit one capsule per disposition, each "
        "opinion_type board_decision. KVKK cites Yönetmelik/Tebliğ constantly: legislation_type "
        "regulation/directive, law_no null, full name in law_name."),
    "rekabet": "PLACEHOLDER -- no Rekabet Kurumu decision body has been extracted yet.",
    "uyusmazlik": "PLACEHOLDER -- no Uyuşmazlık Mahkemesi decision body has been extracted yet.",
}

BASE_INSTRUCTION = """You segment Turkish court and board decisions into retrievable chunks and write one reasoning capsule per legal conclusion.

INPUT. The decision as numbered paragraphs, one per line, each prefixed [p1], [p2], ... That is the WORKING COPY: every reference you make uses these markers. Sometimes a second block follows, "PLAIN TEXT COPY": the same decision as the database's plain text, without paragraph markers (for some courts it is one long line with the breaks removed). Read it if it helps you understand a passage; never cite it. Everything you store points at [pN] markers.

## segments[]

Group CONSECUTIVE paragraphs that share the same legal role into one segment. A segment is a retrievable unit: aim for 500-2,000 characters (code splits longer ones). Do not make one segment per paragraph when neighbouring paragraphs share a role -- a long HUKUKİ DEĞERLENDİRME or ESASIN İNCELENMESİ is a few segments of several paragraphs each, not fifty one-paragraph segments. A heading line joins the segment it introduces. THE ONE EXCEPTION TO GROUPING: the operative disposition (the paragraph or lines with REDDİNE / İPTALİNE / ONANMASINA / BOZULMASINA / İHLAL EDİLDİĞİNE / KABULÜNE / "idari para cezası uygulanmasına" / "yapılacak bir işlem olmadığına" / "... karar verildi/verilmiştir") is ALWAYS its own segment with the conclusion/outcome role, even when it is a single short line and even when it follows the reasoning without a heading. Never let it end a reasoning segment.
- `local_id`: "seg_1", "seg_2", ... in document order.
- `paragraph_refs`: EVERY marker in the segment, e.g. ["p6","p7","p8"], in order, no gaps. The stored chunk text is built by code from exactly these refs. EVERY paragraph of the input must appear in exactly one segment: none left out, none listed twice. Headers, signature lists and one-line headings are paragraphs too and go into a segment. A paragraph you leave out is put into a low-confidence catch-all chunk by code, which is worse than your grouping; a paragraph listed twice is kept only in its first segment.
- `role`: one of the values listed for this document. Three rules are checked:
  1. The DOCKET HEADER (court name, "ANAYASA MAHKEMESİ KARARI", "T.C.", Esas/Karar Sayısı or No, Karar Günü/Tarihi, Resmi Gazete line, "İçtihat Metni", MAHKEMESİ:/SAYISI:/DOSYA NO: lines, Karar Tarihi/Karar No/Konu Özeti rows) is ONE segment with the catch-all role (CATCHALL_ROLE below), never `facts`. The same for the SIGNATURE BLOCK (Başkan / Üye / names) and panel lists. Party lines that follow (İTİRAZ YOLUNA BAŞVURAN, DAVACI, DAVALI, TEMYİZ EDEN) start the content.
  2. `conclusion` / `outcome` is ONLY the operative section: SONUÇ / HÜKÜM / KARAR SONUCU / the decision items -- REDDİNE, İPTALİNE, ONANMASINA, BOZULMASINA, İHLAL EDİLDİĞİNE, KABULÜNE, "... karar verildi/verilmiştir". The sentence closing a reasoning section ("... reddi gerekir.") and per-provision merits sections are application / rule_application. An interim ruling inside the reasoning stays in the reasoning role.
  3. Verbatim statute or Constitution text (quoted provisions, "Madde 5 - ...", the İLGİLİ HUKUK / İPTALİ İSTENEN KANUN HÜKÜMLERİ block) is `rule` (aym) or `rule_application` (courts). The court's paraphrase is `application`.
- `confidence`: "high" when the role is grounded in a marker or unmistakable content, "low" when it is your inference. Be honest.
- `rights` (aym individual applications only): the right(s) THIS paragraph discusses, from the candidate list; null when none.

## cited_legislations[] (inside each segment)

REQUIRED on every segment whose text names a law, decree-law, constitutional article, regulation, directive or communiqué (Tebliğ) -- "6698 sayılı Kanun'un 12'nci maddesi", "Anayasa'nın 20. maddesi", "Kişisel Sağlık Verileri Hakkında Yönetmelik". Empty only when the segment names none. An empty list on a segment that names a law is an error that is checked. `verbatim_mention` exactly as written. `law_no` only when the text states the number (null for the Constitution and for named regulations). `law_short` the abbreviation exactly as written (T.B.K., HMK, İYUK); never convert it to a number. `law_name` if stated at that mention. `article_no` as written ("15", "141/A", "Geçici 3"). `paragraph_no` as a digit (ikinci -> "2"). `law_date` ISO if given. `legislation_type`: statute, decree_law, constitution, regulation, directive, treaty (AİHS and other conventions), or not_legislation. `confidence`: high when an article is identified, low when only the law is named.
CASE LAW IS NOT LEGISLATION: a person's name, "B. No:", "[GK]", "E. 2019/123", "Anayasa Mahkemesi Kararlar Dergisi", "ve diğerleri" mark precedent or doctrine -- leave them out, or give them not_legislation. List each provision at most once per segment. Never infer a field the text does not state: use null.

## capsules[]

One per legal subject decided, PLUS one per dissenting or concurring opinion.
- `outcome`: ONE of the values listed for this document. For the majority: the court's principal disposition, read from the operative ruling. For a dissent: the disposition the dissenter argued for -- a dissenter who would have found a violation says violation, who would have found none says no_violation, who would have annulled says annulled, who would have upheld the provision says denied, who would have reversed says reversed. A dissent that objects only to procedure or jurisdiction names the disposition that objection leads to (dismissed_procedural, no_jurisdiction, remanded). Never the opinion type, never the violation: it names WHAT WAS DECIDED. Several dispositions -> several capsules. "other" is a last resort explained in conclusion_sentence.
- `opinion_type`: majority; dissent (karşı oy / muhalefet şerhi / ayrışık oy); concurring (farklı / ek / değişik gerekçe); board_decision for a KVKK board ruling.
- `dissent_authors`: surnames as printed under a dissent or concurring opinion; empty otherwise.
- `subject_id`: a short snake_case TURKISH slug naming the subject in dispute, in the court's own words (vergi_ziyai_cezasi, kamulastirmasiz_el_atma, veri_guvenligi_ihlali). Never "unspecified".
- `conclusion_sentence`: one sentence IN TURKISH stating what was decided and why, naming the concrete subject.
- `reasoning_summary`: 2-4 sentences IN TURKISH explaining WHY, read from the rule and application segments, keeping the decision's own terms of art verbatim (a lawyer searches with the term of art). Every law number, article number and case number you write must appear in the decision text.
- `supporting_local_ids`: the segments that support this conclusion. A dissent or concurring capsule cites only that opinion's own dissent segments; a majority capsule never cites a dissent segment.

Do not output chunk_id, canonical_id, char_length, citation_granularity, chunk_label, subject_type, content_type, reasoning_stage or reasoning_summary_method: code sets them."""

AUDIT_INSTRUCTION = """You are AUDITING a first-pass segmentation of a Turkish court or board decision made by another model. You receive the decision as numbered paragraphs, the first pass's segments (local_id, paragraph range, role) and its capsules (index, opinion type, outcome, subject, conclusion sentence).

Return ONLY what must change. Empty lists mean the first pass is right. A fix is for a clear error; prefer the first pass when in doubt. Do not re-segment.

role_fixes -- ONLY these four clear cases, nothing else: (1) a docket header or signature block not in the catch-all role; (2) the paragraph carrying the operative disposition not in conclusion/outcome; (3) kvkk: Board evaluations tagged outcome, or disposition lines tagged analysis; (4) a KARŞI OY / MUHALEFET ŞERHİ / AYRIŞIK OY / FARKLI GEREKÇE opinion not tagged dissent. Every other role call belongs to the first pass -- leave it. NEVER move the paragraph that carries the operative disposition (ONANMASINA, BOZULMASINA, REDDİNE, İPTALİNE, İHLAL EDİLDİĞİNE, KABULÜNE, "karar verildi/verilmiştir") out of conclusion/outcome, even if that paragraph also contains reasoning; a decision without a conclusion segment is wrong. The docket header and the signature block take the catch-all role, never `facts`. `conclusion`/`outcome` is only the operative section (SONUÇ / HÜKÜM / KARAR SONUCU / decision items + "karar verildi/verilmiştir"); the sentence closing a reasoning section and per-provision merits sections are application / rule_application. For kvkk, the Board's evaluations under "Kararı ile;" are analysis; only the disposition lines are outcome. Verbatim statute text is rule (aym) / rule_application (courts). A KARŞI OY / MUHALEFET ŞERHİ / AYRIŞIK OY / FARKLI GEREKÇE opinion is dissent; "X bu görüşe katılmamıştır" inside the majority is not.

outcome_fixes -- a capsule whose outcome does not name what the operative ruling decided for THAT capsule's subject and opinion (a dissent's outcome is what the dissenter argued for). Use only the allowed values below.

missing_capsules -- a disposition in the operative ruling that has no capsule at all (a fine AND an instruction, but only the fine has a capsule).

missing_citations -- a law, decree-law, constitutional article, regulation, directive or communiqué named in a segment's text that the first pass did not cite there. Give the segment's local_id and the citation fields exactly as the text states them (law_no only if the number is written; law_name for named regulations; verbatim_mention copied from the text). Each segment's existing citations are listed, so add only what is missing.

One sentence of reason in Turkish for every fix."""

# Gold-file values with no honest mapping into the closed vocabulary.
_GOLD_OUTCOME = {"i̇hlal": "violation", "ihlal": "violation", "denied_on_merits": "denied",
                 "appeal_denied_lower_ruling_affirmed": "affirmed", "denied_procedural": "dismissed_procedural",
                 "no_action_needed": "no_action", "onama": "affirmed"}
_GOLD_DISSENT = {"danistay": "reversed"}


def _short_ref(pid, counter):
    m = re.search(r"-p(\d+)$", str(pid))
    if m:
        return "p" + m.group(1), counter
    return "p" + str(counter + 1), counter + 1


def build_fewshot(source, kind, case_no, candidate_rights, outcomes):
    """A gold document rendered in the response shape. None for norm review (the
    individual-application example taught `facts` on the header) and for kinds
    with no structurally similar example."""
    if kind == "aym_norm_review":
        return None
    path = FEWSHOT_DIR / (source + ".json")
    if not path.is_file():
        path = FEWSHOT_DIR / ({"yargitay": "bam"}.get(source, source) + ".json")
        if not path.is_file():
            return None
    gold = json.loads(path.read_text(encoding="utf-8"))
    keys = ("law_no", "law_short", "law_name", "article_no", "paragraph_no", "law_date",
            "legislation_type", "confidence", "verbatim_mention")
    segments, label, counter = [], {}, 0
    for i, ch in enumerate(gold.get("chunks", []), 1):
        lid = f"seg_{i}"
        label[ch.get("chunk_id")] = lid
        refs = []
        for pid in ch.get("source_paragraph_ids") or []:
            ref, counter = _short_ref(pid, counter)
            refs.append(ref)
        role = ch.get("firac_role") or ch.get("court_reasoning_role") or ch.get("regulatory_role") or ch.get("role")
        seg = {"local_id": lid, "paragraph_refs": refs, "role": role,
               "confidence": ch.get("confidence") or "high",
               "cited_legislations": [{k: c.get(k) for k in keys} for c in ch.get("cited_legislations") or []]}
        for c in seg["cited_legislations"]:
            c["legislation_type"] = c["legislation_type"] or "statute"
            c["confidence"] = c["confidence"] or "low"
        if source == "aym":
            seg["rights"] = [r for r in ch.get("rights") or [] if r in candidate_rights] or None
        segments.append(seg)
    capsules = []
    for cap in gold.get("reasoning_capsules", []):
        kind_, _, author = (cap.get("opinion_type") or "majority").partition(":")
        outcome = _GOLD_OUTCOME.get((cap.get("outcome") or "").replace("̇", ""), cap.get("outcome"))
        if outcome not in outcomes:
            outcome = _GOLD_DISSENT.get(source, "other") if kind_ == "dissent" else "other"
        subject = cap.get("subject_id") or "unspecified"
        if candidate_rights and subject not in candidate_rights:
            subject = candidate_rights[0]
        capsules.append({"outcome": outcome, "opinion_type": kind_,
                         "dissent_authors": [author] if author else [],
                         "subject_id": subject,
                         "conclusion_sentence": cap.get("conclusion_sentence"),
                         "reasoning_summary": cap.get("reasoning_summary"),
                         "supporting_local_ids": [label[c] for c in cap.get("supporting_chunk_ids") or [] if c in label]})
    return {"segments": segments, "capsules": capsules}


def build_system_instruction(source, kind, case_no, candidate_rights, examined, vocab, corrections=None):
    role_field, roles = ROLE_VOCAB[source]
    parts = [BASE_INSTRUCTION, "", "## This document", "",
             f"source_type: {source}", f"kind: {kind}", f"case_no: {case_no}",
             f"Role field `{role_field}`, one of: {', '.join(roles)}",
             f"CATCHALL_ROLE for headers and signature blocks: {CATCHALL.get(source, 'other')}", "",
             KIND_NOTES[kind], "", "## Allowed values", "", "`outcome` (with its Turkish meaning):"]
    parts += [f"  - {v}: {vocab['gloss'][v]}" for v in vocab["outcomes"]]
    parts += [f"`opinion_type`: {', '.join(vocab['opinion_kinds'])}",
              f"`legislation_type`: {', '.join(vocab['legislation_types'])}", ""]
    if candidate_rights:
        parts += ["Candidate rights from the court's own metadata. `rights` and `subject_id` must come "
                  "from this list only; one capsule per right the court decided:"]
        parts += [f"  - {r}" for r in candidate_rights] + [""]
    if examined:
        parts += ["Provisions under review, from the record's own metadata (subject_id names one of these):"]
        parts += [f"  - {e['law']} m.{e['article']}" + (f"/{e['clause']}" if e.get("clause") else "")
                  + (f"  -> {e['result']}" if e.get("result") else "") for e in examined[:12]] + [""]
    if corrections:
        parts += ["## Corrections to your previous answer", "",
                  "Your previous answer for THIS document was rejected by automated checks. Produce "
                  "the whole answer again, fixing every item below and changing nothing else:"]
        parts += [f"  - {c[:400]}" for c in corrections] + [""]
    few = build_fewshot(source, kind, case_no, candidate_rights, vocab["outcomes"])
    if few:
        parts += ["## Worked example", "",
                  f"A hand-verified response for a different {source} document. Match its structure and "
                  "level of detail, not its facts. Its conclusion_sentence predates the Turkish rule: "
                  "yours must be Turkish.", "", json.dumps(few, ensure_ascii=False, indent=1)]
    return "\n".join(parts)


def build_user_content(paragraphs, plain_copy=None):
    lines = ["Decision text:", ""] + [f"[p{i}] {t}" for i, t in enumerate(paragraphs, 1)]
    if plain_copy:
        lines += ["", "PLAIN TEXT COPY (same decision, no paragraph markers; read only, never cite):",
                  plain_copy]
    return "\n".join(lines)


def build_audit_instruction(source, kind, roles, outcomes, gloss):
    parts = [AUDIT_INSTRUCTION, "", "## This document", "", f"source_type: {source}", f"kind: {kind}",
             f"Allowed roles: {', '.join(roles)}",
             f"Catch-all role: {CATCHALL.get(source, 'other')}", "", KIND_NOTES[kind], "",
             "Allowed `outcome` values:"]
    parts += [f"  - {v}: {gloss[v]}" for v in outcomes]
    return "\n".join(parts)


def build_audit_content(paragraphs, segments, capsules, metadata_hint=None, no_ruling=False):
    lines = ["Decision text:", ""] + [f"[p{i}] {t}" for i, t in enumerate(paragraphs, 1)]
    if no_ruling:
        lines += ["", "ALERT: the first pass has NO conclusion/outcome segment. Every decision has an "
                  "operative disposition. Find the segment that contains it (REDDİNE / İPTALİNE / "
                  "ONANMASINA / BOZULMASINA / İHLAL EDİLDİĞİNE / KABULÜNE / 'idari para cezası "
                  "uygulanmasına' / 'yapılacak bir işlem olmadığına' / '... karar verildi/verilmiştir' / "
                  "DEVRİNE / GÖNDERİLMESİNE) and return a role_fix moving it to conclusion (or outcome "
                  "for kvkk). If that segment also holds reasoning, still move it: the disposition wins."]
    lines += ["", "FIRST-PASS SEGMENTS (local_id: paragraphs -> role):"]
    for seg in segments:
        refs = [str(r) for r in getattr(seg, "paragraph_refs", [])]
        rng = (refs[0] + ("-" + refs[-1] if len(refs) > 1 else "")) if refs else "(none)"
        cites = []
        for c in getattr(seg, "cited_legislations", []) or []:
            law = getattr(c, "law_no", None) or getattr(c, "law_short", None) or (getattr(c, "law_name", None) or "?")[:40]
            art = getattr(c, "article_no", None)
            cites.append(f"{law}" + (f" m.{art}" if art else ""))
        lines.append(f"  {seg.local_id}: {rng} -> {seg.role}" + (f"  | cites: {'; '.join(cites)}" if cites else "  | cites: none"))
    lines += ["", "FIRST-PASS CAPSULES (index: opinion_type | outcome | subject_id :: conclusion_sentence):"]
    for k, cap in enumerate(capsules):
        lines.append(f"  [{k}] {getattr(cap, 'opinion_type', None)} | {getattr(cap, 'outcome', None)} | "
                     f"{getattr(cap, 'subject_id', None)} :: {(getattr(cap, 'conclusion_sentence', None) or '')[:300]}")
    if metadata_hint:
        lines += ["", "THE COURT'S OWN METADATA (structured verdicts; authoritative unless the text "
                  "plainly says otherwise):", metadata_hint]
    return "\n".join(lines)
