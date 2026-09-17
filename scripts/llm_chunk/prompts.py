"""
prompts.py -- what the model is told. Three kinds of independent request per
document: STRUCTURE (segments and roles), CAPSULES (one per decision and per
separate opinion; allowed values with Turkish glosses) and LAWS (every
legislation citation, paragraph by paragraph, sent in windows). None needs
another's answer, so all go into the same batch. Worked examples come from
fewshot/. No imports from chunker.py: the vocabularies are passed in.
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
        "issue; DANIŞTAY TETKİK HAKİMİ … DÜŞÜNCESİ / SAVCI DÜŞÜNCESİ -> other (never dissent); İLGİLİ MEVZUAT (quoted "
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
        "opinion_type board_decision."),
    "rekabet": "PLACEHOLDER -- no Rekabet Kurumu decision body has been extracted yet.",
    "uyusmazlik": "PLACEHOLDER -- no Uyuşmazlık Mahkemesi decision body has been extracted yet.",
}

STRUCTURE_INSTRUCTION = """You segment Turkish court and board decisions into retrievable chunks and give every segment its legal role. You do NOT write capsules and do NOT list law citations: separate readers do that.

INPUT. The decision as numbered paragraphs, one per line, each prefixed [p1], [p2], ... That is the WORKING COPY: every reference you make uses these markers. Sometimes a second block follows, "PLAIN TEXT COPY": the same decision as the database's plain text, without paragraph markers (for some courts it is one long line with the breaks removed). Read it if it helps you understand a passage; never cite it. Everything you store points at [pN] markers.

## segments[]

Group CONSECUTIVE paragraphs that share the same legal role into one segment. A segment is a retrievable unit: aim for 500-2,000 characters (code splits longer ones). Do not make one segment per paragraph when neighbouring paragraphs share a role -- a long HUKUKİ DEĞERLENDİRME or ESASIN İNCELENMESİ is a few segments of several paragraphs each, not fifty one-paragraph segments. A heading line joins the segment it introduces: a bare section heading ("B. Diğer İhlal İddiaları", "III. HÜKÜM", "KARAR", "Bu itibarla;", "V. TEMYİZ") is never a segment of its own. THE ONE EXCEPTION TO GROUPING: the operative disposition (the paragraph or lines with REDDİNE / İPTALİNE / ONANMASINA / BOZULMASINA / İHLAL EDİLDİĞİNE / KABULÜNE / "idari para cezası uygulanmasına" / "yapılacak bir işlem olmadığına" / "... karar verildi/verilmiştir") is ALWAYS its own segment with the conclusion/outcome role, even when it is a single short line and even when it follows the reasoning without a heading. The whole operative ruling -- its heading, every numbered item, costs and fees, the closing "karar verildi" line -- is ONE segment, not one segment per line. Never let it end a reasoning segment.
- `local_id`: "seg_1", "seg_2", ... in document order.
- `paragraph_refs`: EVERY marker in the segment, e.g. ["p6","p7","p8"], in order, no gaps. The stored chunk text is built by code from exactly these refs. EVERY paragraph of the input must appear in exactly one segment: none left out, none listed twice. Headers, signature lists and one-line headings are paragraphs too and go into a segment. A paragraph you leave out is put into a low-confidence catch-all chunk by code, which is worse than your grouping; a paragraph listed twice is kept only in its first segment.
- `role`: one of the values listed for this document. Three rules are checked:
  1. The DOCKET HEADER (court name, "ANAYASA MAHKEMESİ KARARI", "T.C.", Esas/Karar Sayısı or No, Karar Günü/Tarihi, Resmi Gazete line, "İçtihat Metni", MAHKEMESİ:/SAYISI:/DOSYA NO: lines, Karar Tarihi/Karar No/Konu Özeti rows) is ONE segment with the catch-all role (CATCHALL_ROLE below), never `facts`. The same for the SIGNATURE BLOCK (Başkan / Üye / names) and panel lists. Party lines that follow (İTİRAZ YOLUNA BAŞVURAN, DAVACI, DAVALI, TEMYİZ EDEN) start the content. The header segment ENDS where content starts: never put the header and the parties, the claims or the first prose paragraph into one segment.
  2. `conclusion` / `outcome` is ONLY the operative section: SONUÇ / HÜKÜM / KARAR SONUCU / the decision items -- REDDİNE, İPTALİNE, ONANMASINA, BOZULMASINA, İHLAL EDİLDİĞİNE, KABULÜNE, "... karar verildi/verilmiştir". The sentence closing a reasoning section ("... reddi gerekir.") and per-provision merits sections are application / rule_application. An interim ruling inside the reasoning stays in the reasoning role. A paragraph that REPORTS an earlier decision in the case history ("Daire, ... bozmuş", "Mahkemece ... karar verilmiş", a "Karar sonucu:" line inside YARGILAMA SÜRECİ) is `facts`, not the ruling of this decision.
  3. Verbatim statute or Constitution text (quoted provisions, "Madde 5 - ...", the İLGİLİ HUKUK / İPTALİ İSTENEN KANUN HÜKÜMLERİ block) is `rule` (aym) or `rule_application` (courts). The court's paraphrase is `application`.
  4. `dissent` is ONLY a judge or member of THIS court disagreeing with THIS decision (KARŞI OY, MUHALEFET ŞERHİ, AYRIŞIK OY, FARKLI/EK/DEĞİŞİK GEREKÇE), written after the operative ruling. The examining judge's or the prosecutor's opinion (TETKİK HAKİMİ DÜŞÜNCESİ, SAVCI DÜŞÜNCESİ, Cumhuriyet Savcısı görüşü) is NOT a dissent: catch-all role. "X bu görüşe katılmamıştır" inside the majority text is not a dissent segment either.
- `confidence`: "high" when the role is grounded in a marker or unmistakable content, "low" when it is your inference. Be honest.
- `rights` (aym individual applications only): the right(s) THIS paragraph discusses, from the candidate list; null when none.

Do not output capsules, cited_legislations, chunk_id, canonical_id, char_length, citation_granularity, chunk_label, subject_type, content_type or reasoning_stage: code sets them."""

CAPSULES_INSTRUCTION = """You write the reasoning capsules of a Turkish court or board decision: one per legal decision in its operative ruling, plus one per dissenting or concurring opinion. Nothing else: no segments, no roles, no citations.

INPUT. The decision as numbered paragraphs, one per line, each prefixed [p1], [p2], ... Sometimes a "PLAIN TEXT COPY" follows: read it if it helps, never cite it.

HOW TO READ IT. First find the operative ruling of THIS court (SONUÇ / HÜKÜM / KARAR SONUCU / KARAR, or the closing disposition sentences) and read it item by item. Then find every separate opinion (KARŞI OY / MUHALEFET ŞERHİ / AYRIŞIK OY / FARKLI or EK or DEĞİŞİK GEREKÇE) and who wrote it -- a long decision can carry several, each written by one or more members. Only then write the capsules. A paragraph that reports an earlier decision in the case history is not this ruling, and the examining judge's or the prosecutor's opinion is not a separate opinion.

FIRST LIST THE RULING. `ruling_items`: every item of THIS court's operative ruling, in order -- `text` is the item's own words, shortened to at most 150 characters but keeping its capitalised verb (ONANMASINA, İPTALİNE, REDDİNE, TEVDİİNE, ...) and any amount; `kind` is decision, cost_or_fee, or forwarding_or_service. Then write majority capsules ONLY for decision items (items on the same legal subject share one capsule), plus one per separate opinion. A total of amounts already given item by item ("toplam 330.000 TL") is not a new item.

WHAT IS NOT A DECISION. Court costs and fees and their refund (harç, yargılama gideri, vekâlet ücreti, gider avansı), serving or sending copies of the decision, and forwarding the file (dosyanın ... gönderilmesine, Başsavcılığa TEVDİİNE) when they accompany a main disposition: no capsule for them. A ruling whose only disposition is sending the file somewhere is one capsule (transferred or remitted).

## capsules[]

One per legal subject decided, PLUS one per dissenting or concurring opinion. Before writing them, read the operative ruling line by line: every separate disposition -- each administrative fine with its amount, each instruction to the data controller, each "yapılacak bir işlem olmadığına", each count or provision decided -- gets its own capsule. Do not stop after the first one. Capsules NEVER overlap: one capsule per separate decision. A claim decided in parts (KISMEN KABUL / KISMEN RED, kısmen bozma) is ONE capsule with partially_granted -- no extra capsules for its parts, and no summary capsule on top of per-claim capsules. Two majority capsules only when the ruling decides different things (a fine AND an instruction; one party's appeal affirmed and another party's appeal dismissed).
- `outcome`: ONE of the values listed for this document. For the majority: the disposition of THIS court, read from the VERB of its own operative items (the capitalised word that ends each item). Appeal courts (Yargıtay, Danıştay, BAM): ONANMASINA -> affirmed; DÜZELTİLEREK ONANMASINA -> corrected_affirmed; BOZULMASINA -> reversed; istinaf/temyiz başvurusunun KABULÜNE with the decision's KALDIRILMASINA -> granted, or remanded when the file is sent back to be decided again (… GÖNDERİLMESİNE / GERİ ÇEVRİLMESİNE), or no_decision_needed when the ruling then says the case itself needs no decision (dava hakkında karar verilmesine yer olmadığına); esastan REDDİNE -> denied; the case ended by waiver or settlement (davanın feragat / kabul / sulh nedeniyle reddine or düşmesine) -> abated; TEVDİİNE -> remitted; a petition rejected for its amount, time limit or form (miktardan / süreden / usulden REDDİNE, dilekçenin reddi) -> dismissed_procedural. What the LOWER court decided, written inside the item, is not this court's outcome: "davanın iptaline ilişkin İdare Mahkemesi kararının ONANMASINA" is affirmed, not annulled. Use a value only if it is listed for this document. For a dissent: the disposition the dissenter argued for -- a dissenter who would have found a violation says violation, who would have found none says no_violation, who would have annulled says annulled, who would have upheld the provision says denied, who would have reversed says reversed. A dissent that objects only to procedure or jurisdiction names the disposition that objection leads to (dismissed_procedural, no_jurisdiction, remanded). A dissent that only says the application should have been examined, without saying how it should then be decided, is other, explained in conclusion_sentence. A dissent's conclusion_sentence and reasoning_summary state the DISSENTER's view -- what the dissenter would have decided and why, never the majority's reasoning -- and agree with its outcome. Never the opinion type, never the violation: it names WHAT WAS DECIDED. Several dispositions -> several capsules. "other" is a last resort explained in conclusion_sentence.
- `opinion_type`: majority; dissent (karşı oy / muhalefet şerhi / ayrışık oy); concurring (farklı / ek / değişik gerekçe); board_decision for a KVKK board ruling.
- `dissent_authors`: surnames as printed under a dissent or concurring opinion; empty otherwise.
- `subject_id`: a short snake_case TURKISH slug naming the subject in dispute, in the court's own words (vergi_ziyai_cezasi, kamulastirmasiz_el_atma, veri_guvenligi_ihlali). Never "unspecified".
- `conclusion_sentence`: one sentence IN THE LANGUAGE OF THE DECISION (Turkish for a Turkish decision, English for an English one) stating what was decided and why, naming the concrete subject; at most about 300 characters -- the disposition and its main reason, not every procedural instruction.
- `reasoning_summary`: 2-4 sentences IN THE LANGUAGE OF THE DECISION explaining WHY, read from the reasoning the decision gives, keeping the decision's own terms of art verbatim (a lawyer searches with the term of art). Every law number, article number and case number you write must appear in the decision text.
- `supporting_paragraph_refs`: the paragraphs this capsule rests on -- its ruling item and the reasoning behind it, e.g. ["p40","p41","p57"]. A dissent or concurring capsule cites only that opinion's own paragraphs; a majority capsule never cites a separate opinion's paragraphs.

Do not output segments, cited_legislations, chunk_id, canonical_id, subject_type or reasoning_summary_method: code sets them."""

LAWS_INSTRUCTION = """You list every legislation citation in a Turkish court or board decision, paragraph by paragraph. Nothing else: no roles, no summaries, no segments.

INPUT. The decision as numbered paragraphs, one per line, each prefixed [p1], [p2], ... Sometimes a "PLAIN TEXT COPY" follows: read it if it helps, never cite it.

OUTPUT. `paragraphs`: one entry for EVERY paragraph of YOUR PART (named at the end of the input), in order -- {"ref": "pN", "cited_legislations": [...]}. The list is empty when that paragraph names no legislation. Do not skip paragraphs and do not stop early. Paragraphs before your part are there only so that back-references can be resolved ("Kanun" defined earlier): do not list them.

WHAT COUNTS. Every mention of a law, decree-law (KHK, Cumhurbaşkanlığı Kararnamesi), the Constitution, a regulation (Yönetmelik), directive or communiqué (Tebliğ), or a treaty -- in any form:
- full: "6698 sayılı Kişisel Verilerin Korunması Kanunu'nun 12'nci maddesi", "Anayasa'nın 20. maddesi"
- abbreviation: "HMK'nın 353/1-b-1 maddesi", "TCK 86/1", "İYUK m.49"
- BACK-REFERENCE to a law named earlier: "Kanun'un 10'uncu maddesi", "aynı Kanun'un 12. maddesi", "anılan Yönetmeliğin 5. maddesi"
- inside the operative ruling: "... 6100 sayılı HMK'nın 353/1-b-1 maddesi uyarınca ... REDDİNE"
- a law named without an article: "2577 sayılı İdari Yargılama Usulü Kanunu"
- quoted statute text: "Madde 5 - ..." under a heading that names the law
One entry per provision per paragraph: the same article named twice in one paragraph is listed once; different articles are separate entries.

FIELDS. `verbatim_mention`: copied character for character from THAT paragraph (when this paragraph words it differently from an earlier one, copy this paragraph's words). `law_no`: only when this decision states the number -- at this mention, or where the decision defined the short name a back-reference uses ("6698 sayılı Kişisel Verilerin Korunması Kanunu (Kanun)" -> "Kanun'un 10'uncu maddesi" has law_no "6698"); null for the Constitution, for named regulations, and when the decision never gives the number. `law_short`: the abbreviation exactly as written (T.B.K., HMK, İYUK); never turn it into a number. A word such as "Kanun" is not an abbreviation: null. `law_name`: the law's name as this decision writes it ("Türk Borçlar Kanunu"); a bare word such as "Kanun", "Anayasa" or "Yönetmelik" is not a name, and never expand an abbreviation into a name the decision does not write: null. `article_no`: as written ("15", "141/A", "Geçici 3", "353/1-b-1"). `paragraph_no`: the fıkra as a digit (ikinci -> "2"), else null. `law_date`: ISO date if written at that mention, else null. `legislation_type`: statute, decree_law, constitution, regulation, directive, treaty (AİHS and other conventions), or not_legislation. Never infer a field the text does not state: null.

NOT LEGISLATION. Case law and doctrine -- a person's name, "B. No:", "[GK]", "E. 2019/123 K. 2020/45", "Anayasa Mahkemesi Kararlar Dergisi", "ve diğerleri", a court's decision number, a Resmî Gazete date or number on its own, the name of a court or institution ("Anayasa Mahkemesi", "Yargıtay", "Danıştay", "Kurul"), a guide or handbook (Rehber, Kılavuz). Leave them out."""

LAWS_NOTES = {
    "kvkk": ("KVKK decisions cite Yönetmelik and Tebliğ constantly: legislation_type regulation or directive, "
             "law_no null, the full name in law_name. \"6698 sayılı Kişisel Verilerin Korunması Kanunu (Kanun)\" "
             "defines \"Kanun\" for the whole decision. \"Kişisel Veri Güvenliği Rehberi\" and the Board's other "
             "guides (Rehber) are not legislation: leave them out."),
    "aym": ("Constitution articles (Anayasa'nın N. maddesi): legislation_type constitution, law_no null. Articles "
            "of the European Convention (Sözleşme, AİHS): treaty. Earlier Constitutional Court judgments "
            "(B. No., E./K. numbers) are case law, not legislation."),
}

LAWS_LIMIT_NOTE = ("Your previous answers overflowed. List at most 10 cited_legislations per paragraph, "
                   "each provision once, verbatim_mention under 120 characters.")
STRUCTURE_LIMIT_NOTE = "Your previous answers overflowed. Use fewer, longer segments."
CAPSULES_LIMIT_NOTE = ("Your previous answers overflowed. One conclusion_sentence per capsule and at most 4 "
                       "sentences of reasoning_summary.")

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
    segments, refs_of, counter = [], {}, 0
    for i, ch in enumerate(gold.get("chunks", []), 1):
        lid = f"seg_{i}"
        refs = []
        for pid in ch.get("source_paragraph_ids") or []:
            ref, counter = _short_ref(pid, counter)
            refs.append(ref)
        refs_of[ch.get("chunk_id")] = refs
        role = ch.get("firac_role") or ch.get("court_reasoning_role") or ch.get("regulatory_role") or ch.get("role")
        seg = {"local_id": lid, "paragraph_refs": refs, "role": role,
               "confidence": ch.get("confidence") or "high"}
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
                         "supporting_paragraph_refs": list(dict.fromkeys(
                             r for c in cap.get("supporting_chunk_ids") or [] for r in refs_of.get(c, [])))})
    return {"segments": segments, "capsules": capsules}


_EXAMPLES = None


def example_documents():
    """{(source, case_no or doc_id)} of every document a worked example was made
    from, read from fewshot/*.json itself (no hand-kept list to go stale)."""
    global _EXAMPLES
    if _EXAMPLES is None:
        keys = set()
        for path in FEWSHOT_DIR.glob("*.json"):
            gold = json.loads(path.read_text(encoding="utf-8"))
            for item in gold.get("chunks", []) + gold.get("reasoning_capsules", []):
                src = item.get("source_type") or path.stem
                if item.get("case_no"):
                    keys.add((src, str(item["case_no"])))
                label = str(item.get("chunk_label") or "")
                if "-p" in label:
                    keys.add((src, label.split("-p", 1)[0]))
        _EXAMPLES = keys
    return _EXAMPLES


def is_example_document(source, case_no, doc_id=None):
    ex = example_documents()
    return (source, str(case_no)) in ex or (doc_id is not None and (source, str(doc_id)) in ex)


def _document_header(source, kind, case_no):
    return ["", "## This document", "", f"source_type: {source}", f"kind: {kind}", f"case_no: {case_no}"]


def build_structure_instruction(source, kind, case_no, candidate_rights, vocab, with_example=True):
    role_field, roles = ROLE_VOCAB[source]
    parts = [STRUCTURE_INSTRUCTION] + _document_header(source, kind, case_no) + [
        f"Role field `{role_field}`, one of: {', '.join(roles)}",
        f"CATCHALL_ROLE for headers and signature blocks: {CATCHALL.get(source, 'other')}", "",
        KIND_NOTES[kind], ""]
    if candidate_rights:
        parts += ["Candidate rights from the court's own metadata. `rights` must come from this list only:"]
        parts += [f"  - {r}" for r in candidate_rights] + [""]
    few = build_fewshot(source, kind, case_no, candidate_rights, vocab["outcomes"]) if with_example else None
    if few:
        parts += ["## Worked example", "",
                  f"A hand-verified answer for a different {source} document. Match its segmentation and level "
                  "of detail, not its facts.", "", json.dumps({"segments": few["segments"]}, ensure_ascii=False, indent=1)]
    return "\n".join(parts)


def build_capsules_instruction(source, kind, case_no, candidate_rights, examined, vocab, with_example=True,
                               metadata_hint=None):
    parts = [CAPSULES_INSTRUCTION] + _document_header(source, kind, case_no) + [
        "", KIND_NOTES[kind], "", "## Allowed values", "", "`outcome` (with its Turkish meaning):"]
    parts += [f"  - {v}: {vocab['gloss'][v]}" for v in vocab["outcomes"]]
    parts += [f"`opinion_type`: {', '.join(vocab['opinion_kinds'])}", ""]
    if candidate_rights:
        parts += ["Candidate rights from the court's own metadata. `subject_id` must come from this list only; "
                  "one capsule per right the court decided:"]
        parts += [f"  - {r}" for r in candidate_rights] + [""]
    if examined:
        parts += ["Provisions under review, from the record's own metadata (subject_id names one of these):"]
        parts += [f"  - {e['law']} m.{e['article']}" + (f"/{e['clause']}" if e.get("clause") else "")
                  + (f"  -> {e['result']}" if e.get("result") else "") for e in examined[:12]] + [""]
    if metadata_hint:
        parts += ["The court's own recorded verdicts, from its metadata (authoritative unless the text "
                  "plainly says otherwise; each maps to the outcome after the arrow):", metadata_hint, ""]
    few = build_fewshot(source, kind, case_no, candidate_rights, vocab["outcomes"]) if with_example else None
    if few and few["capsules"]:
        parts += ["## Worked example", "",
                  f"A hand-verified answer for a different {source} document. Match its structure and level of "
                  "detail, not its facts. Capsule sentences are in the language of THIS decision.", "",
                  json.dumps({"capsules": few["capsules"]}, ensure_ascii=False, indent=1)]
    return "\n".join(parts)


def build_user_content(paragraphs, plain_copy=None):
    lines = ["Decision text:", ""] + [f"[p{i}] {t}" for i, t in enumerate(paragraphs, 1)]
    if plain_copy:
        lines += ["", "PLAIN TEXT COPY (same decision, no paragraph markers; read only, never cite):",
                  plain_copy]
    return "\n".join(lines)


def build_laws_content(paragraphs, first, last, plain_copy=None):
    """The decision up to `last` (earlier paragraphs resolve back-references) and
    which paragraphs this request must answer for."""
    lines = ["Decision text:", ""] + [f"[p{i}] {t}" for i, t in enumerate(paragraphs[:last], 1)]
    if plain_copy:
        lines += ["", "PLAIN TEXT COPY (same decision, no paragraph markers; read only, never cite):", plain_copy]
    lines += ["", f"YOUR PART: paragraphs p{first} to p{last} ({last - first + 1} paragraphs). List every one of "
                  "them in `paragraphs`, in order."
              + (f" Paragraphs before p{first} are context only: do not list them." if first > 1 else "")]
    return "\n".join(lines)


def build_laws_instruction(source):
    parts = [LAWS_INSTRUCTION]
    if LAWS_NOTES.get(source):
        parts += ["", "## This court", "", LAWS_NOTES[source]]
    return "\n".join(parts)
