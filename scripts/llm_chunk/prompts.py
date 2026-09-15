"""
prompts.py -- system instructions and few-shot construction for chunk_generate.py.

Kept separate so the prompt text can be read and edited without scrolling past the
pipeline logic. Few-shot examples come from llm_chunk/fewshot/, which holds the five
hand-verified gold documents (ground truth already confirmed correct). They are
converted back into the *response* shape we ask Gemini for -- segments/capsules with
local_ids -- not the post-processed output shape, so the example matches the task.
"""

import json
import re
from pathlib import Path

FEWSHOT_DIR = Path(__file__).resolve().parents[2] / "fewshot"

# Role field name and allowed values per source. aym uses FIRAC; the other four do
# not (docs 14.1). "dissent" is in both vocabularies per the dissent-role +
# opinion_type decision.
ROLE_VOCAB = {
    "aym": ("firac_role",
            ["facts", "issue", "rule", "application", "conclusion", "dissent", "unknown"]),
    "bam": ("court_reasoning_role",
            ["facts", "issue", "rule_application", "conclusion", "dissent", "other"]),
    "danistay": ("court_reasoning_role",
                 ["facts", "issue", "rule_application", "conclusion", "dissent", "other"]),
    "first_degree": ("court_reasoning_role",
                     ["facts", "issue", "rule_application", "conclusion", "dissent", "other"]),
    "kvkk": ("regulatory_role", ["background", "analysis", "outcome"]),
    # Yargitay reuses bam's vocabulary unchanged. No cassation-specific role value
    # is needed: the six values already cover every function a cassation decision
    # performs, and `other` absorbs the header blocks and file-routing directives
    # that make up much of a short one.
    "yargitay": ("court_reasoning_role",
                 ["facts", "issue", "rule_application", "conclusion", "dissent", "other"]),
    # Sources 7 and 8. PROVISIONAL -- no rekabet or uyusmazlik decision body has
    # ever been seen (every row is pending_extraction upstream), so these values
    # are inherited from the closest institution rather than observed. Nothing
    # reaches Gemini: chunk_generate.run() refuses any source in
    # TEXT_PENDING_SOURCES before a request is built. They exist so the module is
    # complete and the schema builder can be exercised.
    #
    # rekabet is an authority, not a court, so it takes kvkk's regulatory_role.
    # Likely too thin once text lands -- Rekabet decisions are long, carry karsi
    # oy dissents and a formal operative section -- but widening it now would be
    # inventing values for documents nobody has read.
    "rekabet": ("regulatory_role", ["background", "analysis", "outcome"]),
    # uyusmazlik rules on WHICH court has jurisdiction rather than on the merits,
    # but it is still a court writing a reasoned decision, so it reuses the
    # existing court vocabulary unchanged. Deliberately NOT a new role field:
    # adding one would extend the chunk schema on a guess.
    "uyusmazlik": ("court_reasoning_role",
                   ["facts", "issue", "rule_application", "conclusion", "dissent", "other"]),
}

# When a source has no worked example of its own, borrow the structurally closest
# one. Yargitay reviews an appellate ruling, exactly like bam, and shares its role
# vocabulary -- see build_fewshot's stand-in handling.
# rekabet borrows kvkk's, the only regulatory example that exists. uyusmazlik
# gets NO stand-in on purpose: nothing in fewshot/ resembles a ruling on forum
# rather than merits, and a structurally wrong example teaches a wrong shape more
# firmly than no example at all -- the yargitay few-shot leaked its own case_no
# into a generated document when it was merely the wrong archetype.
FEWSHOT_STAND_IN = {"yargitay": "bam", "rekabet": "kvkk"}

SOURCE_NOTES = {
    "aym": (
        "AYM (Constitutional Court) decisions have REAL court-numbered paragraphs. Use "
        "FIRAC roles. Sections are marked by Roman numerals and headings such as OLAY VE "
        "OLGULAR (facts), ILGILI HUKUK / GENEL ILKELER (rule), ESAS / DEGERLENDIRME "
        "(application), HUKUM (conclusion), KARSIOY (dissent). Scope `rights` PER "
        "PARAGRAPH -- pick only the right(s) that paragraph actually discusses, not every "
        "right the case mentions."
    ),
    "bam": (
        "BAM (regional appellate) decisions use keyword section markers: DAVA, CEVAP, ILK "
        "DERECE MAHKEMESI KARARININ OZETI, ISTINAF NEDENLERI, GEREKCE, HUKUM. There are no "
        "official paragraph numbers -- your segments are synthetic. Rule and Application "
        "are fused in GEREKCE: use `rule_application`."
    ),
    "danistay": (
        "Danistay (Council of State) decisions use markers: ISTEMIN KONUSU, YARGILAMA "
        "SURECI, TEMYIZ EDENIN IDDIALARI, KARSI TARAFIN SAVUNMASI, HUKUKI DEGERLENDIRME, "
        "KARAR SONUCU, and KARSI OY for dissent. HUKUKI DEGERLENDIRME fuses Rule and "
        "Application: use `rule_application`. Note YARGILAMA SURECI often embeds the "
        "entire lower-court ruling and can be very long -- segment it sensibly rather "
        "than emitting one huge block."
    ),
    "first_degree": (
        "First-instance decisions have the thinnest structure: DAVA, then GEREGI "
        "DUSUNULDU, then one long undifferentiated reasoning block, then H U K U M "
        "(letter-spaced). There are no internal markers inside the reasoning block, so "
        "use `confidence: low` for role calls you cannot ground in a marker."
    ),
    # Written around FUNCTION, not section markers, on purpose. Yargitay has ~46
    # chambers and the 48 documents we hold show at least four different layouts;
    # the most common one (23 of 45) has no section labels at all. Listing markers
    # for 46 chambers is a losing game, and the next export will show layouts these
    # 48 do not. Semantics generalise where markers do not -- which is exactly the
    # case docs 15 says this pipeline exists for.
    "yargitay": (
        "Yargitay (Court of Cassation) reviews a lower court's ruling on points of "
        "law; it does not retry the facts. Decisions are SHORT -- typically 6 to 8 "
        "paragraphs.\n\n"
        "THERE IS NO FIXED STRUCTURE. Different chambers write differently, and most "
        "decisions have no section headings at all. Do NOT look for a template. "
        "Identify each paragraph by WHAT IT DOES:\n"
        "  - recounts the charge, the claim, or what the courts below decided -> facts\n"
        "  - states what the appellant argues, or the legal question -> issue\n"
        "  - the court's own reasoning and its application of law -> rule_application\n"
        "  - the operative disposition, almost always the LAST paragraph -> conclusion\n"
        "  - header lines, file-routing directives, anything else -> other\n"
        "Use `confidence: low` whenever the role is your inference rather than "
        "something the document states. That is the honest answer here and it is "
        "expected often.\n\n"
        "Headings, where they appear at all, take many forms: a Roman numeral series "
        "(I. DAVA ... VI. KARAR), a labelled field (SUC :, Suc :, HUKUM :, DAVA TURU :), "
        "or a LETTER-SPACED line such as 'Y A R G I T A Y  K A R A R I' or 'K A R A R'. "
        "Letter-spaced lines ARE headings -- read them with the spaces removed.\n\n"
        "The opening lines are a HEADER BLOCK, not content: chamber and case numbers, "
        "the literal scrape artifact \"Ictihat Metni\" glued to whatever follows it, "
        "MAHKEMESI :, SAYISI :, ILK DERECE MAHKEMESI :. Tag these `other`, "
        "`confidence: low`. A labelled field's VALUE may continue onto the next one or "
        "two paragraphs as bare lines -- include those continuation paragraphs in the "
        "same segment.\n\n"
        "The disposition lives in the final paragraph. Words like KABULUNE or REDDINE "
        "appearing EARLIER in the text usually describe what the LOWER court did, not "
        "what Yargitay decided -- do not read those as this court's outcome. Real "
        "dispositions: ONANMASINA (affirmed), BOZULMASINA (reversed), DUZELTILEREK "
        "ONANMASINA (corrected then affirmed), REDDINE (denied), TEVDIINE (file "
        "remitted), DUSMESINE (abated), GERI CEVRILMESINE (returned on a procedural "
        "defect). A single decision often carries SEVERAL of these at once -- part "
        "reversed, part affirmed. Name the court's principal disposition in `outcome` "
        "and describe the full picture in `conclusion_sentence`.\n\n"
        "Party names are redacted as '...' in the published text. That is the source, "
        "not missing data -- never treat a redaction as a value."
    ),
    "kvkk": (
        "KVKK (data protection authority) decisions have NO section headings and no "
        "paragraph numbers -- a short narrative decision summary. Use the three "
        "regulatory stages only. KVKK cites Yonetmelik (regulations) and Yonerge "
        "(directives) constantly; these have NO law number at all -- set law_no to null "
        "and put the full official name in law_name, with legislation_type 'regulation' "
        "or 'directive'. Capturing these is a specific goal here: the regex extractor "
        "structurally cannot find them."
    ),
    # PLACEHOLDERS -- never sent. chunk_generate.run() refuses any source in
    # TEXT_PENDING_SOURCES before a request is built, because no decision body
    # for either court has ever been extracted. Writing a real source note means
    # describing a layout, and describing a layout nobody has read is how the
    # yargitay few-shot ended up teaching an archetype that fitted 3 of 48
    # documents. These exist so the registry is complete and every source's
    # schema can be built and tested.
    "rekabet": (
        "PLACEHOLDER -- not a usable prompt. Rekabet Kurumu (Competition Authority) "
        "decisions have never been extracted; the bodies are PDFs. Before this is "
        "written, read 2-3 real decisions and establish: whether section headings "
        "exist, whether karsi oy dissents appear and how they are marked, how the "
        "operative part is introduced, and whether the three regulatory stages are "
        "enough or a procedural/dissent value is needed."
    ),
    "uyusmazlik": (
        "PLACEHOLDER -- not a usable prompt. Uyusmazlik Mahkemesi decides WHICH "
        "court has jurisdiction, not who wins, so the usual reasoning roles may map "
        "awkwardly. Before this is written, read 2-3 real decisions and establish: "
        "how the dispute type (olumlu gorev uyusmazligi / olumsuz gorev uyusmazligi "
        "/ hukum uyusmazligi) is stated -- it is the natural subject_id and appears "
        "in NO structured field -- and how the two competing courts are named."
    ),
}

BASE_INSTRUCTION = """You segment Turkish court decisions into retrievable chunks and write one reasoning capsule per legal conclusion.

You will receive a decision as numbered paragraphs, one per line, each prefixed with a marker like [p1], [p2].

## segments[]

Group CONSECUTIVE paragraphs that share the same legal role into one segment.

- `local_id`: "seg_1", "seg_2", ... in document order. These are your own labels for
  linking capsules to segments. Never invent any other kind of id.
- `paragraph_refs`: EVERY marker whose text belongs in this segment, e.g.
  ["p6","p7","p8"]. Use exactly the marker names from the input, without the brackets.
  THIS IS THE MOST IMPORTANT FIELD YOU PRODUCE: the stored chunk text is assembled by
  code from exactly the paragraphs you list here, so a missing ref silently drops real
  content, and a wrong ref pulls in content that does not belong. List them all, in
  order, with no gaps.
- `text`: the merged paragraph text, COPIED VERBATIM from the input. Do not paraphrase,
  summarise, translate, fix typos, change capitalisation, or reorder. Join merged
  paragraphs with a single newline. Do not include the [pN] markers themselves. This is
  compared against the paragraphs you referenced, and any difference is reported -- so
  it is a check on whether your refs and your reading agree, not a place to tidy the
  text up.
- `role`: REQUIRED on every single segment, and it must be one of the values listed
  for this source below. Never omit it and never leave it null. If the section is
  genuinely unclassifiable use the source's catch-all value ("unknown" for aym,
  "other" for the others) with `confidence: low` -- an honest catch-all is useful,
  a missing role makes the chunk unfilterable.
- `content_type`: "ruling" for the operative ruling (conclusion / outcome roles),
  "reasoning" for everything else.
- `reasoning_stage`: "background", "analysis" or "outcome". Universal across sources.
- `confidence`: "high" when the role is clearly grounded in a section marker or
  unmistakable content; "low" when you are unsure. Be honest -- "low" is useful
  signal, a wrong "high" is not.
- `source_type` and `case_no`: copy the values given to you below, exactly as given.
  Turkish court documents are often published with identifying details redacted as
  "..." -- including the case-number lines themselves ("DOSYA NO : ...", "KARAR NO :
  ..."), and also names, notary numbers and plate numbers. A redaction is not a value:
  never write "...", "N/A", "unknown" or any other placeholder into `case_no`. Use the
  number given below even when the document's own header is redacted. The one time to
  write something different is when the document plainly states a DIFFERENT real case
  number (e.g. "ESAS NO : 2016/1359" where a different number was given to you) -- then
  report what the document says, because that disagreement is worth knowing about.

Aim for segments under 2000 characters. Do not pad to reach a size, and do not split
mid-sentence to avoid one -- oversized segments are split automatically afterwards.
Segment on legal role, not on length.

## cited_legislations[] (inside each segment)

Every law, decree-law, constitutional article, regulation or directive cited in THAT
segment's own text. Empty array if none.

- `verbatim_mention`: the exact citation text as written, copied from the segment.
  Also substring-checked.
- `law_no`: the number before "sayili". null for constitution citations and for
  regulations/directives, which are named rather than numbered. Set it ONLY when
  the court states the number in the text.
- `law_short`: the abbreviation exactly as the court wrote it, when it cites a law
  that way -- "T.B.K.", "HMK", "İYUK", "TTK", "KVKK". null if no abbreviation is
  used. NEVER convert an abbreviation into a law number yourself: "BK" meant Law
  818 before 2012 and Law 6098 after, and "TMK" is usually the Civil Code but
  sometimes the Anti-Terror Law. If the court wrote only "T.B.K. madde 56", then
  law_short="T.B.K.", law_no=null, article_no="56".
- `law_name`: the name if stated at that mention, else null.
- `article_no`: as written -- "15", "141/A", "Gecici 3", "Ek 5". null if not stated.
  Keep Gecici/Ek articles distinct from plain articles of the same number.
- `paragraph_no`: the fikra number as a digit, converting Turkish ordinal words
  (ikinci -> "2"). null if absent.
- `law_date`: ISO (1994-12-07) if the text gives a date, else null.
- `legislation_type`: "statute" (Kanun), "decree_law" (KHK / Kanun Hukmunde
  Kararname), "constitution" (Anayasa), "regulation" (Yonetmelik), "directive"
  (Yonerge).
- `confidence`: "high" when a specific article is identified, "low" when only the law
  is named.

Use ONLY those five legislation_type values.

CASE LAW IS NOT LEGISLATION, and this is the single most common mistake made here.
A court citing its own earlier decisions -- "Mehmet Serif Ay (B. No: 2012/1181,
17/9/2013)", "Nihat Akbulak ([GK], B. No: 2015/10131)", "Ibrahim Er ve digerleri",
or a bare party name like "Osman Kizilcan" -- is citing PRECEDENT, not a statute.
The giveaways are a person's name, "B. No:", "[GK]", "E. 2019/123", "K. 2020/456",
a chamber name, or "ve digerleri". None of these belong in cited_legislations. Leave
them out entirely; do not invent a type such as "other" to hold them. References to
academic doctrine are excluded for the same reason.

A citation belongs here only if it names a LAW: a numbered statute or decree-law,
the Constitution, or a named regulation or directive. If you emit a regulation or
directive you MUST give its full official name in law_name, because with no law_no
and no name there is nothing to identify the provision by.

List each distinct provision AT MOST ONCE per segment. If the same article is cited
several times in one segment, emit it once. Never repeat an identical citation object
-- an aym response once emitted the same Anayasa article 561 times and exhausted the
output budget before finishing the document.

Never infer a field the text does not state -- use null. A null means the court did
not say it, never that you failed.

## capsules[]

One per legal subject decided, PLUS one per dissenting opinion.

- `outcome`: a short snake_case label grounded in the ruling text.
- `conclusion_sentence`: one sentence IN TURKISH stating what the court decided and
  why, naming the concrete subject matter -- not a restatement of the label. MUST be
  Turkish: this text is searched by Turkish queries alongside reasoning_summary, so an
  English sentence here is dead weight and is automatically rejected.
- `reasoning_summary`: 2-4 sentences IN TURKISH explaining WHY the court decided this,
  read from the rule and application segments. This is the field a lawyer's question
  gets matched against, so it must carry real factual content: what the dispute was
  about, what test the court applied, what tipped it. MUST be Turkish -- an English
  summary is automatically rejected. Never write a placeholder.
  KEEP THE LEGAL TERM OF ART VERBATIM. Write the specific named concepts the decision
  itself uses -- "sermaye piyasasi mevzuati", "duzenleme ortaklik payi", "vergi ziyai
  cezasi", "kamulastirmasiz el atma", "belirsiz alacak davasi", "adli yardim" -- in the
  court's own words. Do NOT paraphrase them into everyday language. A lawyer searches
  with the term of art, so a summary that says "ortak olmak amaciyla para verdigi"
  instead of naming "sermaye piyasasi mevzuatina aykiri para toplanmasi" cannot be
  found by the very question it answers. This is measured, not hypothetical: two of
  fourteen topic queries missed the top result for exactly this reason, because the
  summary had dropped the term the question was asked with. Summarise the REASONING,
  never the VOCABULARY.
- `opinion_type`: "majority", or "dissent:<judge surname>" for a dissent, or
  "board_decision" for a KVKK board ruling.
- `supporting_local_ids`: the `local_id`s of the segments that actually support this
  conclusion -- typically the rule, application and ruling segments. A dissent capsule
  must reference the dissent segments, not the majority's.
- `case_no`, `subject_id`: echo the values given below.

Do not output chunk_id, canonical_id, char_length, citation_granularity, chunk_label,
subject_type or reasoning_summary_method. Those are set by code and any value you
supply for them is discarded."""


def _short_ref(paragraph_id, counter):
    """Gold files use full ids (uuid-...-p7, facts_p1, seg3). The response format wants
    the short [pN] markers we hand the model, so aym's real paragraph numbers are
    extracted and the other sources are numbered sequentially."""
    m = re.search(r"-p(\d+)$", str(paragraph_id))
    if m:
        return "p" + m.group(1), counter
    counter += 1
    return "p" + str(counter), counter


def build_fewshot(source, echo_source=None, echo_case_no=None):
    """Load a gold document and convert it into the response shape, so the worked
    example is in exactly the format we are asking Gemini to produce.

    A source with no example of its own borrows the structurally closest one
    (FEWSHOT_STAND_IN). When it does, `echo_source`/`echo_case_no` rewrite the two
    echoed fields in the example, because a bam example shown under a yargitay
    instruction says source_type "bam" and a bam case_no on every segment -- which
    directly contradicts the "echo the values given to you below" rule and would
    produce a source_type_mismatch flag on every segment of the first run. The
    example is there to teach structure, not to supply facts.
    """
    path = FEWSHOT_DIR / (source + ".json")
    if not path.is_file():
        stand_in = FEWSHOT_STAND_IN.get(source)
        path = FEWSHOT_DIR / (stand_in + ".json") if stand_in else path
        if not stand_in or not path.is_file():
            return None
        source = stand_in
    gold = json.loads(path.read_text(encoding="utf-8"))
    role_field = ROLE_VOCAB[source][0]
    # Gold files predate law_short; .get() yields null for it, which is the
    # correct value to show since those citations state the law number.
    cite_keys = ("law_no", "law_short", "law_name", "article_no", "paragraph_no",
                 "law_date", "legislation_type", "confidence", "verbatim_mention")

    segments, label_for_chunk, counter = [], {}, 0
    for i, ch in enumerate(gold.get("chunks", []), 1):
        local_id = "seg_" + str(i)
        label_for_chunk[ch.get("chunk_id")] = local_id
        refs = []
        for pid in ch.get("source_paragraph_ids") or []:
            ref, counter = _short_ref(pid, counter)
            refs.append(ref)
        segments.append({
            "local_id": local_id,
            "role": ch.get(role_field),
            "paragraph_refs": refs,
            "text": ch.get("text"),
            "content_type": ch.get("content_type"),
            "reasoning_stage": ch.get("reasoning_stage"),
            "rights": ch.get("rights"),
            "confidence": ch.get("confidence"),
            "source_type": echo_source or ch.get("source_type"),
            "case_no": echo_case_no or ch.get("case_no"),
            "cited_legislations": [
                {k: c.get(k) for k in cite_keys}
                for c in (ch.get("cited_legislations") or [])
            ],
        })

    capsules = []
    for cap in gold.get("reasoning_capsules", []):
        capsules.append({
            "outcome": cap.get("outcome"),
            "conclusion_sentence": cap.get("conclusion_sentence"),
            "reasoning_summary": cap.get("reasoning_summary"),
            "opinion_type": cap.get("opinion_type"),
            "supporting_local_ids": [label_for_chunk[c]
                                     for c in (cap.get("supporting_chunk_ids") or [])
                                     if c in label_for_chunk],
            "case_no": echo_case_no or cap.get("case_no"),
            "subject_id": cap.get("subject_id"),
        })
    return {"segments": segments, "capsules": capsules}


def build_system_instruction(source, case_no, candidate_rights, subject_hint=None):
    role_field, role_values = ROLE_VOCAB[source]
    parts = [BASE_INSTRUCTION, "", "## This document", ""]
    parts.append("source_type: " + source)
    parts.append("case_no: " + str(case_no))
    parts.append("Role field: `" + role_field + "`, one of: " + ", ".join(role_values))
    parts.append("")
    parts.append(SOURCE_NOTES[source])
    parts.append("")

    if source == "aym":
        if candidate_rights:
            parts.append(
                "Candidate rights for this case, taken from the court's own "
                "examination_results metadata. `rights` and `subject_id` must be chosen "
                "from this list ONLY -- never invent a right that is not here. Pick the "
                "one(s) each paragraph actually discusses; use null where a paragraph "
                "discusses none:")
            for r in candidate_rights:
                parts.append("  - " + r)
            parts.append("")
            parts.append("Emit one capsule per right in that list that the court "
                         "actually decided, plus one per dissenting opinion.")
        else:
            parts.append(
                "This record has NO examination_results in its metadata (it is a "
                "norm-review decision, not an individual application). Set `rights` to "
                "null on every segment and `subject_id` to \"unspecified\" on every "
                "capsule.")
    else:
        # "unspecified" used to be offered here as an alternative, and the model
        # took it on 4 of 14 capsules -- including a securities case, a tax-fraud
        # case and a carriage-damage case, all of which plainly state what they
        # are about. subject_id is a retrieval field: "unspecified" makes the
        # capsule unfindable by subject. The escape hatch is removed rather than
        # merely discouraged, and the model is told to infer where no label is
        # printed, because every contested case HAS a subject matter.
        parts.append("Set `rights` to null on every segment -- this source has no "
                     "constitutional-rights concept. Set `subject_id` to a short "
                     "snake_case TURKISH label naming the subject matter in dispute, "
                     "taken from the court's own vocabulary -- for example "
                     "`vergi_ziyai_cezasi`, `kamulastirmasiz_el_atma`, "
                     "`sermaye_piyasasi_mevzuatina_aykirilik`, `tasima_hasari_rucu`. "
                     "Every contested case is ABOUT something, so derive the label "
                     "from the dispute even when the decision prints no subject line. "
                     "Do NOT write \"unspecified\": it is not a subject, it makes the "
                     "decision unfindable by subject, and it is treated as a failure "
                     "to read the document rather than as a property of the document.")
        if subject_hint:
            # The document states its own subject. Same mechanism as aym's
            # candidate rights: code supplies, the model narrows. Offered as a
            # hint rather than a constraint -- the values are free text, and
            # 9. Hukuk uses "DAVA :" for the full prayer for relief rather than
            # a short subject tag.
            kind, value = subject_hint
            parts.append("")
            parts.append(f"This decision states its own subject on a \"{kind}\" line: "
                         f"\"{value}\". Base `subject_id` on that, as a short "
                         f"snake_case Turkish slug, unless the text plainly "
                         f"contradicts it. If it names several subjects, pick the one "
                         f"this decision actually turns on.")

    stand_in = FEWSHOT_STAND_IN.get(source) if not (FEWSHOT_DIR / (source + ".json")).is_file() else None
    # The example's echoed fields are ALWAYS rewritten to this document's values,
    # not only for a borrowed example. Verified necessary: yargitay document
    # 1209647600 (case 2025/18047) came back with case_no 2025/18072 on three
    # segments and its capsule -- the case number of the worked example, copied
    # straight out of it. The instruction says "echo the values given below", so
    # showing the model a DIFFERENT case_no in the example contradicts it and
    # invites exactly that copy. The example teaches structure; the facts must
    # come from the document being read.
    few = build_fewshot(source, echo_source=source, echo_case_no=str(case_no))
    if few:
        parts += ["", "## Worked example", "",
                  ("A correct response for a different document from the " + stand_in +
                   " court, shown because " + source + " has no worked example yet. "
                   "MATCH ITS STRUCTURE, NOT ITS FACTS -- the case it describes is "
                   "unrelated to yours, and its role labels follow the same vocabulary "
                   "you were given above."
                   if stand_in else
                   "A hand-verified correct response for a different " + source +
                   " document. Match this level of detail and this exact structure -- "
                   "note how reasoning_summary carries real facts rather than a label."),
                  "",
                  "ONE EXCEPTION: this example predates the Turkish-language "
                  "requirement and its `conclusion_sentence` is written in English. Do "
                  "NOT copy that. Yours must be in Turkish, as specified above. Follow "
                  "the instructions over the example wherever they disagree.",
                  "", json.dumps(few, ensure_ascii=False, indent=2)]
    return "\n".join(parts)


def build_user_content(paragraphs):
    """Numbered paragraph markers -- the same markers paragraph_refs must use, and the
    same normalized text the grounding check runs against."""
    lines = ["[p" + str(i) + "] " + t for i, t in enumerate(paragraphs, 1)]
    return "Decision text:\n\n" + "\n".join(lines)
