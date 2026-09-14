"""
chunk_lib.py -- proven chunking/extraction logic, COPIED VERBATIM.

PROVENANCE
  Source:  Chunk_test_newdata/scripts/multi_court_chunker.py
  Copied:  2026-09-11
  Reason:  llm_chunk/ must be fully self-contained -- no imports or path
           references outside this folder. These functions are copied rather
           than reimplemented so that bugs already found and fixed once are
           not reintroduced. Each fix is documented in the comments below,
           which were copied along with the code.

DO NOT "clean up" or rewrite anything in this file casually. Every comment
marked FIX records a real bug found by checking real court documents:
  - tr_upper/tr_lower: Python's .upper()/.lower() mangle Turkish i/I/ı/İ.
    "Ihlal".lower() yields a dotted-i + combining dot artifact.
  - Gecici/Ek articles kept as their own numbering so "Gecici Madde 3" never
    collides with plain Article 3.
  - Apostrophe + Turkish ordinal-suffix article numbers (23'uncu).
  - Turkish ordinal WORD paragraph numbers (ikinci -> 2).
  - _ORDINAL_WORD_ALT must stay defined before RE_PARAGRAPH, which is built
    by interpolating it.

Contents: 8 functions + 11 module-level objects (the complete dependency
closure of split_by_size_cap, extract_cited_legislations, tr_lower, tr_upper
and aym_extract_paragraph_texts -- computed mechanically, not by eye).
"""

import re


MAX_CHUNK_CHARS = 2000


# ---------------------------------------------------------------------------
# Size-cap splitting -- proven on AYM, extended here with real fallbacks
# found necessary during verification (KVKK text with no period breaks,
# only semicolons, needed a semicolon-level fallback; a final hard-cut
# fallback guarantees nothing is ever left over the cap).
# ---------------------------------------------------------------------------

def split_by_size_cap(text, cap=MAX_CHUNK_CHARS):
    if len(text) <= cap:
        return [text]

    def hard_split(s, limit):
        if len(s) <= limit:
            return [s]
        clauses = re.split(r'(?<=;)\s+', s)
        if len(clauses) > 1:
            pieces, current = [], ""
            for cl in clauses:
                if current and len(current) + 1 + len(cl) > limit:
                    pieces.append(current.strip())
                    current = cl
                else:
                    current = (current + " " + cl).strip() if current else cl
            if current:
                pieces.append(current.strip())
            out = []
            for p in pieces:
                out.extend(hard_split(p, limit) if len(p) > limit else [p])
            return out
        cut = s.rfind(" ", 0, limit)
        cut = cut if cut > 0 else limit
        return [s[:cut].strip(), *hard_split(s[cut:].strip(), limit)]

    sentences = re.split(r'(?<=[.!?])\s+', text)
    pieces, current = [], ""
    for sent in sentences:
        if current and len(current) + 1 + len(sent) > cap:
            pieces.append(current.strip())
            current = sent
        else:
            current = (current + " " + sent).strip() if current else sent
    if current:
        pieces.append(current.strip())

    final_pieces = []
    for p in pieces:
        final_pieces.extend(hard_split(p, cap) if len(p) > cap else [p])
    return final_pieces


def tr_upper(s):
    """Python's str.upper() does not handle Turkish i/ı correctly (turns
    'idari' into 'IDARI' with an ASCII I, not the Turkish 'İDARİ'), which
    silently broke every outcome-pattern match built with Turkish capital
    İ. Same fix already used for AYM's role detection, applied here too."""
    return s.replace("i", "İ").replace("ı", "I").upper()


def tr_lower(s):
    """Turkish-safe lowercase, same reasoning as tr_upper(): Python's
    str.lower() mishandles İ/I the same way in reverse. Used wherever a
    Turkish word is compared case-insensitively (e.g. "Geçici" vs "geçici"
    vs "GEÇİCİ" in article-number extraction)."""
    return s.replace("İ", "i").replace("I", "ı").lower()


def aym_extract_paragraph_texts(html_content):
    from html import unescape
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(unescape(html_content), "html.parser")
    texts = []
    for p in soup.find_all("p"):
        txt = " ".join(p.get_text(separator=" ").split())
        if txt:
            texts.append(txt)
    return texts


# "3/7/2005 tarihli ve 5393 sayılı Belediye Kanunu'nun ... 15. maddesi"
#
# FIX (legislation_type bug): law type is now a CAPTURING group (was
# non-capturing), so extract_cited_legislations can tell "Kanun" apart from
# "KHK"/"Kanun Hükmünde Kararname" instead of hardcoding "statute" for both.
#
# Article-number capture was split OUT of this regex into _extract_article()
# below -- see the FIX note there for why (Geçici/Ek/141-A/ordinal-suffix
# handling needs several distinct sub-patterns tried in priority order).
RE_STATUTE = re.compile(
    r"(?:(\d{1,2}/\d{1,2}/\d{4})\s+tarihli\s+ve\s+)?"      # optional law date
    r"(\d{3,5})\s+sayılı\s+"                                  # law number
    r"((?:mülga\s+)?[^,;:]{0,120}?)"                          # optional law name
    r"(Kanun\s+Hükmünde\s+Kararname|KHK|Kanun)[^\s]{0,6}",    # law type -- now captured
    re.IGNORECASE
)

# "Anayasa'nın 35. maddesi" / "Anayasa'nin 35. maddesinin (1) numaralı fıkrası"
RE_CONSTITUTION = re.compile(
    r"Anayasa[''`]?n[ıi]n\s+(\d+)\s*\.?\s*madde[^\s]{0,10}"
    r"(?:[^.]{0,30}?\((\d+)\)\s*numaralı\s+fıkra)?",
    re.IGNORECASE
)

# "...maddesinin (1) numaralı fıkrası" / "...maddesinin 1. fıkrası" /
# "...maddenin ikinci fıkrasında" (Turkish ordinal WORD, not a digit)
#
# FIX: added the ordinal-word alternative. Verified missing against real
# text: "344'üncü maddenin ikinci fıkrasında" -- "ikinci" (second) never
# matched the old digit-only / (N)-numaralı forms, so paragraph_no came
# back null despite the paragraph being stated in plain words. Only the
# ordinals actually seen in real citation text are included; extend this
# list if a paragraph number beyond onuncu (10th) is ever found.
_TR_ORDINAL_WORD_TO_DIGIT = {
    "birinci": "1", "ikinci": "2", "üçüncü": "3", "dördüncü": "4",
    "beşinci": "5", "altıncı": "6", "yedinci": "7", "sekizinci": "8",
    "dokuzuncu": "9", "onuncu": "10",
}
_ORDINAL_WORD_ALT = "|".join(_TR_ORDINAL_WORD_TO_DIGIT.keys())
RE_PARAGRAPH = re.compile(
    r"madde[^\s]{0,10}\s*(?:\((\d+)\)\s*numaralı|(\d+)\s*\.?\s*|(" + _ORDINAL_WORD_ALT + r")\s*)\s*fıkra",
    re.IGNORECASE
)


def _clean_law_name(raw):
    if not raw:
        return None
    name = raw.strip().strip("\"'`,;:").strip()
    name = re.sub(r"\s+", " ", name)
    # strip leading connector words that aren't part of the actual name
    name = re.sub(r"^(?:ve|ile|olan|adlı)\s+", "", name, flags=re.IGNORECASE)
    if len(name) < 3:
        return None
    return name or None


def _iso_date(tr_date):
    """'3/7/2005' -> '2005-07-03'. Returns None if unparseable."""
    if not tr_date:
        return None
    try:
        d, m, y = tr_date.split("/")
        return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
    except Exception:
        return None


_ARTICLE_PATTERN_GECICI_EK_WORD_FIRST = re.compile(r"(Geçici|Ek)\s+[Mm]adde\s*(\d+)", re.IGNORECASE)
_ARTICLE_PATTERN_GECICI_EK_NUM_FIRST = re.compile(r"(Geçici|Ek)\s+(\d+)\s*\.?\s*madde", re.IGNORECASE)
_ARTICLE_PATTERN_LETTER_SUFFIX = re.compile(r"(\d+)\s*/\s*([A-ZÇĞİÖŞÜ])\s*\.?\s*madde", re.IGNORECASE)
# FIX: "23'üncü maddesinin", "6'ıncı maddesiyle", "341'inci maddesinde" --
# apostrophe + Turkish ordinal suffix, instead of "23. maddesi". Verified
# missing against real text: the SAME article (2575 art. 23) was cited as
# "23. maddesinin" by a majority opinion and "23'üncü maddesinin" by the
# dissent in the same document -- the old pattern caught only the first
# form. All four Turkish ordinal-suffix vowel-harmony variants included,
# with both straight (') and curly (') apostrophes since real text used
# both across different mentions in the same document.
_ARTICLE_PATTERN_ORDINAL_SUFFIX = re.compile(r"(\d+)['’ʼ]?(?:inci|ıncı|uncu|üncü|nci|ncı)\s+madde", re.IGNORECASE)
# article may be separated from the law name by a quoted section title (e.g.
# 5393 sayılı Belediye Kanunu'nun "Belediyenin yetkileri..." kenar başlıklı
# 15. maddesi), so allow a wide window before the article number
_ARTICLE_PATTERN_PLAIN = re.compile(r"[^.]{0,120}?(\d+)\s*\.?\s*madde", re.IGNORECASE)


def _extract_article(window):
    """Returns (article_no_as_written, match_end_offset) or (None, 0).
    Tries Geçici/Ek forms, the letter-suffix form, and the ordinal-suffix
    form before the plain digit-only form, so e.g. "Geçici 3" is never
    mistaken for plain "3", and "23'üncü" is caught before falling through
    to a much later, wrong "plain" match elsewhere in the window.

    KNOWN NOT HANDLED (found during verification, not yet fixed):
      - law type words beyond Kanun/KHK, e.g. "...Yasası" (a real synonym
        for Kanun) -- RE_STATUTE's law-type group doesn't include it, so
        that citation isn't attempted at all, not just mis-parsed.
      - parenthesized article numbers, e.g. "( 23. ) maddesi" -- the ")"
        breaks _ARTICLE_PATTERN_PLAIN's \\s*\\.?\\s* gap.
      - multiple articles cited in one mention via plural "maddeleri",
        e.g. "114 (2) ve 115. maddeleri" -- only one article is captured.
      - regulations (Yönetmelik) and directives (Yönerge), which KVKK
        decisions cite constantly and which have no law_no at all (named,
        not numbered) -- RE_STATUTE requires Kanun/KHK, so these never
        match. Biggest known gap: verified on one real KVKK document where
        8 of 10 real legislation citations were regulations/directives.
      - citations that rely on a law named in an EARLIER sentence ("Kanun'un
        ... maddesinde", or a bare "359'uncu maddede" with no law restated)
        -- this needs the extractor to track "last law mentioned" as state
        across the whole document, a design change, not a regex fix.
    """
    for pattern in (_ARTICLE_PATTERN_GECICI_EK_WORD_FIRST, _ARTICLE_PATTERN_GECICI_EK_NUM_FIRST):
        m = pattern.search(window)
        if m:
            prefix = "Geçici" if tr_lower(m.group(1)) == "geçici" else "Ek"
            return f"{prefix} {m.group(2)}", m.end()
    m = _ARTICLE_PATTERN_LETTER_SUFFIX.search(window)
    if m:
        return f"{m.group(1)}/{m.group(2).upper()}", m.end()
    m = _ARTICLE_PATTERN_ORDINAL_SUFFIX.search(window)
    if m:
        return m.group(1), m.end()
    m = _ARTICLE_PATTERN_PLAIN.search(window)
    if m:
        return m.group(1), m.end()
    return None, 0


def extract_cited_legislations(text):
    """Returns a list of structured legislation citations found in `text`.
    Every field is either directly grounded in the text or explicitly null.
    Nothing is inferred or guessed."""
    if not text:
        return []
    found = []
    seen = set()

    # --- Constitution citations ---
    for m in RE_CONSTITUTION.finditer(text):
        article = m.group(1)
        paragraph = m.group(2)
        canonical = f"constitution-art-{article}" + (f"-p{paragraph}" if paragraph else "")
        if canonical in seen:
            continue
        seen.add(canonical)
        found.append({
            "canonical_id": canonical,
            "legislation_type": "constitution",
            "law_no": None,
            "law_name": "Türkiye Cumhuriyeti Anayasası",
            "article_no": article,
            "paragraph_no": paragraph,
            "verbatim_mention": m.group(0).strip(),
            "law_date": None,
            "confidence": "high",
        })

    # --- Statute / KHK citations ---
    for m in RE_STATUTE.finditer(text):
        law_date_raw, law_no, law_name_raw, law_type_raw = m.group(1), m.group(2), m.group(3), m.group(4)
        if not law_no:
            continue
        law_name = _clean_law_name(law_name_raw)

        # FIX: legislation_type now reflects what was actually matched
        # (Kanun -> statute, KHK/Kanun Hükmünde Kararname -> decree_law)
        # instead of being hardcoded to "statute" regardless.
        legislation_type = "decree_law" if "khk" in tr_lower(law_type_raw) or "kararname" in tr_lower(law_type_raw) else "statute"

        # article number, searched in the window right after the law-type match
        window = text[m.end():m.end() + 150]
        article, article_end = _extract_article(window)
        match_end = m.end() + article_end if article else m.end()

        # FIX: canonical_id now uses the article number AS WRITTEN (slugified),
        # so "Geçici 3" / "Ek 5" / "141/A" each get their own distinct key
        # instead of colliding with plain "3" / "5" / "141".
        if article:
            article_slug = tr_lower(article).replace(" ", "-").replace("/", "-")
            canonical = f"law-{law_no}-art-{article_slug}"
        else:
            canonical = f"law-{law_no}-unknown-article"

        # look for a paragraph reference in the text right after this match.
        # FIX: previously searched (m.group(0) + " " + tail), which inserts
        # an artificial space that doesn't exist in the real text -- this
        # broke RE_PARAGRAPH whenever the article match ended mid-word (e.g.
        # "...49. madde" + inserted-space + "sinin 1. fıkrası", when the
        # real text reads "...49. maddesinin 1. fıkrası" as one continuous
        # word). Verified null paragraph_no on two separate real citations
        # because of this. Fix: search a real, continuous slice of the
        # original text instead of a reconstructed string.
        paragraph = None
        pm = RE_PARAGRAPH.search(text[m.start():match_end + 80])
        if pm:
            paragraph = pm.group(1) or pm.group(2) or (_TR_ORDINAL_WORD_TO_DIGIT.get(tr_lower(pm.group(3))) if pm.group(3) else None)
        if paragraph:
            canonical += f"-p{paragraph}"

        if canonical in seen:
            continue
        seen.add(canonical)
        found.append({
            "canonical_id": canonical,
            "legislation_type": legislation_type,
            "law_no": law_no,
            "law_name": law_name,                    # null if not stated at this mention
            "article_no": article,                   # null if not stated at this mention; "as written" (e.g. "Geçici 3", "141/A")
            "paragraph_no": paragraph,
            "verbatim_mention": text[m.start():match_end].strip(),
            "law_date": _iso_date(law_date_raw),     # null if not stated at this mention
            # confidence is LOW when the article number wasn't stated, since the
            # citation then identifies a whole law rather than a specific provision
            "confidence": "high" if article else "low",
        })

    return found
