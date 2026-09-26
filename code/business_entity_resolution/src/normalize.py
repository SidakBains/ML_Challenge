"""Text normalization for business names and addresses.

Everything here is rule-based and country-agnostic: the same functions run on
US, India, France (unseen in training) or any other country label. Rules only
canonicalize spelling variants (abbreviations, legal forms, state names,
scripts); no external lookup of any kind is performed.
"""
import re
import unicodedata

from anyascii import anyascii

# ----------------------------------------------------------------------------
# Transliteration. `TRANSLIT` can be filled with a learned {non-latin token ->
# latin token} dictionary (see translit_dict.py); anything not in it falls back
# to anyascii's generic romanization.
# ----------------------------------------------------------------------------
TRANSLIT = {}
_WS = re.compile(r"[ \t]+")
_TOKEN_SPLIT = re.compile(r"(\s+|,)")
_NONLATIN = re.compile(r"[^\x00-ɏḀ-ỿ]")  # beyond Latin / Latin-extended
ASCII_PUNCT = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"


def is_nonlatin(tok):
    return bool(_NONLATIN.search(tok))


def _map_token(p):
    """Replace a non-latin token by its learned latin form (punctuation kept)."""
    core = p.strip(ASCII_PUNCT)
    rep = TRANSLIT.get(core)
    return p.replace(core, " " + rep + " ") if rep else p


def set_translit_dict(d):
    TRANSLIT.clear()
    TRANSLIT.update(d)


def translit(s):
    """Lowercase + romanize. Non-latin words use the learned dictionary first."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    if s.isascii():
        return s.lower()
    if TRANSLIT:
        parts = _TOKEN_SPLIT.split(s)
        s = "".join(_map_token(p) if not p.isascii() else p for p in parts)
    return _WS.sub(" ", anyascii(s).lower())


# ----------------------------------------------------------------------------
# Name normalization
# ----------------------------------------------------------------------------
LEGAL_MAP = {
    "corporation": "corp", "corp": "corp", "incorporated": "inc", "inc": "inc",
    "limited": "ltd", "ltd": "ltd", "private": "pvt", "pvt": "pvt", "pte": "pvt",
    "company": "co", "co": "co", "llc": "llc", "llp": "llp", "lp": "lp",
    "plc": "plc", "pllc": "pllc", "pc": "pc", "ltda": "ltd",
    # French legal forms
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl", "sa": "sa",
    "sci": "sci", "snc": "snc", "scop": "scop", "selarl": "selarl", "cie": "cie",
    "gmbh": "gmbh", "ag": "ag",
}
LEGAL = set(LEGAL_MAP.values())
# Titles / filler that sources prepend or append at random.
NAME_DROP = {"the", "and", "et", "dr", "mr", "mrs", "ms", "smt", "shri", "sri", "shree", "m", "s", "of", "a"}
# "X formerly Y", "X DBA: Y", "X aka Y": the part after the marker is the name.
ALIAS_MARKER = re.compile(r"\b(?:d/?b/?a|doing business as|formerly(?: known as)?|f/?k/?a|a/?k/?a|trading as|t/a)\b[:\s]*")
DOMAIN = re.compile(r"^(?:https?://)?(?:www\.)?([a-z0-9\-]+)\.(?:com|net|org|in|co\.in|co|fr|biz|info|us|io)\b")
ACRONYM_DOTS = re.compile(r"\b(?:[a-z]\.)+[a-z]\b\.?")
NON_ALNUM = re.compile(r"[^a-z0-9]+")
LEADING_JUNK = re.compile(r"^[^a-z0-9]+")


def _collapse_acronyms(s):
    """'l.l.c.' -> 'llc', 's.a.s' -> 'sas'."""
    return ACRONYM_DOTS.sub(lambda m: m.group(0).replace(".", ""), s)


def norm_name(raw):
    """Return (name_full, name_core, name_sq, legal, is_domain).

    name_full: all tokens, legal forms canonicalized.
    name_core: without legal forms / titles (falls back to full if empty).
    name_sq:   name_core with spaces removed (matches web-domain style names).
    legal:     sorted legal-form tokens.
    """
    s = LEADING_JUNK.sub("", translit(raw)).strip()
    is_domain = 0
    m = DOMAIN.match(s.replace(" ", ""))
    if m and " " not in s.strip():
        s = m.group(1).replace("-", " ")
        is_domain = 1
    s = s.replace("&", " and ").replace("+", " plus ")
    m = ALIAS_MARKER.search(s)
    if m and m.end() < len(s):
        s = s[m.end():]
    s = _collapse_acronyms(s)
    toks = NON_ALNUM.sub(" ", s).split()
    full, core, legal = [], [], []
    for t in toks:
        t = LEGAL_MAP.get(t, t)
        if t in LEGAL:
            legal.append(t)
            full.append(t)
        elif t in NAME_DROP:
            continue
        else:
            full.append(t)
            core.append(t)
    if not core:
        core = full or toks
    return " ".join(full), " ".join(core), "".join(core), " ".join(sorted(set(legal))), is_domain


# ----------------------------------------------------------------------------
# Address normalization
# ----------------------------------------------------------------------------
US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv", "new hampshire": "nh",
    "new jersey": "nj", "new mexico": "nm", "new york": "ny", "north carolina": "nc",
    "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa",
    "rhode island": "ri", "south carolina": "sc", "south dakota": "sd", "tennessee": "tn",
    "texas": "tx", "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy", "district of columbia": "dc",
    "puerto rico": "pr",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr",
    "himachal pradesh": "hp", "jharkhand": "jh", "karnataka": "ka", "kerala": "kl",
    "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml",
    "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od", "punjab": "pb",
    "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "telangana": "tg", "tripura": "tr",
    "uttar pradesh": "up", "uttarakhand": "uk", "uttaranchal": "uk", "west bengal": "wb",
    "delhi": "dl", "jammu and kashmir": "jk", "jammu kashmir": "jk", "ladakh": "la",
    "chandigarh": "ch", "puducherry": "py", "pondicherry": "py",
    "dadra and nagar haveli and daman and diu": "dn", "andaman and nicobar islands": "an",
    "lakshadweep": "ld",
}
IN_CODE_ALIASES = {"ts": "tg", "or": "od", "ct": "cg", "ut": "uk", "orissa": "od"}
CITY_ALIASES = {"bangalore": "bengaluru", "bangaluru": "bengaluru", "gurgaon": "gurugram",
                "bombay": "mumbai", "madras": "chennai", "calcutta": "kolkata", "poona": "pune"}
# Street-type / unit words -> one canonical short form (US + India + France).
ADDR_MAP = {
    "street": "st", "str": "st", "saint": "st", "st": "st", "road": "rd", "rd": "rd",
    "avenue": "ave", "ave": "ave", "av": "ave", "drive": "dr", "dr": "dr", "lane": "ln",
    "ln": "ln", "boulevard": "blvd", "blvd": "blvd", "bd": "blvd", "bvd": "blvd",
    "circle": "cir", "cir": "cir", "court": "ct", "ct": "ct", "place": "pl", "pl": "pl",
    "parkway": "pkwy", "pkwy": "pkwy", "highway": "hwy", "hwy": "hwy", "terrace": "ter",
    "square": "sq", "sq": "sq", "trail": "trl", "way": "wy", "suite": "ste", "ste": "ste",
    "sainte": "ste", "apartment": "apt", "apt": "apt", "building": "bldg", "bldg": "bldg",
    "floor": "fl", "flr": "fl", "north": "n", "south": "s", "east": "e", "west": "w",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
    "township": "twp", "twp": "twp", "mount": "mt", "fort": "ft", "point": "pt",
    "number": "no", "nos": "no", "sector": "sec", "sec": "sec", "nagar": "ngr", "ngr": "ngr",
    "marg": "marg", "colony": "col", "near": "nr", "nr": "nr", "opposite": "opp", "opp": "opp",
    # French
    "rue": "rue", "r": "rue", "allee": "all", "all": "all", "impasse": "imp", "imp": "imp",
    "chemin": "ch", "ch": "ch", "route": "rte", "rte": "rte", "faubourg": "fbg", "fbg": "fbg",
    "quai": "qu", "qu": "qu", "cours": "crs", "crs": "crs", "bis": "bis", "ter": "ter",
}
ORDINAL_WORDS = {
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5", "sixth": "6",
    "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10", "eleventh": "11",
    "twelfth": "12", "thirteenth": "13", "fourteenth": "14", "fifteenth": "15",
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
    "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12",
}
ADDR_DROP = {"n/a", "na", "null", "none", "unit", "h", "no", "door", "house", "hno", "of", "the",
             "and", "de", "du", "des", "la", "le", "les", "d", "l", "po", "box"}

_STATE_RE = re.compile(
    r"\b(" + "|".join(sorted(map(re.escape, {**US_STATES, **IN_STATES}), key=len, reverse=True)) + r")\b"
)
_STATES = {**US_STATES, **IN_STATES}
_ORDINAL_NUM = re.compile(r"^0*(\d+)(?:st|nd|rd|th)$")
_HAS_DIGIT = re.compile(r"\d")
_DIGITS = re.compile(r"\d+")


def norm_addr(raw):
    """Return (addr_norm, addr_nums).

    addr_norm: canonical word/number tokens joined by spaces (component order kept).
    addr_nums: space-joined sorted set of number keys: each alnum chunk with its
               separators removed ('C-2-08' -> 'c208') plus each digit run
               without leading zeros ('00427' -> '427').
    """
    s = translit(raw)
    if not s:
        return "", ""
    s = _collapse_acronyms(s)
    s = _STATE_RE.sub(lambda m: _STATES[m.group(1)], s)
    nums = set()
    out = []
    for chunk in re.split(r"[\s,;]+", s):
        if not chunk:
            continue
        if _HAS_DIGIT.search(chunk):
            joined = NON_ALNUM.sub("", chunk)
            m = _ORDINAL_NUM.match(joined)
            if m:
                joined = m.group(1)
            joined = joined.lstrip("0") or "0"
            nums.add(joined)
            for d in _DIGITS.findall(chunk):
                nums.add(d.lstrip("0") or "0")
        for t in NON_ALNUM.sub(" ", chunk).split():
            m = _ORDINAL_NUM.match(t)
            if m:
                t = m.group(1)
            elif t.isdigit():
                t = t.lstrip("0") or "0"
            t = ADDR_MAP.get(t, t)
            t = ORDINAL_WORDS.get(t, t)
            t = CITY_ALIASES.get(t, t)
            t = IN_CODE_ALIASES.get(t, t)
            if t in ADDR_DROP:
                continue
            out.append(t)
    return " ".join(out), " ".join(sorted(nums))


if __name__ == "__main__":
    for n in ["Espinoza & Davis Industrials L.L.C.", "... jonesintegrateddesign.com", "Avikor DBA: Life Investments",
              "Onyxveo formerly Espinoza & Davis Industrials LLC", "Sri MAGPPIE PARK PVT LTD", "#empiremolecular",
              "लाइफ इन्वेस्टमेंट्स", "Thermal & Fils SASU", "The Á Strategic Environmental"]:
        print(repr(n), "->", norm_name(n))
    for a in ["2 Muldowney Circle, Unit Apartment B, Poughkeepsie, NY", "00427 Twelfth St, Errie, Illinois",
              "C-2-08 Jainam Arcade, Mumbai, MH", "B-#20, Regency Plaza, Kalyan, Thane, महाराष्ट्र",
              "63 R. DE DIEPPE, LILLE, Hauts-de-France", "5425 152ND AVENUE, RAMSEY, MN", ""]:
        print(repr(a), "->", norm_addr(a))
