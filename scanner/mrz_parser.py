import re
from dataclasses import dataclass, field

@dataclass
class MRZResult:
    cnp: str | None = None
    birth_date: str | None = None
    surname: str | None = None
    given_names: list[str] = field(default_factory=list)
    confidence: float = 0.0

# CNP digit 0: sex + century
_CENTURY = {'1': '19', '2': '19', '3': '18', '4': '18', '5': '20', '6': '20'}

_TRANS = str.maketrans({
    ' ': '<', '\t': '<', '\r': None, '\n': None,
    'Ö': 'O', 'Ü': 'U', 'Á': 'A', 'É': 'E',
    'Î': 'I', 'Â': 'A', 'Ș': 'S', 'Ț': 'T',
})

def _clean(line: str) -> str:
    # Normalize text, replace diacritics and strip invalid characters
    return re.sub(r'[^A-Z0-9<]', '', line.upper().translate(_TRANS))

def _fix_digits(s: str) -> str:
    # Fix common mistakes O->0 and I->1 in CNP positions(15-27)
    chars = list(s.ljust(30, '<')[:30])
    for i in range(15, 28):
        if chars[i] == 'O': chars[i] = '0'
        elif chars[i] == 'I': chars[i] = '1'
    return ''.join(chars)

def _extract_cnp(line1: str) -> str | None:
    # Extracts 13 consecutive digits from MRZ
    m = re.search(r'\d{13}', _fix_digits(line1)[15:])
    return m.group() if m else None

def _birth_date(cnp: str | None) -> str | None:
    if not cnp or len(cnp) != 13 or cnp[0] not in _CENTURY:
        return None
    return f"{_CENTURY[cnp[0]]}{cnp[1:3]}-{cnp[3:5]}-{cnp[5:7]}"

def _has_vowel(s: str) -> bool:
    return any(c in 'AEIOU' for c in s)

def _strip_k(t: str) -> str:
    # Removes trailing OCR-B('K') if there is a vowel inside it
    if len(t) > 2 and t.endswith('K') and _has_vowel(t[:-1]):
        return t[:-1]
    else:
        return t

def _extract_names(line3: str) -> tuple[str, list[str]]:
    line3 = re.sub(r'^ID[A-Z]{3}', '', line3.rstrip('<'))  # Strip merged L1 prefix
    line3 = re.sub(r'<{3,}.*', '', line3)                   # Strip trailing garbage

    parts = line3.split('<<', 1) if '<<' in line3 else line3.split('<', 1)
    surname = parts[0].replace('<', ' ').strip()
    given_names = []

    if len(parts) > 1:
        for name in parts[1].split('<'):
            name = _strip_k(name)
            if name and name.isalpha() and _has_vowel(name) and len(name) > 1:
                given_names.append(name)

    return surname, given_names


def parse_mrz(ocr_text: str) -> MRZResult:
    if not ocr_text.strip():
        return MRZResult()

    cleaned = []
    raw_lines = re.split(r'[\n\r]+', ocr_text)

    for l in raw_lines:
        c = _clean(l)
        if len(c) > 15:
            cleaned.append(c)

    full = ''.join(cleaned)

    match = re.search(r'IDROU[A-Z0-9<]{25}', full)

    if match:
        line1 = match.group()
    elif cleaned:
        line1 = cleaned[0]
    else:
        line1 = ''

    cnp = _extract_cnp(line1)

    l3_candidates = []

    for line in cleaned:
        if '<<' in line:
            letter_count = 0
            number_count = 0
            for char in line:
                if char.isdigit():
                    number_count += 1
                elif char.isalpha():
                    letter_count += 1
            if letter_count > number_count:
                l3_candidates.append(line)

    if l3_candidates:
        line3 = l3_candidates[0]
    elif len(cleaned) ==3:
        line3 = cleaned[-1]
    else:
        line3 = ''


    if line3:
        surname, given_names = _extract_names(line3)
    else:
        surname = ''
        given_names = []


    if cnp and surname:
        confidence = 1.0
    elif cnp:
        confidence = 0.5
    else:
        confidence = 0.0

    return MRZResult(
        cnp=cnp,
        birth_date=_birth_date(cnp),
        surname=surname or None,
        given_names=given_names,
        confidence=confidence,
    )