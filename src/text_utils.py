#Serbian Cyrillic -> Latin transliteration.

#Three Cyrillic letters map to two-character Latin forms (digraphs).
_DOUBLES = {
    "Љ": "Lj", "љ": "lj",
    "Њ": "Nj", "њ": "nj",
    "Џ": "Dž", "џ": "dž",
}

#Everything else is one-to-one.
_SINGLES = {
    "А": "A", "Б": "B", "В": "V", "Г": "G", "Д": "D", "Ђ": "Đ", "Е": "E",
    "Ж": "Ž", "З": "Z", "И": "I", "Ј": "J", "К": "K", "Л": "L", "М": "M",
    "Н": "N", "О": "O", "П": "P", "Р": "R", "С": "S", "Т": "T", "Ћ": "Ć",
    "У": "U", "Ф": "F", "Х": "H", "Ц": "C", "Ч": "Č", "Ш": "Š",
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "ђ": "đ", "е": "e",
    "ж": "ž", "з": "z", "и": "i", "ј": "j", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "ћ": "ć",
    "у": "u", "ф": "f", "х": "h", "ц": "c", "ч": "č", "ш": "š",
}


def transliterate(text: str) -> str:
    #Digraphs first
    for cyr, lat in _DOUBLES.items():
        text = text.replace(cyr, lat)
    #Then single-char substitution
    return "".join(_SINGLES.get(ch, ch) for ch in text)


def is_cyrillic(text: str) -> bool:
    return transliterate(text) != text
