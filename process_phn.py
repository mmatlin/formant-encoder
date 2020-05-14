from typing import List, Tuple

# Dict converting TIMIT monophthong codes to IPA
# Reference: https://en.wikipedia.org/wiki/ARPABET
phn_vowel_ipa = {
    "iy": "i",
    "ih": "ɪ",
    "eh": "ɛ",
    "ae": "æ",
    "aa": "ɑ",
    "ah": "ʌ",
    "ao": "ɔ",
    "uh": "ʊ",
    "uw": "u",
    "ux": "ʉ",
    "er": "ɝ",
    "ax": "ə",
    "ix": "ɨ",
    "axr": "ɚ"
}

def extract_monophthong_times(phn_path: str) -> List[Tuple[str, int, int]]:
    out = []
    with open(phn_path) as phn:
        lines = phn.readlines()
        for line in lines:
            vals = line.split()
            if vals[-1] in phn_vowel_ipa.keys():
                new = [phn_vowel_ipa[vals[-1]]]
                new.extend([int(vals[0]), int(vals[1])])
                out.append(tuple(new))
    return out

def get_num_monophthongs():
    return len(phn_vowel_ipa)