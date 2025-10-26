import re
import pymorphy3
from pathlib import Path
from typing import List, Tuple


TEXT_PATH = Path("text.txt")
PAIRS_PATH = Path("pairs.txt")

WORD_RE = re.compile(r"[\w-]+", re.UNICODE)
MORPH = pymorphy3.MorphAnalyzer()


def tokenize(text: str) -> List[str]:
    return WORD_RE.findall(text.lower())


def extract_tag(word: str):
    parse = MORPH.parse(word)
    return parse[0] if parse != [] else None


def is_noun_or_adj(p) -> bool:
    # Существительное или прилагательное(полное или краткое) https://pymorphy2.readthedocs.io/en/stable/user/grammemes.html
    return p is not None and (p.tag.POS in {"NOUN", "ADJF", "ADJS"})


def same_gram(p1, p2) -> bool:
    case1, case2 = p1.tag.case, p2.tag.case
    num1, num2 = p1.tag.number, p2.tag.number
    gen1, gen2 = p1.tag.gender, p2.tag.gender

    if not case1 or not case2 or case1 != case2:
        return False
    if not num1 or not num2 or num1 != num2:
        return False
    # Обработка множественного числа
    if num1 == "plur":
        return True
    return bool(gen1 and gen2 and gen1 == gen2)


def is_candidate_pair(p_left, p_right) -> bool:
    if not (is_noun_or_adj(p_left) and is_noun_or_adj(p_right)):
        return False
    return same_gram(p_left, p_right)


def to_lemma(p_left, p_right) -> Tuple[str, str]:
    left_lemma = p_left.normal_form
    right_lemma = p_right.normal_form
    return left_lemma, right_lemma


text = TEXT_PATH.read_text(encoding="utf-8", errors="ignore")
words = tokenize(text)

pairs = set()
for w1, w2 in zip(words, words[1:]):
    p1 = extract_tag(w1)
    p2 = extract_tag(w2)

    if is_candidate_pair(p1, p2):
        pairs.add(to_lemma(p1, p2))

print(f"Количество пар: {len(pairs)}")
PAIRS_PATH.write_text("".join(f"{pair[0]} {pair[1]}\n" for pair in pairs))
