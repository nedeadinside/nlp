"""
Мне кажется я переборщил с этим заданием. По идее оно должно решаться в 15 строк кода, но изначально я понял его
как автоматизированный поиск, а когда дошло, решил не переделывать.

Что нужно сделать. Есть два целевых слова, нам требуется найти такую пару слов, что в результате линейной комбинации, они выдадут вектор,
который в top10 содержит оба целевых слова.

В чем идея. Если задаются пары слов, который будут находиться в топе 10 похожих, в результате линейной комбинации других слов, то
сходство между этими словами должно быть достаточно высоким. Тогда мы можем посчитать норму вектора суммы целевых слов и найти направление в векторном пространстве.
После этого мы получаем все существующие существительные в словаре и также их нормируем.
Теперь основной момент, у нас есть вектор направления, и есть нормированные векторы всех существительных, мы можем посчитать скалярное произведение данных векторов с вектором направления,
что в результате даст косинусное сходство между вектором направления и каждым из существительных.

Теперь мы выбираем топ_k по положительному сходству и топ_k по отрицательному сходству, перебираем пары, считаем most_similar и проверяем, есть ли в top10 целевые слова, если да - возвращаем результат.
(тут стоит сказать, что в данной реализации под линейной комбинацией понимается разность векторов)
"""

from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Any, Optional

import gensim
import numpy as np
from gensim.models import KeyedVectors

NOUN_TAG = "_NOUN"


def load_model(path: str) -> KeyedVectors:
    return gensim.models.KeyedVectors.load_word2vec_format(path, binary=False)


def is_noun_token(token: str) -> bool:
    return token.endswith(NOUN_TAG)


def to_noun_token(token: str) -> str:
    if is_noun_token(token):
        return token
    return f"{token}{NOUN_TAG}"


def ensure_in_vocab(model: KeyedVectors, token: str) -> None:
    if token not in model.key_to_index:
        raise KeyError()


def normalize_noun_inputs(model: KeyedVectors, tokens: Iterable[str]) -> List[str]:
    norm = [to_noun_token(t) for t in tokens]
    for t in norm:
        ensure_in_vocab(model, t)
    return norm


def most_similar_nouns(
    model: KeyedVectors,
    positive: str,
    negative: str,
    topn: int = 10,
) -> List[Tuple[str, float]]:
    raw_topn = topn * 5

    res = model.most_similar(positive=[positive], negative=[negative], topn=raw_topn)
    banned = {positive, negative}

    only_nouns = [(w, s) for (w, s) in res if is_noun_token(w) and w not in banned]
    return only_nouns[:topn]


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + 1e-12)


def _noun_matrix(model: KeyedVectors) -> Tuple[List[str], np.ndarray]:
    keys = []
    vecs = []
    for w in model.index_to_key:
        if is_noun_token(w):
            keys.append(w)
            vecs.append(model.get_vector(w))
    if not vecs:
        return [], np.empty((0, model.vector_size), dtype=np.float32)

    X = np.stack(vecs, axis=0)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return keys, X


def _topk_idx(x: np.ndarray, k: int) -> np.ndarray:
    k = min(k, len(x))
    return np.argsort(-x)[:k]


def search_pairs_by_direction(
    model: KeyedVectors,
    target1: str,
    target2: str,
    k_pos: int = 300,
    k_neg: int = 300,
    topn: int = 10,
) -> Optional[Dict[str, Any]]:

    t1, t2 = normalize_noun_inputs(model, [target1, target2])
    v1 = model.get_vector(t1)
    v2 = model.get_vector(t2)
    q = _unit(v1 + v2)

    keys, X = _noun_matrix(model)

    banned = {t1, t2}

    sim_pos = X @ q
    sim_neg = X @ (-q)

    pos_idx = _topk_idx(sim_pos, k_pos)
    neg_idx = _topk_idx(sim_neg, k_neg)

    pos_words = [keys[i] for i in pos_idx if keys[i] not in banned]
    neg_words = [keys[i] for i in neg_idx if keys[i] not in banned]

    for p in pos_words:
        for n in neg_words:
            if p == n:
                continue
            neighbors = most_similar_nouns(model, positive=p, negative=n, topn=topn)

            rank1 = next((i for i, (w, _) in enumerate(neighbors, 1) if w == t1), None)
            rank2 = next((i for i, (w, _) in enumerate(neighbors, 1) if w == t2), None)

            if rank1 is not None and rank2 is not None:
                marked_neighbors = []
                for word, _ in neighbors:
                    mark = ""
                    if word == t1:
                        mark = "[t1] "
                    elif word == t2:
                        mark = "[t2] "
                    marked_neighbors.append(f"{mark}{word}")

                result = {
                    "positive": p,
                    "negative": n,
                    "neighbors": marked_neighbors,
                }
                return result


if __name__ == "__main__":
    model_path = str(Path().cwd() / "cbow.txt")

    TARGET_1 = "король"
    TARGET_2 = "королева"

    SEARCH_K = 200
    SEARCH_TOPN = 10

    model = load_model(model_path)

    result = search_pairs_by_direction(
        model,
        target1=TARGET_1,
        target2=TARGET_2,
        k_pos=SEARCH_K,
        k_neg=SEARCH_K,
        topn=SEARCH_TOPN,
    )
    if result is not None:
        print(f"\n{result['positive']} - {result['negative']} =>")
        for neighbor in result["neighbors"]:
            print(f"\t{neighbor}")
