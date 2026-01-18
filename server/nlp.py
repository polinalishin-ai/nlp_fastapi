from __future__ import annotations
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable
import numpy as np
import nltk
from nltk.stem.snowball import SnowballStemmer
import pymorphy2
from gensim.models import Word2Vec


_FALLBACK_RU_STOPWORDS = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то",
    "все", "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за",
    "бы", "по", "ее", "мне", "есть", "они", "тут", "где", "когда", "или",
    "ни", "быть", "был", "были", "была", "это", "от", "для", "из", "мы",
    "вас", "нас", "при", "над", "под", "до", "после"
}


def load_ru_stopwords() -> set[str]:
    """
    Пытаемся взять nltk.corpus.stopwords (русские).
    Если ресурса нет, пробуем скачать. Если не получилось — используем fallback.
    """
    try:
        from nltk.corpus import stopwords
        try:
            words = set(stopwords.words("russian"))
            if words:
                return words
        except LookupError:
            nltk.download("stopwords", quiet=True)
            words = set(stopwords.words("russian"))
            if words:
                return words
    except Exception:
        pass
    return set(_FALLBACK_RU_STOPWORDS)


RU_STOPWORDS = load_ru_stopwords()
_RU_WORD_RE = re.compile(r"[а-яё]+", re.IGNORECASE)
_RAW_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё]+")


def preprocess_text(text: str, *, remove_stopwords: bool = True) -> list[str]:
    """
    Возвращает список токенов (только слова), в нижнем регистре,
    без пунктуации/цифр и (опционально) без стоп-слов.
    """
    text = text.lower()
    tokens = _RU_WORD_RE.findall(text)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in RU_STOPWORDS]
    return tokens


def tokenize_raw_keep_case(text: str) -> list[str]:
    """
    Токенизация без lower: нужна для простого NER-эвристического анализа.
    """
    return _RAW_WORD_RE.findall(text)


def build_vocab(docs_tokens: list[list[str]], *, min_df: int = 1) -> tuple[list[str], dict[str, int], np.ndarray]:
    """
    Строим словарь (vocab) по документам.
    min_df — минимальная документная частота (сколько документов должно содержать слово).
    Возвращаем:
      vocab_list: список слов по индексам
      vocab_index: слово -> индекс
      doc_freq: массив df по vocab_list
    """
    n_docs = len(docs_tokens)
    df_counter: Counter[str] = Counter()

    for tokens in docs_tokens:
        df_counter.update(set(tokens))

    vocab_list = [w for w, df in df_counter.items() if df >= min_df]
    vocab_list.sort()

    vocab_index = {w: i for i, w in enumerate(vocab_list)}
    doc_freq = np.array([df_counter[w] for w in vocab_list], dtype=np.int32)

    return vocab_list, vocab_index, doc_freq


def bow_matrix(docs_tokens: list[list[str]], vocab_index: dict[str, int]) -> np.ndarray:
    """
    Count BoW: матрица [n_docs, vocab_size] с int счетчиками.
    """
    n_docs = len(docs_tokens)
    vocab_size = len(vocab_index)
    mat = np.zeros((n_docs, vocab_size), dtype=np.int32)

    for di, tokens in enumerate(docs_tokens):
        for t in tokens:
            idx = vocab_index.get(t)
            if idx is not None:
                mat[di, idx] += 1
    return mat


def tfidf_matrix_from_bow(
    bow: np.ndarray,
    doc_freq: np.ndarray,
    *,
    smooth: bool = True,
    l2_normalize: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Делает TF-IDF из BoW и df.
    TF: count / total_tokens_in_doc (если документ пустой => все нули)
    IDF: log((N + 1)/(df + 1)) + 1  (сглаживание)
    """
    n_docs = bow.shape[0]
    row_sums = bow.sum(axis=1, keepdims=True).astype(np.float32)
    tf = np.divide(bow.astype(np.float32), row_sums, out=np.zeros_like(bow, dtype=np.float32), where=row_sums != 0)

    df = doc_freq.astype(np.float32)
    if smooth:
        idf = np.log((n_docs + 1.0) / (df + 1.0)) + 1.0
    else:
        idf = np.log(n_docs / np.maximum(df, 1.0))

    tfidf = tf * idf[np.newaxis, :]

    if l2_normalize:
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        tfidf = np.divide(tfidf, norms, out=np.zeros_like(tfidf), where=norms != 0)

    return tfidf, idf


def select_top_terms_by_frequency(vocab_list: list[str], bow: np.ndarray, top_n: int) -> tuple[list[str], np.ndarray]:
    """
    Берем top_n терминов по суммарной частоте в корпусе.
    Возвращаем (new_vocab_list, new_bow_submatrix).
    """
    if top_n <= 0 or top_n >= len(vocab_list):
        return vocab_list, bow

    term_freq = bow.sum(axis=0)
    top_idx = np.argsort(-term_freq)[:top_n]
    top_idx_sorted = np.sort(top_idx)

    new_vocab = [vocab_list[i] for i in top_idx_sorted]
    new_bow = bow[:, top_idx_sorted]
    return new_vocab, new_bow


_STEMMER = SnowballStemmer("russian")
_MORPH = pymorphy2.MorphAnalyzer()


def stem_tokens(tokens: list[str]) -> list[str]:
    return [_STEMMER.stem(t) for t in tokens]


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    return [_MORPH.parse(t)[0].normal_form for t in tokens]


def pos_tag_tokens(tokens: list[str]) -> list[dict]:
    """
    POS на pymorphy2: возвращаем список объектов вида
    { "token": "...", "lemma": "...", "pos": "...", "tag": "..." }
    """
    out = []
    for t in tokens:
        p = _MORPH.parse(t)[0]
        out.append(
            {
                "token": t,
                "lemma": p.normal_form,
                "pos": str(p.tag.POS) if p.tag.POS else None,
                "tag": str(p.tag),
            }
        )
    return out


_ORG_HINTS = {"ооо", "ао", "зао", "оао", "университет", "банк", "министерство", "компания", "группа"}
_LOC_HINTS = {"город", "г", "область", "край", "республика", "район"}


def ner_heuristic(text: str) -> list[dict]:
    """
    Очень простой "NER" по заглавным буквам + ключевым словам.
    Возвращает список сущностей: {"text": "...", "label": "...", "start": int, "end": int}
    start/end — позиции в строке (если нашли через re.search; упрощенно).
    """
    tokens = tokenize_raw_keep_case(text)
    if not tokens:
        return []

    positions: list[tuple[str, int, int]] = []
    cursor = 0
    for tok in tokens:
        m = re.search(re.escape(tok), text[cursor:])
        if not m:
            continue
        start = cursor + m.start()
        end = cursor + m.end()
        positions.append((tok, start, end))
        cursor = end

    entities: list[dict] = []
    i = 0
    while i < len(positions):
        tok, s, e = positions[i]
        if tok[:1].isupper() and len(tok) > 1:
            j = i
            parts = []
            start = positions[i][1]
            end = positions[i][2]
            while j < len(positions) and len(parts) < 3:
                t2, s2, e2 = positions[j]
                if t2[:1].isupper() and len(t2) > 1:
                    parts.append(t2)
                    end = e2
                    j += 1
                else:
                    break

            ent_text = " ".join(parts)
            ent_lower = ent_text.lower()

            label = "MISC"
            if any(h in ent_lower for h in _ORG_HINTS):
                label = "ORG"
            elif any(h in ent_lower for h in _LOC_HINTS):
                label = "LOC"
            else:
                if len(parts) >= 2:
                    label = "PERSON"

            entities.append({"text": ent_text, "label": label, "start": start, "end": end})
            i = j
        else:
            i += 1

    return entities


@dataclass
class Word2VecParams:
    vector_size: int = 100
    window: int = 5
    min_count: int = 1
    sg: int = 1
    workers: int = 1
    epochs: int = 10


def train_word2vec(docs_tokens: list[list[str]], params: Word2VecParams) -> Word2Vec:
    """
    Тренируем Word2Vec по токенизированным документам.
    """
    model = Word2Vec(
        sentences=docs_tokens,
        vector_size=params.vector_size,
        window=params.window,
        min_count=params.min_count,
        sg=params.sg,
        workers=params.workers,
    )
    model.train(docs_tokens, total_examples=len(docs_tokens), epochs=params.epochs)
    return model


def word_vectors_dict(model: Word2Vec, words: Iterable[str] | None = None) -> dict[str, list[float]]:
    """
    Возвращаем dict: слово -> вектор (list[float]).
    Если words=None, берем все слова из словаря модели.
    """
    if words is None:
        words = model.wv.index_to_key

    out: dict[str, list[float]] = {}
    for w in words:
        if w in model.wv:
            out[w] = model.wv[w].astype(float).tolist()
    return out
