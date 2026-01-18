from __future__ import annotations
from datetime import datetime
from typing import Literal, Optional
from fastapi import FastAPI
from pydantic import BaseModel, Field
from . import nlp


app = FastAPI(
    title="NLP Microservice",
    docs_url=None,
    redoc_url=None,
)


# модели запросов JSON
DetailMode = Literal["full", "partial"]


class CorpusRequest(BaseModel):
    texts: list[str] = Field(..., description="Список документов/строк корпуса")
    detail: DetailMode = Field("full", description="full = полный вывод, partial = обрезанный")
    top_n: int = Field(30, description="В partial режиме: сколько слов оставить (по частоте)")
    include_preprocessed: bool = Field(False, description="Если true — вернуть токены после предобработки")


class TfidfRequest(CorpusRequest):
    l2_normalize: bool = Field(False, description="L2-нормализация TF-IDF по документам")


class Word2VecRequest(CorpusRequest):
    vector_size: int = Field(100, description="Размер вектора")
    window: int = Field(5, description="Окно контекста")
    min_count: int = Field(1, description="Минимальная частота слова")
    sg: int = Field(1, description="1=skip-gram, 0=CBOW")
    epochs: int = Field(10, description="Эпохи обучения")
    workers: int = Field(1, description="Потоки (для простоты оставь 1)")


class TextRequest(BaseModel):
    text: str


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "service": "nlp-fastapi",
        "time": datetime.now().isoformat(timespec="seconds"),
        "endpoints": [
            "POST /bag-of-words",
            "POST /tf-idf",
            "POST /word2vec",
            "POST /text_nltk/tokenize",
            "POST /text_nltk/stem",
            "POST /text_nltk/lemmatize",
            "POST /text_nltk/pos",
            "POST /text_nltk/ner",
        ],
    }


@app.post("/bag-of-words")
def bag_of_words(req: CorpusRequest):
    docs_tokens = [nlp.preprocess_text(t, remove_stopwords=True) for t in req.texts]

    vocab_list, vocab_index, doc_freq = nlp.build_vocab(docs_tokens, min_df=1)
    bow = nlp.bow_matrix(docs_tokens, vocab_index)

    if req.detail == "partial":
        vocab_list, bow = nlp.select_top_terms_by_frequency(vocab_list, bow, req.top_n)

    response = {
        "vocab": vocab_list,
        "shape": [int(bow.shape[0]), int(bow.shape[1])],
        "matrix": bow.tolist(),
    }

    if req.include_preprocessed:
        response["preprocessed"] = docs_tokens

    return response


@app.post("/tf-idf")
def tf_idf(req: TfidfRequest):
    docs_tokens = [nlp.preprocess_text(t, remove_stopwords=True) for t in req.texts]

    vocab_list, vocab_index, doc_freq = nlp.build_vocab(docs_tokens, min_df=1)
    bow = nlp.bow_matrix(docs_tokens, vocab_index)

    if req.detail == "partial":
        vocab_list, bow = nlp.select_top_terms_by_frequency(vocab_list, bow, req.top_n)
        doc_freq = (bow > 0).sum(axis=0).astype("int32")  # doc_freq под новый vocab

    tfidf, idf = nlp.tfidf_matrix_from_bow(bow, doc_freq, smooth=True, l2_normalize=req.l2_normalize)

    response = {
        "vocab": vocab_list,
        "shape": [int(tfidf.shape[0]), int(tfidf.shape[1])],
        "idf": [float(x) for x in idf.tolist()],
        "matrix": tfidf.round(6).tolist(),  # округляем
    }

    if req.include_preprocessed:
        response["preprocessed"] = docs_tokens

    return response


@app.post("/word2vec")
def word2vec(req: Word2VecRequest):
    docs_tokens = [nlp.preprocess_text(t, remove_stopwords=True) for t in req.texts]

    params = nlp.Word2VecParams(
        vector_size=req.vector_size,
        window=req.window,
        min_count=req.min_count,
        sg=req.sg,
        workers=req.workers,
        epochs=req.epochs,
    )
    model = nlp.train_word2vec(docs_tokens, params)

    freq = {}  # расчет частот для partial
    for doc in docs_tokens:
        for w in doc:
            freq[w] = freq.get(w, 0) + 1

    if req.detail == "partial":
        top_words = sorted(freq.items(), key=lambda x: -x[1])[: max(req.top_n, 1)]
        words = [w for w, _ in top_words if w in model.wv]
    else:
        words = None

    vectors = nlp.word_vectors_dict(model, words=words)

    response = {
        "params": {
            "vector_size": req.vector_size,
            "window": req.window,
            "min_count": req.min_count,
            "sg": req.sg,
            "epochs": req.epochs,
        },
        "vocab_size": len(model.wv.index_to_key),
        "returned_words": len(vectors),
        "vectors": vectors,
    }

    if req.include_preprocessed:
        response["preprocessed"] = docs_tokens

    return response


@app.post("/text_nltk/tokenize")
def text_tokenize(req: TextRequest):
    tokens = nlp.preprocess_text(req.text, remove_stopwords=True)
    return {"tokens": tokens}


@app.post("/text_nltk/stem")
def text_stem(req: TextRequest):
    tokens = nlp.preprocess_text(req.text, remove_stopwords=True)
    stems = nlp.stem_tokens(tokens)
    return {"tokens": tokens, "stems": stems}


@app.post("/text_nltk/lemmatize")
def text_lemmatize(req: TextRequest):
    tokens = nlp.preprocess_text(req.text, remove_stopwords=True)
    lemmas = nlp.lemmatize_tokens(tokens)
    return {"tokens": tokens, "lemmas": lemmas}


@app.post("/text_nltk/pos")
def text_pos(req: TextRequest):
    tokens = nlp.preprocess_text(req.text, remove_stopwords=True)
    tags = nlp.pos_tag_tokens(tokens)
    return {"tokens": tokens, "pos": tags}


@app.post("/text_nltk/ner")
def text_ner(req: TextRequest):
    entities = nlp.ner_heuristic(req.text)
    return {
        "method": "rule_based_heuristic",
        "entities": entities,
        "note": "Это эвристический NER без обученных моделей (для русского без внешних NLP-пакетов).",
    }
