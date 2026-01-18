from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List
import requests


SESSION = requests.Session()
SESSION.trust_env = False


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def split_corpus(raw: str, mode: str) -> List[str]:
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")

    if mode == "lines":
        return [line.strip() for line in raw.split("\n") if line.strip()]

    parts = [p.strip() for p in raw.split("\n\n") if p.strip()]
    return [" ".join(p.split()) for p in parts]


def http_get(base_url: str, endpoint: str) -> Dict[str, Any]:
    url = base_url.rstrip("/") + endpoint
    r = SESSION.get(url, timeout=30)
    _raise_for_status(r)
    return r.json()


def http_post(base_url: str, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = base_url.rstrip("/") + endpoint
    r = SESSION.post(url, json=payload, timeout=120)
    _raise_for_status(r)
    return r.json()


def _raise_for_status(r: requests.Response) -> None:
    if 200 <= r.status_code < 300:
        return
    try:
        data = r.json()
        raise RuntimeError(f"HTTP {r.status_code}: {json.dumps(data, ensure_ascii=False)}")
    except Exception:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text}")


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def print_full(data: Dict[str, Any]) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def print_summary(endpoint: str, data: Dict[str, Any]) -> None:
    print(f"== Ответ от {endpoint} ==")

    if endpoint in ("/bag-of-words", "/tf-idf"):
        vocab = data.get("vocab", [])
        shape = data.get("shape", None)
        matrix = data.get("matrix", [])

        print(f"Размер словаря: {len(vocab)}")
        print(f"Shape: {shape}")
        if vocab:
            print("Первые 15 терминов vocab:", vocab[:15])

        if matrix:
            rows_show = min(2, len(matrix))
            cols_show = min(10, len(matrix[0]) if matrix else 0)
            print(f"Фрагмент matrix: первые {rows_show} строк(и), первые {cols_show} столбцов:")
            for i in range(rows_show):
                print(matrix[i][:cols_show])

        if endpoint == "/tf-idf":
            idf = data.get("idf", [])
            if idf:
                print("Первые 10 значений idf:", idf[:10])

        if "preprocessed" in data:
            print("Preprocessed[0..1]:", data["preprocessed"][:2])
        return

    if endpoint == "/word2vec":
        params = data.get("params", {})
        vocab_size = data.get("vocab_size", None)
        returned_words = data.get("returned_words", None)
        vectors = data.get("vectors", {})

        print("Params:", params)
        print("vocab_size (в модели):", vocab_size)
        print("returned_words (в ответе):", returned_words)

        items = list(vectors.items())
        show_n = min(5, len(items))
        if show_n > 0:
            print(f"Первые {show_n} слов(а) и первые 8 компонент вектора:")
            for w, vec in items[:show_n]:
                print(w, "->", vec[:8])

        if "preprocessed" in data:
            print("Preprocessed[0..1]:", data["preprocessed"][:2])
        return

    if endpoint.startswith("/text_nltk/"):
        print("Ключи ответа:", list(data.keys()))
        if "tokens" in data:
            print("tokens:", data["tokens"][:40])
        if "stems" in data:
            print("stems:", data["stems"][:40])
        if "lemmas" in data:
            print("lemmas:", data["lemmas"][:40])
        if "pos" in data:
            print("pos[0..5]:", data["pos"][:5])
        if "entities" in data:
            print("entities:", data["entities"])
        return

    if endpoint == "/":
        print("status:", data.get("status"))
        print("service:", data.get("service"))
        print("time:", data.get("time"))
        print("endpoints:", data.get("endpoints"))
        return

    print_full(data)


def make_corpus_payload(texts: List[str], detail: str, top_n: int, include_preprocessed: bool) -> Dict[str, Any]:
    return {
        "texts": texts,
        "detail": detail,
        "top_n": top_n,
        "include_preprocessed": include_preprocessed,
    }


def make_tfidf_payload(
    texts: List[str],
    detail: str,
    top_n: int,
    include_preprocessed: bool,
    l2_normalize: bool,
) -> Dict[str, Any]:
    payload = make_corpus_payload(texts, detail, top_n, include_preprocessed)
    payload["l2_normalize"] = l2_normalize
    return payload


def make_word2vec_payload(
    texts: List[str],
    detail: str,
    top_n: int,
    include_preprocessed: bool,
    vector_size: int,
    window: int,
    min_count: int,
    sg: int,
    epochs: int,
    workers: int,
) -> Dict[str, Any]:
    payload = make_corpus_payload(texts, detail, top_n, include_preprocessed)
    payload.update(
        {
            "vector_size": vector_size,
            "window": window,
            "min_count": min_count,
            "sg": sg,
            "epochs": epochs,
            "workers": workers,
        }
    )
    return payload


def make_text_payload(text: str) -> Dict[str, Any]:
    return {"text": text}


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--server", default="http://127.0.0.1:8000", help="Базовый URL сервера")
    parser.add_argument(
        "--print",
        dest="print_mode",
        choices=["summary", "full"],
        default="summary",
        help="Как печатать ответ: summary (кратко) или full (полный JSON)",
    )
    parser.add_argument("--save", type=str, default=None, help="Сохранить полный ответ в JSON-файл")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="client.py",
        description="Клиент для NLP FastAPI сервера (BoW/TF-IDF/Word2Vec + text_nltk/*).",
    )

    add_common_args(p)

    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("health", help="GET / (health check)")
    add_common_args(sp)

    for name in ("bow", "tfidf", "word2vec"):
        sp = sub.add_parser(name, help=f"Запрос к /{_cmd_to_endpoint(name).lstrip('/')}")
        add_common_args(sp)

        sp.add_argument("--corpus-file", default=str(Path(__file__).with_name("text.txt")),
                        help="Путь к файлу корпуса (utf-8)")
        sp.add_argument("--split", choices=["paragraphs", "lines"], default="paragraphs",
                        help="Как делить corpus-file на документы")
        sp.add_argument("--detail", choices=["full", "partial"], default="full",
                        help="Какие данные запросить у сервера: full или partial")
        sp.add_argument("--top-n", type=int, default=30, help="Для detail=partial: сколько терминов/слов вернуть")
        sp.add_argument("--include-preprocessed", action="store_true",
                        help="Вернуть токены после предобработки (может раздувать JSON)")

        if name == "tfidf":
            sp.add_argument("--l2-normalize", action="store_true", help="Включить L2-нормализацию TF-IDF")
        if name == "word2vec":
            sp.add_argument("--vector-size", type=int, default=100)
            sp.add_argument("--window", type=int, default=5)
            sp.add_argument("--min-count", type=int, default=1)
            sp.add_argument("--sg", type=int, choices=[0, 1], default=1, help="1=skip-gram, 0=CBOW")
            sp.add_argument("--epochs", type=int, default=10)
            sp.add_argument("--workers", type=int, default=1)

    for name in ("tokenize", "stem", "lemmatize", "pos", "ner"):
        sp = sub.add_parser(name, help=f"Запрос к /text_nltk/{name}")
        add_common_args(sp)

        sp.add_argument("--text", type=str, default=None, help="Текст прямо в аргументе")
        sp.add_argument("--text-file", type=str, default=None, help="Файл с текстом (utf-8)")

    return p


def _cmd_to_endpoint(cmd: str) -> str:
    if cmd == "bow":
        return "/bag-of-words"
    if cmd == "tfidf":
        return "/tf-idf"
    if cmd == "word2vec":
        return "/word2vec"
    if cmd in ("tokenize", "stem", "lemmatize", "pos", "ner"):
        return f"/text_nltk/{cmd}"
    if cmd == "health":
        return "/"
    raise ValueError(f"Unknown cmd: {cmd}")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    base_url: str = args.server
    endpoint = _cmd_to_endpoint(args.cmd)

    try:
        if args.cmd == "health":
            data = http_get(base_url, endpoint)

        elif args.cmd in ("bow", "tfidf", "word2vec"):
            raw = read_text_file(Path(args.corpus_file))
            texts = split_corpus(raw, args.split)
            if not texts:
                raise RuntimeError("Корпус пустой: проверь файл и режим --split")

            if args.cmd == "bow":
                payload = make_corpus_payload(texts, args.detail, args.top_n, args.include_preprocessed)
                data = http_post(base_url, endpoint, payload)

            elif args.cmd == "tfidf":
                payload = make_tfidf_payload(texts, args.detail, args.top_n, args.include_preprocessed, args.l2_normalize)
                data = http_post(base_url, endpoint, payload)

            else:
                payload = make_word2vec_payload(
                    texts,
                    args.detail,
                    args.top_n,
                    args.include_preprocessed,
                    args.vector_size,
                    args.window,
                    args.min_count,
                    args.sg,
                    args.epochs,
                    args.workers,
                )
                data = http_post(base_url, endpoint, payload)

        else:
            text = None
            if args.text is not None:
                text = args.text
            elif args.text_file is not None:
                text = read_text_file(Path(args.text_file))
            else:
                raise RuntimeError("Нужен либо --text, либо --text-file")

            payload = make_text_payload(text)
            data = http_post(base_url, endpoint, payload)

        if args.print_mode == "full":
            print_full(data)
        else:
            print_summary(endpoint, data)

        if args.save:
            out_path = Path(args.save)
            save_json(data, out_path)
            print(f"\nСохранено в: {out_path.resolve()}")

        return 0

    except requests.RequestException as e:
        print(f"Ошибка сети/соединения: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
