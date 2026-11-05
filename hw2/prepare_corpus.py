"""Build the HW2 Wikipedia corpus and match complete HW1 evidence quotes."""
import argparse
import hashlib
import json
import re
import time
import unicodedata
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urlencode, urlsplit
from urllib.request import Request, urlopen

from freshqa.dataset import load_rows


DATA = Path(__file__).resolve().parent / "data"
ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "data" / "fresh.jsonl"


def normalize_evidence(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"\w+", text))


def fetch_article(url):
    title = unquote(urlsplit(url).path.removeprefix("/wiki/")).replace("_", " ")
    query = urlencode({
        "action": "query", "prop": "extracts", "explaintext": 1,
        "titles": title, "redirects": 1, "format": "json",
    }, quote_via=quote)
    request = Request("https://en.wikipedia.org/w/api.php?" + query,
                      headers={"User-Agent": "agents-course-seminar02/1.0 (educational corpus)"})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=30) as response:
                pages = json.load(response)["query"]["pages"]
            page = next(iter(pages.values()))
            if not page.get("extract", "").strip():
                raise ValueError("пустой текст")
            return {"page": page["title"], "url": url, "text": page["extract"]}
        except (URLError, TimeoutError, ValueError, KeyError, StopIteration) as error:
            if attempt == 2:
                raise RuntimeError(f"Не удалось загрузить {url}: {error}") from error
            delay = 2 ** attempt
            if isinstance(error, HTTPError) and error.code == 429:
                retry = error.headers.get("Retry-After", "")
                delay = max(delay, int(retry) if retry.isdigit() else 60)
            time.sleep(delay)


def write_rows(path, rows):
    handle = NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False)
    pending = Path(handle.name)
    try:
        with handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


def download_corpus(urls, cache):
    # Keep finished articles when a later request fails or is interrupted.
    cache.mkdir(parents=True, exist_ok=True)
    corpus = []
    for index, url in enumerate(urls, start=1):
        path = cache / (hashlib.sha256(url.encode()).hexdigest() + ".jsonl")
        if path.exists():
            article, = load_rows(path)
        else:
            article = fetch_article(url)
            write_rows(path, [article])
            time.sleep(1)
        print(f"[{index}/{len(urls)}] {article['page']}", flush=True)
        corpus.append(article)
    return corpus


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true",
                        help="заново скачать статьи и пересобрать выборку")
    args = parser.parse_args()
    rows = load_rows(QUESTIONS)
    candidates = [r for r in rows if urlsplit(r["url"]).hostname == "en.wikipedia.org"
                  and urlsplit(r["url"]).path.startswith("/wiki/")]
    urls = sorted({r["url"] for r in candidates})
    corpus_path = DATA / "corpus.jsonl"
    rebuilding = args.refresh or not corpus_path.exists()
    cache = ROOT / "tmp" / "hw2-download"
    if rebuilding:
        corpus = download_corpus(urls, cache)
    else:
        corpus = load_rows(corpus_path)
    texts = {r["url"]: " " + normalize_evidence(r["text"]) + " " for r in corpus}
    if set(texts) != set(urls) or len(texts) != len(corpus):
        parser.exit(1, "Снимок и источники разошлись. Обновите снимок с --refresh.\n")
    matched, missing = [], []
    fields = ("id", "source", "question", "answer", "answer_aliases", "evidence", "page", "url", "event_date")
    for row in candidates:
        evidence = normalize_evidence(row["evidence"])
        if evidence and " " + evidence + " " in texts[row["url"]]:
            matched.append({k: row[k] for k in fields if k in row})
        else:
            missing.append(row["id"])
    if len(matched) < 30:
        parser.exit(1, f"Только {len(matched)} вопросов с полной цитатой. Проверьте источники.\n")
    DATA.mkdir(exist_ok=True)
    if rebuilding:
        write_rows(corpus_path, corpus)
    write_rows(DATA / "questions.jsonl", matched)
    if rebuilding:
        for url in urls:
            (cache / (hashlib.sha256(url.encode()).hexdigest() + ".jsonl")).unlink(missing_ok=True)
    print(f"Вопросов ДЗ-1: {len(rows)}, из них по Wikipedia: {len(candidates)}. "
          f"Загружено статей в снимке: {len(corpus)}. Полная цитата найдена: {len(matched)}.")
    if missing:
        print("Цитата не совпала:", ", ".join(missing))


if __name__ == "__main__":
    main()
