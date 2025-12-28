# utils.py
import pandas as pd
import requests
import re
from io import BytesIO
from sentence_transformers import SentenceTransformer, util
import pymorphy2
import functools
import os

def get_github_headers():
    """
    Возвращает headers для GitHub requests.
    Если GITHUB_TOKEN есть в env — используем его.
    Если нет — работаем без авторизации (для публичных repo).
    """
    token = os.getenv("GITHUB_TOKEN")
    if token:
        return {"Authorization": f"token {token}"}
    return {}

# ---------- модель и морфологический разбор ----------
@functools.lru_cache(maxsize=1)
def get_model():
    hf_model_id = "skatzR/USER-BGE-M3-MiniLM-L12-v2-Distilled"
    return SentenceTransformer(hf_model_id)

@functools.lru_cache(maxsize=1)
def get_morph():
    return pymorphy2.MorphAnalyzer()

# ---------- служебные функции ----------
def preprocess(text):
    return re.sub(r"\s+", " ", str(text).lower().strip())

def lemmatize(word):
    return get_morph().parse(word)[0].normal_form

@functools.lru_cache(maxsize=10000)
def lemmatize_cached(word):
    return lemmatize(word)

SYNONYM_GROUPS = []
SYNONYM_DICT = {}
for group in SYNONYM_GROUPS:
    lemmas = {lemmatize(w.lower()) for w in group}
    for lemma in lemmas:
        SYNONYM_DICT[lemma] = lemmas

GITHUB_CSV_URLS = [
    "https://raw.githubusercontent.com/skatzrskx55q/data-assistant-vfiziki/main/data6.xlsx",
    "https://raw.githubusercontent.com/skatzrsk/semantic-assistant/main/data21.xlsx",
    "https://raw.githubusercontent.com/skatzrsk/semantic-assistant/main/data31.xlsx"
]

# ---------- разбор фраз с | и / ----------
@functools.lru_cache(maxsize=5000)
def split_by_slash(phrase: str):
    phrase = phrase.strip()
    segments = [seg.strip() for seg in phrase.split("|")]
    all_phrases = []

    for segment in segments:
        parts = []
        last_idx = 0

        for m in re.finditer(r'\b[\w-]+(?:/[\w-]+)+\b', segment):
            if m.start() > last_idx:
                prefix = segment[last_idx:m.start()].strip()
                if prefix:
                    parts.append([prefix])

            options = [opt.strip() for opt in m.group(0).split("/") if opt.strip()]
            parts.append(options)
            last_idx = m.end()

        if last_idx < len(segment):
            suffix = segment[last_idx:].strip()
            if suffix:
                parts.append([suffix])

        for combination in product(*parts):
            combined = " ".join(combination)
            combined = re.sub(r"\s+", " ", combined).strip()
            if combined:
                all_phrases.append(combined)

    return all_phrases

def load_excel(url):
    resp = requests.get(url, headers=get_github_headers())
    if resp.status_code != 200:
        raise ValueError(f"Ошибка загрузки {url}")
    df = pd.read_excel(BytesIO(resp.content))
    topic_cols = [c for c in df.columns if c.lower().startswith("topics")]
    if not topic_cols:
        raise KeyError("Не найдены колонки topics")
    df["topics"] = df[topic_cols].astype(str).agg(lambda x: [v for v in x if v and v != "nan"], axis=1)
    df["phrase_full"] = df["phrase"]
    df["phrase_list"] = df["phrase"].apply(split_by_slash)
    df = df.explode("phrase_list", ignore_index=True)
    df["phrase"] = df["phrase_list"]
    df["phrase_proc"] = df["phrase"].apply(preprocess)
    df["phrase_lemmas"] = df["phrase_proc"].apply(
        lambda t: {lemmatize_cached(w) for w in re.findall(r"\w+", t)}
    )
    # Эмбеддинги вычисляются в load_all_excels() для всего датасета
    if "comment" not in df.columns:
        df["comment"] = ""
    return df[["phrase", "phrase_proc", "phrase_full", "phrase_lemmas", "topics", "comment"]]

def load_all_excels():
    dfs = []
    for url in GITHUB_CSV_URLS:
        try:
            dfs.append(load_excel(url))
        except Exception as e:
            print(f"⚠️ Ошибка с {url}: {e}")
    if not dfs:
        raise ValueError("Не удалось загрузить ни одного файла")
    df = pd.concat(dfs, ignore_index=True)
    
    # Вычисляем эмбеддинги для семантического поиска (один раз)
    model = get_model()
    df.attrs["phrase_embs"] = model.encode(df["phrase_proc"].tolist(), convert_to_tensor=True)
    
    return df

# ---------- удаление дублей ----------
def _score_of(item):
    """Возвращает числовой score из кортежа результата."""
    return item[0] if len(item) == 4 else 1.0

def _phrase_full_of(item):
    """Возвращает phrase_full из кортежа результата."""
    return item[1] if len(item) == 4 else item[0]

def deduplicate_results(results):
    """
    Удаляет дубликаты по phrase_full, сохраняя кортеж в исходном формате
    (4-элемента для semantic, 3-элемента для keyword) и оставляя
    наиболее высокий score при коллизии.
    """
    best = {}
    for item in results:
        key = _phrase_full_of(item)
        score = _score_of(item)
        if key not in best or score > _score_of(best[key]):
            best[key] = item
    return list(best.values())

# ---------- поиск ----------
def semantic_search(query, df, top_k=5, threshold=0.5):
    model = get_model()
    query_proc = preprocess(query)
    query_emb = model.encode(query_proc, convert_to_tensor=True)
    phrase_embs = df.attrs["phrase_embs"]
    sims = util.pytorch_cos_sim(query_emb, phrase_embs)[0]
    results = [
        (float(score), df.iloc[idx]["phrase_full"], df.iloc[idx]["topics"], df.iloc[idx]["comment"])
        for idx, score in enumerate(sims) if float(score) >= threshold
    ]
    results = sorted(results, key=lambda x: x[0], reverse=True)[:top_k]
    return deduplicate_results(results)

def keyword_search(query, df):
    query_proc = preprocess(query)
    query_words = re.findall(r"\w+", query_proc)
    query_lemmas = [lemmatize_cached(w) for w in query_words]
    matched = []
    for row in df.itertuples():
        lemma_match = all(
            any(ql in SYNONYM_DICT.get(pl, {pl}) for pl in row.phrase_lemmas)
            for ql in query_lemmas
        )
        partial_match = all(q in row.phrase_proc for q in query_words)
        if lemma_match or partial_match:
            matched.append((row.phrase_full, row.topics, row.comment))
    return deduplicate_results(matched)

# ---------- фильтрация (если понадобится) ----------
def filter_by_topics(results, selected_topics):
    if not selected_topics:
        return results
    filtered = []
    for item in results:
        if isinstance(item, tuple) and len(item) == 4:
            score, phrase, topics, comment = item
            if set(topics) & set(selected_topics):
                filtered.append((score, phrase, topics, comment))
        elif isinstance(item, tuple) and len(item) == 3:
            phrase, topics, comment = item
            if set(topics) & set(selected_topics):
                filtered.append((phrase, topics, comment))
    return filtered
