import matplotlib
matplotlib.use("Agg")

import json
import spacy
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from scipy.stats import kendalltau, spearmanr, pearsonr
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.inter_rater import fleiss_kappa
from pingouin import intraclass_corr
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
from natasha import DatesExtractor, MorphVocab
import re
from collections import Counter
import math
from tqdm import tqdm
import json


# --- Инициализация моделей ---
nlp = spacy.load("ru_core_news_sm")
legal_ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
np.random.seed(42)
nltk.download('stopwords')
stop_words = set(stopwords.words("russian"))
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
sentiment_analyzer = pipeline("sentiment-analysis", model="DeepPavlov/rubert-base-cased-conversational", tokenizer="DeepPavlov/rubert-base-cased-conversational")
dates_extractor = DatesExtractor(MorphVocab())

# --- Функции предобработки ---
def tokenize(text):
    return [token.lemma_ for token in nlp(text.lower()) if token.text not in stop_words and token.is_alpha]

def custom_analyzer(text):
    return tokenize(text)

def get_key_terms(text, n=50):
    vectorizer = TfidfVectorizer(analyzer=custom_analyzer)
    tfidf = vectorizer.fit_transform([text])
    terms = vectorizer.get_feature_names_out()
    scores = tfidf.toarray()[0]
    return [terms[i] for i in scores.argsort()[-n:]]

def extract_citations(text):
    ner_results = legal_ner(text)
    entities = set()
    for entity in ner_results:
        if entity['entity_group'] in ['ORG', 'PER', 'LOC']:
            entities.add(entity['word'])
    pattern = r'(?:Статья|Ст\.)\s*\d+\s*(?:ГК\s*РФ|КоАП\s*РФ|УК\s*РФ|ФЗ\s*№\s*\d+)'
    entities.update(re.findall(pattern, text, re.IGNORECASE))
    key_terms = get_key_terms(text, n=20)
    entities.update(key_terms)
    return entities

def extract_dates(text):
    dates = [match.fact.as_json for match in dates_extractor(text)]
    formatted_dates = []
    for date in dates:
        year = date.get('year', 0)
        month = date.get('month', 1)
        day = date.get('day', 1)
        if year:
            formatted_dates.append(f"{year:04d}-{month:02d}-{day:02d}")
    return sorted(formatted_dates)

def normalize_score(score, min_val, max_val):
    return score


def analyze_block_sentiment(block_text):
    sentences = [sent.text for sent in nlp(block_text).sents]  # разбиваем на предложения
    positive_scores, negative_scores = [], []

    for sent in sentences:
        if not sent.strip():
            continue
        result = sentiment_analyzer(sent, truncation=True, max_length=512)
        label = result[0]['label']
        score = result[0]['score']
        if label == 'POSITIVE':
            positive_scores.append(score)
            negative_scores.append(0)
        elif label == 'NEGATIVE':
            negative_scores.append(score)
            positive_scores.append(0)
        else:
            # если модель умеет NEUTRAL
            positive_scores.append(0)
            negative_scores.append(0)

    # усредняем по всем предложениям
    avg_positive = np.mean(positive_scores) if positive_scores else 0
    avg_negative = np.mean(negative_scores) if negative_scores else 0
    neutrality = 1 - abs(avg_positive - avg_negative)  # чем ближе к 1, тем более нейтрально
    return normalize_score(neutrality, 0, 1)



def aggregate_expert_scores(expert_ratings):
    mean_ratings = [np.mean(ratings) for ratings in expert_ratings]
    kappa_data = [[ratings.count(i) for i in range(1, 6)] for ratings in [[int(r + 0.5) for r in ratings] for ratings in expert_ratings]]
    kappa = fleiss_kappa(kappa_data) if kappa_data else 0
    icc_data = []
    for i, ratings in enumerate(expert_ratings):
        for j, rating in enumerate(ratings):
            icc_data.append([i, j, rating])
    icc_df = pd.DataFrame(icc_data, columns=['subject', 'rater', 'rating'])
    icc = intraclass_corr(data=icc_df, targets='subject', raters='rater', ratings='rating')
    icc_value = icc[icc['Type'] == 'ICC2k']['ICC'].iloc[0] if not icc.empty else 0
    return mean_ratings, kappa, icc_value

def bootstrap_ci(data1, data2, stat_func, n_iterations=1000, ci=0.95):
    stats = []
    n = len(data1)
    for _ in range(n_iterations):
        indices = np.random.choice(n, n, replace=True)
        sample1 = [data1[i] for i in indices]
        sample2 = [data2[i] for i in indices]
        stat, _ = stat_func(sample1, sample2)
        stats.append(stat)
    lower = np.percentile(stats, (1 - ci) / 2 * 100)
    upper = np.percentile(stats, (1 + ci) / 2 * 100)
    return lower, upper



# --- Основная функция расчета метрик без экспертных оценок ---
def compute_metrics(data):
    blocks = [
        "Требования истца", "Аргументы истца", "Аргументы ответчика",
        "Оценка судом представленных сторонами доказательств",
        "Логические шаги в рассуждениях судьи и промежуточные выводы",
        "Применимые в судебном деле нормы права", "Решение суда"
    ]
    source_text = data['source_text']
    all_blocks_text = " ".join(data[block]['text'] for block in blocks)

    # --- Кэш для эмбеддингов ---
    block_embed_cache = {}
    sentence_embed_cache = {}

    def get_block_embedding(text):
        if text not in block_embed_cache:
            block_embed_cache[text] = sentence_model.encode(text)
        return block_embed_cache[text]

    def get_sentence_embedding(text):
        if text not in sentence_embed_cache:
            sentence_embed_cache[text] = sentence_model.encode(text)
        return sentence_embed_cache[text]

    # --- Ключевые термины и сущности ---
    key_terms = get_key_terms(source_text, n=50)
    extracted_terms = set(tokenize(all_blocks_text))
    citations_d = extract_citations(source_text)
    citations_e = extract_citations(all_blocks_text)
    dates_d = extract_dates(source_text)
    dates_e = extract_dates(all_blocks_text)

    # TF-IDF векторы
    vectorizer = TfidfVectorizer(analyzer=custom_analyzer)
    tfidf_matrix = vectorizer.fit_transform([source_text, all_blocks_text])
    v_d = tfidf_matrix[0].toarray()[0]
    v_e = tfidf_matrix[1].toarray()[0]

    # Векторные представления блоков
    block_embeddings = {block: get_block_embedding(data[block]['text']) for block in blocks}

    # --- Документ-уровневые метрики ---
    metrics = {}
    metrics['Coverage Ratio'] = normalize_score(
        len(set(key_terms) & extracted_terms) / len(key_terms) if key_terms else 0, 0, 1)
    metrics['Redundancy Penalty'] = normalize_score(
        sum(np.dot(block_embeddings[blocks[i]], block_embeddings[blocks[j]]) /
            (np.linalg.norm(block_embeddings[blocks[i]]) * np.linalg.norm(block_embeddings[blocks[j]]))
            for i in range(7) for j in range(i + 1, 7)) / 21, 0, 1)
    metrics['Compression Ratio'] = normalize_score(
        sum(len(tokenize(data[block]['text'])) for block in blocks) / len(tokenize(source_text)) if source_text else 0, 0, 2)
    metrics['Term Frequency Coherence'] = normalize_score(
        np.dot(v_d, v_e) / (np.linalg.norm(v_d) * np.linalg.norm(v_e)) if np.linalg.norm(v_d) * np.linalg.norm(v_e) != 0 else 0, 0, 1)
    metrics['Citation Coverage'] = normalize_score(
        len(citations_d & citations_e) / len(citations_d) if citations_d else 0, 0, 1)
    metrics['Semantic Entropy'] = normalize_score(
        -sum((count / total_words) * math.log2(count / total_words)
             for count in Counter(tokenize(all_blocks_text)).values()
             if count > 0) if (total_words := sum(Counter(tokenize(all_blocks_text)).values())) > 0 else 0, 0, 10)

    # --- Raw Cosine Similarity ---
    combined_vector = get_block_embedding(all_blocks_text)
    source_vector = get_block_embedding(source_text)
    raw_cos_sim = np.dot(combined_vector, source_vector) / (np.linalg.norm(combined_vector) * np.linalg.norm(source_vector))
    metrics['Raw Cosine Similarity'] = normalize_score(raw_cos_sim, 0, 1)

    # --- Block Order Consistency ---
    all_blocks_tokens = tokenize(all_blocks_text)
    source_tokens = tokenize(source_text)
    block_positions, source_positions = [], []
    source_token_indices = {}
    for idx, token in enumerate(source_tokens):
        source_token_indices.setdefault(token, []).append(idx)
    used_source_indices = {}
    for token in all_blocks_tokens:
        if token in source_token_indices:
            for idx in source_token_indices[token]:
                if used_source_indices.get(token, -1) < idx:
                    block_positions.append(len(block_positions))
                    source_positions.append(idx)
                    used_source_indices[token] = idx
                    break
    metrics['Block Order Consistency'] = normalize_score(
        (kendalltau(block_positions, source_positions)[0] + 1) / 2, 0, 1
    ) if len(block_positions) >= 2 else 0.0

    # --- Monotonicity Score ---
    metrics['Monotonicity Score'] = normalize_score(1 if dates_e == sorted(dates_e) else 0, 0, 1)

    # --- Документные метрики Block Completeness и Keyword-Based Pseudo-F1 ---
    all_blocks_token_set = set(all_blocks_tokens)
    metrics['Block Completeness'] = normalize_score(
        len(set(key_terms) & all_blocks_token_set) / len(key_terms) if key_terms else 0, 0, 1)
    precision = len(set(key_terms) & all_blocks_token_set) / len(all_blocks_token_set) if all_blocks_token_set else 0
    recall = len(set(key_terms) & all_blocks_token_set) / len(key_terms) if key_terms else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    metrics['Keyword-Based Pseudo-F1'] = normalize_score(f1, 0, 1)

    # --- Блоковые метрики ---
    block_metrics = {}
    for block in blocks:
        block_text = data[block]['text']
        sentences = [sent.text for sent in nlp(block_text).sents]

        # Intra-block coherence
        if len(sentences) < 2:
            coherence = 1.0
        else:
            sentence_embeds = [get_sentence_embedding(sent) for sent in sentences]
            coherence = sum(
                np.dot(sentence_embeds[i], sentence_embeds[j]) /
                (np.linalg.norm(sentence_embeds[i]) * np.linalg.norm(sentence_embeds[j]))
                for i in range(len(sentences)) for j in range(i + 1, len(sentences))
            ) / (len(sentences) * (len(sentences) - 1) / 2)

        # Inter-block distinctiveness
        distinctiveness = sum(
            1 - np.dot(block_embeddings[block], block_embeddings[other]) /
            (np.linalg.norm(block_embeddings[block]) * np.linalg.norm(block_embeddings[other]))
            for other in blocks if other != block
        ) / (len(blocks) - 1)

        # Neutrality Bias
        neutrality = analyze_block_sentiment(block_text)

        # Legal Term Density
        legal_density = len(set(tokenize(block_text)) & citations_e) / len(tokenize(block_text)) if block_text else 0

        block_metrics[block] = {
            'Intra-Block Coherence': normalize_score(coherence, 0, 1),
            'Inter-Block Distinctiveness': distinctiveness,
            'Neutrality Bias': neutrality,
            'Legal Term Density': legal_density
        }

    return block_metrics, metrics



def custom_serializer(obj):
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    raise TypeError(f'Object of type {type(obj).__name__} is not JSON serializable')

def process_json(input_file, output_file, checkpoint_interval=100):
    with open(input_file, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    output_list = []

    for i, data in enumerate(tqdm(data_list, desc="Processing documents")):
        block_metrics, document_metrics = compute_metrics(data)
        data['document_metrics'] = document_metrics
        for block in block_metrics:
            data[block]['metrics'] = block_metrics[block]
        output_list.append(data)

        # Сохранение каждые checkpoint_interval документов
        if (i + 1) % checkpoint_interval == 0:
            checkpoint_file = f"{output_file}.checkpoint_{i + 1}.json"
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(output_list, f, ensure_ascii=False, indent=2, default=custom_serializer)
            print(f"Checkpoint saved at {checkpoint_file}")

    # Финальное сохранение
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_list, f, ensure_ascii=False, indent=2, default=custom_serializer)
    print(f"Final output saved to {output_file}")

# --- Пример запуска ---
if __name__ == "__main__":
    # sample_data = {
    #     "id_sud_resh": "54321",
    #     "Требования истца": {
    #         "text": "Истец, ООО 'Ромашка', требует взыскания 500000 рублей с ответчика за неисполнение договора поставки от 10.01.2023, ссылаясь на Статью 309 ГК РФ и Статью 516 ГК РФ. Также истец требует компенсации убытков в размере 100000 рублей за упущенную выгоду.",
    #         "expert_eval": [4.5, 4.0, 4.8],
    #         "metrics": {}
    #     },
    #     "Аргументы истца": {
    #         "text": "Истец предоставил договор поставки от 10.01.2023, акт сверки от 15.02.2023, показания свидетелей (Иванов И.И., Петров П.П.), подтверждающие факт неисполнения обязательств ответчиком. Также представлено заключение независимого эксперта от 20.02.2023.",
    #         "expert_eval": [4.2, 4.0, 4.5],
    #         "metrics": {}
    #     },
    #     "Аргументы ответчика": {
    #         "text": "Ответчик, ЗАО 'Лютик', утверждает, что неисполнение договора произошло из-за форс-мажорных обстоятельств (пожар на складе 12.01.2023), ссылаясь на Статью 401 ГК РФ. Ответчик предоставил справку от МЧС и фотографии поврежденного имущества.",
    #         "expert_eval": [3.5, 3.8, 3.2],
    #         "metrics": {}
    #     },
    #     "Оценка судом представленных сторонами доказательств": {
    #         "text": "Суд рассмотрел договор поставки, акт сверки, свидетельские показания и заключение эксперта. Установлено, что пожар на складе ответчика 12.01.2023 не освобождает его от ответственности, так как не был своевременно уведомлен истец (Статья 401 ГК РФ).",
    #         "expert_eval": [4.0, 4.3, 4.1],
    #         "metrics": {}
    #     },
    #     "Логические шаги в рассуждениях судьи и промежуточные выводы": {
    #         "text": "Суд установил, что ответчик нарушил условия договора от 10.01.2023, не уведомив истца о форс-мажоре в установленный срок. На основании Статьи 309 ГК РФ и Статьи 516 ГК РФ обязательство должно быть исполнено. Убытки подтверждены актом от 15.02.2023.",
    #         "expert_eval": [4.3, 4.5, 4.2],
    #         "metrics": {}
    #     },
    #     "Применимые в судебном деле нормы права": {
    #         "text": "Суд применил Статью 309 ГК РФ (обязательства должны исполняться надлежащим образом), Статью 516 ГК РФ (расчеты по договору поставки), Статью 401 ГК РФ (ответственность за неисполнение обязательств) и Федеральный закон №44-ФЗ от 05.04.2013.",
    #         "expert_eval": [4.0, 4.2, 4.0],
    #         "metrics": {}
    #     },
    #     "Решение суда": {
    #         "text": "Суд постановил взыскать с ЗАО 'Лютик' в пользу ООО 'Ромашка' 450000 рублей за неисполнение договора и 80000 рублей убытков. Решение принято 25.03.2023 на основании Статьи 309 ГК РФ и Статьи 516 ГК РФ.",
    #         "expert_eval": [4.8, 4.7, 4.9],
    #         "metrics": {}
    #     },
    #     "source_text": (
    #         "Судебное решение по делу №54321 от 25.03.2023. Истец, ООО 'Ромашка', требует взыскания 500000 рублей за неисполнение договора поставки от 10.01.2023, ссылаясь на Статью 309 ГК РФ и Статью 516 ГК РФ, а также 100000 рублей убытков. Истец предоставил договор, акт сверки от 15.02.2023, свидетельские показания (Иванов И.И., Петров П.П.) и заключение эксперта от 20.02.2023. Ответчик, ЗАО 'Лютик', ссылается на форс-мажор (пожар 12.01.2023, Статья 401 ГК РФ), предоставив справку МЧС. Суд установил нарушение условий договора ответчиком, так как уведомление о форс-мажоре не было направлено вовремя. Применены Статья 309 ГК РФ, Статья 516 ГК РФ, Статья 401 ГК РФ и Федеральный закон №44-ФЗ. Суд взыскал 450000 рублей и 80000 рублей убытков."
    #     )
    # }

    # expert_ratings = [
    #     sample_data[block]["expert_eval"] for block in [
    #         "Требования истца", "Аргументы истца", "Аргументы ответчика",
    #         "Оценка судом представленных сторонами доказательств",
    #         "Логические шаги в рассуждениях судьи и промежуточные выводы",
    #         "Применимые в судебном деле нормы права", "Решение суда"
    #     ]
    # ]

    # # with open("input.json", "w", encoding="utf-8") as f:
    # #     json.dump(sample_data, f, ensure_ascii=False, indent=2)

    process_json("benchmark_input_grouped.json", "output.json")