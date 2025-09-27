import json
import numpy as np
from collections import defaultdict, Counter
from scipy.stats import pearsonr

# Файл для чтения и вывода
input_file = 'merged_output_with_expert_eval_all.json'
output_file = 'metrics_analysis.json'

# Загрузка данных
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Определение блочных метрик (ключи в metrics каждого блока)
block_metrics = ['Intra-Block Coherence', 'Inter-Block Distinctiveness', 'Neutrality Bias', 'Legal Term Density', 'LLM_Evaluation_Score']

# Блоки (названия) — для per-block
blocks = [
    'Требования истца', 'Аргументы истца', 'Аргументы ответчика',
    'Оценка судом представленных сторонами доказательств',
    'Логические шаги в рассуждениях судьи и промежуточные выводы',
    'Применимые в судебном деле нормы права', 'Решение суда'
]

# Документные метрики
doc_metrics = [
    'Coverage Ratio', 'Redundancy Penalty', 'Compression Ratio',
    'Term Frequency Coherence', 'Citation Coverage', 'Semantic Entropy',
    'Raw Cosine Similarity', 'Block Order Consistency', 'Monotonicity Score',
    'Block Completeness', 'Keyword-Based Pseudo-F1'
]

# 1. Сбор всех raw значений + пар для корреляций (per-block и doc-level, игнор 0)
all_values = {metric: [] for metric in block_metrics + doc_metrics}
block_values = {block_name: {metric: [] for metric in block_metrics} for block_name in blocks}
block_correlation_pairs = {block_name: {metric: [] for metric in block_metrics} for block_name in blocks}  # Per-block пары
doc_correlation_pairs_block_metrics = {metric: [] for metric in block_metrics}  # Doc-level для блочных метрик
doc_correlation_pairs = {metric: [] for metric in doc_metrics}  # Для документных

# Для expert: avg_expert и num_experts (только >0)
expert_avgs_all = []
expert_avgs_per_block = {block_name: [] for block_name in blocks}
expert_lens_all = []  # Только >0
expert_lens_per_block = {block_name: [] for block_name in blocks}  # Только >0
expert_lens_histogram = Counter()  # Только >0, global

# Для doc-level блочных: doc_avg_metric per doc
doc_avg_metrics = {metric: [] for metric in block_metrics}
doc_avg_experts = []

# Обработка каждого документа
for doc in data:
    block_expert_avgs = {}
    block_values_per_doc = {block_name: {metric: None for metric in block_metrics} for block_name in blocks}  # Temp для doc_avg
    valid_blocks_count = 0
    for block_name in blocks:
        if block_name in doc:
            block = doc[block_name]
            expert_evals = block.get('expert_eval', [])
            num_experts = len(expert_evals)
            if num_experts > 0:  # Игнор 0: сбор только >0
                avg_expert = np.mean(expert_evals)
                block_expert_avgs[block_name] = avg_expert
                expert_avgs_per_block[block_name].append(avg_expert)
                expert_avgs_all.append(avg_expert)
                expert_lens_per_block[block_name].append(num_experts)
                expert_lens_all.append(num_experts)
                expert_lens_histogram[num_experts] += 1
            else:
                block_expert_avgs[block_name] = np.nan
            # Метрики (всегда, даже без expert)
            metrics = block.get('metrics', {})
            for metric in block_metrics:
                value = metrics.get(metric)
                if value is not None:
                    block_values[block_name][metric].append(value)
                    all_values[metric].append(value)
                    block_values_per_doc[block_name][metric] = value  # Для doc_avg
                # Per-block пара (только если >0)
                avg_expert = block_expert_avgs.get(block_name, np.nan)
                if not np.isnan(avg_expert) and value is not None:
                    block_correlation_pairs[block_name][metric].append((value, avg_expert))
            valid_blocks_count += 1 if num_experts > 0 else 0

    # Doc-level avg_expert (только если valid >0)
    valid_block_avgs = [v for v in block_expert_avgs.values() if not np.isnan(v)]
    doc_avg_expert = np.mean(valid_block_avgs) if valid_block_avgs else np.nan
    doc_avg_experts.append(doc_avg_expert)

    # Doc-level avg_metric для блочных (только если valid >0)
    if not np.isnan(doc_avg_expert) and valid_blocks_count > 0:
        for metric in block_metrics:
            valid_block_vals = [block_values_per_doc[b][metric] for b in blocks if block_values_per_doc[b][metric] is not None]
            if valid_block_vals:
                doc_avg_metric = np.mean(valid_block_vals)
                doc_avg_metrics[metric].append(doc_avg_metric)
                doc_correlation_pairs_block_metrics[metric].append((doc_avg_metric, doc_avg_expert))

    # Документные метрики и пары (только если doc_avg_expert валиден, т.е. >0 в доке)
    doc_metrics_dict = doc.get('document_metrics', {})
    for metric in doc_metrics:
        value = doc_metrics_dict.get(metric)
        if value is not None:
            all_values[metric].append(value)
        if value is not None and not np.isnan(doc_avg_expert):
            doc_correlation_pairs[metric].append((value, doc_avg_expert))

# Функции
def compute_stats(values, include_var_std=True):
    values = [v for v in values if not np.isnan(v)]
    n = len(values)
    if n > 0:
        mean_val = np.mean(values)
        if include_var_std and n > 1:
            var_val = np.var(values, ddof=1)
            std_val = np.std(values, ddof=1)
        else:
            var_val = std_val = None
    else:
        mean_val = var_val = std_val = np.nan
    stats = {
        'Число оценок': n,
        'Средняя оценка': float(mean_val) if not np.isnan(mean_val) else None
    }
    if include_var_std and var_val is not None:
        stats['Дисперсия'] = float(var_val)
        stats['Ст. отклонение'] = float(std_val)
    return stats

def normalize(values):
    if not values or len(values) == 0:
        return []
    min_v = min(v for v in values if not np.isnan(v))
    max_v = max(v for v in values if not np.isnan(v))
    range_v = max_v - min_v
    return [(v - min_v) / range_v if range_v > 0 and not np.isnan(v) else 0 for v in values]

def compute_correlation(pairs):
    pairs = [(m, e) for m, e in pairs if not (np.isnan(m) or np.isnan(e))]
    if len(pairs) > 1:
        metrics_vals, expert_vals = zip(*pairs)
        norm_metrics = normalize(list(metrics_vals))
        norm_experts = normalize(list(expert_vals))
        if np.std(norm_metrics) == 0 or np.std(norm_experts) == 0:
            return None, 1.0
        corr, p_value = pearsonr(norm_metrics, norm_experts)
        return float(corr), float(p_value)
    return None, None

# 2. Статистики для блочных метрик: global (с doc_level corr) + per_block (с num_experts)
block_stats = {}
for metric in block_metrics:
    # Global stats + doc_level corr
    global_stats = compute_stats(all_values[metric])
    corr_doc, p_doc = compute_correlation(doc_correlation_pairs_block_metrics[metric])
    global_stats['Корреляция Пирсона с экспертной оценкой'] = corr_doc
    global_stats['p-value'] = p_doc
    block_stats[metric] = {'global': global_stats}
    
    # Per-block stats + corr + среднее num_experts
    per_block = {}
    for block_name in blocks:
        stats = compute_stats(block_values[block_name][metric])
        corr, p_val = compute_correlation(block_correlation_pairs[block_name][metric])
        stats['Корреляция Пирсона с экспертной оценкой'] = corr
        stats['p-value'] = p_val
        # Добавляем среднее число экспертов для блока
        avg_num = np.mean(expert_lens_per_block[block_name]) if expert_lens_per_block[block_name] else np.nan
        stats['Среднее число экспертов'] = float(avg_num) if not np.isnan(avg_num) else None
        per_block[block_name] = stats
    block_stats[metric]['per_block'] = per_block

# 3. Объединённые expert_stats: avg_expert + num_experts
expert_stats = {}
# Global
global_expert_stats = compute_stats(expert_avgs_all)
avg_num_experts_global = np.mean(expert_lens_all) if expert_lens_all else np.nan
global_expert_stats['Среднее число экспертов'] = float(avg_num_experts_global) if not np.isnan(avg_num_experts_global) else None
# Гистограмма с русскими ключами
hist_global = {}
for k, v in expert_lens_histogram.items():
    if k == 1:
        hist_global['На одно судебное решение были оценки от 1-го эксперта'] = v
    elif k == 2:
        hist_global['На одно судебное решение были оценки от 2-х эксперта'] = v
    elif k == 3:
        hist_global['На одно судебное решение были оценки от 3-х эксперта'] = v
    elif k == 4:
        hist_global['На одно судебное решение были оценки от 4-х эксперта'] = v
    elif k == 5:
        hist_global['На одно судебное решение были оценки от 5-ти эксперта'] = v
    else:
        # Для >5 суммировать в 5-ти, если нужно
        pass
global_expert_stats.update(hist_global)
expert_stats['global'] = global_expert_stats

# Per-block
per_block_expert = {}
for block_name in blocks:
    block_avg_stats = compute_stats(expert_avgs_per_block[block_name])
    avg_num_experts_block = np.mean(expert_lens_per_block[block_name]) if expert_lens_per_block[block_name] else np.nan
    block_avg_stats['Среднее число экспертов'] = float(avg_num_experts_block) if not np.isnan(avg_num_experts_block) else None
    # Гистограмма per-block (Counter для блока)
    hist_block_counter = Counter(expert_lens_per_block[block_name])
    hist_block = {}
    for k, v in hist_block_counter.items():
        if k == 1:
            hist_block['На одно судебное решение были оценки от 1-го эксперта'] = v
        elif k == 2:
            hist_block['На одно судебное решение были оценки от 2-х эксперта'] = v
        elif k == 3:
            hist_block['На одно судебное решение были оценки от 3-х эксперта'] = v
        elif k == 4:
            hist_block['На одно судебное решение были оценки от 4-х эксперта'] = v
        elif k == 5:
            hist_block['На одно судебное решение были оценки от 5-ти эксперта'] = v
    block_avg_stats.update(hist_block)
    per_block_expert[block_name] = block_avg_stats
expert_stats['per_block'] = per_block_expert

# 4. Doc stats с corr встроенными
doc_stats = {}
for metric in doc_metrics:
    stats = compute_stats(all_values[metric])
    corr, p_val = compute_correlation(doc_correlation_pairs[metric])
    stats['Корреляция Пирсона с экспертной оценкой'] = corr
    stats['p-value'] = p_val
    doc_stats[metric] = stats

doc_expert_stats = compute_stats(doc_avg_experts)

# Сохранение
output_data = {
    'block_stats': block_stats,
    'expert_stats': expert_stats,
    'doc_stats': doc_stats,
    'doc_expert_stats': doc_expert_stats
}

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2, default=str)

# Краткий вывод (пример для expert_stats)
print("Expert global:", expert_stats['global'])
print("Expert per_block (пример 'Требования истца'):", expert_stats['per_block']['Требования истца'])
print(f"Единый файл сохранён: {output_file}")