import json
import numpy as np
from collections import defaultdict, Counter
from scipy.stats import pearsonr, spearmanr, kendalltau
import statsmodels.formula.api as smf
import statsmodels.api as sm
import pandas as pd
import logging
import sys
import time
import re  # NEW: для sanitize
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt  # NEW: для графиков


# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Файл для чтения и вывода
input_file = 'merged_output_with_expert_eval_all.json'
output_file = 'metrics_analysis(2).json'

# Загрузка данных
try:
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
except Exception as e:
    logger.error(f"Ошибка загрузки данных: {e}")
    sys.exit(1)

# Определение блочных и документных метрик
block_metrics = ['Intra-Block Coherence', 'Inter-Block Distinctiveness', 'Neutrality Bias', 'Legal Term Density', 'LLM_Evaluation_Score']
blocks = [
    'Требования истца', 'Аргументы истца', 'Аргументы ответчика',
    'Оценка судом представленных сторонами доказательств',
    'Логические шаги в рассуждениях судьи и промежуточные выводы',
    'Применимые в судебном деле нормы права', 'Решение суда'
]
doc_metrics = [
    'Coverage Ratio', 'Redundancy Penalty', 'Compression Ratio',
    'Term Frequency Coherence', 'Citation Coverage', 'Semantic Entropy',
    'Raw Cosine Similarity', 'Block Order Consistency', 'Monotonicity Score',
    'Block Completeness', 'Keyword-Based Pseudo-F1'
]
penalty_metrics = ['Redundancy Penalty']

# Сбор сырых данных
block_all_values = {metric: [] for metric in block_metrics}
doc_all_values = {metric: [] for metric in doc_metrics}
block_values = {block_name: {metric: [] for metric in block_metrics} for block_name in blocks}
block_correlation_pairs = {block_name: {metric: [] for metric in block_metrics} for block_name in blocks}
doc_correlation_pairs_block_metrics = {metric: [] for metric in block_metrics}
doc_correlation_pairs = {metric: [] for metric in doc_metrics}
expert_avgs_all = []
expert_avgs_per_block = {block_name: [] for block_name in blocks}
expert_lens_all = []
expert_lens_per_block = {block_name: [] for block_name in blocks}
expert_lens_histogram = Counter()
doc_avg_metrics = {metric: [] for metric in block_metrics}
doc_avg_experts = []
expert_long = []
doc_count_per_block = {block_name: 0 for block_name in blocks}

def minmax_scale_array(arr, min_v=None, max_v=None):
    arr = np.asarray(arr, dtype=float)
    mask = ~np.isnan(arr)
    if not mask.any():
        return np.full(arr.shape, np.nan)
    if min_v is None:
        min_v = np.nanmin(arr[mask])
    if max_v is None:
        max_v = np.nanmax(arr[mask])
    if np.isnan(min_v) or np.isnan(max_v):
        return np.full(arr.shape, np.nan)
    if min_v == max_v:
        # KEEP AS IS per user request: возвращаем константу 0.5 на валидных позициях
        out = np.full(arr.shape, np.nan)
        out[mask] = 0.5
        return out
    out = (arr - min_v) / (max_v - min_v)
    out[~mask] = np.nan
    return out

def to_numeric(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan

# CHG: при сборе экспертов добавляем 'rater_pos' — позиция рецензента в документе (1..n)
doc_id_counter = 0
for doc in data:
    doc_id = doc_id_counter
    doc_id_counter += 1
    block_expert_avgs = {}
    block_values_per_doc = {block_name: {metric: None for metric in block_metrics} for block_name in blocks}
    valid_blocks_count = 0
    for block_name in blocks:
        if block_name in doc:
            block = doc[block_name]
            expert_evals = block.get('expert_eval', [])
            # сохраняем порядок — rater_pos это позиция в списке
            expert_evals = [to_numeric(e) for e in expert_evals if not pd.isna(e)]
            num_experts = len(expert_evals)
            if num_experts > 0:
                doc_count_per_block[block_name] += 1
                avg_expert = np.mean(expert_evals) if expert_evals else np.nan
                block_expert_avgs[block_name] = avg_expert
                expert_avgs_per_block[block_name].append(avg_expert)
                expert_avgs_all.append(avg_expert)
                expert_lens_per_block[block_name].append(num_experts)
                expert_lens_all.append(num_experts)
                expert_lens_histogram[num_experts] += 1
                for rater_pos, score in enumerate(expert_evals, 1):  # CHG: rater_pos добавлен
                    # global_rater_id больше не используется для корреляций; сохраняем позицию
                    row = {
                        'doc_id': doc_id,
                        'block_name': block_name,
                        'rater_pos': rater_pos,          # NEW
                        'expert_score': score,
                        'num_experts': num_experts
                    }
                    metrics = block.get('metrics', {})
                    for m in block_metrics:
                        value = to_numeric(metrics.get(m))
                        if not np.isnan(value):
                            if m in penalty_metrics:
                                value = 1.0 - value
                        row[m] = value
                    expert_long.append(row)
            metrics = block.get('metrics', {})
            for metric in block_metrics:
                value = to_numeric(metrics.get(metric))
                if not np.isnan(value):
                    if metric in penalty_metrics:
                        value = 1.0 - value
                    block_values[block_name][metric].append(value)
                    block_all_values[metric].append(value)
                    block_values_per_doc[block_name][metric] = value
                if not np.isnan(block_expert_avgs.get(block_name, np.nan)) and not np.isnan(value):
                    block_correlation_pairs[block_name][metric].append((value, block_expert_avgs[block_name]))
            valid_blocks_count += 1 if num_experts > 0 else 0

    valid_block_avgs = [v for v in block_expert_avgs.values() if not np.isnan(v)]
    doc_avg_expert = np.mean(valid_block_avgs) if valid_block_avgs else np.nan
    doc_avg_experts.append(doc_avg_expert)

    if not np.isnan(doc_avg_expert) and valid_blocks_count > 0:
        for metric in block_metrics:
            valid_block_vals = [block_values_per_doc[b][metric] for b in blocks if block_values_per_doc[b][metric] is not None]
            if valid_block_vals:
                doc_avg_metric = np.mean(valid_block_vals)
                doc_avg_metrics[metric].append(doc_avg_metric)
                doc_correlation_pairs_block_metrics[metric].append((doc_avg_metric, doc_avg_expert))

    doc_metrics_dict = doc.get('document_metrics', {})
    for metric in doc_metrics:
        value = to_numeric(doc_metrics_dict.get(metric))
        if not np.isnan(value):
            if metric in penalty_metrics:
                value = 1.0 - value
            doc_all_values[metric].append(value)
        if not np.isnan(value) and not np.isnan(doc_avg_expert):
            doc_correlation_pairs[metric].append((value, doc_avg_expert))

# Логирование числа документов
logger.info(f"Число документов по блокам: {doc_count_per_block}")

# Вычисление глобальных min/max для нормализации
metric_ranges = {}
for metric in block_metrics:
    arr = np.array(block_all_values[metric], dtype=float)
    metric_ranges[metric] = (np.nanmin(arr) if len(arr) > 0 else np.nan, np.nanmax(arr) if len(arr) > 0 else np.nan)
for metric in doc_metrics:
    arr = np.array(doc_all_values[metric], dtype=float)
    metric_ranges[metric] = (np.nanmin(arr) if len(arr) > 0 else np.nan, np.nanmax(arr) if len(arr) > 0 else np.nan)
expert_scores_all = [row['expert_score'] for row in expert_long]
metric_ranges['expert_score'] = (np.nanmin(expert_scores_all) if expert_scores_all else np.nan, np.nanmax(expert_scores_all) if expert_scores_all else np.nan)
logger.info(f"Диапазоны метрик: {metric_ranges}")

# Диагностика: обёрнута проверками
if expert_long:
    expert_df = pd.DataFrame(expert_long)
    if not expert_df.empty and 'block_name' in expert_df.columns:
        for block_name in blocks:
            block_data = expert_df[expert_df['block_name'] == block_name]
            if 'expert_score' in block_data.columns:
                logger.info(f"Блок {block_name}, expert_score stats: {block_data['expert_score'].describe()}")
            if not block_data.empty and 'rater_pos' in block_data.columns:
                logger.info(f"Блок {block_name}, число экспертов (по doc): {block_data.groupby('doc_id')['rater_pos'].count().value_counts()}")

# Вспомогательные функции
def compute_stats(values, metric_name, include_var_std=True):
    values = np.array([v for v in values if not np.isnan(v) and v is not None], dtype=float)
    if len(values) == 0:
        return {'Число оценок': 0, 'Средняя оценка': None}
    min_v, max_v = metric_ranges.get(metric_name, (None, None))
    norm_values = minmax_scale_array(values, min_v, max_v)
    n = len(norm_values)
    mean_val = np.nanmean(norm_values) if n > 0 else np.nan
    stats = {'Число оценок': n, 'Средняя оценка': float(mean_val) if not np.isnan(mean_val) else None}
    if include_var_std and n > 1:
        var_val = np.nanvar(norm_values, ddof=1)
        stats['Дисперсия'] = float(var_val) if not np.isnan(var_val) else None
        stats['Ст. отклонение'] = float(np.sqrt(var_val)) if not np.isnan(var_val) else None
    return stats

def compute_correlation(pairs, metric_name):
    pairs = [(m, e) for m, e in pairs if not (np.isnan(m) or np.isnan(e) or m is None or e is None)]
    if len(pairs) <= 1:
        return {'Pearson': (None, None), 'Spearman': (None, None), 'Kendall': (None, None)}
    metrics_vals, expert_vals = zip(*pairs)
    min_v_m, max_v_m = metric_ranges.get(metric_name, (None, None))
    min_v_e, max_v_e = metric_ranges.get('expert_score', (None, None))
    norm_metrics = minmax_scale_array(metrics_vals, min_v_m, max_v_m)
    norm_experts = minmax_scale_array(expert_vals, min_v_e, max_v_e)
    if np.nanstd(norm_metrics) == 0 or np.nanstd(norm_experts) == 0:
        logger.warning(f"Нулевая дисперсия для метрики {metric_name}")
        return {'Pearson': (None, None), 'Spearman': (None, None), 'Kendall': (None, None)}
    pearson_corr, pearson_p = pearsonr(norm_metrics, norm_experts)
    spearman_corr, spearman_p = spearmanr(norm_metrics, norm_experts)
    kendall_corr, kendall_p = kendalltau(norm_metrics, norm_experts)
    return {
        'Pearson': (float(pearson_corr), float(pearson_p)),
        'Spearman': (float(spearman_corr), float(spearman_p)),
        'Kendall': (float(kendall_corr), float(kendall_p))
    }

def compute_expert_correlations(block_df):
    """Вычисление корреляций между экспертами для документов с ≥2 экспертами.
       Pivot — по rater_pos (позициям рецензентов в документе)."""
    if block_df.empty:
        return {'Pearson': (None, None), 'Spearman': (None, None), 'Kendall': (None, None)}
    valid_docs = block_df.groupby('doc_id')['rater_pos'].count()
    valid_docs = valid_docs[valid_docs >= 2].index
    block_df = block_df[block_df['doc_id'].isin(valid_docs)]
    
    if block_df.empty or len(block_df['doc_id'].unique()) < 2:
        return {'Pearson': (None, None), 'Spearman': (None, None), 'Kendall': (None, None)}
    
    # Pivot по rater_pos
    try:
        pivot_df = block_df.pivot(index='doc_id', columns='rater_pos', values='expert_score')
    except Exception as e:
        logger.error(f"Ошибка при pivot в compute_expert_correlations: {e}")
        return {'Pearson': (None, None), 'Spearman': (None, None), 'Kendall': (None, None)}
    pivot_df = pivot_df.dropna(how='all')
    
    if pivot_df.shape[1] < 2:
        return {'Pearson': (None, None), 'Spearman': (None, None), 'Kendall': (None, None)}
    
    correlations = {'Pearson': [], 'Spearman': [], 'Kendall': []}
    col_idxs = list(pivot_df.columns)
    for i in range(len(col_idxs)):
        for j in range(i + 1, len(col_idxs)):
            col_i = col_idxs[i]
            col_j = col_idxs[j]
            scores_i = pivot_df[col_i].dropna()
            scores_j = pivot_df[col_j].reindex(scores_i.index).dropna()
            scores_i = scores_i.loc[scores_j.index]
            if len(scores_i) > 1 and np.std(scores_i) > 0 and np.std(scores_j) > 0:
                pearson_r, pearson_p = pearsonr(scores_i, scores_j)
                spearman_r, spearman_p = spearmanr(scores_i, scores_j)
                kendall_r, kendall_p = kendalltau(scores_i, scores_j)
                correlations['Pearson'].append((pearson_r, pearson_p))
                correlations['Spearman'].append((spearman_r, spearman_p))
                correlations['Kendall'].append((kendall_r, kendall_p))
    result = {}
    for metric in ['Pearson', 'Spearman', 'Kendall']:
        if correlations[metric]:
            r_values, p_values = zip(*correlations[metric])
            result[metric] = (float(np.mean(r_values)), float(np.mean(p_values)))
        else:
            result[metric] = (None, None)
    return result

def plot_bland_altman(x, y, title, filename):
    x = np.array(x)
    y = np.array(y)
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        logger.warning(f"Недостаточно данных для графика Bland-Altman: {filename}")
        return
    mean = (x + y) / 2
    diff = x - y
    bias = np.mean(diff)
    sd = np.std(diff)
    plt.figure()
    plt.scatter(mean, diff)
    plt.axhline(bias, color='red', label='Bias')
    plt.axhline(bias + 1.96 * sd, color='gray', linestyle='--', label='Upper LoA')
    plt.axhline(bias - 1.96 * sd, color='gray', linestyle='--', label='Lower LoA')
    plt.title(title)
    plt.xlabel('Mean')
    plt.ylabel('Difference')
    plt.legend()
    plt.savefig(filename)
    plt.close()
    logger.info(f"Сохранён график Bland-Altman: {filename}")

# 2. Статистики для блочных метрик
block_stats = {}
for metric in block_metrics:
    global_stats = compute_stats(block_all_values[metric], metric)
    corr_dict = compute_correlation(doc_correlation_pairs_block_metrics[metric], metric)
    for key, (r, p) in corr_dict.items():
        global_stats[f'{key} r'] = r
        global_stats[f'{key} p'] = p
    
    # Дополнительно: CCC, MAE и Bland-Altman для глобальных (усреднённых по документу) блочных метрик
    pairs = doc_correlation_pairs_block_metrics[metric]
    pairs = [(m, e) for m, e in pairs if not (np.isnan(m) or np.isnan(e))]
    if len(pairs) > 1:
        metrics_vals, expert_vals = zip(*pairs)
        min_v_m, max_v_m = metric_ranges.get(metric, (None, None))
        min_v_e, max_v_e = metric_ranges.get('expert_score', (None, None))
        norm_metrics = minmax_scale_array(metrics_vals, min_v_m, max_v_m)
        norm_experts = minmax_scale_array(expert_vals, min_v_e, max_v_e)
        if np.nanstd(norm_metrics) > 0 and np.nanstd(norm_experts) > 0:
            pearson_corr = pearsonr(norm_metrics, norm_experts)[0]
            mu_m = np.nanmean(norm_metrics)
            mu_e = np.nanmean(norm_experts)
            sigma_m = np.nanstd(norm_metrics)
            sigma_e = np.nanstd(norm_experts)
            ccc = 2 * pearson_corr * sigma_m * sigma_e / (sigma_m**2 + sigma_e**2 + (mu_m - mu_e)**2)
            global_stats['Lin CCC'] = float(ccc) if not np.isnan(ccc) else None
        else:
            global_stats['Lin CCC'] = None
        mae = np.nanmean(np.abs(norm_metrics - norm_experts))
        global_stats['MAE'] = float(mae) if not np.isnan(mae) else None
        
        safe_metric = re.sub(r'\W+', '_', metric)
        filename = f"bland_altman_global_{safe_metric}.png"
        plot_bland_altman(norm_metrics, norm_experts, f"Bland-Altman Plot for Global Avg {metric}", filename)
    else:
        global_stats['Lin CCC'] = None
        global_stats['MAE'] = None

    block_stats[metric] = {'global': global_stats}
    per_block = {}
    for block_name in blocks:
        stats = compute_stats(block_values[block_name][metric], metric)
        corr_dict = compute_correlation(block_correlation_pairs[block_name][metric], metric)
        for key, (r, p) in corr_dict.items():
            stats[f'{key} r'] = r
            stats[f'{key} p'] = p
        avg_num = np.mean(expert_lens_per_block[block_name]) if expert_lens_per_block[block_name] else np.nan
        stats['Среднее число экспертов'] = float(avg_num) if not np.isnan(avg_num) else None
        
        # Дополнительно: CCC, MAE и Bland-Altman для per_block
        pairs = block_correlation_pairs[block_name][metric]
        pairs = [(m, e) for m, e in pairs if not (np.isnan(m) or np.isnan(e))]
        if len(pairs) > 1:
            metrics_vals, expert_vals = zip(*pairs)
            min_v_m, max_v_m = metric_ranges.get(metric, (None, None))
            min_v_e, max_v_e = metric_ranges.get('expert_score', (None, None))
            norm_metrics = minmax_scale_array(metrics_vals, min_v_m, max_v_m)
            norm_experts = minmax_scale_array(expert_vals, min_v_e, max_v_e)
            if np.nanstd(norm_metrics) > 0 and np.nanstd(norm_experts) > 0:
                pearson_corr = pearsonr(norm_metrics, norm_experts)[0]
                mu_m = np.nanmean(norm_metrics)
                mu_e = np.nanmean(norm_experts)
                sigma_m = np.nanstd(norm_metrics)
                sigma_e = np.nanstd(norm_experts)
                ccc = 2 * pearson_corr * sigma_m * sigma_e / (sigma_m**2 + sigma_e**2 + (mu_m - mu_e)**2)
                stats['Lin CCC'] = float(ccc) if not np.isnan(ccc) else None
            else:
                stats['Lin CCC'] = None
            mae = np.nanmean(np.abs(norm_metrics - norm_experts))
            stats['MAE'] = float(mae) if not np.isnan(mae) else None
            
            safe_block = re.sub(r'\W+', '_', block_name)
            safe_metric = re.sub(r'\W+', '_', metric)
            filename = f"bland_altman_{safe_block}_{safe_metric}.png"
            plot_bland_altman(norm_metrics, norm_experts, f"Bland-Altman Plot for {metric} in {block_name}", filename)
        else:
            stats['Lin CCC'] = None
            stats['MAE'] = None

        per_block[block_name] = stats
    block_stats[metric]['per_block'] = per_block

# 3. Expert stats
expert_stats = {}
global_expert_stats = compute_stats(expert_avgs_all, 'expert_score')
avg_num_experts_global = np.mean(expert_lens_all) if expert_lens_all else np.nan
global_expert_stats['Среднее число экспертов'] = float(avg_num_experts_global) if not np.isnan(avg_num_experts_global) else None
hist_global = {}
for k, v in expert_lens_histogram.items():
    hist_global[f'На одно судебное решение были оценки от {k}-го эксперта' + ('ов' if k > 1 else '')] = v
global_expert_stats.update(hist_global)
global_expert_stats['Fleiss κ'] = None  # оставлено как None: если нужно — добавим расчёт

# ICC(2,k) — считаем по подмножеству документов с одинаковым числом рецензентов (мода)
if expert_long:
    expert_df = pd.DataFrame(expert_long)
    logger.info(f"expert_df shape: {expert_df.shape}, columns: {expert_df.columns.tolist()}")
    if 'num_experts' in expert_df.columns and 'expert_score' in expert_df.columns:
        # modal number of raters
        mode_raters = int(expert_df['num_experts'].mode().iloc[0]) if not expert_df['num_experts'].mode().empty else 0
        if mode_raters > 1:
            subset_doc_ids = expert_df.groupby('doc_id')['num_experts'].first()
            subset_doc_ids = subset_doc_ids[subset_doc_ids == mode_raters].index
            df_icc = expert_df[expert_df['doc_id'].isin(subset_doc_ids)].copy()
            if not df_icc.empty and df_icc['doc_id'].nunique() > 1:
                # нормализация и ANOVA для ICC(2,k)
                norm_expert_scores = minmax_scale_array(df_icc['expert_score'].values, *metric_ranges['expert_score'])
                df_icc['norm_expert_score'] = norm_expert_scores
                try:
                    model = smf.ols('norm_expert_score ~ C(doc_id) + C(block_name)', data=df_icc).fit()
                    anova_table = sm.stats.anova_lm(model, typ=2)
                    msb = anova_table.loc['C(doc_id)', 'sum_sq'] / anova_table.loc['C(doc_id)', 'df'] if 'C(doc_id)' in anova_table.index else 0
                    mse = anova_table.loc['Residual', 'sum_sq'] / anova_table.loc['Residual', 'df'] if 'Residual' in anova_table.index else 0
                    n_raters = mode_raters
                    icc2k = (msb - mse) / (msb + (n_raters - 1) * mse) if mse > 0 else np.nan
                    global_expert_stats['ICC(2,k)'] = float(icc2k) if not np.isnan(icc2k) else None
                    global_expert_stats['ICC_n_raters_used'] = int(n_raters)
                    global_expert_stats['ICC_docs_used'] = int(df_icc['doc_id'].nunique())
                except Exception as e:
                    logger.error(f"Ошибка при расчёте ICC(2,k): {e}")
                    global_expert_stats['ICC(2,k)'] = None
            else:
                global_expert_stats['ICC(2,k)'] = None
        else:
            global_expert_stats['ICC(2,k)'] = None
    else:
        global_expert_stats['ICC(2,k)'] = None
else:
    global_expert_stats['ICC(2,k)'] = None

# Per-block expert stats
per_block_expert = {}
for block_name in blocks:
    block_data = [row for row in expert_long if row['block_name'] == block_name]
    if block_data:
        block_df = pd.DataFrame(block_data)
        block_avg_stats = compute_stats(block_df['expert_score'], 'expert_score')
        avg_num_experts_block = np.mean(block_df['num_experts']) if 'num_experts' in block_df else np.nan
        block_avg_stats['Среднее число экспертов'] = float(avg_num_experts_block) if not np.isnan(avg_num_experts_block) else None
        hist_block_counter = Counter(block_df['num_experts'])
        hist_block = {}
        for k, v in hist_block_counter.items():
            hist_block[f'На одно судебное решение были оценки от {k}-го эксперта' + ('ов' if k > 1 else '')] = v
        block_avg_stats.update(hist_block)
        

        
        # ICC(2,k) для блока: используем модальное число рецензентов внутри блока
        if 'num_experts' in block_df.columns:
            try:
                mode_raters_block = int(block_df['num_experts'].mode().iloc[0]) if not block_df['num_experts'].mode().empty else 0
            except Exception:
                mode_raters_block = 0
        else:
            mode_raters_block = 0
        if mode_raters_block > 1:
            subset_doc_ids = block_df.groupby('doc_id')['num_experts'].first()
            subset_doc_ids = subset_doc_ids[subset_doc_ids == mode_raters_block].index
            df_icc_block = block_df[block_df['doc_id'].isin(subset_doc_ids)].copy()
            if not df_icc_block.empty and df_icc_block['doc_id'].nunique() > 1:
                norm_block_scores = minmax_scale_array(df_icc_block['expert_score'].values, *metric_ranges['expert_score'])
                df_icc_block['norm_expert_score'] = norm_block_scores
                try:
                    model_block = smf.ols('norm_expert_score ~ C(doc_id)', data=df_icc_block).fit()
                    anova_table_block = sm.stats.anova_lm(model_block, typ=2)
                    msb_block = anova_table_block.loc['C(doc_id)', 'sum_sq'] / anova_table_block.loc['C(doc_id)', 'df'] if 'C(doc_id)' in anova_table_block.index else 0
                    mse_block = anova_table_block.loc['Residual', 'sum_sq'] / anova_table_block.loc['Residual', 'df'] if 'Residual' in anova_table_block.index else 0
                    n_raters_block = mode_raters_block
                    icc2k_block = (msb_block - mse_block) / (msb_block + (n_raters_block - 1) * mse_block) if mse_block > 0 else np.nan
                    block_avg_stats['ICC(2,k)'] = float(icc2k_block) if not np.isnan(icc2k_block) else None
                except Exception as e:
                    logger.error(f"Ошибка при расчёте ICC блока {block_name}: {e}")
                    block_avg_stats['ICC(2,k)'] = None
            else:
                block_avg_stats['ICC(2,k)'] = None
        else:
            block_avg_stats['ICC(2,k)'] = None

        per_block_expert[block_name] = block_avg_stats
    else:
        per_block_expert[block_name] = {'Число оценок': 0, 'Средняя оценка': None, 'ICC(2,k)': None}

# 4. Документные статистики
doc_stats = {}
for metric in doc_metrics:
    stats = compute_stats(doc_all_values[metric], metric)
    corr_dict = compute_correlation(doc_correlation_pairs[metric], metric)
    for key, (r, p) in corr_dict.items():
        stats[f'{key} r'] = r
        stats[f'{key} p'] = p
    
    # Дополнительно: CCC, MAE и Bland-Altman для документных метрик
    pairs = doc_correlation_pairs[metric]
    pairs = [(m, e) for m, e in pairs if not (np.isnan(m) or np.isnan(e))]
    if len(pairs) > 1:
        metrics_vals, expert_vals = zip(*pairs)
        min_v_m, max_v_m = metric_ranges.get(metric, (None, None))
        min_v_e, max_v_e = metric_ranges.get('expert_score', (None, None))
        norm_metrics = minmax_scale_array(metrics_vals, min_v_m, max_v_m)
        norm_experts = minmax_scale_array(expert_vals, min_v_e, max_v_e)
        if np.nanstd(norm_metrics) > 0 and np.nanstd(norm_experts) > 0:
            pearson_corr = pearsonr(norm_metrics, norm_experts)[0]
            mu_m = np.nanmean(norm_metrics)
            mu_e = np.nanmean(norm_experts)
            sigma_m = np.nanstd(norm_metrics)
            sigma_e = np.nanstd(norm_experts)
            ccc = 2 * pearson_corr * sigma_m * sigma_e / (sigma_m**2 + sigma_e**2 + (mu_m - mu_e)**2)
            stats['Lin CCC'] = float(ccc) if not np.isnan(ccc) else None
        else:
            stats['Lin CCC'] = None
        mae = np.nanmean(np.abs(norm_metrics - norm_experts))
        stats['MAE'] = float(mae) if not np.isnan(mae) else None
        
        safe_metric = re.sub(r'\W+', '_', metric)
        filename = f"bland_altman_doc_{safe_metric}.png"
        plot_bland_altman(norm_metrics, norm_experts, f"Bland-Altman Plot for Doc {metric}", filename)
    else:
        stats['Lin CCC'] = None
        stats['MAE'] = None

    doc_stats[metric] = stats

doc_expert_stats = compute_stats(doc_avg_experts, 'expert_score')

# Сохранение
output_data = {
    'block_stats': block_stats,
    'expert_stats': expert_stats,
    'doc_stats': doc_stats,
    'doc_expert_stats': doc_expert_stats,
    'per_block_expert': per_block_expert
}

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2, default=str)

# Краткий вывод
logger.info(f"JSON сохранён: {output_file}")
for block_name in blocks:
    logger.info(f"Корреляции для блока {block_name}: {per_block_expert[block_name]}")
    logger.info(f"ICC(2,k) ({block_name}): {per_block_expert[block_name].get('ICC(2,k)', 'N/A')}")
