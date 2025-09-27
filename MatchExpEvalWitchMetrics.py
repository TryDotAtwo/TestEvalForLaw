import json
from collections import defaultdict

# Файлы для чтения
output_file = 'output_with_llm_eval_gpt-4.1-mini.json'
evaluations_file = 'evaluations_with_source.json'
output_merged_file = 'merged_output_with_expert_eval_all.json'  # Новый файл для версии с всеми оценками

# Маппинг блоков из output к ключам в evaluation
block_to_eval_key = {
    'Требования истца': 'plaintiff_claims',
    'Аргументы истца': 'plaintiff_arguments',
    'Аргументы ответчика': 'defendant_arguments',
    'Оценка судом представленных сторонами доказательств': 'evaluation_of_evidence',
    'Логические шаги в рассуждениях судьи и промежуточные выводы': 'intermediate_conclusions',
    'Применимые в судебном деле нормы права': 'applicable_laws',
    'Решение суда': 'judgment_summary'
}

# Загрузка данных
with open(output_file, 'r', encoding='utf-8') as f:
    output_data = json.load(f)

with open(evaluations_file, 'r', encoding='utf-8') as f:
    evaluations_data = json.load(f)

# Создание словаря: для каждого source — список всех evaluation (для поддержки нескольких экспертов)
evaluations_by_source = defaultdict(list)
for item in evaluations_data:
    evaluations_by_source[item['source']].append(item['evaluation'])

# Инициализация счетчиков для отладки
total_objects = len(output_data)
matched_objects = 0
total_sections = 0
filled_sections = 0
unfilled_sections = 0
total_expert_evals = 0  # Общее число экспертных оценок (с учётом дубликатов)

# Обработка каждого объекта в output
merged_data = []
for item in output_data:
    source_text = item.get('source_text', '')
    if source_text in evaluations_by_source:
        matched_objects += 1
        all_evals = evaluations_by_source[source_text]
        total_expert_evals += len(all_evals)  # Считаем все оценки для этого source
        
        # Копируем item
        merged_item = item.copy()
        
        # Заполняем expert_eval для каждого блока: собираем все scores по ключу из всех evals
        for block_name, block_data in merged_item.items():
            if block_name in block_to_eval_key and isinstance(block_data, dict) and 'expert_eval' in block_data:
                eval_key = block_to_eval_key[block_name]
                all_scores = []
                for single_eval in all_evals:
                    score = single_eval.get(eval_key)
                    if score is not None:
                        all_scores.append(score)
                if all_scores:  # Если есть хотя бы одна оценка
                    block_data['expert_eval'] = all_scores  # Список всех оценок, напр. [5, 4, 5]
                    filled_sections += 1
                else:
                    unfilled_sections += 1
                total_sections += 1
        merged_data.append(merged_item)
    else:
        # Если нет совпадения, добавляем как есть
        merged_data.append(item.copy())
        # Для нематчинга считаем секции приблизительно
        total_sections += len([k for k in item if k in block_to_eval_key])

# Сохранение merged JSON с ВСЕМИ оценками
with open(output_merged_file, 'w', encoding='utf-8') as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=2)

# Отладка (обновлённая)
print(f"Общее количество объектов в output: {total_objects}")
print(f"Объектов с совпадающим source_text: {matched_objects}")
print(f"Объектов без совпадения: {total_objects - matched_objects}")
print(f"Общее количество экспертных оценок (с дубликатами): {total_expert_evals}")
print(f"Общее количество секций для заполнения: {total_sections}")
print(f"Секций, куда успешно добавлены экспертные оценки (все от всех экспертов): {filled_sections}")
print(f"Секций, которые не удалось заполнить (нет оценок по ключу): {unfilled_sections}")

if unfilled_sections > 0:
    print("Предупреждение: Некоторые секции не заполнены из-за отсутствующих ключей в evaluations (нормально).")
else:
    print("Все возможные экспертные оценки были успешно добавлены.")

# Дополнительно: Проверка на дубликаты sources в evaluations
from collections import Counter
source_counts = Counter(item['source'] for item in evaluations_data)
duplicates = sum(count - 1 for count in source_counts.values() if count > 1)
print(f"\nДубликатов source в evaluations (число лишних оценок): {duplicates}")
if duplicates > 0:
    print("Эти дубликаты теперь учтены как оценки от нескольких экспертов.")