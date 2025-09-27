from pathlib import Path
import json

# Путь к директории с экспертными оценками
expert_eval_dir = Path("Expert_eval")

# Шаг 1: Собираем ВСЕ оценки (с дубликатами) в список с ID
all_evaluations = []
for subdir in expert_eval_dir.iterdir():
    if subdir.is_dir():
        for json_file in subdir.glob("*.json"):
            with open(json_file, 'r', encoding='utf-8') as f:
                eval_data = json.load(f)
                for eval_id, eval_obj in eval_data.items():
                    all_evaluations.append({
                        "id": eval_id,
                        "evaluation": eval_obj
                    })

total_evaluations = len(all_evaluations)
print(f"Найдено оценок (с дубликатами): {total_evaluations}")

# Отладка: Выводим примеры eval_id
print("Примеры eval_id из оценок:")
for i, eval_item in enumerate(all_evaluations[:3]):
    print(f"  {eval_item['id']}")

# Шаг 2: Загружаем inputTEXT.json для source по ID
input_file = Path("inputTEXT.json")
if not input_file.exists():
    print("Ошибка: Файл inputTEXT.json не найден!")
    exit(1)

with open(input_file, 'r', encoding='utf-8') as f:
    input_data = json.load(f)

# Создаем словарь {id: source_value} для быстрого поиска
source_map = {}
for item in input_data:
    item_id = item["id"]
    source_value = item.get("source", "")  # Берем значение поля "source"
    source_map[item_id] = source_value

print(f"В source_map {len(source_map)} ID")

# Отладка: Выводим примеры из source_map
print("Примеры из source_map:")
for i, (item_id, source) in enumerate(list(source_map.items())[:3]):
    print(f"  {item_id[:20]}... -> '{source}'")

# Проверка пересечения eval_ids и source_map
eval_ids = set(item["id"] for item in all_evaluations)
source_ids = set(source_map.keys())
intersection = eval_ids.intersection(source_ids)
print(f"Пересечение ID: {len(intersection)}")
if intersection:
    print("Пример пересекающегося ID:", list(intersection)[0])
else:
    print("Нет пересекающихся ID между оценками и inputTEXT.")

# Шаг 3: Подвязываем source к каждой оценке напрямую по ID
matched = 0
not_matched = 0
debug_not_matched = []  # Для отладки первых 5 несовпадений
output_data = []
for eval_item in all_evaluations:
    eval_id = eval_item["id"]
    source_value = source_map.get(eval_id, "")
    if source_value != "":
        matched += 1
        output_item = {
            "id": eval_id,
            "source": source_value,
            "evaluation": eval_item["evaluation"]
        }
    else:
        not_matched += 1
        output_item = {
            "id": eval_id,
            "source": "",
            "evaluation": eval_item["evaluation"]
        }
        # Отладка: собираем первые 5
        if len(debug_not_matched) < 5:
            debug_not_matched.append(f"{eval_id[:20]}... (no source_map)")

    output_data.append(output_item)

print(f"Совпало source с оценками: {matched}")
print(f"Не совпало source с оценками: {not_matched}")

if debug_not_matched:
    print("Примеры несовпадений (первые 5):")
    for debug in debug_not_matched:
        print(f"  {debug}")

# Шаг 4: Сохраняем в единый JSON файл
output_file = Path("evaluations_with_source.json")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print(f"Новый JSON с оценками и подвязанными source сохранен в {output_file}")