from datasets import load_dataset
import json
from collections import defaultdict

# Загружаем датасет
dataset = load_dataset("lawful-good-project/sud-resh-benchmark", split="train")

# Соответствие инструкций блокам
instruction_to_block = {
    "На основе вводных данных определите требования истца по рассматриваемому делу.": "Требования истца",
    "На основе вводных данных определите аргументы истца по рассматриваемому делу.": "Аргументы истца",
    "На основе вводных данных определите аргументы ответчика по рассматриваемому делу.": "Аргументы ответчика",
    "На основе вводных данных определите оценку судом представленных сторонами доказательств по рассматриваемому делу.": "Оценка судом представленных сторонами доказательств",
    "На основе вводных данных определите логику рассуждений судьи по рассматриваемому делу.": "Логические шаги в рассуждениях судьи и промежуточные выводы",
    "На основе вводных данных определите применимые нормы права по рассматриваемому делу.": "Применимые в судебном деле нормы права",
    "На основе вводных данных определите краткое содержание окончательного решения судьи по рассматриваемому делу.": "Решение суда"
}

blocks = [
    "Требования истца", "Аргументы истца", "Аргументы ответчика",
    "Оценка судом представленных сторонами доказательств",
    "Логические шаги в рассуждениях судьи и промежуточные выводы",
    "Применимые в судебном деле нормы права", "Решение суда"
]

# Группируем по source
source_groups = defaultdict(list)
for example in dataset:
    source = example.get("source", "")
    source_groups[source].append(example)

output_list = []

for idx, (source_text, examples) in enumerate(source_groups.items(), start=1):
    data = {"id_sud_resh": f"SR{idx:05d}"}
    
    # Инициализация всех блоков пустыми
    for block in blocks:
        data[block] = {
            "text": "",
            "expert_eval": [],
            "metrics": {}
        }
    
    # Заполняем блоки из всех примеров с этим source
    for example in examples:
        block_name = instruction_to_block.get(example["instruction"])
        if block_name:
            data[block_name]["text"] = example.get("correct_answer", "")
    
    # Источник текста
    data["source_text"] = source_text
    
    output_list.append(data)

# Сохраняем JSON
with open("benchmark_input_grouped.json", "w", encoding="utf-8") as f:
    json.dump(output_list, f, ensure_ascii=False, indent=2)

print(f"Сгенерировано {len(output_list)} уникальных примеров по source.")
