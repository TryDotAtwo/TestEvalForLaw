import json
import asyncio
import aiofiles
import g4f
from tqdm.asyncio import tqdm_asyncio
import re
from typing import Dict, Any, Optional, Tuple
import logging
import glob
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
# g4f.debug.logging = True
# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("llm_evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

import os
import certifi

# Заставляем curl использовать актуальный файл сертификатов
os.environ["CURL_CA_BUNDLE"] = "D:\cacert.pem"


# --- Blocks for Evaluation ---
BLOCKS = [
    "Требования истца",
    "Аргументы истца",
    "Аргументы ответчика",
    "Оценка судом представленных сторонами доказательств",
    "Логические шаги в рассуждениях судьи и промежуточные выводы",
    "Применимые в судебном деле нормы права",
    "Решение суда"
]

# --- Prompt Template (тексты блоков вставляются "как есть") ---
PROMPT_TEMPLATE = """
Перед вами данные, которые были извлечены большой языковой моделью из текста судебных решений.

Проверяются следующие значения:

Требования истца (plaintiff_claims): на основе вводных данных определите требования истца по рассматриваемому делу
Аргументы истца (plaintiff_arguments): на основе вводных данных определите аргументы истца по рассматриваемому делу
Аргументы ответчика (defendant_arguments): на основе вводных данных определите аргументы ответчика по рассматриваемому делу
Оценка судом представленных сторонами доказательств (evaluation_of_evidence): на основе вводных данных определите оценку судом представленных сторонами доказательств по рассматриваемому делу
Логика рассуждений судьи (intermediate_conclusions): на основе вводных данных определите логику рассуждений судьи по рассматриваемому делу
Нормы права, которые применяются в судебном деле (applicable_laws): на основе вводных данных определите применимые нормы права по рассматриваемому делу
Окончательное решение суда (judgment_summary): на основе вводных данных определите краткое содержание окончательного решения судьи по рассматриваемому делу
Оригинальный текст судебного дела (source) показан в начале каждого образца.

**Оригинальный текст судебного дела (source):**
{source_text}

**Извлеченные данные:**
"Требования истца": \"\"\"{plaintiff_claims}\"\"\"
"Аргументы истца": \"\"\"{plaintiff_arguments}\"\"\"
"Аргументы ответчика": \"\"\"{defendant_arguments}\"\"\"
"Оценка судом представленных сторонами доказательств": \"\"\"{evaluation_of_evidence}\"\"\"
"Логические шаги в рассуждениях судьи и промежуточные выводы": \"\"\"{intermediate_conclusions}\"\"\"
"Применимые в судебном деле нормы права": \"\"\"{applicable_laws}\"\"\"
"Решение суда": \"\"\"{judgment_summary}\"\"

Ваша задача:
Сверить данные, которые были извлечены моделью, с содержанием оригинального текста судебного дела и оценить по шкале от 1 до 5 баллов, насколько хорошо модель справилась со своей задачей:

1: модель не выполнила задание
2: модель плохо выполнила задание
3: модель частично выполнила задание
4: модель хорошо выполнила задание
5: модель полностью выполнила задание


Верните ответ строго в формате JSON с блоками:
{{
    "Требования истца": {{"score": integer_from_1_to_5, "reason": "краткое объяснение"}},
    "Аргументы истца": {{"score": integer_from_1_to_5, "reason": "краткое объяснение"}},
    "Аргументы ответчика": {{"score": integer_from_1_to_5, "reason": "краткое объяснение"}},
    "Оценка судом представленных сторонами доказательств": {{"score": integer_from_1_to_5, "reason": "краткое объяснение"}},
    "Логические шаги в рассуждениях судьи и промежуточные выводы": {{"score": integer_from_1_to_5, "reason": "краткое объяснение"}},
    "Применимые в судебном деле нормы права": {{"score": integer_from_1_to_5, "reason": "краткое объяснение"}},
    "Решение суда": {{"score": integer_from_1_to_5, "reason": "краткое объяснение"}}
}}
{retry_message}
"""

def is_valid_block(block):
    # Проверяем, что внутри есть score и reason
    if not isinstance(block, dict):
        return False
    if "score" not in block or "reason" not in block:
        return False
    # score — int 1..5
    if not isinstance(block["score"], int) or not (1 <= block["score"] <= 5):
        return False
    # reason — str
    if not isinstance(block["reason"], str):
        return False
    return True

def extract_json_from_response(response: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    expected_keys = BLOCKS

    # Очистка ответа от оберток
    cleaned_response = response.strip()
    if cleaned_response.startswith("```json"):
        cleaned_response = cleaned_response[7:]
    if cleaned_response.endswith("```"):
        cleaned_response = cleaned_response[:-3]
    cleaned_response = cleaned_response.strip()

    # Проверяем, является ли весь ответ JSON
    try:
        if cleaned_response.startswith("{") and cleaned_response.endswith("}"):
            data = json.loads(cleaned_response)
            if not all(k in data for k in expected_keys):
                missing_keys = [k for k in expected_keys if k not in data]
                return None, f"Отсутствуют ключи: {', '.join(missing_keys)}. Верните ТОЛЬКО валидный JSON с ключами {', '.join(expected_keys)}."
            for block in expected_keys:
                if not is_valid_block(data[block]):
                    return None, f"Неверная структура блока '{block}': ожидается {{'score': int (1-5), 'reason': str}}. Верните ТОЛЬКО валидный JSON."
            return data, None
    except json.JSONDecodeError as e:
        logger.debug(f"Failed to parse entire response: {e}")

    # Поиск JSON-объектов с учетом вложенных скобок
    def find_json_objects(text: str) -> list[str]:
        objects = []
        stack = []
        start_idx = None
        i = 0
        while i < len(text):
            char = text[i]
            if char == "{":
                if not stack:
                    start_idx = i
                stack.append(char)
            elif char == "}":
                if stack:
                    stack.pop()
                    if not stack and start_idx is not None:
                        candidate = text[start_idx:i+1]
                        try:
                            json.loads(candidate)  # Проверяем, валиден ли JSON
                            objects.append(candidate)
                        except json.JSONDecodeError:
                            pass
                        start_idx = None
            elif char == '"' and i > 0 and text[i-1] != "\\":  # Обработка строк
                i += 1
                while i < len(text) and (text[i] != '"' or text[i-1] == "\\"):
                    i += 1
            i += 1
        return objects

    matches = find_json_objects(cleaned_response)
    logger.debug(f"Found {len(matches)} JSON objects: {matches}")

    for match in matches:
        try:
            data = json.loads(match)
            if not all(k in data for k in expected_keys):
                missing_keys = [k for k in expected_keys if k not in data]
                logger.debug(f"Skipping JSON with missing keys: {missing_keys}")
                continue
            for block in expected_keys:
                if not is_valid_block(data[block]):
                    logger.debug(f"Invalid block structure for '{block}' in JSON: {match}")
                    return None, f"Неверная структура блока '{block}': ожидается {{'score': int (1-5), 'reason': str}}. Верните ТОЛЬКО валидный JSON."
            return data, None
        except json.JSONDecodeError as e:
            logger.debug(f"Failed to parse JSON match: {match}, error: {e}")
            continue

    return None, f"Не найден валидный JSON с ключами {', '.join(expected_keys)}. Верните ТОЛЬКО валидный JSON без лишнего текста."


# --- Parse LLM response ---
def parse_llm_response(response: str) -> Tuple[Dict[str, Dict[str, Any]], Optional[str]]:
    data, error_message = extract_json_from_response(response)
    if data is not None:
        return data, None
    return {block: {"score": 3, "reason": "Invalid JSON response"} for block in BLOCKS}, error_message

# --- Async LLM call ---
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(Exception),
    before_sleep=lambda retry_state: logger.warning(f"Retry attempt {retry_state.attempt_number} after error...")
)


async def call_llm(prompt: str) -> str:
    # Оборачиваем синхронный вызов в отдельный поток
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,  # None → использует ThreadPoolExecutor по умолчанию
        lambda: g4f.ChatCompletion.create(
            model="gpt-5-nano",
            # provider="DeepInfra",
            messages=[{"role": "user", "content": prompt}],
            timeout=60
        )
    )
    # print("\n----- RAW LLM RESPONSE -----", flush=True)
    # print(response, flush=True)
    # print("-----------------------------\n", flush=True)
    if not response:
        raise ValueError("Empty response from LLM")
    return response

# --- Evaluate a single case ---
async def evaluate_case(data: Dict[str, Any], retry_count: int = 0) -> Dict[str, Dict[str, Any]]:
    if retry_count >= 2:
        logger.error(f"Max retries reached for case {data['id_sud_resh']}. Returning default evaluation.")
        return {block: {"score": 3, "reason": "Invalid JSON response after retry"} for block in BLOCKS}

    retry_message = ""

    try:
        prompt = PROMPT_TEMPLATE.format(
            source_text=data['source_text'],
            retry_message=retry_message,
            plaintiff_claims=data['Требования истца']['text'],
            plaintiff_arguments=data['Аргументы истца']['text'],
            defendant_arguments=data['Аргументы ответчика']['text'],
            evaluation_of_evidence=data['Оценка судом представленных сторонами доказательств']['text'],
            intermediate_conclusions=data['Логические шаги в рассуждениях судьи и промежуточные выводы']['text'],
            applicable_laws=data['Применимые в судебном деле нормы права']['text'],
            judgment_summary=data['Решение суда']['text']
        )

        response = await call_llm(prompt)
        parsed, error_message = parse_llm_response(response)
        if error_message is None:
            return parsed
        
        # Формируем новый промпт с указанием ошибки
        retry_message = f"Произошла ошибка в предыдущем ответе: {error_message} Верните ТОЛЬКО валидный JSON с указанной структурой."
        logger.warning(f"Invalid JSON for case {data['id_sud_resh']}: {error_message}. Retrying (attempt {retry_count + 1})...")

        prompt = PROMPT_TEMPLATE.format(
            source_text=data['source_text'],
            retry_message=retry_message,
            plaintiff_claims=data['Требования истца']['text'],
            plaintiff_arguments=data['Аргументы истца']['text'],
            defendant_arguments=data['Аргументы ответчика']['text'],
            evaluation_of_evidence=data['Оценка судом представленных сторонами доказательств']['text'],
            intermediate_conclusions=data['Логические шаги в рассуждениях судьи и промежуточные выводы']['text'],
            applicable_laws=data['Применимые в судебном деле нормы права']['text'],
            judgment_summary=data['Решение суда']['text']
        )

        response = await call_llm(prompt)
        parsed, error_message = parse_llm_response(response)
        if error_message is None:
            return parsed
        logger.warning(f"Retry failed for case {data['id_sud_resh']}: {error_message}. Returning default evaluation.")
        return {block: {"score": 3, "reason": "Invalid JSON response after retry"} for block in BLOCKS}

    except Exception as e:
        logger.warning(f"Error evaluating case {data['id_sud_resh']}: {e}. Retrying (attempt {retry_count + 1})...")
        return await evaluate_case(data, retry_count=retry_count + 1)

# --- Process a single case ---
async def process_case(data: Dict[str, Any]) -> Dict[str, Any]:
    evaluations = await evaluate_case(data)
    for block, eval_data in evaluations.items():
        if block in data:
            data[block]["metrics"]["LLM_Evaluation_Score"] = eval_data["score"]
            data[block]["metrics"]["LLM_Evaluation_Reason"] = eval_data["reason"]
    return data

# --- Find latest checkpoint ---
def find_latest_checkpoint(output_file: str) -> tuple[str, list[Dict[str, Any]]]:
    checkpoint_pattern = f"{output_file}.checkpoint_*.json"
    checkpoints = sorted(glob.glob(checkpoint_pattern),
                         key=lambda x: int(re.search(r'checkpoint_(\d+)', x).group(1)) if re.search(r'checkpoint_(\d+)', x) else 0)
    if checkpoints:
        latest_checkpoint = checkpoints[-1]
        with open(latest_checkpoint, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Found latest checkpoint: {latest_checkpoint}")
        return latest_checkpoint, data
    return None, []
from asyncio import Semaphore

# --- Process JSON file with real-time progress ---
async def process_json(input_file: str, output_file: str, max_concurrent: int = 60, checkpoint_interval: int = 100):
    latest_checkpoint, processed_data = find_latest_checkpoint(output_file)
    processed_ids = {item['id_sud_resh'] for item in processed_data}

    async with aiofiles.open(input_file, 'r', encoding='utf-8') as f:
        data_list = json.loads(await f.read())

    data_list = [data for data in data_list if data['id_sud_resh'] not in processed_ids]
    output_list = processed_data
    start_index = len(processed_data)

    semaphore = asyncio.Semaphore(max_concurrent)  # Ограничение параллельных задач

    async def sem_task(data):
        async with semaphore:
            return await process_case(data)

    tasks = [sem_task(data) for data in data_list]

    from tqdm.asyncio import tqdm_asyncio
    completed_count = 0

    for f in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Processing cases"):
        result = await f
        output_list.append(result)
        completed_count += 1

        # Сохраняем чекпоинт каждые checkpoint_interval кейсов
        if completed_count % checkpoint_interval == 0:
            checkpoint_file = f"{output_file}.checkpoint_{start_index + completed_count}.json"
            async with aiofiles.open(checkpoint_file, 'w', encoding='utf-8') as cf:
                await cf.write(json.dumps(output_list, ensure_ascii=False, indent=2))
            logger.info(f"Checkpoint saved at {checkpoint_file}")

    # Финальный результат
    async with aiofiles.open(output_file, 'w', encoding='utf-8') as f:
        await f.write(json.dumps(output_list, ensure_ascii=False, indent=2))
    logger.info(f"Final output saved to {output_file}")

# --- Main execution ---
if __name__ == "__main__":
    asyncio.run(process_json(
        input_file="output — копия.json",
        output_file="output_with_llm_eval.json",
        checkpoint_interval=100
    ))