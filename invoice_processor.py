import os
import re
import csv
import json
import logging
from io import BytesIO
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import pytesseract
import fitz  # PyMuPDF
import requests
import base64
from fuzzywuzzy import fuzz, process  # Для нечеткого сравнения

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class InvoiceProcessor:
    def __init__(self, config_dir="config", templates_dir="templates"):
        self.config_dir = config_dir
        self.templates_dir = templates_dir
        self.validation_rules = self.load_validation_rules()

        # Инициализация Яндекс OCR API
        self.yandex_api_key = os.getenv("YANDEX_API_KEY")
        if self.yandex_api_key:
            logger.info("Яндекс OCR API ключ найден")
        else:
            logger.warning("Яндекс OCR API ключ не найден. Установите переменную окружения YANDEX_API_KEY")

        # Создаем директорию для сохранения JSON ответов
        self.debug_dir = "debug_responses"
        os.makedirs(self.debug_dir, exist_ok=True)

        # Загрузка шаблонов таблиц
        self.table_templates = self.load_table_templates()

    def load_table_templates(self) -> List[Dict]:
        """Загружает шаблоны таблиц из конфигурационного файла."""
        config_path = os.path.join(self.config_dir, "table_templates.json")
        templates = []
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    templates = data.get("templates", [])
                logger.info(f"Загружено {len(templates)} шаблонов таблиц")
            else:
                logger.warning(f"Файл конфигурации шаблонов не найден: {config_path}")
                # Создаем базовый шаблон, если файл отсутствует
                templates = [{
                    "name": "Стандартный шаблон",
                    "headers": ["№", "Артикул", "Товары (работы, услуги)", "Кол-во", "Ед.", "Цена", "Сумма"],
                    "required_headers": ["Товары (работы, услуги)", "Кол-во", "Цена", "Сумма"],
                    "min_similarity": 85
                }]
        except Exception as e:
            logger.error(f"Ошибка загрузки шаблонов таблиц: {e}")
            templates = []
        return templates

    def load_validation_rules(self):
        """Загружает правила валидации из CSV файла."""
        rules = {}
        csv_path = os.path.join(self.config_dir, "validation_rules.csv")

        if not os.path.exists(csv_path):
            self.create_default_validation_rules(csv_path)

        try:
            with open(csv_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file, delimiter='|')
                for row in reader:
                    rules[row['field']] = {
                        'validation_type': row['validation_type'],
                        'rule': row['rule'],
                        'message': row['message']
                    }
        except Exception as e:
            logger.error(f"Ошибка загрузки правил валидации: {e}")

        return rules

    def create_default_validation_rules(self, csv_path):
        """Создает файл правил валидации по умолчанию."""
        default_rules = [
            {
                'field': 'price',
                'validation_type': 'number',
                'rule': '>0',
                'message': 'Цена должна быть положительным числом'
            },
            {
                'field': 'quantity',
                'validation_type': 'number',
                'rule': '>0',
                'message': 'Количество должно быть положительным числом'
            }
        ]

        try:
            with open(csv_path, 'w', encoding='utf-8', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=['field', 'validation_type', 'rule', 'message'], delimiter='|')
                writer.writeheader()
                for rule in default_rules:
                    writer.writerow(rule)
        except Exception as e:
            logger.error(f"Ошибка создания правил валидации: {e}")

    def ocr_image(self, image, lang='ru'):
        """Распознаёт текст с изображения."""
        try:
            # Если доступен Яндекс OCR API, используем его
            if self.yandex_api_key:
                logger.info("Использую Яндекс OCR API для распознавания")
                return self.yandex_ocr_api(image)
            else:
                # Используем Tesseract OCR
                logger.info("Использую Tesseract OCR для распознавания")
                custom_config = r'--oem 3 --psm 6'
                text = pytesseract.image_to_string(image, lang=lang, config=custom_config)
                return text.strip()
        except Exception as e:
            logger.error(f"Ошибка OCR: {e}")
            return ""

    def yandex_ocr_api(self, image):
        """Распознаёт текст с изображения с помощью Яндекс OCR API."""
        try:
            if not self.yandex_api_key:
                logger.warning("API ключ Яндекс OCR не установлен")
                return ""

            # Конвертируем изображение в JPEG
            if isinstance(image, str):
                image = Image.open(image)

            # Убедимся, что изображение в RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Сохраняем изображение в байты
            img_byte_arr = BytesIO()
            image.save(img_byte_arr, format='JPEG', quality=95)
            img_bytes = img_byte_arr.getvalue()

            # Кодируем в base64
            encoded_image = base64.b64encode(img_bytes).decode('utf-8')

            # Подготавливаем запрос
            url = "https://ocr.api.cloud.yandex.net/ocr/v1/recognizeText"
            headers = {
                "Authorization": f"Api-Key {self.yandex_api_key}",
                "Content-Type": "application/json",
                "x-data-logging-enabled": "true"
            }

            data = {
                "mimeType": "JPEG",
                "languageCodes": ["ru", "en"],
                "model": "table",
                "content": encoded_image
            }

            # Отправляем запрос
            response = requests.post(url, headers=headers, data=json.dumps(data))
            response.raise_for_status()

            # Обрабатываем ответ
            result = response.json()

            # Сохраняем полный JSON ответ в файл
            self.save_yandex_response(result, img_bytes)

            # Извлекаем текст из результата
            text = self.extract_text_from_yandex_response(result)

            return text

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                logger.error("Ошибка 403: Проверьте API ключ и права доступа к Яндекс OCR API")
            else:
                logger.error(f"HTTP ошибка при использовании Яндекс OCR API: {e}")
            return ""
        except Exception as e:
            logger.error(f"Ошибка при использовании Яндекс OCR API: {e}")
            return ""

    def extract_text_from_yandex_response(self, response_data: Dict) -> str:
        """Извлекает весь текст из JSON-ответа Яндекс OCR."""
        text_parts = []
        try:
            blocks = response_data.get("result", {}).get("blocks", [])
            for block in blocks:
                # Извлекаем текст из текстовых блоков
                if "text" in block:
                    text_parts.append(block["text"])
                # Извлекаем текст из таблиц
                elif "table" in block:
                    table = block["table"]
                    for row in table.get("rows", []):
                        for cell in row.get("cells", []):
                            if "text" in cell:
                                text_parts.append(cell["text"])
            return "\n".join(text_parts)
        except Exception as e:
            logger.error(f"Ошибка извлечения текста из ответа Yandex OCR: {e}")
            return ""


    def save_yandex_response(self, response_data, image_bytes):
        """Сохраняет полный JSON ответ от Яндекс OCR API в файл."""
        try:
            # Создаем уникальное имя файла с временной меткой
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"yandex_ocr_response_{timestamp}.json"
            filepath = os.path.join(self.debug_dir, filename)

            # Сохраняем JSON ответ
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(response_data, f, ensure_ascii=False, indent=2)

            # Также сохраняем информацию о размере изображения
            image_info = {
                "image_size_bytes": len(image_bytes),
                "saved_at": timestamp,
                "response_file": filename
            }

            info_filename = f"yandex_ocr_info_{timestamp}.json"
            info_filepath = os.path.join(self.debug_dir, info_filename)
            with open(info_filepath, 'w', encoding='utf-8') as f:
                json.dump(image_info, f, ensure_ascii=False, indent=2)

            logger.info(f"Полный JSON ответ от Яндекс OCR API сохранен в: {filepath}")

        except Exception as e:
            logger.error(f"Ошибка сохранения JSON ответа: {e}")


    def find_table_by_template(self, response_data: Dict) -> Optional[Tuple[List[Dict], str]]:
        """
        Ищет таблицу в JSON-ответе по загруженным шаблонам.
        Возвращает кортеж (данные_таблицы, имя_шаблона) или None.
        """
        # Убедитесь, что импортировали typing.Dict или используйте dict для Python 3.9+
        # from typing import Dict
        # или просто используйте dict
        try:
            blocks = response_data.get("result", {}).get("blocks", [])

            for template in self.table_templates:
                template_name = template["name"]
                expected_headers = template["headers"]
                required_headers = template.get("required_headers", expected_headers)
                min_similarity = template.get("min_similarity", 80)

                logger.debug(f"Поиск таблицы по шаблону: {template_name}")

                for block in blocks:
                    if "table" not in block:
                        continue

                    table = block["table"]
                    rows = table.get("rows", [])
                    if not rows:
                        continue

                    # Проверяем первую строку как возможные заголовки
                    header_row = rows[0]
                    found_headers = []
                    header_indices = {}  # Сопоставление индекса ячейки -> найденный заголовок

                    for i, cell in enumerate(header_row.get("cells", [])):
                        cell_text = cell.get("text", "").strip()
                        if cell_text:
                            # Используем fuzzy matching для поиска соответствия
                            match, score = process.extractOne(cell_text, expected_headers, scorer=fuzz.token_sort_ratio)
                            if match and score >= min_similarity:
                                found_headers.append(match)
                                header_indices[i] = match
                                logger.debug(f"Найдено совпадение заголовка: '{cell_text}' -> '{match}' (score: {score})")

                    # Проверяем, найдены ли все обязательные заголовки
                    found_required = all(
                        any(fuzz.token_sort_ratio(req, fh) >= min_similarity for fh in found_headers)
                        for req in required_headers
                    )

                    if found_required and len(found_headers) >= len(expected_headers) * 0.5:  # Хотя бы половина
                        logger.info(f"Найдена таблица по шаблону '{template_name}'")

                        # Извлекаем данные из строк таблицы (кроме заголовка)
                        table_data = []
                        for row in rows[1:]:
                            row_data = {}
                            for i, cell in enumerate(row.get("cells", [])):
                                # Используем сопоставленный заголовок, если он есть
                                header_key = header_indices.get(i, f"Столбец_{i + 1}")
                                row_data[header_key] = cell.get("text", "").strip()
                            if any(row_data.values()):  # Добавляем только непустые строки
                                table_data.append(row_data)

                        return table_data, template_name

            logger.info("Подходящая таблица по шаблонам не найдена")
            return None
        except Exception as e:
            logger.error(f"Ошибка поиска таблицы по шаблону: {e}")
            return None


    def validate_field(self, field_name, value):
        """Валидирует значение поля по правилам."""
        if field_name not in self.validation_rules:
            return True, ""

        rule = self.validation_rules[field_name]
        message = rule['message']

        try:
            if rule['validation_type'] == 'number':
                # Убираем пробелы и заменяем запятые на точки
                clean_value = str(value).replace(' ', '').replace(',', '.')
                num_value = float(clean_value)

                if rule['rule'] == '>0' and num_value <= 0:
                    return False, message

        except (ValueError, TypeError):
            return False, f"Неверный формат значения для поля {field_name}"

        return True, ""


    def extract_items_traditional(self, text_lines):
        """Традиционное извлечение строк товаров (резервный метод)."""
        items = []
        # ... (оставляем как есть, если нужно для резервного метода)
        return items


    def process_pdf(self, pdf_bytes, lang='ru'):
        """Обрабатывает PDF из байтов с использованием pymupdf."""
        try:
            import fitz  # PyMuPDF

            logger.info("Начинаю обработку PDF")

            # Открываем PDF из байтов
            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            all_items = []
            all_text = ""  # Для отладки - собираем весь текст

            logger.info(f"PDF содержит {len(pdf_document)} страниц")

            # Обрабатываем каждую страницу
            for page_num in range(len(pdf_document)):
                logger.info(f"Обрабатываем страницу {page_num + 1}")

                page = pdf_document.load_page(page_num)

                # Получаем pixmap (изображение страницы)
                mat = fitz.Matrix(2.0, 2.0)  # Увеличиваем для лучшего OCR
                pix = page.get_pixmap(matrix=mat)

                # Конвертируем в PIL Image
                img_data = pix.tobytes("ppm")
                image = Image.open(BytesIO(img_data))

                # Предобработка изображения (если нужно)
                # processed_image = self.preprocess_image(image)

                # OCR
                text = self.ocr_image(image, lang)
                all_text += text + "\n\n"  # Собираем весь текст для отладки

                logger.info(f"Распознанный текст со страницы {page_num + 1} (первые 500 символов):")
                logger.info(text[:500] if len(text) > 500 else text)

                # text_lines = text.splitlines() # Не используется, если используем JSON

            pdf_document.close()

            # Если использовался Yandex OCR, у нас есть сохраненный JSON файл
            # Найдем последний сохраненный файл для обработки
            # В реальной реализации лучше передавать response_data напрямую
            # Здесь для демонстрации будем искать последний файл
            latest_response_file = self.get_latest_response_file()
            if latest_response_file:
                with open(latest_response_file, 'r', encoding='utf-8') as f:
                    response_data = json.load(f)
                table_result = self.find_table_by_template(response_data)
                if table_result:
                    items, template_name = table_result
                    logger.info(f"Извлечено {len(items)} позиций по шаблону '{template_name}'")
                    # Валидация извлеченных данных (опционально)
                    validated_items = []
                    for item in items:
                        is_valid = True
                        for field_name, field_value in item.items():
                            valid, _ = self.validate_field(field_name.lower().replace(' ', '_'), field_value)
                            if not valid:
                                is_valid = False
                                break
                        if is_valid:
                            validated_items.append(item)
                        else:
                            logger.warning(f"Строка не прошла валидацию: {item}")
                    return validated_items, all_text
                else:
                    logger.warning("Таблица по шаблонам не найдена, используем традиционный метод")
                    # text_lines = all_text.splitlines() # Если нужно для традиционного метода
                    # items = self.extract_items_traditional(text_lines) # Реализовать при необходимости
                    # return items, all_text
            else:
                logger.warning("Файл ответа Yandex OCR не найден")

            # Логируем общий результат
            logger.info(f"Обработка PDF завершена. Найдено {len(all_items)} элементов")
            logger.debug(f"Полный распознанный текст:\n{all_text}")

            return all_items, all_text  # Возвращаем и элементы, и текст для отладки

        except ImportError:
            error_msg = "PyMuPDF не установлен. Установите: pip install pymupdf"
            logger.error(error_msg)
            return [], ""
        except Exception as e:
            error_msg = f"Ошибка обработки PDF: {e}"
            logger.error(error_msg)
            return [], ""


    def get_latest_response_file(self) -> Optional[str]:
        """Находит последний сохраненный JSON-файл ответа Yandex OCR."""
        try:
            files = [f for f in os.listdir(self.debug_dir) if f.startswith("yandex_ocr_response_") and f.endswith(".json")]
            if not files:
                return None
            files.sort()
            latest_file = files[-1]
            return os.path.join(self.debug_dir, latest_file)
        except Exception as e:
            logger.error(f"Ошибка поиска последнего файла ответа: {e}")
            return None


    def process_image(self, image_bytes, lang='ru'):
        """Обрабатывает изображение из байтов."""
        try:
            logger.info("Начинаю обработку изображения")

            image = Image.open(BytesIO(image_bytes))

            # Предобработка изображения (если нужно)
            # processed_image = self.preprocess_image(image)

            # OCR
            text = self.ocr_image(image, lang)

            logger.info("Распознанный текст с изображения (первые 500 символов):")
            logger.info(text[:500] if len(text) > 500 else text)

            # text_lines = text.splitlines() # Не используется, если используем JSON

            # Если использовался Yandex OCR, у нас есть сохраненный JSON файл
            latest_response_file = self.get_latest_response_file()
            if latest_response_file:
                with open(latest_response_file, 'r', encoding='utf-8') as f:
                    response_data = json.load(f)
                table_result = self.find_table_by_template(response_data)
                if table_result:
                    items, template_name = table_result
                    logger.info(f"Извлечено {len(items)} позиций по шаблону '{template_name}'")
                    # Валидация извлеченных данных (опционально)
                    validated_items = []
                    for item in items:
                        is_valid = True
                        for field_name, field_value in item.items():
                            valid, _ = self.validate_field(field_name.lower().replace(' ', '_'), field_value)
                            if not valid:
                                is_valid = False
                                break
                        if is_valid:
                            validated_items.append(item)
                        else:
                            logger.warning(f"Строка не прошла валидацию: {item}")
                    return validated_items, text
                else:
                    logger.warning("Таблица по шаблонам не найдена")
            else:
                logger.warning("Файл ответа Yandex OCR не найден")

            logger.info(f"Обработка изображения завершена.")

            return [], text  # Возвращаем и элементы, и текст для отладки

        except Exception as e:
            error_msg = f"Ошибка обработки изображения: {e}"
            logger.error(error_msg)
            return [], ""


    def process_file(self, file_bytes, file_extension, lang='ru+eng'):
        """Определяет тип файла и вызывает соответствующую функцию."""
        file_extension = file_extension.lower()

        if file_extension == '.pdf':
            return self.process_pdf(file_bytes, lang)
        elif file_extension in ['.jpg', '.jpeg', '.png']:
            return self.process_image(file_bytes, lang)
        else:
            raise ValueError(f"Неподдерживаемый формат файла: {file_extension}")

# Создаем экземпляр для использования
processor = InvoiceProcessor()

