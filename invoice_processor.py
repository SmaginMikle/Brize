import os
import re
import csv
from io import BytesIO
import logging
from PIL import Image
import pytesseract
import fitz  # PyMuPDF
from google.cloud import vision
import io
import json

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

        # Инициализация клиента Google Cloud Vision
        try:
            # Убедитесь, что у вас установлен файл учетных данных Google Cloud
            # и установлена переменная окружения GOOGLE_APPLICATION_CREDENTIALS
            self.vision_client = vision.ImageAnnotatorClient()
            logger.info("Google Cloud Vision клиент инициализирован")
        except Exception as e:
            logger.error(f"Ошибка инициализации Google Cloud Vision: {e}")
            self.vision_client = None

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

    def ocr_image(self, image, lang='rus+eng'):
        """Распознаёт текст с изображения с помощью Tesseract OCR."""
        try:
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(image, lang=lang, config=custom_config)
            return text.strip()
        except Exception as e:
            logger.error(f"Ошибка OCR: {e}")
            return ""

    def preprocess_image(self, image):
        """Предобработка изображения для улучшения OCR."""
        if isinstance(image, str):
            image = Image.open(image)

        # Увеличиваем резкость
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2.0)

        # Увеличиваем контрастность
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)

        return image

    def recognize_layout_with_google_vision(self, image_bytes):
        """Использует Google Cloud Vision для распознавания текста и структуры."""
        if not self.vision_client:
            logger.warning("Google Cloud Vision не инициализирован, пропускаю распознавание")
            return []

        try:
            # Создаем объект изображения для Google Cloud Vision
            image = vision.Image(content=image_bytes)

            # Настраиваем функции распознавания
            features = [
                vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION),
                vision.Feature(type_=vision.Feature.Type.TABLE_DETECTION)
            ]

            # Создаем запрос
            request = vision.AnnotateImageRequest(image=image, features=features)

            # Выполняем запрос
            response = self.vision_client.annotate_image(request=request)

            # Обрабатываем результаты
            text_annotations = response.full_text_annotation
            table_annotations = response.table_annotations

            logger.info(f"Распознано {len(text_annotations.pages) if text_annotations.pages else 0} страниц текста")
            logger.info(f"Найдено {len(table_annotations)} таблиц")

            return response

        except Exception as e:
            logger.error(f"Ошибка при использовании Google Cloud Vision: {e}")
            return None

    def extract_table_data_from_vision_response(self, response):
        """Извлекает данные таблицы из ответа Google Cloud Vision."""
        if not response or not response.table_annotations:
            return []

        tables_data = []

        try:
            for table in response.table_annotations:
                table_data = []
                for row in table.header_rows:
                    row_data = []
                    for cell in row.cells:
                        cell_text = ""
                        for paragraph in cell.layout.text_anchor:
                            # Извлечение текста из ячейки
                            # Это упрощенная реализация, может потребоваться доработка
                            cell_text += str(paragraph)  # Заглушка
                        row_data.append(cell_text)
                    table_data.append(row_data)

                for row in table.body_rows:
                    row_data = []
                    for cell in row.cells:
                        cell_text = ""
                        # Извлечение текста из ячейки
                        # Реализация зависит от структуры ответа API
                        row_data.append(cell_text)
                    table_data.append(row_data)

                tables_data.append(table_data)

            return tables_data

        except Exception as e:
            logger.error(f"Ошибка извлечения данных таблицы: {e}")
            return []

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
        """Традиционное извлечение строк товаров."""
        items = []

        # Пропускаем строки с заголовками
        header_keywords = ['наименование', 'товар', 'услуга', 'цена', 'колич', 'сумма', 'итого', '№']

        patterns = [
            # Таблица товаров с ГОСТом
            re.compile(r'^\s*(\d+)\s+(.+?)\s+ГОСТ\s+(\d+-\d+)\s+\((.+?)\)\s+(\w+)\s+(\d+)$'),
            # Таблица товаров без ГОСТа
            re.compile(r'^\s*(\d+)\s+(.+?)\s+(\w+)\s+(\d+)$'),
            # Код | Наименование | Цена | Кол-во | Сумма
            re.compile(r'^\s*(\d{4,})\s+(.+?)\s+([\d\s]+[.,]\d{2})\s+(\d+)\s+([\d\s]+[.,]\d{2})'),
            # Наименование | Цена | Кол-во | Сумма
            re.compile(r'^\s*(.+?)\s+([\d\s]+[.,]\d{2})\s+(\d+)\s+([\d\s]+[.,]\d{2})'),
            # Код | Наименование | Кол-во | Цена
            re.compile(r'^\s*(\d{4,})\s+(.+?)\s+(\d+)\s+([\d\s]+[.,]\d{2})'),
            # Наименование | Кол-во | Цена
            re.compile(r'^\s*(.+?)\s+(\d+)\s+([\d\s]+[.,]\d{2})'),
        ]

        for line in text_lines:
            # Пропускаем строки с заголовками
            if any(word in line.lower() for word in header_keywords):
                continue

            for i, pattern in enumerate(patterns):
                match = pattern.search(line)
                if match:
                    groups = match.groups()
                    item = {}

                    if i == 0:  # Таблица товаров с ГОСТом
                        item = {
                            'номер': groups[0],
                            'наименование': groups[1].strip(),
                            'гост': groups[2],
                            'описание': groups[3].strip(),
                            'ед_изм': groups[4],
                            'количество': groups[5]
                        }
                    elif i == 1:  # Таблица товаров без ГОСТа
                        item = {
                            'номер': groups[0],
                            'наименование': groups[1].strip(),
                            'ед_изм': groups[2],
                            'количество': groups[3]
                        }
                    elif i == 2:  # Полный формат
                        item = {
                            'code': groups[0],
                            'name': groups[1].strip(),
                            'price': groups[2].replace(' ', ''),
                            'quantity': groups[3],
                            'amount': groups[4].replace(' ', '')
                        }
                    elif i == 3:  # Без кода
                        item = {
                            'name': groups[0].strip(),
                            'price': groups[1].replace(' ', ''),
                            'quantity': groups[2],
                            'amount': groups[3].replace(' ', '')
                        }
                    elif i == 4:  # Код, наименование, кол-во, цена
                        item = {
                            'code': groups[0],
                            'name': groups[1].strip(),
                            'quantity': groups[2],
                            'price': groups[3].replace(' ', '')
                        }
                    elif i == 5:  # Наименование, кол-во, цена
                        item = {
                            'name': groups[0].strip(),
                            'quantity': groups[1],
                            'price': groups[2].replace(' ', '')
                        }

                    # Валидация
                    is_valid = True
                    for field_name, field_value in item.items():
                        valid, _ = self.validate_field(field_name, field_value)
                        if not valid:
                            is_valid = False
                            break

                    if is_valid:
                        items.append(item)
                    break  # Найдено совпадение

        return items

    def process_pdf(self, pdf_bytes, lang='rus+eng'):
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

                # Предобработка изображения
                processed_image = self.preprocess_image(image)

                # Конвертируем обратно в байты для Google Cloud Vision
                img_byte_arr = io.BytesIO()
                processed_image.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()

                # Распознавание с помощью Google Cloud Vision (если доступно)
                vision_response = self.recognize_layout_with_google_vision(img_byte_arr)

                # Извлечение данных таблицы
                table_data = self.extract_table_data_from_vision_response(vision_response)

                # OCR
                text = self.ocr_image(processed_image, lang)
                all_text += text + "\n\n"  # Собираем весь текст для отладки

                logger.info(f"Распознанный текст со страницы {page_num + 1} (первые 500 символов):")
                logger.info(text[:500] if len(text) > 500 else text)

                text_lines = text.splitlines()

                # Извлечение данных
                items = self.extract_items_traditional(text_lines)
                all_items.extend(items)

            pdf_document.close()

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

    def process_image(self, image_bytes, lang='rus+eng'):
        """Обрабатывает изображение из байтов."""
        try:
            logger.info("Начинаю обработку изображения")

            image = Image.open(BytesIO(image_bytes))

            # Предобработка изображения
            processed_image = self.preprocess_image(image)

            # Распознавание с помощью Google Cloud Vision (если доступно)
            vision_response = self.recognize_layout_with_google_vision(image_bytes)

            # Извлечение данных таблицы
            table_data = self.extract_table_data_from_vision_response(vision_response)

            # OCR
            text = self.ocr_image(processed_image, lang)

            logger.info("Распознанный текст с изображения (первые 500 символов):")
            logger.info(text[:500] if len(text) > 500 else text)

            text_lines = text.splitlines()

            # Извлечение данных
            items = self.extract_items_traditional(text_lines)

            logger.info(f"Обработка изображения завершена. Найдено {len(items)} элементов")

            return items, text  # Возвращаем и элементы, и текст для отладки

        except Exception as e:
            error_msg = f"Ошибка обработки изображения: {e}"
            logger.error(error_msg)
            return [], ""

    def process_file(self, file_bytes, file_extension, lang='rus+eng'):
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