import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib
import os
import json

class FieldExtractor:
    def __init__(self, model_path="models/field_model.pkl"):
        self.model_path = model_path
        self.models = {}
        self.is_trained = False
        self.training_data = []
        self.load_models()
    
    def simple_field_classification(self, text_line):
        """Простая классификация полей на основе правил."""
        text_lower = text_line.lower().strip()
        
        # Правила для разных типов полей
        if any(keyword in text_lower for keyword in ['дата', 'date', 'от']):
            if re.search(r'\d{1,2}[.\-/\s]\d{1,2}[.\-/\s]\d{2,4}', text_line):
                return 'date'
        
        if any(keyword in text_lower for keyword in ['номер', '№', 'no']):
            if re.search(r'[А-ЯA-Z\d\-\/]+', text_line):
                return 'document_number'
        
        if any(keyword in text_lower for keyword in ['поставщик', 'supplier', 'компани', 'ooo', 'ооо']):
            return 'supplier'
        
        if any(keyword in text_lower for keyword in ['покупатель', 'customer', 'клиент']):
            return 'customer'
        
        # Проверка на строку таблицы (товары)
        if re.search(r'\d+\s+.+?\s+\d+[.,]\d{2}', text_line):
            return 'item_row'
        
        # Проверка на сумму
        if re.search(r'\d+[.,]\d{2}\s*₽?$', text_line.strip()):
            return 'amount'
        
        return 'unknown'
    
    def extract_structured_data(self, text_lines):
        """Извлекает структурированные данные из строк текста."""
        structured_data = {
            'metadata': {},
            'items': []
        }
        
        for line in text_lines:
            field_type = self.classify_field_type(line)
            
            if field_type == 'date':
                date_match = re.search(r'\d{1,2}[.\-/\s]\d{1,2}[.\-/\s]\d{2,4}', line)
                if date_match:
                    structured_data['metadata']['date'] = date_match.group()
            
            elif field_type == 'document_number':
                number_match = re.search(r'[А-ЯA-Z\d\-\/]+', line)
                if number_match:
                    structured_data['metadata']['document_number'] = number_match.group()
            
            elif field_type == 'supplier':
                structured_data['metadata']['supplier'] = line.strip()
            
            elif field_type == 'customer':
                structured_data['metadata']['customer'] = line.strip()
            
            elif field_type == 'item_row':
                item = self.parse_item_row(line)
                if item:
                    structured_data['items'].append(item)
        
        return structured_data
    
    def parse_item_row(self, line):
        """Парсит строку таблицы товаров."""
        # Шаблоны для строк товаров
        patterns = [
            # Код | Наименование | Цена | Кол-во | Сумма
            re.compile(r'^\s*(\d{4,})\s+(.+?)\s+([\d\s]+[.,]\d{2})\s+(\d+)\s+([\d\s]+[.,]\d{2})'),
            # Наименование | Цена | Кол-во | Сумма
            re.compile(r'^\s*(.+?)\s+([\d\s]+[.,]\d{2})\s+(\d+)\s+([\d\s]+[.,]\d{2})'),
            # Код | Наименование | Кол-во | Цена
            re.compile(r'^\s*(\d{4,})\s+(.+?)\s+(\d+)\s+([\d\s]+[.,]\d{2})'),
            # Наименование | Кол-во | Цена
            re.compile(r'^\s*(.+?)\s+(\d+)\s+([\d\s]+[.,]\d{2})'),
        ]
        
        for i, pattern in enumerate(patterns):
            match = pattern.search(line)
            if match:
                groups = match.groups()
                if i == 0:  # Полный формат
                    return {
                        'code': groups[0],
                        'name': groups[1].strip(),
                        'price': groups[2].replace(' ', ''),
                        'quantity': groups[3],
                        'amount': groups[4].replace(' ', '')
                    }
                elif i == 1:  # Без кода
                    return {
                        'name': groups[0].strip(),
                        'price': groups[1].replace(' ', ''),
                        'quantity': groups[2],
                        'amount': groups[3].replace(' ', '')
                    }
                elif i == 2:  # Код, наименование, кол-во, цена
                    return {
                        'code': groups[0],
                        'name': groups[1].strip(),
                        'quantity': groups[2],
                        'price': groups[3].replace(' ', '')
                    }
                elif i == 3:  # Наименование, кол-во, цена
                    return {
                        'name': groups[0].strip(),
                        'quantity': groups[1],
                        'price': groups[2].replace(' ', '')
                    }
        
        return None
    
    def classify_field_type(self, text_line):
        """Классифицирует тип поля в строке текста."""
        if not self.is_trained:
            # Используем простые правила если модель не обучена
            return self.simple_field_classification(text_line)
        
        try:
            # Здесь будет код классификации с использованием обученной модели
            return self.simple_field_classification(text_line)
        except Exception as e:
            print(f"Ошибка классификации поля: {e}")
            return "unknown"
    
    def add_training_sample(self, text_line, field_type):
        """Добавляет образец для дообучения."""
        self.training_data.append({
            'text': text_line,
            'type': field_type
        })
    
    def save_training_data(self, data_path="training_data"):
        """Сохраняет данные для дообучения."""
        os.makedirs(data_path, exist_ok=True)
        timestamp = np.random.randint(100000, 999999)
        file_path = os.path.join(data_path, f"training_data_{timestamp}.json")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_data, f, ensure_ascii=False, indent=2)
    
    def load_models(self):
        """Загружает предобученные модели."""
        if os.path.exists(self.model_path):
            try:
                self.models = joblib.load(self.model_path)
                self.is_trained = True
                print("Модели полей загружены")
            except Exception as e:
                print(f"Ошибка загрузки моделей: {e}")
        else:
            print("Файлы моделей не найдены, будут использованы правила")
    
    def train_models(self):
        """Обучает модели на доступных данных."""
        if len(self.training_data) < 5:  # Минимальное количество образцов
            return False
        
        try:
            # Здесь будет код обучения моделей
            print(f"Обучение моделей на {len(self.training_data)} образцах")
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Ошибка обучения моделей: {e}")
            return False

field_extractor = FieldExtractor()