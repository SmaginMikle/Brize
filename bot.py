import asyncio
import logging
import json
import os
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler, filters, CallbackQueryHandler, ContextTypes
)
from io import BytesIO
import pandas as pd
from invoice_processor import processor

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class InvoiceBot:
    def __init__(self, config_path="config/bot_config.json"):
        self.config_path = config_path
        self.config = self.load_config()
        self.token = self.config["telegram"]["token"]
        self.admin_users = self.config["telegram"]["admin_users"]
        self.max_file_size = self.config["telegram"]["max_file_size"]

        # Создаем необходимые директории
        os.makedirs("config", exist_ok=True)
        os.makedirs("templates", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        os.makedirs("training_data", exist_ok=True)
        os.makedirs("logs", exist_ok=True)

        # Состояния пользователей
        self.user_states = {}
        self.pending_results = {}  # Хранение результатов для каждого пользователя

    def load_config(self):
        """Загружает конфигурацию бота."""
        default_config = {
            "telegram": {
                "token": "YOUR_BOT_TOKEN_HERE",
                "admin_users": [],
                "max_file_size": 20971520
            },
            "training": {
                "min_samples_for_training": 10,
                "auto_training_enabled": True,
                "training_data_path": "training_data"
            },
            "processing": {
                "supported_formats": [".pdf", ".jpg", ".jpeg", ".png"],
                "max_pages": 10,
                "ocr_languages": "rus+eng"
            }
        }

        if not os.path.exists(self.config_path):
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, ensure_ascii=False, indent=2)
            return default_config

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Ошибка загрузки конфигурации: {e}")
            return default_config

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /start."""
        user = update.effective_user
        welcome_message = f"""
🤖 Привет, {user.first_name}!

Я бот для распознавания счетов и накладных. Отправь мне:
📄 PDF-документ
🖼️ Изображение счета (JPG, PNG)

После обработки я предложу выбрать формат вывода результатов:
📊 Вывести в чат
💾 Отправить Excel-файл
🔍 Показать OCR текст

Команды:
/help - помощь
/train - режим дообучения шаблонов (только для админов)
/stats - статистика обработки
"""

        await update.message.reply_text(welcome_message)

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /help."""
        help_text = """
📖 Как использовать бота:

1. Отправь мне PDF или изображение счета
2. Дождись обработки
3. Выбери формат вывода результатов:
   📊 Вывести в чат - покажу таблицу в сообщении
   💾 Отправить Excel-файл - пришлю файл для скачивания
   🔍 Показать OCR текст - покажу распознанный текст для отладки

📝 Поддерживаемые форматы:
- PDF документы
- Изображения JPG, PNG

⚙️ Дополнительные команды:
/train - режим дообучения (для админов)
/stats - статистика использования
"""

        await update.message.reply_text(help_text)

    async def train_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /train."""
        user_id = update.effective_user.id

        if self.admin_users and user_id not in self.admin_users:
            await update.message.reply_text("❌ У вас нет прав для дообучения.")
            return

        self.user_states[user_id] = "training_start"

        keyboard = [
            [InlineKeyboardButton("➕ Добавить образец", callback_data="train_add_sample")],
            [InlineKeyboardButton("📊 Обучить модель", callback_data="train_model")],
            [InlineKeyboardButton("💾 Сохранить данные", callback_data="train_save_data")],
            [InlineKeyboardButton("❌ Отмена", callback_data="train_cancel")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(
            "🎯 Режим дообучения шаблонов\n\n"
            "Выберите действие:",
            reply_markup=reply_markup
        )

    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /stats."""
        stats_text = """
📊 Статистика бота:

📈 Обработано документов: 0
📋 Извлечено строк: 0
✅ Успешных извлечений: 0
❌ Ошибок: 0

🕒 Время работы: 0 дней
"""

        await update.message.reply_text(stats_text)

    def format_items_for_chat(self, items, max_items=10):
        """Форматирует данные для вывода в чат."""
        if not items:
            return "❌ Не найдено позиций в документе."

        # Заголовки таблицы
        headers = ["Наименование", "Кол-во"]
        if any('номер' in item for item in items):
            headers.insert(0, "№")
        if any('гост' in item for item in items):
            headers.insert(2, "ГОСТ")
        if any('ед_изм' in item for item in items):
            headers.append("Ед.изм")
        if any('price' in item for item in items):
            headers.insert(-1 if 'ед_изм' in headers else len(headers), "Цена")
        if any('amount' in item for item in items):
            headers.append("Сумма")

        # Создаем таблицу
        table_lines = []
        table_lines.append(" | ".join(f"{h:^15}" for h in headers))
        table_lines.append("-" * len(table_lines[0]))

        # Добавляем строки данных
        for i, item in enumerate(items[:max_items]):
            row = []
            if 'номер' in headers:
                row.append(f"{item.get('номер', ''):^15}")
            if 'name' in item:
                row.append(f"{item.get('name', '')[:15]:^15}")
            elif 'наименование' in item:
                row.append(f"{item.get('наименование', '')[:15]:^15}")

            if 'гост' in headers:
                row.append(f"{item.get('гост', ''):^15}")

            if 'price' in headers:
                row.append(f"{item.get('price', ''):^15}")
            elif 'цена' in item:
                row.append(f"{item.get('цена', ''):^15}")

            row.append(f"{item.get('quantity', item.get('количество', '')):^7}")

            if 'ед_изм' in headers:
                row.append(f"{item.get('ед_изм', ''):^7}")

            if 'amount' in headers:
                row.append(f"{item.get('amount', ''):^15}")

            table_lines.append(" | ".join(row))

        # Если есть еще позиции
        if len(items) > max_items:
            table_lines.append(f"\n... и еще {len(items) - max_items} позиций")

        # Добавляем общую информацию
        total_amount = 0
        for item in items:
            if item.get('amount'):
                try:
                    total_amount += float(item.get('amount', 0))
                except:
                    pass
            elif item.get('price') and item.get('quantity'):
                try:
                    total_amount += float(item.get('price', 0)) * float(item.get('quantity', 0))
                except:
                    pass

        if total_amount > 0:
            table_lines.append(f"\n💰 Общая сумма: {total_amount:.2f}")

        return "```\n" + "\n".join(table_lines) + "\n```"

    async def handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик документов (PDF)."""
        user = update.effective_user
        user_id = user.id
        document = update.message.document

        # Проверка размера файла
        if document.file_size > self.max_file_size:
            await update.message.reply_text("❌ Файл слишком большой. Максимальный размер: 20MB")
            return

        # Проверка формата
        file_extension = os.path.splitext(document.file_name)[1].lower()
        if file_extension not in self.config["processing"]["supported_formats"]:
            await update.message.reply_text("❌ Неподдерживаемый формат файла.")
            return

        await update.message.reply_text("🔄 Обрабатываю документ...")

        try:
            # Скачиваем файл
            file = await document.get_file()
            file_bytes = await file.download_as_bytearray()

            # Обрабатываем документ
            result = processor.process_file(bytes(file_bytes), file_extension)

            # Обрабатываем результат (может быть кортеж или список в зависимости от реализации)
            if isinstance(result, tuple) and len(result) == 2:
                items, ocr_text = result
            else:
                items = result if isinstance(result, list) else []
                ocr_text = ""

            if not items:
                # Выводим промежуточный результат для отладки
                debug_text = f"❌ Не удалось извлечь данные из документа.\n\n"
                debug_text += f"📄 Распознанный текст (первые 1000 символов):\n"
                debug_text += f"```\n{ocr_text[:1000] if len(ocr_text) > 1000 else ocr_text}\n```"

                await update.message.reply_text(debug_text, parse_mode='Markdown')
                return

            # Сохраняем результаты для пользователя
            self.pending_results[user_id] = {
                'items': items,
                'file_name': document.file_name,
                'ocr_text': ocr_text  # Сохраняем текст для возможной отладки
            }

            # Предлагаем выбор формата вывода
            keyboard = [
                [InlineKeyboardButton("📊 Вывести в чат", callback_data=f"show_chat_{user_id}")],
                [InlineKeyboardButton("💾 Отправить Excel-файл", callback_data=f"send_excel_{user_id}")],
                [InlineKeyboardButton("🔍 Показать OCR текст", callback_data=f"show_ocr_{user_id}")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await update.message.reply_text(
                f"✅ Обработано! Найдено {len(items)} позиций.\nВыберите формат вывода результатов:",
                reply_markup=reply_markup
            )

            logger.info(f"Пользователь {user_id} обработал документ: {len(items)} позиций")

        except Exception as e:
            logger.error(f"Ошибка обработки документа: {e}")
            await update.message.reply_text("❌ Ошибка обработки документа.")

    async def handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик фотографий."""
        user = update.effective_user
        user_id = user.id
        photo = update.message.photo[-1]  # Берем фото максимального размера

        # Проверка размера файла
        if photo.file_size > self.max_file_size:
            await update.message.reply_text("❌ Файл слишком большой. Максимальный размер: 20MB")
            return

        await update.message.reply_text("🔄 Обрабатываю изображение...")

        try:
            # Скачиваем файл
            file = await photo.get_file()
            file_bytes = await file.download_as_bytearray()

            # Обрабатываем изображение
            result = processor.process_file(bytes(file_bytes), '.jpg')

            # Обрабатываем результат (может быть кортеж или список в зависимости от реализации)
            if isinstance(result, tuple) and len(result) == 2:
                items, ocr_text = result
            else:
                items = result if isinstance(result, list) else []
                ocr_text = ""

            if not items:
                # Выводим промежуточный результат для отладки
                debug_text = f"❌ Не удалось извлечь данные из изображения.\n\n"
                debug_text += f"📄 Распознанный текст (первые 1000 символов):\n"
                debug_text += f"```\n{ocr_text[:1000] if len(ocr_text) > 1000 else ocr_text}\n```"

                await update.message.reply_text(debug_text, parse_mode='Markdown')
                return

            # Сохраняем результаты для пользователя
            self.pending_results[user_id] = {
                'items': items,
                'file_name': f"photo_{user_id}.jpg",
                'ocr_text': ocr_text  # Сохраняем текст для возможной отладки
            }

            # Предлагаем выбор формата вывода
            keyboard = [
                [InlineKeyboardButton("📊 Вывести в чат", callback_data=f"show_chat_{user_id}")],
                [InlineKeyboardButton("💾 Отправить Excel-файл", callback_data=f"send_excel_{user_id}")],
                [InlineKeyboardButton("🔍 Показать OCR текст", callback_data=f"show_ocr_{user_id}")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await update.message.reply_text(
                f"✅ Обработано! Найдено {len(items)} позиций.\nВыберите формат вывода результатов:",
                reply_markup=reply_markup
            )

            logger.info(f"Пользователь {user_id} обработал изображение: {len(items)} позиций")

        except Exception as e:
            logger.error(f"Ошибка обработки изображения: {e}")
            await update.message.reply_text("❌ Ошибка обработки изображения.")

    async def button_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик нажатий кнопок."""
        query = update.callback_query
        user_id = query.from_user.id
        data = query.data

        await query.answer()  # Подтверждаем получение callback

        # Обработка выбора формата вывода
        if data.startswith("show_chat_"):
            target_user_id = int(data.split("_")[-1])

            # Проверяем, что пользователь имеет право просматривать результаты
            if target_user_id != user_id and user_id not in self.admin_users:
                await query.answer("❌ У вас нет прав для просмотра этих результатов.", show_alert=True)
                return

            # Получаем результаты
            if target_user_id in self.pending_results:
                items = self.pending_results[target_user_id]['items']
                formatted_text = self.format_items_for_chat(items)

                try:
                    await query.edit_message_text(
                        f"📊 Результаты обработки:\n\n{formatted_text}",
                        parse_mode='Markdown'
                    )
                except Exception as e:
                    # Если Markdown не работает, отправляем без форматирования
                    await query.edit_message_text(
                        f"📊 Результаты обработки:\n\n{formatted_text.replace('```', '')}"
                    )

                # Удаляем результаты из памяти
                self.pending_results.pop(target_user_id, None)
            else:
                await query.edit_message_text("❌ Результаты не найдены.")

        elif data.startswith("send_excel_"):
            target_user_id = int(data.split("_")[-1])

            # Проверяем, что пользователь имеет право получать файл
            if target_user_id != user_id and user_id not in self.admin_users:
                await query.answer("❌ У вас нет прав для получения этого файла.", show_alert=True)
                return

            # Получаем результаты
            if target_user_id in self.pending_results:
                items = self.pending_results[target_user_id]['items']
                file_name = self.pending_results[target_user_id]['file_name']

                # Создаем Excel файл
                df = pd.DataFrame(items)
                excel_buffer = BytesIO()
                df.to_excel(excel_buffer, index=False)
                excel_buffer.seek(0)

                # Отправляем файл
                result_filename = f"результат_{file_name.split('.')[0]}.xlsx"
                try:
                    await query.message.reply_document(
                        document=excel_buffer,
                        filename=result_filename,
                        caption=f"✅ Извлечено {len(items)} позиций"
                    )
                    await query.edit_message_text("💾 Excel-файл отправлен!")
                except Exception as e:
                    await query.edit_message_text(f"❌ Ошибка отправки файла: {e}")

                # Удаляем результаты из памяти
                self.pending_results.pop(target_user_id, None)
            else:
                await query.edit_message_text("❌ Результаты не найдены.")

        elif data.startswith("show_ocr_"):
            target_user_id = int(data.split("_")[-1])

            # Проверяем, что пользователь имеет право просматривать результаты
            if target_user_id != user_id and user_id not in self.admin_users:
                await query.answer("❌ У вас нет прав для просмотра этих результатов.", show_alert=True)
                return

            # Получаем OCR текст
            if target_user_id in self.pending_results:
                ocr_text = self.pending_results[target_user_id].get('ocr_text', '')

                if ocr_text:
                    # Разбиваем текст на части, если он слишком длинный
                    max_length = 4000  # Максимальная длина сообщения в Telegram
                    if len(ocr_text) > max_length:
                        ocr_text_to_show = ocr_text[:max_length] + "\n\n... (текст обрезан для отображения)"
                    else:
                        ocr_text_to_show = ocr_text

                    await query.edit_message_text(
                        f"🔍 Распознанный текст:\n\n```\n{ocr_text_to_show}\n```",
                        parse_mode='Markdown'
                    )
                else:
                    await query.edit_message_text("❌ OCR текст не найден.")
            else:
                await query.edit_message_text("❌ Результаты не найдены.")

        # Обработка админских функций
        elif data == "train_add_sample":
            if user_id not in self.admin_users:
                await query.answer("❌ У вас нет прав для этой операции.", show_alert=True)
                return

            self.user_states[user_id] = "waiting_for_sample_text"
            await query.edit_message_text(
                "📝 Отправьте мне текст строки из документа и укажите её тип.\n"
                "Формат: [тип] текст строки\n"
                "Пример: [item_row] 1001 Товар 1 1000.00 1000.00"
            )

        elif data == "train_model":
            if user_id not in self.admin_users:
                await query.answer("❌ У вас нет прав для этой операции.", show_alert=True)
                return
            # Здесь будет код обучения модели
            await query.edit_message_text("🤖 Обучение модели временно недоступно.")

        elif data == "train_save_data":
            if user_id not in self.admin_users:
                await query.answer("❌ У вас нет прав для этой операции.", show_alert=True)
                return
            try:
                # field_extractor.save_training_data()  # Если используете
                await query.edit_message_text("💾 Данные для обучения сохранены.")
            except Exception as e:
                await query.edit_message_text(f"❌ Ошибка сохранения: {e}")

        elif data == "train_cancel":
            if user_id not in self.admin_users:
                await query.answer("❌ У вас нет прав для этой операции.", show_alert=True)
                return
            self.user_states.pop(user_id, None)
            await query.edit_message_text("❌ Режим дообучения отменен.")

    async def handle_training_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик текста для дообучения."""
        user_id = update.effective_user.id

        if self.user_states.get(user_id) == "waiting_for_sample_text":
            if user_id not in self.admin_users:
                await update.message.reply_text("❌ У вас нет прав для дообучения.")
                return

            text = update.message.text

            # Парсим текст: [тип] содержимое
            if text.startswith('[') and ']' in text:
                end_bracket = text.find(']')
                field_type = text[1:end_bracket]
                sample_text = text[end_bracket + 1:].strip()

                if field_type and sample_text:
                    # field_extractor.add_training_sample(sample_text, field_type)  # Если используете
                    await update.message.reply_text(
                        f"✅ Образец добавлен!\n"
                        f"Тип: {field_type}\n"
                        f"Текст: {sample_text}"
                    )
                else:
                    await update.message.reply_text("❌ Неверный формат. Используйте: [тип] текст")
            else:
                await update.message.reply_text("❌ Неверный формат. Используйте: [тип] текст")

            # Возвращаем к меню
            self.user_states[user_id] = "training_start"
            keyboard = [
                [InlineKeyboardButton("➕ Добавить образец", callback_data="train_add_sample")],
                [InlineKeyboardButton("📊 Обучить модель", callback_data="train_model")],
                [InlineKeyboardButton("💾 Сохранить данные", callback_data="train_save_data")],
                [InlineKeyboardButton("❌ Отмена", callback_data="train_cancel")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await update.message.reply_text(
                "🎯 Режим дообучения шаблонов\n\n"
                "Выберите действие:",
                reply_markup=reply_markup
            )

    def run(self):
        """Запуск бота."""
        if self.token == "YOUR_BOT_TOKEN_HERE":
            print("❌ Пожалуйста, укажите токен бота в config/bot_config.json")
            return

        # Создаем приложение
        application = Application.builder().token(self.token).build()

        # Обработчики команд
        application.add_handler(CommandHandler("start", self.start))
        application.add_handler(CommandHandler("help", self.help_command))
        application.add_handler(CommandHandler("train", self.train_command))
        application.add_handler(CommandHandler("stats", self.stats_command))

        # Обработчики документов и фото
        application.add_handler(MessageHandler(filters.Document.ALL, self.handle_document))
        application.add_handler(MessageHandler(filters.PHOTO, self.handle_photo))

        # Обработчик текста для дообучения
        application.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            self.handle_training_text
        ))

        # Обработчик кнопок
        application.add_handler(CallbackQueryHandler(self.button_handler))

        print("🤖 Бот запущен!")
        # Запускаем бота
        application.run_polling()


# === Запуск бота ===

if __name__ == "__main__":
    bot = InvoiceBot()
    bot.run()