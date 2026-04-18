# anomaly_sniffer/core/data_processor.py
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import logging
import joblib
import os

logger = logging.getLogger(__name__)


class DataProcessor:
    def __init__(self):
        self.scaler = None

    def preprocess_data(self, data):
        """Предварительная обработка и нормализация данных в реальном времени."""
        # Приводим входные данные к 2D-массиву, если пришел одномерный список
        if isinstance(data, (list, tuple)) and not isinstance(data[0], (list, tuple)):
            data = [data]

        if not isinstance(data, pd.DataFrame):
            # Теперь мы уверены, что pandas создаст 1 строку и N столбцов
            data = pd.DataFrame(data)

        # Удаляем временной столбец, если он есть (по имени)
        if isinstance(data.columns[0], str) and data.columns[0].lower().startswith('time'):
            data = data.iloc[:, 1:]

        # Критическая проверка для онлайн-режима
        if self.scaler is None:
            logger.warning("Scaler не инициализирован! Применяется fit на лету (только для тестов).")
            self.scaler = MinMaxScaler()
            scaled_data = self.scaler.fit_transform(data)
        else:
            # В боевом режиме используем ТОЛЬКО transform, применяя прошлые веса
            scaled_data = self.scaler.transform(data)

        return scaled_data

    def create_sequences(self, data, time_step):
        """Создание последовательностей для нейронной сети."""
        xs = []
        # data: 2D массив (samples, num_features)
        for i in range(len(data) - time_step + 1):
            xs.append(data[i:(i + time_step), :])

        # Возвращаем 3D массив (samples, time_step, num_features)
        return np.array(xs)

    def load_and_preprocess_training_data(self, file_path, headers_list, fit_scaler=True):
        try:
            # Читаем файл с автоопределением разделителя
            df = pd.read_csv(file_path, sep=None, engine='python')
            logger.info(f"Загружен файл. Строк: {len(df)}, Колонок: {len(df.columns)}")

            # 1. Оставляем только те колонки из HEADERS, которые реально есть в файле
            existing_headers = [h for h in headers_list if h in df.columns]
            if not existing_headers:
                logger.error(f"В файле нет ни одной колонки из списка HEADERS!")
                return None

            data = df[existing_headers].copy()

            # 2. ДИАГНОСТИКА: Ищем текстовые колонки
            numeric_cols = []
            for col in data.columns:
                # Пытаемся конвертировать колонку
                converted = pd.to_numeric(data[col], errors='coerce')
                # Если в колонке больше 50% — это мусор (текст), выкидываем её из обучения
                if converted.isna().sum() > len(data) * 0.5:
                    logger.warning(f"Колонка '{col}' содержит текст (например, IP). ИСКЛЮЧАЕМ её из обучения.")
                else:
                    data[col] = converted
                    numeric_cols.append(col)

            # 3. Очистка
            data = data[numeric_cols].dropna()
            logger.info(f"После очистки осталось колонок: {len(numeric_cols)}, строк: {len(data)}")

            if data.empty:
                logger.error("ДАННЫХ НЕТ. Проверь: возможно, все выбранные колонки содержат текст.")
                return None

            # 4. Нормализация
            if fit_scaler:
                self.scaler = MinMaxScaler()
                scaled_data = self.scaler.fit_transform(data)
            else:
                scaled_data = self.scaler.transform(data)

            return scaled_data

        except Exception as e:
            logger.error(f"Критическая ошибка: {e}")
            return None

    # НОВЫЕ МЕТОДЫ ДЛЯ СОХРАНЕНИЯ КОНТЕКСТА (Обязательно для диплома)
    def save_scaler(self, path="scaler.pkl"):
        """Сохраняет обученный scaler на диск."""
        if self.scaler is not None:
            # Создаем директорию, если её нет
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
            joblib.dump(self.scaler, path)
            logger.info(f"Scaler успешно сохранен в {path}")
        else:
            logger.error("Попытка сохранить пустой scaler!")

    def load_scaler(self, path="scaler.pkl"):
        """Загружает обученный scaler с диска."""
        if os.path.exists(path):
            self.scaler = joblib.load(path)
            logger.info(f"Scaler успешно загружен из {path}")
            return True
        else:
            logger.error(f"Файл scaler'а не найден: {path}")
            return False