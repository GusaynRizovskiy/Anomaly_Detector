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
            # Читаем файл
            df = pd.read_csv(file_path, sep=None, engine='python')
            logger.info(f"Загружен файл. Строк: {len(df)}, Колонок: {len(df.columns)}")

            # 1. Жестко фиксируем список колонок.
            # Если в файле нет какой-то колонки из HEADERS, мы создадим её заполненную нулями.
            # Это гарантирует, что на выходе ВСЕГДА будет 26 признаков.
            data = pd.DataFrame(index=df.index)

            for col in headers_list:
                if col in df.columns:
                    # Пытаемся превратить в число.
                    # errors='coerce' превратит IP-адреса или текст в NaN
                    converted = pd.to_numeric(df[col], errors='coerce')

                    # Считаем количество ошибок для лога
                    na_count = converted.isna().sum()
                    if na_count > 0:
                        logger.warning(
                            f"Колонка '{col}': обнаружено {na_count} некорректных значений (текст/IP). Заменяем на 0.")

                    # Заполняем пропуски нулями, чтобы не терять строки
                    data[col] = converted.fillna(0)
                else:
                    # Если колонки вообще нет в CSV, создаем её (набор из 26 должен быть полным)
                    logger.error(f"КРИТИЧЕСКАЯ ОШИБКА: Колонка {col} отсутствует в файле! Заполняю нулями.")
                    data[col] = 0.0

            # 2. Проверка размерности
            if data.shape[1] != len(headers_list):
                logger.error(f"Ошибка размерности! Ожидалось {len(headers_list)}, получено {data.shape[1]}")
                return None

            # 3. Финальная очистка от пустых строк (если весь файл битый)
            data = data.dropna()
            logger.info(f"Подготовка завершена. Итого признаков: {data.shape[1]}, строк: {len(data)}")

            # 4. Нормализация
            if fit_scaler:
                self.scaler = MinMaxScaler()
                scaled_data = self.scaler.fit_transform(data)
                logger.info("Скейлер обучен на 26 признаках.")
            else:
                if self.scaler is None:
                    raise ValueError("Скейлер не инициализирован для трансформации!")
                scaled_data = self.scaler.transform(data)

            return scaled_data

        except Exception as e:
            logger.error(f"Критическая ошибка при подготовке данных: {e}")
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