# anomaly_sniffer/core/anomaly_detector.py
# -*- coding: utf-8 -*-
import numpy as np
import os
import matplotlib.pyplot as plt
import logging
import joblib  # Используем для загрузки скейлера, если нужно
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv1D, LSTM, RepeatVector
from tensorflow.keras.losses import MeanSquaredError as mse_loss

logger = logging.getLogger(__name__)


class AnomalyDetector:
    def __init__(self, time_step, num_features=1):
        self.model = None
        self.time_step = time_step
        self.num_features = num_features
        self.loss_metric = mse_loss()

    def build_model(self):
        """Создание архитектуры нейронной сети."""
        inputs = Input(shape=(self.time_step, self.num_features))

        # Энкодер
        x = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(inputs)
        x = LSTM(16, return_sequences=True, activation='relu')(x)
        x = LSTM(8, activation='relu')(x)

        # Реконструктор
        x = RepeatVector(self.time_step)(x)

        # Декодер
        x = LSTM(8, return_sequences=True, activation='relu')(x)
        x = LSTM(16, return_sequences=True, activation='relu')(x)
        x = Conv1D(filters=self.num_features, kernel_size=3, padding='same', activation='linear')(x)

        self.model = Model(inputs=inputs, outputs=x)
        self.model.compile(optimizer='adam', loss=self.loss_metric)

    def train_model(self, X_train, epochs=50, batch_size=32, validation_split=0.1, model_path=None):
        """
        Обучение модели. model_path теперь в конце и не обязателен для вызова.
        """
        if self.model is None:
            self.build_model()

        logger.info(f"Начало обучения: epochs={epochs}, batch_size={batch_size}")

        # Обучаем
        history = self.model.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )

        # Если путь передан, сохраняем сразу (но в main.py у нас отдельный вызов save_model, так что это опционально)
        if model_path:
            self.save_model(model_path)

        return history

    def save_model(self, model_path):
        """
        Сохранение обученной модели на диск.
        """
        if self.model is None:
            logger.error("Ошибка: Попытка сохранить пустую модель (модель не была построена или обучена).")
            return

        try:
            # Создаем папку для моделей, если её нет
            directory = os.path.dirname(model_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
                logger.info(f"Создана директория: {directory}")

            # Сохраняем модель в формате .h5 или в стандартном формате Keras
            self.model.save(model_path)
            logger.info(f"Модель успешно сохранена по адресу: {model_path}")
        except Exception as e:
            logger.error(f"Не удалось сохранить модель: {e}")
            
    # ИСПРАВЛЕНО: Теперь метод находится на уровне класса (вровень с другими def)
    def _save_training_plot(self, history, show=False):
        """Вспомогательный метод для построения графиков."""
        os.makedirs('plots', exist_ok=True)
        plt.figure(figsize=(8, 5))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.grid(True)
        plot_path = 'plots/training_history.png'
        plt.savefig(plot_path)
        if show:
            plt.show()
        else:
            plt.close()
        logger.info(f"График обучения сохранён: {plot_path}")

    # ИСПРАВЛЕНО: Уточнена логика для 3D-массива (окна)
    def calculate_reconstruction_error(self, X):
        """
        Вычисляет среднюю ошибку реконструкции для окна данных.
        X должен быть 3D: (1, time_step, num_features)
        """
        if self.model is None:
            logger.error("Модель не загружена.")
            return 0

        # Получаем предсказание (реконструкцию)
        reconstruction = self.model.predict(X, verbose=0)

        # Считаем MSE вручную для корректного получения одного числа
        # (X - reconstruction)^2 -> берем среднее по всем осям
        mse = np.mean(np.power(X - reconstruction, 2))

        return float(mse)

    def load_model(self, model_path):
        """Загрузка обученной модели."""
        try:
            # Важно: если вы использовали кастомный лосс, его нужно указать
            self.model = load_model(model_path)
            logger.info("Модель успешно загружена.")
        except Exception as e:
            logger.error(f"Не удалось загрузить модель: {e}")
            self.model = None

    # МЕТОДЫ save_scaler и load_scaler УДАЛЕНЫ (перенесены в DataProcessor)