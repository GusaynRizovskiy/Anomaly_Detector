# anomaly_sniffer/core/anomaly_detector.py
# -*- coding: utf-8 -*-
import os
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv1D, LSTM, RepeatVector
from tensorflow.keras.losses import MeanSquaredError as mse_loss
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle
import logging
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import os

logger = logging.getLogger(__name__)


class AnomalyDetector:
    # ИЗМЕНЕНО: Добавлен num_features
    def __init__(self, time_step, num_features=1):
        self.model = None
        self.time_step = time_step
        # НОВОЕ: Сохраняем количество признаков
        self.num_features = num_features
        self.loss_metric = mse_loss()

    def build_model(self):
        """Создание архитектуры нейронной сети."""
        # ИЗМЕНЕНО: Форма входного слоя теперь (time_step, num_features)
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

        # ИЗМЕНЕНО: Выходной Conv1D слой должен реконструировать num_features каналов
        x = Conv1D(filters=self.num_features, kernel_size=3, padding='same', activation='linear')(x)

        self.model = Model(inputs=inputs, outputs=x)
        self.model.compile(optimizer='adam', loss=self.loss_metric)

    def train_model(self, X_train, epochs, batch_size, model_path):
        """Обучение модели с контролем переобучения и сохранением метрик."""
        if self.model is None:
            self.build_model()

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)
        ]

        try:
            # Теперь сохраняем результат обучения в переменную history
            history = self.model.fit(
                X_train, X_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.1,
                callbacks=callbacks,
                verbose=2
            )

            # Добавляем визуализацию результатов для защиты
            self._save_training_plot(history)

            logger.info("Обучение завершено. Модель сохранена, графики построены.")
        except Exception as e:
            logger.error(f"Ошибка в процессе обучения: {e}")

    def _save_training_plot(self, history):
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
        plt.savefig('plots/training_history.png')
        plt.close()

    def calculate_reconstruction_error(self, X):
        """Вычисляет ошибку реконструкции для одного образца."""
        if self.model is None:
            logger.error("Модель не загружена.")
            return 0

        reconstruction = self.model.predict(X, verbose=0)
        # Для многомерных данных mse - это средняя ошибка по всем time_step и num_features
        mse = self.loss_metric(X, reconstruction).numpy()
        return mse

    def load_model(self, model_path):
        """Загрузка обученной модели."""
        try:
            # Для корректной загрузки модели Keras часто не нужно явно указывать loss,
            # но для автокодировщика это может быть полезно
            self.model = load_model(
                model_path,
                custom_objects={'mse_loss': mse_loss()}
            )
            logger.info("Модель загружена.")
        except Exception as e:
            logger.error(f"Не удалось загрузить модель с {model_path}: {e}")
            self.model = None

    def save_scaler(self, scaler, scaler_path):
        """Сохранение объекта нормализатора."""
        try:
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            logger.info(f"Нормализатор сохранен в {scaler_path}")
        except Exception as e:
            logger.error(f"Ошибка сохранения нормализатора: {e}")

    def load_scaler(self, scaler_path):
        """Загрузка объекта нормализатора."""
        try:
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            logger.info(f"Нормализатор загружен из {scaler_path}")
            return scaler
        except FileNotFoundError:
            logger.error(f"Файл нормализатора не найден: {scaler_path}")
            return None
        except Exception as e:
            logger.error(f"Ошибка загрузки нормализатора: {e}")
            return None