# -*- coding: utf-8 -*-
# anomaly_sniffer/main.py

import argparse
import os
import numpy as np
import time
from datetime import datetime
import logging
import collections
import pandas as pd
import json
import matplotlib.pyplot as plt

from core.anomaly_detector import AnomalyDetector
from core.sniffer import Sniffer
from core.data_processor import DataProcessor

# Настройка заголовков (должны совпадать с порядком в сниффере)
HEADERS = [
    'total_packets', 'total_loopback', 'total_multicast', 'total_udp',
    'total_tcp', 'total_options', 'total_fragment', 'total_fin', 'total_syn',
    'total_intensity', 'input_packets', 'input_udp', 'input_tcp',
    'input_options', 'input_fragment', 'input_fin', 'input_syn',
    'input_intensity', 'output_packets', 'output_udp', 'output_tcp',
    'output_options', 'output_fragment', 'output_fin', 'output_syn',
    'output_intensity'
]
NUM_FEATURES = len(HEADERS)

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Глобальные переменные для режима TEST
data_buffer = collections.deque(maxlen=None)
threshold = None


def log_anomaly(anomaly_data, event_type="NETWORK_ANOMALY_DETECTED"):
    """Запись данных об аномалии в JSON-файл."""
    try:
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        filename = datetime.now().strftime("anomaly_log_%Y-%m-%d.json")
        filepath = os.path.join(log_dir, filename)

        record = {
            "timestamp": datetime.now().isoformat(),
            "level": "CRITICAL",  # Можно менять на INFO для валидации, если хочешь
            "event_id": event_type,  # <--- ТЕПЕРЬ МЫ МОЖЕМ ЭТО МЕНЯТЬ
            "description": "Обнаружена аномалия (валидация файла)" if event_type == "OFFLINE_VALIDATION" else "Обнаружена сетевая аномалия",
            "details": anomaly_data
        }

        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

        # Убрал лишний warning для валидации, чтобы не засорять консоль тысячами строк
        if event_type != "OFFLINE_VALIDATION":
            logger.warning(f"!!! АНОМАЛИЯ ЗАПИСАНА: MSE: {anomaly_data['mse_error']:.4f}")

    except Exception as e:
        logger.error(f"Ошибка при записи лога аномалии: {e}")


def handle_metrics_for_test(metrics, processor, detector, args):
    """Обработчик для режима TEST (Live Detection)."""
    global data_buffer, threshold

    if threshold is None:
        return

    # Формирование строки данных
    row_data = [
        metrics['total']['packets'], metrics['total']['loopback'], metrics['total']['multicast'],
        metrics['total']['udp'], metrics['total']['tcp'], metrics['total']['options'],
        metrics['total']['fragment'], metrics['total']['fin'], metrics['total']['syn'],
        metrics['total']['intensivity'],
        metrics['input']['packets'], metrics['input']['udp'], metrics['input']['tcp'],
        metrics['input']['options'], metrics['input']['fragment'], metrics['input']['fin'],
        metrics['input']['syn'], metrics['input']['intensivity'],
        metrics['output']['packets'], metrics['output']['udp'], metrics['output']['tcp'],
        metrics['output']['options'], metrics['output']['fragment'], metrics['output']['fin'],
        metrics['output']['syn'], metrics['output']['intensivity']
    ]

    df = pd.DataFrame([row_data], columns=HEADERS)

    # Нормализация (строго transform, без fit)
    try:
        if processor.scaler is None:
            logger.error("Ошибка: Scaler не загружен!")
            return
        scaled_metric = processor.scaler.transform(df)
    except Exception as e:
        logger.error(f"Ошибка нормализации данных: {e}")
        return

    # Добавление в буфер
    data_buffer.append(scaled_metric[0])

    # Проверка заполненности буфера
    if len(data_buffer) < args.time_step:
        logger.info(f"Накопление буфера... {len(data_buffer)}/{args.time_step}")
        return

    # Предсказание
    # Преобразуем буфер в 3D массив (1, time_step, num_features)
    input_data = np.array(list(data_buffer))[np.newaxis, :, :]

    error = detector.calculate_reconstruction_error(input_data)

    if error > threshold:
        log_anomaly({
            "mse_error": error,
            "threshold": threshold,
            "metrics_snapshot": dict(zip(HEADERS, row_data))
        })
    else:
        logger.info(f"Статус OK. MSE: {error:.4f} (Порог: {threshold:.4f})")


def handle_metrics_for_collect(metrics, args):
    """Обработчик для режима COLLECT."""
    headers_with_timestamp = ['timestamp'] + HEADERS

    row_data = [datetime.now().isoformat(),
                metrics['total']['packets'], metrics['total']['loopback'], metrics['total']['multicast'],
                metrics['total']['udp'], metrics['total']['tcp'], metrics['total']['options'],
                metrics['total']['fragment'], metrics['total']['fin'], metrics['total']['syn'],
                metrics['total']['intensivity'],
                metrics['input']['packets'], metrics['input']['udp'], metrics['input']['tcp'],
                metrics['input']['options'], metrics['input']['fragment'], metrics['input']['fin'],
                metrics['input']['syn'], metrics['input']['intensivity'],
                metrics['output']['packets'], metrics['output']['udp'], metrics['output']['tcp'],
                metrics['output']['options'], metrics['output']['fragment'], metrics['output']['fin'],
                metrics['output']['syn'], metrics['output']['intensivity']
                ]

    df = pd.DataFrame([row_data], columns=headers_with_timestamp)
    df.to_csv(args.data_file, mode='a', header=False, index=False)
    logging.info(f"Данные записаны. Пакетов: {metrics['total']['packets']}")


def run_file_validation(args, processor, detector):
    logger.info("--- ЗАПУСК РЕЖИМА ВАЛИДАЦИИ ФАЙЛА ---")
    logger.info(f"Файл данных: {args.data_file}")

    # Загрузка модели и скейлера
    detector.load_model(args.model_path)
    if detector.model is None:
        return
    processor.scaler = detector.load_scaler(args.scaler_path)
    if processor.scaler is None:
        return

    # Загрузка порога
    try:
        with open(args.threshold_file, 'r') as f:
            threshold_val = float(f.read().strip())
        logger.info(f"Порог загружен: {threshold_val:.6f}")
    except Exception as e:
        logger.error(f"Не удалось загрузить порог: {e}")
        return

    # Чтение CSV с автоматическим определением разделителя
    try:
        df = pd.read_csv(args.data_file, sep=None, engine='python')
        logger.info(f"Прочитано: строк={df.shape[0]}, столбцов={df.shape[1]}")

        if df.shape[1] < 2:
            logger.error(f"ОШИБКА: всего {df.shape[1]} столбцов. Неверный разделитель?")
            logger.error(f"Первые 5 строк:\n{df.head()}")
            return

        # Попытка извлечь временную метку (первый столбец, если похож на дату/время)
        timestamp_col = None
        first_col = df.columns[0].lower()
        if 'time' in first_col or 'date' in first_col or 'timestamp' in first_col:
            timestamp_col = df.iloc[:, 0].copy()  # сохраняем копию временных меток
            df = df.iloc[:, 1:]                   # удаляем столбец времени из данных
            logger.info("Первый столбец интерпретирован как временная метка и сохранён для графика.")
        else:
            logger.warning("Не удалось обнаружить столбец с временной меткой. Ось X будет индексом окна.")

        # Проверка количества признаков
        expected_features = processor.scaler.n_features_in_
        if df.shape[1] != expected_features:
            logger.error(f"Несовпадение признаков: в файле {df.shape[1]}, модель ожидает {expected_features}.")
            return

    except Exception as e:
        logger.error(f"Ошибка чтения файла: {e}")
        return

    # Масштабирование (трансформация)
    try:
        scaled_data = processor.scaler.transform(df)
    except Exception as e:
        logger.error(f"Ошибка масштабирования: {e}")
        return

    # Создание окон
    X_val = processor.create_sequences(scaled_data, args.time_step)
    if len(X_val) == 0:
        logger.error("Файл слишком короткий для заданного time_step.")
        return
    logger.info(f"Сформировано окон для анализа: {len(X_val)}")

    # Предсказание
    logger.info("Выполняется предсказание нейросети...")
    try:
        reconstructions = detector.model.predict(X_val, verbose=1)
        # MSE для каждого окна (усреднение по времени и признакам)
        mse_errors = np.mean(np.power(X_val - reconstructions, 2), axis=(1, 2))
    except Exception as e:
        logger.error(f"Ошибка при расчёте: {e}")
        return

    # Поиск аномалий
    anomalies_idx = np.where(mse_errors > threshold_val)[0]
    num_anomalies = len(anomalies_idx)
    logger.info(f"Обнаружено аномалий: {num_anomalies}")

    # --- Сохранение результатов в JSON (как было) ---
    if num_anomalies > 0:
        logger.info(f"Сохранение {num_anomalies} событий в JSON лог...")
        for idx in anomalies_idx:
            anomaly_info = {
                "source_file": args.data_file,
                "window_index": int(idx),
                "mse_error": float(mse_errors[idx]),
                "threshold": float(threshold_val)
            }
            log_anomaly(anomaly_info, event_type="OFFLINE_VALIDATION")
        logger.info("Сохранение завершено.")

    # --- Построение графика ---
    try:
        # Формирование оси X (времени)
        if timestamp_col is not None:
            # Для каждого окна берём временную метку последней строки в окне
            x_values = [str(timestamp_col.iloc[i + args.time_step - 1]) for i in range(len(X_val))]
        else:
            x_values = list(range(len(X_val)))  # просто индекс окна

        plt.figure(figsize=(12, 6))
        plt.plot(x_values, mse_errors, label='MSE ошибка', color='blue', linewidth=1)

        # --- Настройка меток оси X (прореживание при большом количестве окон) ---
        if len(x_values) > 20:
            max_ticks = 10
            step = max(1, len(x_values) // max_ticks)
            tick_positions = list(range(0, len(x_values), step))
            # добавим последнее окно, если оно не попало
            if tick_positions[-1] != len(x_values) - 1:
                tick_positions.append(len(x_values) - 1)
            plt.xticks(tick_positions, [x_values[i] for i in tick_positions], rotation=45, ha='right')
        else:
            plt.xticks(rotation=45, ha='right')
        # ------------------------------------------------------------

        plt.axhline(y=threshold_val, color='red', linestyle='--', label=f'Порог ({threshold_val:.4f})')
        plt.xlabel('Время' if timestamp_col is not None else 'Номер окна')
        plt.ylabel('MSE ошибка')
        plt.title(f'Валидация файла: {os.path.basename(args.data_file)}')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Автоматическая подгонка, чтобы подписи поместились
        plt.tight_layout()

        # Выделение аномальных точек
        if len(anomalies_idx) > 0:
            plt.scatter([x_values[i] for i in anomalies_idx],
                        [mse_errors[i] for i in anomalies_idx],
                        color='red', s=30, label='Аномалии', zorder=5)
            plt.legend()

        # Сохранение графика
        plots_dir = "plots"
        os.makedirs(plots_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(args.data_file))[0]
        plot_path = os.path.join(plots_dir, f"validation_{base_name}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        logger.info(f"График сохранён: {plot_path}")

        # Опционально показать (закомментировано)
        # plt.show()
        plt.close()

    except Exception as e:
        logger.error(f"Ошибка при построении графика: {e}")

    # Вывод статистики в консоль
    print("\n" + "=" * 40)
    print(f"РЕЗУЛЬТАТЫ ВАЛИДАЦИИ")
    print(f"Файл: {args.data_file}")
    print(f"Всего проверено окон: {len(mse_errors)}")
    print(f"Обнаружено аномалий: {num_anomalies}")
    print(f"Процент аномалий:    {(num_anomalies / len(mse_errors)) * 100:.2f}%")
    print("=" * 40 + "\n")

def main():
    global data_buffer, threshold

    # 1. Аргументы командной строки
    parser = argparse.ArgumentParser(description="Диплом: Детектор аномалий (CNN-LSTM Autoencoder).")
    parser.add_argument('mode', choices=['collect', 'train', 'test', 'validate'],
                        help="Режим работы: collect (сбор), train (обучение), test (live), validate (файл).")

    parser.add_argument('-i', '--interface', default='eth0', help="Сетевой интерфейс (для collect/test).")
    parser.add_argument('-n', '--network', default='192.168.1.0/24', help="CIDR локальной сети.")
    parser.add_argument('-t', '--interval', type=int, default=5, help="Интервал сбора (сек).")
    parser.add_argument('-d', '--data-file', default='training_data.csv', help="Файл данных (csv).")
    parser.add_argument('-ts', '--time_step', type=int, default=10, help="Длина окна последовательности.")
    parser.add_argument('-e', '--epochs', type=int, default=50, help="Эпохи обучения.")
    parser.add_argument('-b', '--batch_size', type=int, default=64, help="Размер батча.")
    parser.add_argument('-m', '--model_path', default='anomaly_detector_model.keras', help="Путь к модели.")
    parser.add_argument('-s', '--scaler_path', default='scaler.pkl', help="Путь к скейлеру.")
    parser.add_argument('-thr', '--threshold_file', default='threshold.txt', help="Путь к порогу.")

    args = parser.parse_args()

    # Инициализация компонентов
    processor = DataProcessor()
    detector = AnomalyDetector(time_step=args.time_step, num_features=NUM_FEATURES)

    # --- ВЫБОР РЕЖИМА ---

    if args.mode == 'validate':
        # Режим проверки файла (без сниффера)
        if not os.path.exists(args.data_file):
            logger.error(f"Файл данных не найден: {args.data_file}")
            return
        run_file_validation(args, processor, detector)

    elif args.mode == 'train':
        # Режим обучения
        logger.info(f"Запуск обучения на файле: {args.data_file}")

        # Загрузка + Fit Scaler
        raw_data = processor.load_and_preprocess_training_data(args.data_file, fit_scaler=True)
        if raw_data is None: return

        # Сохранение скейлера
        detector.save_scaler(processor.scaler, args.scaler_path)

        # Создание последовательностей
        X_train = processor.create_sequences(raw_data, args.time_step)

        if X_train.size == 0:
            logger.error("Недостаточно данных для обучения.")
            return

        logger.info(f"Размер обучающей выборки: {X_train.shape}")

        # Обучение
        detector.train_model(X_train, args.epochs, args.batch_size, args.model_path)

        # Расчет порога (Оптимизированный)
        logger.info("Расчет порога ошибки...")
        reconstructions = detector.model.predict(X_train, verbose=0)
        # Быстрый расчет MSE через numpy
        mse_train = np.mean(np.power(X_train - reconstructions, 2), axis=(1, 2))

        # Берем 99-й перцентиль
        new_threshold = np.percentile(mse_train, 99)

        try:
            with open(args.threshold_file, 'w') as f:
                f.write(str(new_threshold))
            logger.info(f"Порог установлен и сохранен: {new_threshold:.6f}")
        except Exception as e:
            logger.error(f"Ошибка сохранения порога: {e}")

    elif args.mode == 'test':
        # Режим тестирования (Live)
        logger.info("Запуск режима Live Detection.")

        # Загрузки
        detector.load_model(args.model_path)
        processor.scaler = detector.load_scaler(args.scaler_path)
        if detector.model is None or processor.scaler is None:
            return

        try:
            with open(args.threshold_file, 'r') as f:
                threshold = float(f.read().strip())
            logger.info(f"Порог загружен: {threshold:.6f}")
        except Exception:
            logger.error("Файл порога не найден. Сначала запустите train.")
            return

        # Инициализация буфера
        data_buffer = collections.deque(maxlen=args.time_step)

        # Запуск сниффера
        sniffer = Sniffer(
            interface=args.interface,
            network_cidr=args.network,
            time_interval=args.interval,
            callback=lambda m: handle_metrics_for_test(m, processor, detector, args)
        )
        sniffer.start_sniffing()

        logger.info("Детектор работает. Нажмите Ctrl+C для остановки.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Остановка...")
            sniffer.stop_sniffing()

    elif args.mode == 'collect':
        # Режим сбора данных
        logger.info(f"Сбор данных в файл: {args.data_file}")

        # Создание файла с заголовками, если нет
        if not os.path.exists(args.data_file) or os.stat(args.data_file).st_size == 0:
            headers_with_timestamp = ['timestamp'] + HEADERS
            pd.DataFrame(columns=headers_with_timestamp).to_csv(args.data_file, index=False)
            logger.info("Создан новый файл с заголовками.")

        sniffer = Sniffer(
            interface=args.interface,
            network_cidr=args.network,
            time_interval=args.interval,
            callback=lambda m: handle_metrics_for_collect(m, args)
        )
        sniffer.start_sniffing()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Остановка сбора.")
            sniffer.stop_sniffing()


if __name__ == "__main__":
    main()