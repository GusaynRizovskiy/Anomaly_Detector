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
import socket

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

def send_alert_to_remote(anomaly_data, host, port):
    """Отправка JSON алертов по TCP сокету."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2.0) # Чтобы программа не зависла, если сервер лежит
            s.connect((host, port))
            s.sendall(json.dumps(anomaly_data).encode('utf-8'))
    except Exception as e:
        logger.error(f"Не удалось отправить алерт на {host}:{port}: {e}")

def log_anomaly(anomaly_data, event_type="NETWORK_ANOMALY_DETECTED", args=None):
    """
    Запись данных об аномалии в JSON-файл и опциональная отправка на удаленный сервер.
    """
    try:
        # 1. Логика записи в файл (остается прежней)
        if event_type == "OFFLINE_DETECTION":
            log_dir = os.path.join("logs", "offline")
        else:
            log_dir = os.path.join("logs", "online")

        os.makedirs(log_dir, exist_ok=True)

        filename = datetime.now().strftime("anomaly_%Y-%m-%d.json")
        filepath = os.path.join(log_dir, filename)

        record = {
            "timestamp": datetime.now().isoformat(),
            "level": "CRITICAL",
            "event_id": event_type,
            "description": "Detected network anomaly",
            "details": anomaly_data
        }

        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

        # Вывод в консоль
        if event_type != "OFFLINE_DETECTION":
            logger.warning(f"!!! ONLINE ANOMALY: MSE {anomaly_data['mse_error']:.4f}")

        # 2. НОВОЕ: Отправка на удаленный сервер, если аргументы переданы
        if args and getattr(args, 'remote_host', None) and getattr(args, 'remote_port', None):
            send_alert_to_remote(record, args.remote_host, args.remote_port)

    except Exception as e:
        logger.error(f"Ошибка при логировании аномалии: {e}")




def handle_metrics_for_test(metrics, processor, detector, args):
    """
    Обработка метрик в реальном времени для режима detect-online.
    """
    global data_buffer, threshold

    # 1. Превращаем словарь метрик в плоский список (вектор)
    row = []
    # Важно: порядок извлечения должен строго соответствовать HEADERS
    row.append(metrics['total']['packets'])
    row.append(metrics['total']['loopback'])
    row.append(metrics['total']['multicast'])
    row.append(metrics['total']['udp'])
    row.append(metrics['total']['tcp'])
    row.append(metrics['total']['options'])
    row.append(metrics['total']['fragment'])
    row.append(metrics['total']['fin'])
    row.append(metrics['total']['syn'])
    row.append(metrics['total']['intensivity'])

    row.append(metrics['input']['packets'])
    row.append(metrics['input']['udp'])
    row.append(metrics['input']['tcp'])
    row.append(metrics['input']['options'])
    row.append(metrics['input']['fragment'])
    row.append(metrics['input']['fin'])
    row.append(metrics['input']['syn'])
    row.append(metrics['input']['intensivity'])

    row.append(metrics['output']['packets'])
    row.append(metrics['output']['udp'])
    row.append(metrics['output']['tcp'])
    row.append(metrics['output']['options'])
    row.append(metrics['output']['fragment'])
    row.append(metrics['output']['fin'])
    row.append(metrics['output']['syn'])
    row.append(metrics['output']['intensivity'])

    # 2. Нормализация данных
    # Подготавливаем данные для скейлера (ожидает 2D массив)
    scaled_row = processor.scaler.transform([row])[0]
    data_buffer.append(scaled_row)

    # 3. Проверка: накопилось ли достаточно данных для окна (time_step)
    if len(data_buffer) == args.time_step:
        sequence = np.array([list(data_buffer)])

        # Предсказание (реконструкция)
        reconstruction = detector.model.predict(sequence, verbose=0)

        # Расчет ошибки MSE
        mse = np.mean(np.power(sequence - reconstruction, 2))

        # 4. Сравнение с порогом
        if mse > threshold:
            # Формируем данные для лога
            anomaly_info = {
                "mse_error": float(mse),
                "threshold": float(threshold),
                "metrics_snapshot": dict(zip(HEADERS, row))
            }
            # Вызываем лог. Тип события по умолчанию направит в logs/online
            log_anomaly(anomaly_info, event_type="NETWORK_ANOMALY_DETECTED",args=args)


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
            log_anomaly(anomaly_info, event_type="OFFLINE_DETECTION",args=args)
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

        if args.show:
            plt.show()
        else:
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

    parser = argparse.ArgumentParser(description="Диплом: Детектор аномалий (CNN-LSTM Autoencoder).")
    # ОБНОВЛЕНО: Новые названия режимов
    parser.add_argument('mode', choices=['collect', 'train', 'detect-online', 'detect-offline'],
                        help="Режим работы: collect (сбор), train (обучение), detect-online (live), detect-offline (файл).")

    parser.add_argument('-i', '--interface', default='eth0', help="Сетевой интерфейс (для collect/detect-online).")
    parser.add_argument('-n', '--network', default='192.168.1.0/24', help="CIDR локальной сети.")
    parser.add_argument('-t', '--interval', type=int, default=5, help="Интервал сбора (сек).")
    parser.add_argument('-d', '--data-file', default='training_data.csv', help="Файл данных (csv для offline).")
    parser.add_argument('-ts', '--time_step', type=int, default=10, help="Длина окна последовательности.")
    parser.add_argument('-e', '--epochs', type=int, default=50, help="Эпохи обучения.")
    parser.add_argument('-b', '--batch_size', type=int, default=64, help="Размер батча.")
    parser.add_argument('-m', '--model_path', default='anomaly_detector_model.keras', help="Путь к модели.")
    parser.add_argument('-s', '--scaler_path', default='scaler.pkl', help="Путь к скейлеру.")
    parser.add_argument('-thr', '--threshold_file', default='threshold.txt', help="Путь к порогу.")
    parser.add_argument('--show', action='store_true', help='Показать график после detect-offline')
    parser.add_argument('--remote-host', help="IP адрес сервера для отправки алертов")
    parser.add_argument('--remote-port', type=int, help="Порт сервера для отправки алертов")

    args = parser.parse_args()

    processor = DataProcessor()
    detector = AnomalyDetector(time_step=args.time_step, num_features=NUM_FEATURES)

    if args.mode == 'detect-offline':
        # Переименованный режим validate
        if not os.path.exists(args.data_file):
            logger.error(f"Файл не найден: {args.data_file}")
            return
        run_file_validation(args, processor, detector)

    elif args.mode == 'detect-online':
        # Новый режим реального времени
        logger.info(f"--- ЗАПУСК РЕЖИМА DETECT-ONLINE (Интерфейс: {args.interface}) ---")

        # Загрузка компонентов ИИ
        detector.load_model(args.model_path)
        processor.scaler = detector.load_scaler(args.scaler_path)

        if detector.model is None or processor.scaler is None:
            logger.error("Необходимые файлы (модель/скейлер) отсутствуют.")
            return

        # Загрузка порога
        try:
            with open(args.threshold_file, 'r') as f:
                threshold = float(f.read().strip())
            logger.info(f"Порог детекции: {threshold:.6f}")
        except Exception:
            logger.error("Файл порога не найден. Проведите обучение (train).")
            return

        # Буфер для формирования временного окна
        data_buffer = collections.deque(maxlen=args.time_step)

        # Запуск захвата трафика. Мы используем handle_metrics_for_test,
        # так как она уже реализует логику накопления буфера и вызова детектора.
        sniffer = Sniffer(
            interface=args.interface,
            network_cidr=args.network,
            time_interval=args.interval,
            callback=lambda m: handle_metrics_for_test(m, processor, detector, args)
        )
        sniffer.start_sniffing()

        logger.info("Сенсор активен. Ожидание аномалий...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Завершение работы...")
            sniffer.stop_sniffing()

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