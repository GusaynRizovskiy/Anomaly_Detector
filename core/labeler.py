# core/labeler.py
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class DataLabeler:
    def __init__(self, pcap_start_time, timezone_offset=6, aggregation_interval=2):
        self.pcap_start_time = pcap_start_time
        self.timezone_offset = timezone_offset
        self.aggregation_interval = aggregation_interval

        # База атак (CIC-IDS2017)
        self.attack_windows = [
            ("Botnet ARES", datetime(2017, 7, 7, 10, 2, 0), datetime(2017, 7, 7, 11, 2, 0)),
            ("PortScan (FW ON)", datetime(2017, 7, 7, 13, 55, 0), datetime(2017, 7, 7, 14, 35, 59)),
            ("PortScan (FW OFF)", datetime(2017, 7, 7, 14, 51, 0), datetime(2017, 7, 7, 15, 29, 0)),
            ("DDoS", datetime(2017, 7, 7, 15, 56, 0), datetime(2017, 7, 7, 16, 16, 0))
        ]

        self.adjusted_attacks = [
            (name, start + timedelta(hours=self.timezone_offset), end + timedelta(hours=self.timezone_offset))
            for name, start, end in self.attack_windows
        ]

    def process(self, input_csv, output_csv):
        logger.info(f"Начало разметки файла {input_csv}...")
        try:
            # ИСПРАВЛЕНИЕ: Добавили sep=None и engine='python' для автоопределения ';'
            df = pd.read_csv(input_csv, sep=None, engine='python')

            # Очистим имена колонок от случайных пробелов
            df.columns = df.columns.str.strip()

            if 'timestamp' not in df.columns:
                logger.error(f"В файле нет колонки 'timestamp'! Найдено: {list(df.columns)}")
                return False

            # Преобразуем время
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            sys_start_time = df['timestamp'].iloc[0]

            # Синхронизация временных шкал
            delta = sys_start_time - self.pcap_start_time
            logger.info(f"Смещение времени синхронизировано. Старт системы: {sys_start_time}")

            def get_label(row_ts):
                # Переводим время нашего эксперимента обратно в 2017 год
                orig_end = row_ts - delta
                orig_start = orig_end - timedelta(seconds=self.aggregation_interval)

                for name, start, end in self.adjusted_attacks:
                    if orig_start <= end and orig_end >= start:
                        return 1, name
                return 0, "Normal"

            # Применяем разметку
            logger.info("Применяю метки атак...")
            labels_info = df['timestamp'].apply(lambda x: pd.Series(get_label(x)))
            df['label'] = labels_info[0]
            df['attack_type'] = labels_info[1]

            # Сохраняем результат
            df.to_csv(output_csv, index=False, sep=';')  # Сохраняем с тем же разделителем
            logger.info(f"Разметка завершена. Файл сохранен: {output_csv}")
            return True

        except Exception as e:
            logger.error(f"Ошибка при разметке: {e}")
            return False