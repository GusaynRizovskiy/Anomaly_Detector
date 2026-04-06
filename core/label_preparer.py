# label_preparer.py
# -*- coding: utf-8 -*-
"""
Вспомогательный скрипт для подготовки размеченных данных из CIC IDS 2017.
Использование:
    python label_preparer.py merge --input file1.csv,file2.csv --output merged.csv
    python label_preparer.py sync --original merged.csv --collected training_data.csv --output labeled.csv --interval 5
"""

import argparse
import pandas as pd
import numpy as np
from datetime import timedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def merge_csvs(input_files, output_file):
    """Объединяет несколько CSV с одинаковой структурой в один."""
    df_list = []
    for f in input_files:
        logger.info(f"Чтение {f}...")
        df = pd.read_csv(f, sep=None, engine='python')
        df_list.append(df)
    merged = pd.concat(df_list, ignore_index=True)
    merged.to_csv(output_file, index=False)
    logger.info(f"Объединённый файл сохранён: {output_file} (строк: {len(merged)})")


def synchronize_labels(original_csv, collected_csv, output_labeled_csv, interval_sec, label_col='label'):
    """
    Синхронизирует оригинальные метки атак (CIC IDS) с собранными данными (интервалы агрегации).
    Для каждого интервала в collected_csv определяет, была ли атака в этот промежуток времени.
    """
    # Загрузка оригинальных данных
    logger.info(f"Загрузка оригинального CSV: {original_csv}")
    df_orig = pd.read_csv(original_csv, sep=None, engine='python')

    # Проверка наличия колонок Timestamp и Label
    if 'Timestamp' not in df_orig.columns or 'Label' not in df_orig.columns:
        raise ValueError("Оригинальный CSV должен содержать колонки 'Timestamp' и 'Label'")

    # Преобразование Timestamp в datetime (формат CIC IDS 2017: '07/07/2017 08:02:12 AM')
    df_orig['Timestamp'] = pd.to_datetime(df_orig['Timestamp'], format='%d/%m/%Y %I:%M:%S %p')
    # Метка: 1 если не BENIGN, иначе 0
    df_orig['is_attack'] = (df_orig['Label'] != 'BENIGN').astype(int)

    # Загрузка собранных данных (выход сниффера)
    logger.info(f"Загрузка собранных данных: {collected_csv}")
    df_collected = pd.read_csv(collected_csv)
    if 'timestamp' not in df_collected.columns:
        raise ValueError("Собранный CSV должен содержать колонку 'timestamp'")
    df_collected['timestamp'] = pd.to_datetime(df_collected['timestamp'])


    # Для каждой строки собранных данных определяем интервал [timestamp - interval, timestamp]
    labels = []
    for idx, row in df_collected.iterrows():
        end = row['timestamp']
        start = end - timedelta(seconds=interval_sec)
        # Ищем атаки в оригинальных данных, попадающие в этот интервал
        mask = (df_orig['Timestamp'] >= start) & (df_orig['Timestamp'] <= end)
        attack_present = df_orig.loc[mask, 'is_attack'].any()
        labels.append(1 if attack_present else 0)

    df_collected[label_col] = labels  # вместо df_collected['label'] = labels
    df_collected.to_csv(output_labeled_csv, index=False)
    logger.info(f"Размеченный файл сохранён с колонкой '{label_col}'")
    logger.info(f"Всего строк: {len(df_collected)}, из них аномалий: {sum(labels)}")
    window_labels = []
    for i in range(len(df_collected) - interval_sec + 1):
        # Берём метку последнего интервала в окне (индекс i+interval_sec-1)
        window_labels.append(labels[i + interval_sec - 1])
    # Сохраняем как новый DataFrame с окнами
    df_windows = pd.DataFrame({'window_index': range(len(window_labels)), 'label': window_labels})
    df_windows.to_csv(output_labeled_csv.replace('.csv', '_windows.csv'), index=False)


def main():
    parser = argparse.ArgumentParser(description="Подготовка размеченных данных для детектора аномалий")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Команда merge
    parser_merge = subparsers.add_parser('merge', help='Объединить несколько CSV в один')
    parser_merge.add_argument('--input', required=True, help='Список файлов через запятую')
    parser_merge.add_argument('--output', required=True, help='Выходной файл')

    # Команда sync
    parser_sync = subparsers.add_parser('sync', help='Синхронизировать метки атак с собранными данными')
    parser_sync.add_argument('--original', required=True, help='Оригинальный CSV CIC IDS (объединённый)')
    parser_sync.add_argument('--collected', required=True, help='CSV, собранный сниффером (режим collect)')
    parser_sync.add_argument('--output', required=True, help='Выходной размеченный CSV')
    parser_sync.add_argument('--interval', type=int, required=True,
                             help='Интервал агрегации (сек), использованный при сборе')
    parser_sync.add_argument('--label-col', default='label', help="Имя колонки для меток (по умолчанию 'label')")

    args = parser.parse_args()

    if args.command == 'merge':
        input_files = [f.strip() for f in args.input.split(',')]
        merge_csvs(input_files, args.output)
    elif args.command == 'sync':
        synchronize_labels(args.original, args.collected, args.output, args.interval)


if __name__ == '__main__':
    main()