# -*- coding: utf-8 -*-
import threading
import time
import logging
from scapy.all import sniff, IP, UDP, TCP
from scapy.layers.inet import IP, UDP, TCP
from scapy.layers.l2 import Ether
import ipaddress
import collections
from collections import Counter

logger = logging.getLogger(__name__)


class Sniffer:
    """Класс для захвата сетевого трафика и сбора метрик."""

    def __init__(self, interface, network_cidr, time_interval, callback):
        self.interface = interface
        self.network_cidr = network_cidr
        self.time_interval = time_interval
        self.callback = callback
        self.is_running = False
        self.sniffer_thread = None
        self.packet_counts = None
        self.lock = threading.Lock()  # Защита от гонок

    def address_in_network(self, ip_address_str, network_cidr_str):
        """Проверка принадлежности IP-адреса к подсети."""
        try:
            ip_addr = ipaddress.ip_address(ip_address_str)
            network_obj = ipaddress.ip_network(network_cidr_str, strict=False)
            return ip_addr in network_obj
        except Exception:
            return False

    def initialize_packet_counts(self):
        """Инициализация счетчиков метрик."""
        return {
            'total': {
                'packets': 0, 'loopback': 0, 'multicast': 0, 'udp': 0,
                'tcp': 0, 'options': 0, 'fragment': 0, 'fin': 0, 'syn': 0
            },
            'input': {
                'packets': 0, 'udp': 0, 'tcp': 0, 'options': 0,
                'fragment': 0, 'fin': 0, 'syn': 0
            },
            'output': {
                'packets': 0, 'udp': 0, 'tcp': 0, 'options': 0,
                'fragment': 0, 'fin': 0, 'syn': 0
            },
            'current_flows': Counter()
        }

    def packet_callback(self, packet):
        """Обработка каждого перехваченного пакета."""
        ip_layer = packet.getlayer(IP)
        if not ip_layer:
            return

        with self.lock:  # Атомарность операций
            src_ip = ip_layer.src
            dst_ip = ip_layer.dst

            # Определение протокола и портов
            proto = "Other"
            sport = 0
            dport = 0

            tcp_layer = packet.getlayer(TCP)
            udp_layer = packet.getlayer(UDP)

            if tcp_layer:
                proto = "TCP"
                sport = tcp_layer.sport
                dport = tcp_layer.dport
            elif udp_layer:
                proto = "UDP"
                sport = udp_layer.sport
                dport = udp_layer.dport

            # Учет потока
            flow_key = (src_ip, sport, dst_ip, dport, proto)
            self.packet_counts['current_flows'][flow_key] += 1

            # Определение направления
            is_local_src = self.address_in_network(src_ip, self.network_cidr)
            is_local_dst = self.address_in_network(dst_ip, self.network_cidr)

            direction = None
            if is_local_dst and not is_local_src:
                direction = 'input'  # Входящий трафик
            elif is_local_src and not is_local_dst:
                direction = 'output'  # Исходящий трафик
            # Если оба в сети или оба вне сети — считаем только в total

            # Обновление метрик
            self._update_metrics(packet, 'total', ip_layer, tcp_layer, udp_layer, dst_ip)
            if direction:
                self._update_metrics(packet, direction, ip_layer, tcp_layer, udp_layer, dst_ip)

    def _update_metrics(self, packet, direction, ip_layer, tcp_layer, udp_layer, dst_ip):
        """
        Атомарное обновление метрик для указанного направления.

        КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: разделена логика для 'total' и направлений.
        """
        counts = self.packet_counts[direction]
        counts['packets'] += 1

        # UDP/TCP
        if udp_layer:
            counts['udp'] += 1
        elif tcp_layer:
            counts['tcp'] += 1
            # TCP флаги (побитовая проверка)
            flags = tcp_layer.flags
            if flags & 0x01:  # FIN
                counts['fin'] += 1
            if flags & 0x02:  # SYN
                counts['syn'] += 1

        # IP опции
        if ip_layer.options:
            counts['options'] += 1

        # Фрагментация (ИСПРАВЛЕНО: проверяем флаг MF и смещение)
        # MF (More Fragments) = 0x1 в поле flags (биты 13-15)
        # Фрагмент есть, если MF=1 ИЛИ Fragment Offset > 0
        if (ip_layer.flags & 0x1) or (ip_layer.frag > 0):
            counts['fragment'] += 1

        # Специфичные метрики только для 'total'
        if direction == 'total':
            # Multicast
            if (dst_ip == "255.255.255.255" or
                    dst_ip.endswith(".255") or
                    dst_ip.startswith(("224.", "239.", "233."))):
                counts['multicast'] += 1

            # Loopback
            elif dst_ip == '127.0.0.1' or dst_ip.startswith('127.'):
                counts['loopback'] += 1

    def _sniff_loop(self):
        """Основной цикл захвата и агрегации."""
        while self.is_running:
            self.packet_counts = self.initialize_packet_counts()

            start_time = time.time()

            # Захват трафика на интервал
            try:
                sniff(
                    filter=f"ip",  # ИСПРАВЛЕНО: убрана привязка к подсети для захвата всего трафика
                    iface=self.interface,
                    prn=self.packet_callback,
                    store=False,
                    timeout=self.time_interval,
                    quiet=True  # Подавление вывода scapy
                )
            except Exception as e:
                logger.error(f"Ошибка при захвате пакетов: {e}")
                continue

            end_time = time.time()
            duration = end_time - start_time

            # Расчет интенсивностей
            with self.lock:
                for key in ['total', 'input', 'output']:
                    if duration > 0:
                        self.packet_counts[key]['intensity'] = \
                            self.packet_counts[key]['packets'] / duration
                    else:
                        self.packet_counts[key]['intensity'] = 0

                # Извлечение самого активного потока
                top_flow = ("0.0.0.0", 0, "0.0.0.0", 0, "None")
                if self.packet_counts['current_flows']:
                    top_flow = self.packet_counts['current_flows'].most_common(1)[0][0]

                self.packet_counts['metadata'] = {
                    "src_ip": top_flow[0],
                    "src_port": top_flow[1],
                    "dst_ip": top_flow[2],
                    "dst_port": top_flow[3],
                    "protocol": top_flow[4]
                }

                # Логирование метрик (для отладки)
                logger.debug(
                    f"Интервал {duration:.2f}s | "
                    f"Total: {self.packet_counts['total']['packets']} | "
                    f"Input: {self.packet_counts['input']['packets']} | "
                    f"Output: {self.packet_counts['output']['packets']}"
                )

            # Передача данных в callback
            if self.is_running:
                try:
                    self.callback(self.packet_counts)
                except Exception as e:
                    logger.error(f"Ошибка в callback: {e}")

    def start_sniffing(self):
        """Запуск сниффера в отдельном потоке."""
        if self.is_running:
            logger.warning("Сниффер уже запущен")
            return

        self.is_running = True
        self.sniffer_thread = threading.Thread(target=self._sniff_loop, daemon=True)
        self.sniffer_thread.start()
        logger.info(f"Сниффер запущен на интерфейсе {self.interface}")

    def stop_sniffing(self):
        """Остановка сниффера."""
        if self.is_running:
            self.is_running = False
            if self.sniffer_thread:
                self.sniffer_thread.join(timeout=self.time_interval + 2)
            logger.info("Сниффер остановлен")