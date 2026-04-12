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
            'total': {'packets': 0, 'loopback': 0, 'multicast': 0, 'udp': 0, 'tcp': 0, 'options': 0, 'fragment': 0,
                      'fin': 0, 'syn': 0},
            'input': {'packets': 0, 'udp': 0, 'tcp': 0, 'options': 0, 'fragment': 0, 'fin': 0, 'syn': 0},
            'output': {'packets': 0, 'udp': 0, 'tcp': 0, 'options': 0, 'fragment': 0, 'fin': 0, 'syn': 0},
            'current_flows': Counter()
        }

    def packet_callback(self, packet):
        # 1. Безопасное получение IP-слоя
        ip_layer = packet.getlayer(IP)
        if not ip_layer:
            return

        src_ip = ip_layer.src
        dst_ip = ip_layer.dst

        # 2. Определение протокола и портов без риска падения
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

        # 3. Формируем метаданные потока
        flow_key = (src_ip, sport, dst_ip, dport, proto)
        self.packet_counts['current_flows'][flow_key] += 1

        # 4. Проверка multicast и loopback
        if dst_ip == "255.255.255.255" or dst_ip.endswith(".255") or dst_ip.startswith(("224.", "23")):
            self.packet_counts['total']['multicast'] += 1
        elif dst_ip == '127.0.0.1':
            self.packet_counts['total']['loopback'] += 1

        # 5. Определение направления
        is_input = self.address_in_network(dst_ip, self.network_cidr) and not self.address_in_network(src_ip, self.network_cidr)
        is_output = self.address_in_network(src_ip, self.network_cidr) and not self.address_in_network(dst_ip, self.network_cidr)

        if is_input:
            self.update_packet_counts_directional(packet, 'input')
        elif is_output:
            self.update_packet_counts_directional(packet, 'output')

        # 6. Общий счетчик
        self.update_packet_counts_directional(packet, 'total')

    def update_packet_counts_directional(self, packet, direction):
        """Обновление счетчиков для заданного направления с защитой от ошибок."""
        self.packet_counts[direction]['packets'] += 1

        # Безопасно получаем слои
        tcp_layer = packet.getlayer(TCP)
        udp_layer = packet.getlayer(UDP)
        ip_layer = packet.getlayer(IP)

        if udp_layer:
            self.packet_counts[direction]['udp'] += 1
        elif tcp_layer:
            self.packet_counts[direction]['tcp'] += 1
            # Проверка флагов через побитовое И
            flags = tcp_layer.flags
            if flags & 0x01:  # FIN
                self.packet_counts[direction]['fin'] += 1
            if flags & 0x02:  # SYN
                self.packet_counts[direction]['syn'] += 1

        if ip_layer:
            # Проверка наличия опций (длина списка > 0)
            if ip_layer.options:
                self.packet_counts[direction]['options'] += 1
            # Проверка флага фрагментации (MF - More Fragments)
            if ip_layer.flags & 0x1:
                self.packet_counts[direction]['fragment'] += 1

    def _sniff_loop(self):
        """Основной цикл захвата и агрегации."""
        while self.is_running:
            self.packet_counts = self.initialize_packet_counts()

            start_time = time.time()
            # Запуск захвата на интервал time_interval
            sniff(
                filter=f"net {self.network_cidr}",
                iface=self.interface,
                prn=self.packet_callback,
                store=False,
                timeout=self.time_interval
            )
            end_time = time.time()

            duration = end_time - start_time

            # Расчет интенсивностей (исправлено название на 'intensity')
            for key in ['total', 'input', 'output']:
                if duration > 0:
                    self.packet_counts[key]['intensity'] = self.packet_counts[key]['packets'] / duration
                else:
                    self.packet_counts[key]['intensity'] = 0

            # Извлечение самого активного потока (Context)
            top_flow = ("0.0.0.0", 0, "0.0.0.0", 0, "None")
            if self.packet_counts['current_flows']:
                # .most_common(1) вернет [((src, sp, dst, dp, pr), count)]
                top_flow = self.packet_counts['current_flows'].most_common(1)[0][0]

            self.packet_counts['metadata'] = {
                "src_ip": top_flow[0],
                "src_port": top_flow[1],
                "dst_ip": top_flow[2],
                "dst_port": top_flow[3],
                "protocol": top_flow[4]
            }

            # Передача данных в callback (в main.py)
            if self.is_running:
                self.callback(self.packet_counts)

    def start_sniffing(self):
        """Запуск сниффера в отдельном потоке."""
        if self.is_running:
            return

        self.is_running = True
        self.sniffer_thread = threading.Thread(target=self._sniff_loop, daemon=True)
        self.sniffer_thread.start()

    def stop_sniffing(self):
        """Остановка сниффера."""
        if self.is_running:
            self.is_running = False
            self.sniffer_thread.join()