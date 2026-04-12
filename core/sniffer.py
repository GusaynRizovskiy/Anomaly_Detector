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
        """
        Обработка каждого перехваченного пакета.
        Просто обновляем счетчики в словаре.
        """
        self.packet_counts['total']['packets'] += 1

        if packet.haslayer(IP):
            if packet.haslayer(IP):
                ip_layer = packet[IP]
                proto = "TCP" if packet.haslayer(TCP) else "UDP" if packet.haslayer(UDP) else "Other"

                # Формируем ключ потока
                flow_key = (
                    ip_layer.src,
                    packet[TCP].sport if packet.haslayer(TCP) else (packet[UDP].sport if packet.haslayer(UDP) else 0),
                    ip_layer.dst,
                    packet[TCP].dport if packet.haslayer(TCP) else (packet[UDP].dport if packet.haslayer(UDP) else 0),
                    proto
                )
                self.packet_counts['current_flows'][flow_key] += 1
            src_ip = packet[IP].src
            dst_ip = packet[IP].dst

            # Проверка multicast и loopback
            if dst_ip == "255.255.255.255" or dst_ip.endswith(".255") or dst_ip.startswith(("224.", "23")):
                self.packet_counts['total']['multicast'] += 1
            elif dst_ip == '127.0.0.1':
                self.packet_counts['total']['loopback'] += 1

            # Определение направления (входящий/исходящий)
            if self.address_in_network(dst_ip, self.network_cidr) and not self.address_in_network(src_ip,
                                                                                                  self.network_cidr):
                self.update_packet_counts_directional(packet, 'input')
            elif self.address_in_network(src_ip, self.network_cidr) and not self.address_in_network(dst_ip,
                                                                                                    self.network_cidr):
                self.update_packet_counts_directional(packet, 'output')

        # Обновление метрик по протоколам и флагам
        self.update_packet_counts_directional(packet, 'total')

    def update_packet_counts_directional(self, packet, direction):
        """Обновление счетчиков для заданного направления."""
        self.packet_counts[direction]['packets'] += 1

        # Обновление счетчиков по протоколам
        if packet.haslayer(UDP):
            self.packet_counts[direction]['udp'] += 1
        elif packet.haslayer(TCP):
            self.packet_counts[direction]['tcp'] += 1
            if packet[TCP].flags & 0x01:  # FIN flag
                self.packet_counts[direction]['fin'] += 1
            if packet[TCP].flags & 0x02:  # SYN flag
                self.packet_counts[direction]['syn'] += 1

        # Проверка опций и фрагментации
        if packet.haslayer(IP):
            if packet[IP].options:
                self.packet_counts[direction]['options'] += 1
            if packet[IP].flags & 0x2:  # MF (More Fragments) flag
                self.packet_counts[direction]['fragment'] += 1

    def _sniff_loop(self):
        """Основной цикл захвата и агрегации."""
        while self.is_running:
            self.packet_counts = self.initialize_packet_counts()

            start_time = time.time()
            sniff(
                filter=f"net {self.network_cidr}",
                iface=self.interface,
                prn=self.packet_callback,
                store=False,
                timeout=self.time_interval
            )
            end_time = time.time()

            # Вычисление интенсивностей после завершения интервала
            duration = end_time - start_time
            if duration > 0:
                self.packet_counts['total']['intensivity'] = self.packet_counts['total']['packets'] / duration
                self.packet_counts['input']['intensivity'] = self.packet_counts['input']['packets'] / duration
                self.packet_counts['output']['intensivity'] = self.packet_counts['output']['packets'] / duration
            else:
                self.packet_counts['total']['intensivity'] = 0
                self.packet_counts['input']['intensivity'] = 0
                self.packet_counts['output']['intensivity'] = 0
            # 1. Находим самый активный поток (тот, у кого больше всего пакетов)
            top_flow = ("0.0.0.0", 0, "0.0.0.0", 0, "None")
            if self.packet_counts['current_flows']:
                # .most_common(1) вернет [((src, sport, dst, dport, proto), count)]
                top_flow = self.packet_counts['current_flows'].most_common(1)[0][0]

            # 2. Сохраняем метаданные в отдельный ключ, чтобы main.py их увидел
            self.packet_counts['metadata'] = {
                "src_ip": top_flow[0],
                "src_port": top_flow[1],
                "dst_ip": top_flow[2],
                "dst_port": top_flow[3],
                "protocol": top_flow[4]
            }
            # Передача агрегированных метрик в основной поток
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