# -*- coding: utf-8 -*-
import json
import requests
import websocket
import logging
import os
from datetime import datetime
from dotenv import load_dotenv

# Настраиваем локальный логгер для этого модуля, чтобы избежать циклического импорта
logger = logging.getLogger(__name__)

class RemoteTransmitter:
    def __init__(self, base_url, login, password):
        # Удаляем https:// для формирования ws url
        self.raw_url = base_url.replace("https://", "").replace("http://", "")
        self.base_url = base_url
        self.login = login
        self.password = password
        self.token = None
        self.ws = None

    def _get_severity_local(self, mse, threshold):
        """Дублируем логику, чтобы не зависеть от main.py"""
        ratio = mse / threshold
        if ratio > 3.0: return "CRITICAL"
        if ratio > 1.5: return "WARNING"
        return "INFO"

    def authenticate(self):
        """Получение accessToken через REST API."""
        try:
            url = f"{self.base_url}/api/auth/login"
            payload = {
                "login": self.login,
                "password": self.password,
                "type": "sensor-user"
            }
            # verify=False для самоподписанных сертификатов
            response = requests.post(url, json=payload, verify=False, timeout=5)
            if response.status_code == 200:
                self.token = response.json().get('accessToken')
                logger.info("Успешная аутентификация на удаленном сервере.")
                return True
            else:
                logger.error(f"Ошибка аутентификации: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Не удалось связаться с сервером аутентификации: {e}")
            return False

    def connect_ws(self):
        """Установка WebSocket соединения."""
        if not self.token:
            if not self.authenticate():
                return

        # Формируем URL динамически на основе base_url
        ws_url = f"wss://{self.raw_url}/integrated-container-ids/connection-integrated-container-ids?token={self.token}"
        try:
            self.ws = websocket.create_connection(
                ws_url,
                sslopt={"cert_reqs": websocket.ssl.CERT_NONE},
                timeout=5
            )
            logger.info("WebSocket соединение установлено.")
        except Exception as e:
            logger.error(f"Ошибка WebSocket: {e}")
            self.ws = None

    def send_event(self, internal_anomaly_data):
        """Преобразование внутреннего формата в формат сервера и отправка. Возвращает True/False."""
        # 1. Если нет токена (не прошли аутентификацию), сразу выходим
        if not self.token:
            return False

        try:
            if not self.ws or not hasattr(self.ws, 'connected') or not self.ws.connected:
                self.connect_ws()

            if self.ws and self.ws.connected:
                severity_map = {"CRITICAL": 3, "WARNING": 2, "INFO": 1}
                level = self._get_severity_local(
                    internal_anomaly_data['mse_error'],
                    internal_anomaly_data['threshold']
                )

                ctx = internal_anomaly_data.get('network_context', {})

                # Формируем структуру как на сервере
                event = {
                    "type": "integratedContainerIds/transmittingEvents",
                    "transmittingEvents": [
                        {
                            "event_type": "alert",
                            "timestamp": datetime.now().isoformat(),
                            "src_ip": ctx.get('src_ip', '0.0.0.0'),
                            "src_port": int(ctx.get('src_port', 0)),
                            "dest_ip": ctx.get('dst_ip', '0.0.0.0'),
                            "dest_port": int(ctx.get('dst_port', 0)),
                            "proto": ctx.get('protocol', 'TCP'),
                            "signature": f"Anomaly Detected (Score: {internal_anomaly_data.get('anomaly_score', 0)}%)",
                            "severity": severity_map.get(level, 1),
                            "category": "Network Anomaly"
                        }
                    ]
                }
                # ИНФОРМАТИВНОЕ СООБЩЕНИЕ ДЛЯ КОНСОЛИ
                print(
                    f"[SERVER SUCCESS] [{datetime.now().strftime('%H:%M:%S')}] Событие успешно передано на SIEM-сервер.")

                logger.info("Событие успешно отправлено на сервер.")
                self.ws.send(json.dumps(event))
                logger.info("Событие успешно отправлено на сервер.")
                return True # Успешно отправлено
            return False # WebSocket не подключен
        except Exception as e:
            logger.error(f"Ошибка при отправке через WS: {e}")
            self.ws = None
            return False # Произошла ошибка