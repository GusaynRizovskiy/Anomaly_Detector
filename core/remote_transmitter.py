import requests
import websocket
import threading
import os
from dotenv import load_dotenv

# Загружаем переменные из .env файла
load_dotenv()

# Извлекаем значения
login = os.getenv("REMOTE_LOGIN")
password = os.getenv("REMOTE_PASSWORD")
base_url = os.getenv("REMOTE_SERVER_URL")

class RemoteTransmitter:
    def __init__(self, base_url, login, password):
        self.base_url = base_url  # https://185.22.155.17:9000
        self.login = login
        self.password = password
        self.token = None
        self.ws = None

    def authenticate(self):
        """Получение accessToken через REST API."""
        try:
            url = f"{self.base_url}/api/auth/login"
            payload = {
                "login": self.login,
                "password": self.password,
                "type": "sensor-user"
            }
            # verify=False если на сервере самоподписанный SSL-сертификат
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

        ws_url = f"wss://185.22.155.17:9000/integrated-container-ids/connection-integrated-container-ids?token={self.token}"
        try:
            self.ws = websocket.create_connection(ws_url, sslopt={"cert_reqs": websocket.ssl.CERT_NONE})
            logger.info("WebSocket соединение установлено.")
        except Exception as e:
            logger.error(f"Ошибка WebSocket: {e}")

    def send_event(self, internal_anomaly_data):
        """Преобразование внутреннего формата в формат сервера и отправка."""
        if not self.ws or not self.ws.connected:
            self.connect_ws()

        if self.ws and self.ws.connected:
            # Маппинг уровней серьезности
            severity_map = {"CRITICAL": 3, "WARNING": 2, "INFO": 1}
            level = get_severity(internal_anomaly_data['mse_error'], internal_anomaly_data['threshold'])

            # Извлекаем метаданные, собранные сниффером
            ctx = internal_anomaly_data.get('network_context', {})

            event = {
                "type": "integratedContainerIds/transmittingEvents",
                "transmittingEvents": [
                    {
                        "event_type": "alert",
                        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f%z"),
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
            try:
                self.ws.send(json.dumps(event))
            except Exception as e:
                logger.error(f"Ошибка при отправке через WS: {e}")
                self.ws = None  # Сброс для переподключения