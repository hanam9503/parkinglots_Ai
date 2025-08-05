import requests
import threading
import time
from collections import deque
from typing import Dict, Any, Optional
from config.settings import config
import logging

logger = logging.getLogger(__name__)

class ServerSync:
    def __init__(self):
        self.server_url = config.server.SYNC_SERVER_URL
        self.session = self._create_session()
        self.is_connected = False
        self.offline_queue = deque(maxlen=200)
        self.queue_lock = threading.Lock()
        
        # Background services
        self.shutdown_event = threading.Event()
        self._start_background_services()
    
    def _create_session(self):
        """Create optimized requests session"""
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=3
        )
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session
    
    def send_event(self, event_data: Dict[str, Any]) -> bool:
        """Send parking event to server"""
        if self.is_connected:
            return self._send_with_retry('/api/events', event_data)
        else:
            self._add_to_queue('event', event_data)
            return False
    
    def send_status(self, status_data: Dict[str, Any]) -> bool:
        """Send parking status to server"""
        if self.is_connected:
            return self._send_with_retry('/api/status', status_data)
        else:
            self._add_to_queue('status', status_data)
            return False
    
    def check_health(self) -> bool:
        """Check server health"""
        try:
            response = self.session.get(
                f"{self.server_url}/api/health",
                timeout=config.server.CONNECTION_TIMEOUT
            )
            
            if response.status_code == 200:
                if not self.is_connected:
                    logger.info(f"✅ Server connection established")
                self.is_connected = True
                return True
            else:
                self._handle_connection_error("Health check failed")
                return False
        except Exception as e:
            self._handle_connection_error(f"Health check error: {e}")
            return False