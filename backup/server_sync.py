"""
server_sync.py - Server Communication Module
============================================
Quản lý giao tiếp với server backend
Bao gồm: đồng bộ events, status updates, offline queue, retry logic
"""

import time
import requests
import threading
from collections import deque
from typing import Dict, Any, Optional, List
import logging
import json
from urllib3.exceptions import InsecureRequestWarning
import warnings

# Suppress SSL warnings
warnings.filterwarnings("ignore", category=InsecureRequestWarning)

logger = logging.getLogger(__name__)

class ServerSync:
    """
    Lớp quản lý đồng bộ dữ liệu với server
    
    Chức năng chính:
    - Gửi events (enter/exit) lên server
    - Gửi status updates cho các ô đỗ
    - Quản lý offline queue khi mất kết nối
    - Auto retry và connection recovery
    - Health check định kỳ
    """
    
    def __init__(self,
                 server_url: str,
                 connection_timeout: float = 5.0,
                 request_timeout: float = 10.0,
                 max_retries: int = 3,
                 offline_queue_size: int = 1000,
                 health_check_interval: float = 30.0):
        """
        Khởi tạo ServerSync
        
        Args:
            server_url (str): URL của server (ví dụ: http://localhost:5000)
            connection_timeout (float): Timeout khi kết nối (seconds)
            request_timeout (float): Timeout cho request (seconds)
            max_retries (int): Số lần retry tối đa
            offline_queue_size (int): Kích thước queue offline
            health_check_interval (float): Khoảng thời gian health check (seconds)
        """
        self.server_url = server_url.rstrip('/')  # Remove trailing slash
        self.connection_timeout = connection_timeout
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.health_check_interval = health_check_interval
        
        # HTTP Session với timeout và headers mặc định
        self.session = requests.Session()
        self.session.timeout = request_timeout
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'MultiCameraParkingSystem/1.0'
        })
        
        # Connection state
        self.is_connected = False
        self.last_health_check = 0
        self.connection_failures = 0
        
        # Offline queue - Lưu trữ dữ liệu khi offline
        self.offline_queue = deque(maxlen=offline_queue_size)
        
        # Statistics
        self.stats = {
            'events_sent': 0,
            'status_sent': 0,
            'failed_requests': 0,
            'connection_recoveries': 0,
            'queue_overflows': 0,
            'last_success': 0,
            'last_failure': 0
        }
        
        # Threading
        self.lock = threading.RLock()
        self.health_check_thread = None
        self.sync_thread = None
        self.stop_event = threading.Event()
        
        # API endpoints
        self.endpoints = {
            'health': '/api/health',
            'events': '/api/events',
            'status': '/api/status',
            'bulk_sync': '/api/bulk-sync'
        }
        
        logger.info(f"🌐 ServerSync initialized for {server_url}")
        
        # Kiểm tra kết nối ban đầu
        self._check_server_health()
        
        # Bắt đầu background threads
        self._start_background_threads()
    
    def _start_background_threads(self):
        """Bắt đầu các background threads"""
        # Health check thread
        self.health_check_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self.health_check_thread.start()
        
        # Sync thread cho offline queue
        self.sync_thread = threading.Thread(target=self._sync_offline_queue_loop, daemon=True)
        self.sync_thread.start()
        
        logger.info("🔄 Started background sync threads")
    
    def _health_check_loop(self):
        """Background loop cho health check"""
        while not self.stop_event.is_set():
            try:
                current_time = time.time()
                
                # Chỉ check nếu đã qua health_check_interval
                if current_time - self.last_health_check >= self.health_check_interval:
                    self._check_server_health()
                
                # Ngủ 1 giây trước khi check lại
                self.stop_event.wait(1.0)
                
            except Exception as e:
                logger.error(f"❌ Health check loop error: {e}")
                self.stop_event.wait(5.0)
    
    def _sync_offline_queue_loop(self):
        """Background loop để sync offline queue"""
        while not self.stop_event.is_set():
            try:
                # Chỉ sync khi có kết nối và có data trong queue
                if self.is_connected and self.offline_queue:
                    self._sync_offline_data()
                
                # Ngủ 5 giây trước khi check lại
                self.stop_event.wait(5.0)
                
            except Exception as e:
                logger.error(f"❌ Offline sync loop error: {e}")
                self.stop_event.wait(10.0)
    
    def _check_server_health(self) -> bool:
        """
        Kiểm tra health của server
        
        Returns:
            bool: True nếu server healthy
        """
        try:
            response = self.session.get(
                f"{self.server_url}{self.endpoints['health']}",
                timeout=self.connection_timeout
            )
            
            was_connected = self.is_connected
            self.is_connected = response.status_code == 200
            self.last_health_check = time.time()
            
            if self.is_connected:
                self.connection_failures = 0
                if not was_connected:
                    # Connection recovered
                    self.stats['connection_recoveries'] += 1
                    logger.info(f"✅ Connected to server: {self.server_url}")
            else:
                self.connection_failures += 1
                if was_connected:
                    logger.warning(f"⚠️ Server returned status {response.status_code}")
            
            return self.is_connected
            
        except Exception as e:
            was_connected = self.is_connected
            self.is_connected = False
            self.connection_failures += 1
            self.last_health_check = time.time()
            
            if was_connected:
                logger.warning(f"⚠️ Lost connection to server: {e}")
            
            return False
    
    def send_event(self, event_data: Dict[str, Any]) -> bool:
        """
        Gửi parking event lên server
        
        Args:
            event_data (Dict): Dữ liệu event
                {
                    'id': str,
                    'camera_id': str,
                    'spot_id': str,
                    'event_type': 'enter' | 'exit',
                    'timestamp': str,
                    'plate_text': str,
                    'location_name': str,
                    ...
                }
                
        Returns:
            bool: True nếu gửi thành công
        """
        return self._send_data('events', event_data, 'event')
    
    def send_status(self, status_data: Dict[str, Any]) -> bool:
        """
        Gửi status update lên server
        
        Args:
            status_data (Dict): Dữ liệu status
                {
                    'spot_id': str,
                    'camera_id': str,
                    'is_occupied': bool,
                    'plate_text': str,
                    'last_update': str,
                    ...
                }
                
        Returns:
            bool: True nếu gửi thành công
        """
        return self._send_data('status', status_data, 'status')
    
    def _send_data(self, endpoint_key: str, data: Dict[str, Any], data_type: str) -> bool:
        """
        Gửi dữ liệu lên server (internal method)
        
        Args:
            endpoint_key (str): Key của endpoint trong self.endpoints
            data (Dict): Dữ liệu cần gửi
            data_type (str): Loại dữ liệu ('event' hoặc 'status')
            
        Returns:
            bool: True nếu thành công
        """
        with self.lock:
            # Kiểm tra kết nối nếu cần
            if not self.is_connected:
                self._check_server_health()
            
            # Nếu vẫn không có kết nối, thêm vào offline queue
            if not self.is_connected:
                return self._add_to_offline_queue(endpoint_key, data, data_type)
            
            # Thử gửi dữ liệu
            for attempt in range(self.max_retries):
                try:
                    response = self.session.post(
                        f"{self.server_url}{self.endpoints[endpoint_key]}",
                        json=data,
                        timeout=self.request_timeout
                    )
                    
                    if response.status_code in [200, 201]:
                        # Thành công
                        if data_type == 'event':
                            self.stats['events_sent'] += 1
                        elif data_type == 'status':
                            self.stats['status_sent'] += 1
                        
                        self.stats['last_success'] = time.time()
                        return True
                    else:
                        logger.warning(f"⚠️ Server returned {response.status_code} for {data_type}")
                        
                except requests.exceptions.Timeout:
                    logger.warning(f"⏰ Timeout sending {data_type} (attempt {attempt + 1})")
                    continue
                    
                except Exception as e:
                    logger.warning(f"❌ Error sending {data_type} (attempt {attempt + 1}): {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(0.5 * (attempt + 1))  # Exponential backoff
                    continue
            
            # Tất cả attempts thất bại
            self.is_connected = False
            self.stats['failed_requests'] += 1
            self.stats['last_failure'] = time.time()
            
            # Thêm vào offline queue
            return self._add_to_offline_queue(endpoint_key, data, data_type)
    
    def _add_to_offline_queue(self, endpoint_key: str, data: Dict[str, Any], data_type: str) -> bool:
        """
        Thêm dữ liệu vào offline queue
        
        Args:
            endpoint_key (str): Endpoint key
            data (Dict): Dữ liệu
            data_type (str): Loại dữ liệu
            
        Returns:
            bool: False (vì không gửi được ngay)
        """
        queue_item = {
            'endpoint': endpoint_key,
            'data': data,
            'type': data_type,
            'timestamp': time.time(),
            'attempts': 0
        }
        
        try:
            self.offline_queue.append(queue_item)
        except:
            # Queue đầy, item cũ nhất sẽ bị loại bỏ
            self.stats['queue_overflows'] += 1
            self.offline_queue.append(queue_item)
        
        return False
    
    def _sync_offline_data(self):
        """Đồng bộ dữ liệu từ offline queue"""
        if not self.offline_queue:
            return
        
        synced_count = 0
        failed_items = []
        
        # Process tối đa 10 items mỗi lần để tránh block
        batch_size = min(10, len(self.offline_queue))
        
        for _ in range(batch_size):
            try:
                item = self.offline_queue.popleft()
            except IndexError:
                break
            
            item['attempts'] += 1
            
            try:
                response = self.session.post(
                    f"{self.server_url}{self.endpoints[item['endpoint']]}",
                    json=item['data'],
                    timeout=self.request_timeout
                )
                
                if response.status_code in [200, 201]:
                    # Thành công
                    synced_count += 1
                    
                    if item['type'] == 'event':
                        self.stats['events_sent'] += 1
                    elif item['type'] == 'status':
                        self.stats['status_sent'] += 1
                else:
                    # Thất bại, retry nếu chưa quá max attempts
                    if item['attempts'] < self.max_retries:
                        failed_items.append(item)
                
            except Exception as e:
                # Lỗi network, retry nếu chưa quá max attempts
                if item['attempts'] < self.max_retries:
                    failed_items.append(item)
                else:
                    logger.warning(f"❌ Dropped offline item after {item['attempts']} attempts: {e}")
        
        # Đưa các failed items trở lại queue
        for item in failed_items:
            try:
                self.offline_queue.appendleft(item)
            except:
                # Queue đầy
                self.stats['queue_overflows'] += 1
        
        if synced_count > 0:
            logger.info(f"📡 Synced {synced_count} offline items to server")
    
    def get_connection_status(self) -> Dict[str, Any]:
        """
        Lấy trạng thái kết nối và thống kê
        
        Returns:
            Dict: Thông tin chi tiết về kết nối
                {
                    'is_connected': bool,
                    'server_url': str,
                    'offline_queue_size': int,
                    'connection_failures': int,
                    'stats': dict,
                    'last_health_check': float
                }
        """
        with self.lock:
            return {
                'is_connected': self.is_connected,
                'server_url': self.server_url,
                'offline_queue_size': len(self.offline_queue),
                'connection_failures': self.connection_failures,
                'stats': self.stats.copy(),
                'last_health_check': self.last_health_check,
                'health_check_interval': self.health_check_interval
            }
    
    def force_sync_offline_queue(self) -> int:
        """
        Ép buộc sync toàn bộ offline queue ngay lập tức
        
        Returns:
            int: Số items đã được sync thành công
        """
        if not self.is_connected:
            logger.warning("⚠️ Cannot force sync: not connected to server")
            return 0
        
        logger.info(f"🔄 Force syncing {len(self.offline_queue)} offline items")
        
        original_size = len(self.offline_queue)
        
        while self.offline_queue and self.is_connected:
            self._sync_offline_data()
            time.sleep(0.1)  # Small delay để tránh overwhelm server
        
        synced_count = original_size - len(self.offline_queue)
        logger.info(f"✅ Force sync completed: {synced_count}/{original_size} items synced")
        
        return synced_count
    
    def clear_offline_queue(self):
        """Xóa toàn bộ offline queue"""
        with self.lock:
            queue_size = len(self.offline_queue)
            self.offline_queue.clear()
            logger.info(f"🗑️ Cleared {queue_size} items from offline queue")
    
    def update_server_url(self, new_url: str):
        """
        Cập nhật server URL
        
        Args:
            new_url (str): URL mới
        """
        with self.lock:
            old_url = self.server_url
            self.server_url = new_url.rstrip('/')
            self.is_connected = False
            
            logger.info(f"🔧 Updated server URL: {old_url} -> {self.server_url}")
            
            # Kiểm tra kết nối với URL mới
            self._check_server_health()
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test kết nối đến server và trả về thông tin chi tiết
        
        Returns:
            Dict: Kết quả test
                {
                    'success': bool,
                    'response_time_ms': float,
                    'status_code': int,
                    'error': str (nếu có)
                }
        """
        start_time = time.time()
        
        try:
            response = self.session.get(
                f"{self.server_url}{self.endpoints['health']}",
                timeout=self.connection_timeout
            )
            
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            
            return {
                'success': response.status_code == 200,
                'response_time_ms': round(response_time, 2),
                'status_code': response.status_code,
                'server_url': self.server_url
            }
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            
            return {
                'success': False,
                'response_time_ms': round(response_time, 2),
                'status_code': 0,
                'error': str(e),
                'server_url': self.server_url
            }
    
    def cleanup(self):
        """Dọn dẹp resources và stop threads"""
        logger.info("🧹 Starting ServerSync cleanup...")
        
        # Signal threads to stop
        self.stop_event.set()
        
        # Wait for threads to finish
        if self.health_check_thread and self.health_check_thread.is_alive():
            self.health_check_thread.join(timeout=5)
        
        if self.sync_thread and self.sync_thread.is_alive():
            self.sync_thread.join(timeout=5)
        
        # Try to sync remaining offline data
        if self.offline_queue and self.is_connected:
            logger.info(f"📡 Syncing {len(self.offline_queue)} remaining items before cleanup")
            self.force_sync_offline_queue()
        
        # Close session
        try:
            self.session.close()
        except:
            pass
        
        logger.info("✅ ServerSync cleanup completed")

# Utility functions for testing and configuration

def test_server_connection(server_url: str, timeout: float = 5.0) -> Dict[str, Any]:
    """
    Utility function để test kết nối server không cần tạo ServerSync instance
    
    Args:
        server_url (str): URL server cần test
        timeout (float): Timeout cho test
        
    Returns:
        Dict: Kết quả test
    """
    start_time = time.time()
    
    try:
        response = requests.get(
            f"{server_url.rstrip('/')}/api/health",
            timeout=timeout
        )
        
        response_time = (time.time() - start_time) * 1000
        
        return {
            'success': response.status_code == 200,
            'response_time_ms': round(response_time, 2),
            'status_code': response.status_code,
            'server_url': server_url
        }
        
    except Exception as e:
        response_time = (time.time() - start_time) * 1000
        
        return {
            'success': False,
            'response_time_ms': round(response_time, 2),
            'status_code': 0,
            'error': str(e),
            'server_url': server_url
        }