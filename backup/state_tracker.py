"""
state_tracker.py - Vehicle State Management Module
==================================================
Quản lý trạng thái các xe trong các ô đỗ
Theo dõi lịch sử phát hiện và xác nhận thay đổi trạng thái
"""

import time
import threading
from collections import defaultdict, deque
from typing import Dict, Optional, Any, List
import logging

logger = logging.getLogger(__name__)

class VehicleStateTracker:
    """
    Lớp theo dõi trạng thái xe trong các ô đỗ
    
    Chức năng chính:
    - Theo dõi lịch sử phát hiện của mỗi ô đỗ
    - Xác nhận thay đổi trạng thái dựa trên consistency
    - Ngăn chặn false positive/negative
    - Quản lý timeout cho các trạng thái cũ
    """
    
    def __init__(self, 
                 min_detections: int = 3,
                 max_history: int = 8,
                 consistency_threshold: float = 0.75,
                 state_timeout: int = 3600):
        """
        Khởi tạo state tracker
        
        Args:
            min_detections (int): Số lần phát hiện tối thiểu để xác nhận trạng thái
            max_history (int): Số lượng detection history tối đa lưu trữ
            consistency_threshold (float): Ngưỡng consistency để xác nhận (0.0-1.0)
            state_timeout (int): Timeout cho trạng thái (seconds)
        """
        self.min_detections = min_detections
        self.max_history = max_history
        self.consistency_threshold = consistency_threshold
        self.state_timeout = state_timeout
        
        # Lịch sử phát hiện cho mỗi ô đỗ
        # Format: spot_id -> deque([{occupied: bool, timestamp: float, car_id: str}])
        self.detection_history = defaultdict(lambda: deque(maxlen=max_history))
        
        # Trạng thái đã được xác nhận
        # Format: spot_id -> bool (True=occupied, False=empty)
        self.confirmed_states = {}
        
        # Timestamp của lần thay đổi trạng thái cuối cùng
        # Format: spot_id -> float
        self.state_timestamps = {}
        
        # Lock để đảm bảo thread safety
        self.lock = threading.RLock()
        
        logger.info(f"🔄 VehicleStateTracker initialized with {min_detections} min detections")
    
    def update_detection(self, spot_id: str, is_occupied: bool, car_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Cập nhật phát hiện cho một ô đỗ
        
        Args:
            spot_id (str): ID của ô đỗ
            is_occupied (bool): Có xe hay không
            car_id (str, optional): ID của xe (nếu có)
            
        Returns:
            Dict hoặc None: Thông tin state change nếu có thay đổi được xác nhận
                {
                    'spot_id': str,
                    'new_state': bool,
                    'previous_state': bool,
                    'car_id': str,
                    'confidence': float,
                    'detection_count': int
                }
        """
        with self.lock:
            current_time = time.time()
            
            # Thêm detection mới vào lịch sử
            detection = {
                'occupied': is_occupied,
                'timestamp': current_time,
                'car_id': car_id
            }
            
            self.detection_history[spot_id].append(detection)
            
            # Kiểm tra xem có thay đổi trạng thái không
            state_change = self._check_state_change(spot_id)
            
            return state_change
    
    def _check_state_change(self, spot_id: str) -> Optional[Dict[str, Any]]:
        """
        Kiểm tra xem có thay đổi trạng thái được xác nhận không
        
        Args:
            spot_id (str): ID ô đỗ cần kiểm tra
            
        Returns:
            Dict hoặc None: Thông tin state change nếu có
        """
        history = self.detection_history[spot_id]
        
        # Cần ít nhất min_detections để xác nhận
        if len(history) < self.min_detections:
            return None
        
        # Lấy các detection gần đây
        recent_detections = list(history)[-self.min_detections:]
        
        # Tính consistency
        occupied_count = sum(1 for d in recent_detections if d['occupied'])
        consistency = occupied_count / len(recent_detections)
        
        # Xác định trạng thái mới dựa trên consistency
        new_state = None
        if consistency >= self.consistency_threshold:
            new_state = True  # Occupied
        elif consistency <= (1.0 - self.consistency_threshold):
            new_state = False  # Empty
        else:
            return None  # Không đủ consistent để xác nhận
        
        # Kiểm tra xem có thay đổi so với trạng thái hiện tại không
        current_state = self.confirmed_states.get(spot_id, None)
        
        if current_state != new_state:
            # Có thay đổi trạng thái
            self.confirmed_states[spot_id] = new_state
            self.state_timestamps[spot_id] = time.time()
            
            # Lấy thông tin xe gần nhất
            latest_detection = recent_detections[-1]
            
            logger.info(f"🔄 State change detected: {spot_id} -> {'OCCUPIED' if new_state else 'EMPTY'}")
            
            return {
                'spot_id': spot_id,
                'new_state': new_state,
                'previous_state': current_state,
                'car_id': latest_detection['car_id'] if new_state else None,
                'confidence': consistency,
                'detection_count': len(recent_detections),
                'timestamp': latest_detection['timestamp']
            }
        
        return None
    
    def get_confirmed_state(self, spot_id: str) -> bool:
        """
        Lấy trạng thái đã được xác nhận của ô đỗ
        
        Args:
            spot_id (str): ID ô đỗ
            
        Returns:
            bool: True nếu có xe, False nếu trống
        """
        with self.lock:
            return self.confirmed_states.get(spot_id, False)
    
    def get_state_duration(self, spot_id: str) -> float:
        """
        Lấy thời gian (giây) từ lần thay đổi trạng thái cuối cùng
        
        Args:
            spot_id (str): ID ô đỗ
            
        Returns:
            float: Thời gian tính bằng giây
        """
        with self.lock:
            if spot_id in self.state_timestamps:
                return time.time() - self.state_timestamps[spot_id]
            return 0.0
    
    def get_detection_stats(self, spot_id: str) -> Dict[str, Any]:
        """
        Lấy thống kê phát hiện cho một ô đỗ
        
        Args:
            spot_id (str): ID ô đỗ
            
        Returns:
            Dict: Thống kê chi tiết
                {
                    'total_detections': int,
                    'recent_occupied_ratio': float,
                    'confirmed_state': bool,
                    'state_duration': float,
                    'last_detection_time': float,
                    'confidence': float
                }
        """
        with self.lock:
            history = self.detection_history[spot_id]
            
            if not history:
                return {
                    'total_detections': 0,
                    'recent_occupied_ratio': 0.0,
                    'confirmed_state': False,
                    'state_duration': 0.0,
                    'last_detection_time': 0.0,
                    'confidence': 0.0
                }
            
            # Tính toán các thống kê
            total_detections = len(history)
            recent_detections = list(history)[-self.min_detections:] if len(history) >= self.min_detections else list(history)
            
            occupied_count = sum(1 for d in recent_detections if d['occupied'])
            recent_occupied_ratio = occupied_count / len(recent_detections) if recent_detections else 0.0
            
            # Tính confidence dựa trên consistency
            if len(recent_detections) >= self.min_detections:
                if recent_occupied_ratio >= self.consistency_threshold:
                    confidence = recent_occupied_ratio
                elif recent_occupied_ratio <= (1.0 - self.consistency_threshold):
                    confidence = 1.0 - recent_occupied_ratio
                else:
                    confidence = 0.5  # Không chắc chắn
            else:
                confidence = 0.0
            
            return {
                'total_detections': total_detections,
                'recent_occupied_ratio': recent_occupied_ratio,
                'confirmed_state': self.get_confirmed_state(spot_id),
                'state_duration': self.get_state_duration(spot_id),
                'last_detection_time': history[-1]['timestamp'],
                'confidence': confidence
            }
    
    def cleanup_old_states(self) -> int:
        """
        Dọn dẹp các trạng thái cũ đã timeout
        
        Returns:
            int: Số lượng trạng thái đã được dọn dẹp
        """
        with self.lock:
            current_time = time.time()
            cleaned_count = 0
            
            # Danh sách các spot_id cần xóa
            spots_to_remove = []
            
            for spot_id, timestamp in self.state_timestamps.items():
                if current_time - timestamp > self.state_timeout:
                    spots_to_remove.append(spot_id)
            
            # Xóa các trạng thái cũ
            for spot_id in spots_to_remove:
                self.confirmed_states.pop(spot_id, None)
                self.state_timestamps.pop(spot_id, None)
                # Giữ lại detection history để tham khảo
                cleaned_count += 1
            
            if cleaned_count > 0:
                logger.info(f"🧹 Cleaned up {cleaned_count} old states")
            
            return cleaned_count
    
    def reset_spot_state(self, spot_id: str):
        """
        Reset trạng thái của một ô đỗ cụ thể
        
        Args:
            spot_id (str): ID ô đỗ cần reset
        """
        with self.lock:
            self.detection_history[spot_id].clear()
            self.confirmed_states.pop(spot_id, None)
            self.state_timestamps.pop(spot_id, None)
            
            logger.info(f"🔄 Reset state for spot: {spot_id}")
    
    def reset_all_states(self):
        """Reset tất cả trạng thái"""
        with self.lock:
            self.detection_history.clear()
            self.confirmed_states.clear()
            self.state_timestamps.clear()
            
            logger.info("🔄 Reset all states")
    
    def get_all_spots_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Lấy trạng thái của tất cả các ô đỗ
        
        Returns:
            Dict: spot_id -> status_dict
        """
        with self.lock:
            status = {}
            
            # Lấy tất cả spot_id từ cả confirmed_states và detection_history
            all_spot_ids = set(self.confirmed_states.keys()) | set(self.detection_history.keys())
            
            for spot_id in all_spot_ids:
                status[spot_id] = self.get_detection_stats(spot_id)
            
            return status
    
    def set_parameters(self, 
                      min_detections: Optional[int] = None,
                      consistency_threshold: Optional[float] = None,
                      state_timeout: Optional[int] = None):
        """
        Cập nhật tham số của state tracker
        
        Args:
            min_detections (int, optional): Số lần phát hiện tối thiểu
            consistency_threshold (float, optional): Ngưỡng consistency
            state_timeout (int, optional): Timeout cho trạng thái (seconds)
        """
        with self.lock:
            if min_detections is not None:
                self.min_detections = min_detections
                logger.info(f"🔧 Updated min_detections to {min_detections}")
            
            if consistency_threshold is not None:
                self.consistency_threshold = consistency_threshold
                logger.info(f"🔧 Updated consistency_threshold to {consistency_threshold}")
            
            if state_timeout is not None:
                self.state_timeout = state_timeout
                logger.info(f"🔧 Updated state_timeout to {state_timeout}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Lấy thống kê tổng quan của state tracker
        
        Returns:
            Dict: Thống kê tổng quan
                {
                    'total_spots_tracked': int,
                    'currently_occupied': int,
                    'total_state_changes': int,
                    'average_confidence': float,
                    'uptime_seconds': float
                }
        """
        with self.lock:
            total_spots = len(set(self.confirmed_states.keys()) | set(self.detection_history.keys()))
            currently_occupied = sum(1 for occupied in self.confirmed_states.values() if occupied)
            
            # Tính average confidence từ tất cả spots
            confidences = []
            for spot_id in self.detection_history.keys():
                stats = self.get_detection_stats(spot_id)
                confidences.append(stats['confidence'])
            
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            return {
                'total_spots_tracked': total_spots,
                'currently_occupied': currently_occupied,
                'occupancy_rate': (currently_occupied / total_spots * 100) if total_spots > 0 else 0.0,
                'average_confidence': avg_confidence,
                'parameters': {
                    'min_detections': self.min_detections,
                    'consistency_threshold': self.consistency_threshold,
                    'state_timeout': self.state_timeout
                }
            }