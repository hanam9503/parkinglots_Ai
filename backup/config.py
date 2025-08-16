"""
config.py - Configuration Management Module
===========================================
Chứa tất cả các lớp cấu hình cho hệ thống đa camera
Bao gồm: CameraConfig, SystemConfig và các utility functions
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import json
import logging

logger = logging.getLogger(__name__)

@dataclass
class CameraConfig:
    """
    Cấu hình cho từng camera riêng lẻ
    
    Attributes:
        camera_id (str): ID duy nhất của camera (tự động thêm prefix CAM_ nếu chưa có)
        name (str): Tên hiển thị của camera
        stream_url (str): Đường dẫn video file hoặc RTSP stream
        location_name (str): Tên vị trí lắp đặt camera
        parking_spots_file (str): Đường dẫn đến file JSON chứa tọa độ các ô đỗ xe
        
        # Thông số phát hiện
        vehicle_conf (float): Ngưỡng confidence cho việc phát hiện xe (0.0-1.0)
        plate_conf (float): Ngưỡng confidence cho việc phát hiện biển số (0.0-1.0)  
        intersection_threshold (float): Ngưỡng giao với ô đỗ để xác định xe đã vào (0.0-1.0)
        
        # Thông số xử lý
        frame_skip (int): Bỏ qua n frame để tăng hiệu suất
        yolo_img_size (int): Kích thước ảnh đưa vào YOLO model
        plate_img_size (int): Kích thước ảnh biển số đưa vào model
        
        # Thông số hiệu suất
        enabled (bool): Camera có được kích hoạt hay không
        priority (int): Độ ưu tiên (1=cao, 2=trung bình, 3=thấp)
        max_fps (float): FPS tối đa để xử lý
    """
    camera_id: str
    name: str
    stream_url: str
    location_name: str
    parking_spots_file: str
    
    # Detection settings - Cài đặt phát hiện
    vehicle_conf: float = 0.6
    plate_conf: float = 0.4
    intersection_threshold: float = 0.3
    
    # Processing settings - Cài đặt xử lý
    frame_skip: int = 2
    yolo_img_size: int = 640
    plate_img_size: int = 416
    
    # Performance settings - Cài đặt hiệu suất
    enabled: bool = True
    priority: int = 1
    max_fps: float = 10.0
    
    def __post_init__(self):
        """Xử lý sau khi khởi tạo - đảm bảo camera_id có định dạng đúng"""
        if not self.camera_id.startswith('CAM_'):
            self.camera_id = f"CAM_{self.camera_id}"

@dataclass
class SystemConfig:
    """
    Cấu hình tổng thể của hệ thống
    
    Attributes:
        # Kết nối server
        sync_enabled (bool): Có đồng bộ với server hay không
        sync_server_url (str): URL của server đồng bộ
        connection_timeout (float): Timeout khi kết nối (seconds)
        request_timeout (float): Timeout cho mỗi request (seconds)
        
        # Tối ưu hiệu suất
        max_concurrent_cameras (int): Số camera tối đa chạy đồng thời
        max_workers (int): Số worker threads tối đa
        memory_limit_mb (int): Giới hạn RAM sử dụng (MB)
        
        # Chế độ xử lý
        save_images (bool): Có lưu hình ảnh hay không
        image_quality (int): Chất lượng ảnh JPEG (1-100)
        image_cleanup_hours (int): Xóa ảnh cũ sau n giờ
        
        # Cài đặt ổn định
        min_detection_frames (int): Số frame tối thiểu để xác nhận trạng thái
        max_missed_frames (int): Số frame miss tối đa trước khi reset
        state_timeout_minutes (int): Timeout cho trạng thái ô đỗ (phút)
        
        # Chế độ offline
        enable_offline_mode (bool): Kích hoạt chế độ offline
        offline_queue_size (int): Kích thước queue offline
        sync_interval (float): Khoảng thời gian đồng bộ (seconds)
        
        # Cài đặt giao diện
        display_refresh_rate (float): Tần suất refresh display (seconds)
        status_update_interval (float): Khoảng thời gian cập nhật status (seconds)
    """
    # Server connection - Kết nối server
    sync_enabled: bool = True
    sync_server_url: str = "http://localhost:5000"
    connection_timeout: float = 5.0
    request_timeout: float = 10.0
    
    # Performance optimization - Tối ưu hiệu suất
    max_concurrent_cameras: int = 4
    max_workers: int = 8
    memory_limit_mb: int = 4096
    
    # Processing mode - Chế độ xử lý
    save_images: bool = True
    image_quality: int = 85
    image_cleanup_hours: int = 24
    
    # Stability settings - Cài đặt ổn định
    min_detection_frames: int = 3
    max_missed_frames: int = 5
    state_timeout_minutes: int = 60
    
    # Offline mode - Chế độ offline
    enable_offline_mode: bool = True
    offline_queue_size: int = 500
    sync_interval: float = 15.0
    
    # UI settings - Cài đặt giao diện
    display_refresh_rate: float = 0.5
    status_update_interval: float = 2.0

# Configuration utility functions
def load_camera_configs_from_json(config_file: str) -> List[CameraConfig]:
    """
    Tải cấu hình camera từ file JSON
    
    Args:
        config_file (str): Đường dẫn đến file JSON cấu hình
        
    Returns:
        List[CameraConfig]: Danh sách cấu hình camera
        
    JSON Format:
    {
        "cameras": [
            {
                "camera_id": "CAM_001",
                "name": "Camera 1",
                "stream_url": "video.mp4",
                "location_name": "Tầng 1",
                "parking_spots_file": "spots1.json",
                "vehicle_conf": 0.6,
                ...
            }
        ]
    }
    """
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        configs = []
        for cam_data in data.get('cameras', []):
            config = CameraConfig(
                camera_id=cam_data['camera_id'],
                name=cam_data['name'],
                stream_url=cam_data['stream_url'],
                location_name=cam_data['location_name'],
                parking_spots_file=cam_data['parking_spots_file'],
                vehicle_conf=cam_data.get('vehicle_conf', 0.6),
                plate_conf=cam_data.get('plate_conf', 0.4),
                intersection_threshold=cam_data.get('intersection_threshold', 0.3),
                frame_skip=cam_data.get('frame_skip', 2),
                yolo_img_size=cam_data.get('yolo_img_size', 640),
                priority=cam_data.get('priority', 2),
                max_fps=cam_data.get('max_fps', 10.0),
                enabled=cam_data.get('enabled', True)
            )
            configs.append(config)
        
        logger.info(f"✅ Loaded {len(configs)} camera configurations from {config_file}")
        return configs
        
    except Exception as e:
        logger.error(f"❌ Failed to load camera configs from {config_file}: {e}")
        return []

def load_system_config_from_json(config_file: str) -> SystemConfig:
    """
    Tải cấu hình hệ thống từ file JSON
    
    Args:
        config_file (str): Đường dẫn đến file JSON cấu hình hệ thống
        
    Returns:
        SystemConfig: Cấu hình hệ thống
    """
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        config = SystemConfig(**data)
        logger.info(f"✅ Loaded system configuration from {config_file}")
        return config
        
    except Exception as e:
        logger.error(f"❌ Failed to load system config from {config_file}: {e}")
        logger.info("🔄 Using default system configuration")
        return SystemConfig()

def save_config_to_json(config, output_file: str):
    """
    Lưu cấu hình ra file JSON
    
    Args:
        config: CameraConfig hoặc SystemConfig object
        output_file (str): Đường dẫn file đầu ra
    """
    try:
        from dataclasses import asdict
        
        config_dict = asdict(config)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Saved configuration to {output_file}")
        
    except Exception as e:
        logger.error(f"❌ Failed to save config to {output_file}: {e}")

def create_default_configs():
    """
    Tạo các cấu hình mặc định cho demo/testing
    
    Returns:
        Tuple[SystemConfig, List[CameraConfig]]: Cấu hình hệ thống và danh sách camera
    """
    # System configuration - Cấu hình hệ thống
    system_config = SystemConfig(
        sync_enabled=True,
        sync_server_url="http://localhost:5000",
        max_concurrent_cameras=3,
        max_workers=8,
        save_images=True,
        enable_offline_mode=True
    )
    
    # Camera configurations - Cấu hình camera
    camera_configs = [
        CameraConfig(
            camera_id="CAM_001",
            name="Tầng 1 - Khu A",
            stream_url="videos/camera1.mp4",
            location_name="Tầng 1 - Khu A",
            parking_spots_file="config/camera1_spots.json",
            vehicle_conf=0.6,
            plate_conf=0.4,
            frame_skip=2,
            priority=1,  # High priority
            max_fps=15.0
        ),
        CameraConfig(
            camera_id="CAM_002", 
            name="Tầng 1 - Khu B",
            stream_url="videos/camera2.mp4",
            location_name="Tầng 1 - Khu B", 
            parking_spots_file="config/camera2_spots.json",
            vehicle_conf=0.6,
            plate_conf=0.4,
            frame_skip=2,
            priority=2,  # Normal priority
            max_fps=12.0
        ),
        CameraConfig(
            camera_id="CAM_003",
            name="Tầng 2 - Khu A", 
            stream_url="videos/camera3.mp4",
            location_name="Tầng 2 - Khu A",
            parking_spots_file="config/camera3_spots.json",
            vehicle_conf=0.5,
            plate_conf=0.4,
            frame_skip=3,
            priority=2,
            max_fps=10.0
        ),
        CameraConfig(
            camera_id="CAM_004",
            name="Lối vào chính",
            stream_url="videos/entrance.mp4", 
            location_name="Lối vào chính",
            parking_spots_file="config/entrance_spots.json",
            vehicle_conf=0.7,
            plate_conf=0.5,
            frame_skip=1,
            priority=1,  # High priority cho lối vào
            max_fps=20.0,
            enabled=False  # Disabled by default
        )
    ]
    
    return system_config, camera_configs

def validate_camera_config(config: CameraConfig) -> List[str]:
    """
    Kiểm tra tính hợp lệ của cấu hình camera
    
    Args:
        config (CameraConfig): Cấu hình camera cần kiểm tra
        
    Returns:
        List[str]: Danh sách các lỗi (empty nếu hợp lệ)
    """
    errors = []
    
    # Kiểm tra các trường bắt buộc
    if not config.camera_id:
        errors.append("camera_id không được để trống")
    
    if not config.name:
        errors.append("name không được để trống")
    
    if not config.stream_url:
        errors.append("stream_url không được để trống")
    
    if not config.parking_spots_file:
        errors.append("parking_spots_file không được để trống")
    
    # Kiểm tra giá trị số
    if not (0.0 <= config.vehicle_conf <= 1.0):
        errors.append("vehicle_conf phải từ 0.0 đến 1.0")
    
    if not (0.0 <= config.plate_conf <= 1.0):
        errors.append("plate_conf phải từ 0.0 đến 1.0")
    
    if not (0.0 <= config.intersection_threshold <= 1.0):
        errors.append("intersection_threshold phải từ 0.0 đến 1.0")
    
    if config.frame_skip < 1:
        errors.append("frame_skip phải >= 1")
    
    if config.yolo_img_size < 32:
        errors.append("yolo_img_size phải >= 32")
    
    if config.max_fps <= 0:
        errors.append("max_fps phải > 0")
    
    if config.priority not in [1, 2, 3]:
        errors.append("priority phải là 1, 2 hoặc 3")
    
    return errors

def validate_system_config(config: SystemConfig) -> List[str]:
    """
    Kiểm tra tính hợp lệ của cấu hình hệ thống
    
    Args:
        config (SystemConfig): Cấu hình hệ thống cần kiểm tra
        
    Returns:
        List[str]: Danh sách các lỗi (empty nếu hợp lệ)
    """
    errors = []
    
    # Kiểm tra các thông số timeout
    if config.connection_timeout <= 0:
        errors.append("connection_timeout phải > 0")
    
    if config.request_timeout <= 0:
        errors.append("request_timeout phải > 0")
    
    # Kiểm tra các thông số hiệu suất
    if config.max_concurrent_cameras < 1:
        errors.append("max_concurrent_cameras phải >= 1")
    
    if config.max_workers < 1:
        errors.append("max_workers phải >= 1")
    
    if config.memory_limit_mb < 512:
        errors.append("memory_limit_mb phải >= 512")
    
    # Kiểm tra chất lượng ảnh
    if not (1 <= config.image_quality <= 100):
        errors.append("image_quality phải từ 1 đến 100")
    
    # Kiểm tra các thông số ổn định
    if config.min_detection_frames < 1:
        errors.append("min_detection_frames phải >= 1")
    
    if config.max_missed_frames < 1:
        errors.append("max_missed_frames phải >= 1")
    
    if config.state_timeout_minutes < 1:
        errors.append("state_timeout_minutes phải >= 1")
    
    # Kiểm tra offline settings
    if config.offline_queue_size < 10:
        errors.append("offline_queue_size phải >= 10")
    
    if config.sync_interval <= 0:
        errors.append("sync_interval phải > 0")
    
    return errors