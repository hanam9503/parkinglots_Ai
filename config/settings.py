"""
Enhanced Configuration Settings
Cấu hình toàn cục cho hệ thống parking
"""

import os
from dataclasses import dataclass
from typing import Tuple, Optional
from ultralytics import YOLO

@dataclass
class EnhancedConfig:
    """Enhanced configuration class with comprehensive settings"""
    
    # System identification
    CAMERA_ID: str = "CAM_001"
    LOCATION_NAME: str = "Bãi đỗ xe tầng 1"
    
    # Model Paths
    VEHICLE_MODEL_PATH: str = "model\yolo11s.pt"
    PLATE_MODEL_PATH: str = "model\license-plate-finetune-v1m.pt"
    ESRGAN_MODEL_PATH: str = "weights/RealESRGAN_x4plus.pth"
    
    # Server Connection
    SYNC_ENABLED: bool = True
    SYNC_SERVER_URL: str = "http://localhost:5000"
    CONNECTION_TIMEOUT: float = 5.0
    REQUEST_TIMEOUT: float = 10.0
    MAX_RETRY_ATTEMPTS: int = 3
    RETRY_DELAY: float = 2.0
    HEALTH_CHECK_INTERVAL: float = 30.0
    
    # Performance Optimization
    FRAME_SKIP: int = 2
    YOLO_IMG_SIZE: int = 640
    PLATE_IMG_SIZE: int = 416
    MAX_WORKERS: int = 4
    
    # Detection thresholds
    VEHICLE_CONF: float = 0.6
    PLATE_CONF: float = 0.4
    OCR_MIN_CONF: float = 0.4
    INTERSECTION_THRESHOLD: float = 0.3
    
    # Enhancement settings
    USE_REAL_ESRGAN: bool = True
    ESRGAN_SCALE: int = 4
    ESRGAN_TILE_SIZE: int = 128
    ENABLE_SMART_ENHANCEMENT: bool = True
    
    # Processing mode
    ENABLE_CACHING: bool = True
    CACHE_SIZE: int = 100
    CACHE_EXPIRE_MINUTES: int = 30
    
    # Offline mode settings
    ENABLE_OFFLINE_MODE: bool = True
    OFFLINE_QUEUE_SIZE: int = 200
    SYNC_INTERVAL: float = 15.0
    BATCH_SYNC_SIZE: int = 10
    
    # Image settings
    MAX_PLATE_SIZE: Tuple[int, int] = (300, 100)
    IMAGE_QUALITY: int = 85
    SAVE_IMAGES: bool = True
    IMAGE_CLEANUP_HOURS: int = 24
    
    # Stability settings
    MIN_DETECTION_FRAMES: int = 3
    MAX_MISSED_FRAMES: int = 5
    STATE_TIMEOUT_MINUTES: int = 60
    
    # Logging settings
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE: str = 'logs/parking_system.log'
    LOG_MAX_BYTES: int = 10 * 1024 * 1024  # 10MB
    LOG_BACKUP_COUNT: int = 5
    
    # Video settings
    DEFAULT_VIDEO_PATH: str = "videos/0723.mp4"
    DISPLAY_WINDOW_SIZE: Tuple[int, int] = (1280, 720)
    
    # Statistics intervals
    SUMMARY_INTERVAL: float = 30.0  # seconds
    STATUS_INTERVAL: float = 120.0  # seconds
    PERFORMANCE_DISPLAY_INTERVAL: int = 30  # frames
    MEMORY_CLEANUP_INTERVAL: int = 200  # frames
    
    # File paths
    PARKING_SPOTS_CONFIG: str = "config/parking_spots.json"
    WEIGHTS_DIR: str = "weights"
    VEHICLE_IMAGES_DIR: str = "vehicle_images"
    PLATE_IMAGES_DIR: str = "plate_images"
    ENHANCED_IMAGES_DIR: str = "enhanced_images"
    LOGS_DIR: str = "logs"
    
    def __post_init__(self):
        """Post-initialization validation and setup"""
        # Validate model paths
        if not os.path.exists(self.VEHICLE_MODEL_PATH):
            print(f"⚠️ Warning: Vehicle model not found at {self.VEHICLE_MODEL_PATH}")
        
        if not os.path.exists(self.PLATE_MODEL_PATH):
            print(f"⚠️ Warning: Plate model not found at {self.PLATE_MODEL_PATH}")
        
        # Validate thresholds
        if not 0 <= self.VEHICLE_CONF <= 1:
            raise ValueError(f"VEHICLE_CONF must be between 0 and 1, got {self.VEHICLE_CONF}")
        
        if not 0 <= self.PLATE_CONF <= 1:
            raise ValueError(f"PLATE_CONF must be between 0 and 1, got {self.PLATE_CONF}")
        
        if not 0 <= self.OCR_MIN_CONF <= 1:
            raise ValueError(f"OCR_MIN_CONF must be between 0 and 1, got {self.OCR_MIN_CONF}")
        
        # Validate intervals
        if self.FRAME_SKIP < 1:
            raise ValueError(f"FRAME_SKIP must be >= 1, got {self.FRAME_SKIP}")
        
        if self.MAX_WORKERS < 1:
            raise ValueError(f"MAX_WORKERS must be >= 1, got {self.MAX_WORKERS}")
        
        # Validate image settings
        if not 1 <= self.IMAGE_QUALITY <= 100:
            raise ValueError(f"IMAGE_QUALITY must be between 1 and 100, got {self.IMAGE_QUALITY}")
    
    @classmethod
    def from_env(cls) -> 'EnhancedConfig':
        """Create config from environment variables"""
        return cls(
            CAMERA_ID=os.getenv('CAMERA_ID', cls.CAMERA_ID),
            LOCATION_NAME=os.getenv('LOCATION_NAME', cls.LOCATION_NAME),
            SYNC_SERVER_URL=os.getenv('SYNC_SERVER_URL', cls.SYNC_SERVER_URL),
            VEHICLE_CONF=float(os.getenv('VEHICLE_CONF', cls.VEHICLE_CONF)),
            PLATE_CONF=float(os.getenv('PLATE_CONF', cls.PLATE_CONF)),
            USE_REAL_ESRGAN=os.getenv('USE_REAL_ESRGAN', str(cls.USE_REAL_ESRGAN)).lower() == 'true',
            ENABLE_OFFLINE_MODE=os.getenv('ENABLE_OFFLINE_MODE', str(cls.ENABLE_OFFLINE_MODE)).lower() == 'true',
            LOG_LEVEL=os.getenv('LOG_LEVEL', cls.LOG_LEVEL),
        )
    
    def get_server_endpoints(self) -> dict:
        """Get server endpoint URLs"""
        return {
            'events': f"{self.SYNC_SERVER_URL}/api/events",
            'status': f"{self.SYNC_SERVER_URL}/api/status",
            'health': f"{self.SYNC_SERVER_URL}/api/health",
            'dashboard': f"{self.SYNC_SERVER_URL}/api/camera-dashboard"
        }
    
    def validate_paths(self) -> bool:
        """Validate all required paths exist"""
        paths_to_check = [
            self.VEHICLE_MODEL_PATH,
            self.PLATE_MODEL_PATH,
            self.PARKING_SPOTS_CONFIG
        ]
        
        all_valid = True
        for path in paths_to_check:
            if not os.path.exists(path):
                print(f"❌ Missing file: {path}")
                all_valid = False
        
        return all_valid
    
    def display_config(self):
        """Display current configuration"""
        print("\n📋 Current Configuration:")
        print(f"   🏢 Location: {self.LOCATION_NAME}")
        print(f"   📹 Camera ID: {self.CAMERA_ID}")
        print(f"   🌐 Server: {self.SYNC_SERVER_URL}")
        print(f"   🎯 Vehicle Confidence: {self.VEHICLE_CONF}")
        print(f"   🎯 Plate Confidence: {self.PLATE_CONF}")
        print(f"   🔧 Enhancement: {'Enabled' if self.USE_REAL_ESRGAN else 'Disabled'}")
        print(f"   📡 Offline Mode: {'Enabled' if self.ENABLE_OFFLINE_MODE else 'Disabled'}")
        print(f"   💾 Caching: {'Enabled' if self.ENABLE_CACHING else 'Disabled'}")
        print(f"   🖼️ Image Quality: {self.IMAGE_QUALITY}%")

# Create global config instance
config = EnhancedConfig()

# Environment-based configuration (if needed)
if os.getenv('USE_ENV_CONFIG', 'false').lower() == 'true':
    config = EnhancedConfig.from_env()

# Display configuration info
if __name__ == "__main__":
    config.display_config()
    print(f"\n✅ Configuration validation: {'Passed' if config.validate_paths() else 'Failed'}")