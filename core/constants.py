"""
Constants và Enums cho Enhanced Parking System
Định nghĩa các hằng số và enum được sử dụng trong toàn hệ thống
"""

from enum import Enum, IntEnum, auto
from typing import Dict, List, Set, Tuple

# Version information
VERSION = "2.0.0"
BUILD_DATE = "2024-08-04"
AUTHOR = "Enhanced Parking System Team"

# Event types
class EventType(Enum):
    ENTER = "enter"
    EXIT = "exit"
    UPDATE = "update"
    ERROR = "error"

# Vehicle types
class VehicleType(Enum):
    CAR = "car"
    TRUCK = "truck"
    BUS = "bus"
    MOTORCYCLE = "motorcycle"
    BICYCLE = "bicycle"
    UNKNOWN = "unknown"

# Parking spot types
class SpotType(Enum):
    STANDARD = "standard"
    VIP = "vip"
    ACCESSIBILITY = "accessibility"
    COMPACT = "compact"
    TRUCK = "truck"
    ELECTRIC = "electric"
    RESERVED = "reserved"

# System states
class SystemState(Enum):
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    MAINTENANCE = "maintenance"

# Connection states
class ConnectionState(Enum):
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    RECONNECTING = "reconnecting"
    FAILED = "failed"
    TIMEOUT = "timeout"

# Processing states
class ProcessingState(Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    QUEUED = "queued"
    CANCELLED = "cancelled"

# Priority levels
class Priority(IntEnum):
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3
    BACKGROUND = 4

# Log levels mapping
LOG_LEVELS = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50
}

# Error codes
class ErrorCode(Enum):
    # System errors
    SYSTEM_INIT_ERROR = "SYSTEM_INIT_ERROR"
    SYSTEM_CONFIG_ERROR = "SYSTEM_CONFIG_ERROR"
    SYSTEM_RESOURCE_ERROR = "SYSTEM_RESOURCE_ERROR"
    
    # Model errors
    MODEL_LOAD_ERROR = "MODEL_LOAD_ERROR"
    MODEL_INFERENCE_ERROR = "MODEL_INFERENCE_ERROR"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    
    # Detection errors
    VEHICLE_DETECTION_ERROR = "VEHICLE_DETECTION_ERROR"
    PLATE_DETECTION_ERROR = "PLATE_DETECTION_ERROR"
    DETECTION_TIMEOUT = "DETECTION_TIMEOUT"
    
    # Processing errors
    IMAGE_PROCESSING_ERROR = "IMAGE_PROCESSING_ERROR"
    OCR_ERROR = "OCR_ERROR"
    ENHANCEMENT_ERROR = "ENHANCEMENT_ERROR"
    
    # Validation errors
    INVALID_EVENT_DATA = "INVALID_EVENT_DATA"
    INVALID_STATUS_DATA = "INVALID_STATUS_DATA"
    INVALID_PLATE = "INVALID_PLATE"
    INVALID_CONFIG = "INVALID_CONFIG"
    
    # File errors
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    FILE_READ_ERROR = "FILE_READ_ERROR"
    FILE_WRITE_ERROR = "FILE_WRITE_ERROR"
    FILE_PERMISSION_ERROR = "FILE_PERMISSION_ERROR"
    
    # Network errors
    SERVER_CONNECTION_ERROR = "SERVER_CONNECTION_ERROR"
    DATA_SYNC_ERROR = "DATA_SYNC_ERROR"
    OFFLINE_QUEUE_ERROR = "OFFLINE_QUEUE_ERROR"
    REQUEST_TIMEOUT = "REQUEST_TIMEOUT"
    
    # Unknown error
    UNKNOWN_ERROR = "UNKNOWN_ERROR"

# HTTP status codes
HTTP_STATUS_CODES = {
    200: "OK",
    201: "Created",
    400: "Bad Request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not Found",
    409: "Conflict",
    500: "Internal Server Error",
    502: "Bad Gateway",
    503: "Service Unavailable",
    504: "Gateway Timeout"
}

# Image formats
SUPPORTED_IMAGE_FORMATS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"
}

# Video formats
SUPPORTED_VIDEO_FORMATS = {
    ".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"
}

# Default configurations
DEFAULT_CONFIG = {
    # System
    "FRAME_SKIP": 2,
    "MAX_WORKERS": 4,
    "DEVICE": "auto",  # auto, cpu, cuda
    
    # Detection thresholds
    "VEHICLE_CONF": 0.6,
    "PLATE_CONF": 0.4,
    "OCR_MIN_CONF": 0.4,
    "INTERSECTION_THRESHOLD": 0.3,
    
    # Image processing
    "YOLO_IMG_SIZE": 640,
    "PLATE_IMG_SIZE": 416,
    "MAX_PLATE_SIZE": (300, 100),
    "IMAGE_QUALITY": 85,
    
    # Enhancement
    "USE_REAL_ESRGAN": True,
    "ESRGAN_SCALE": 4,
    "ESRGAN_TILE_SIZE": 128,
    
    # Caching
    "ENABLE_CACHING": True,
    "CACHE_SIZE": 100,
    "CACHE_EXPIRE_MINUTES": 30,
    
    # Offline mode
    "ENABLE_OFFLINE_MODE": True,
    "OFFLINE_QUEUE_SIZE": 200,
    "SYNC_INTERVAL": 15.0,
    "BATCH_SYNC_SIZE": 10,
    
    # Timeouts
    "CONNECTION_TIMEOUT": 5.0,
    "REQUEST_TIMEOUT": 10.0,
    "HEALTH_CHECK_INTERVAL": 30.0,
    
    # Retry settings
    "MAX_RETRY_ATTEMPTS": 3,
    "RETRY_DELAY": 2.0,
    
    # State tracking
    "MIN_DETECTION_FRAMES": 3,
    "MAX_MISSED_FRAMES": 5,
    "STATE_TIMEOUT_MINUTES": 60,
    
    # File cleanup
    "IMAGE_CLEANUP_HOURS": 24,
    "LOG_CLEANUP_DAYS": 7,
    
    # Display
    "DISPLAY_WINDOW_SIZE": (1280, 720),
    "SUMMARY_INTERVAL": 30.0,
    "STATUS_INTERVAL": 120.0
}

# Vietnamese province codes for license plates
VIETNAMESE_PROVINCE_CODES = {
    # Hanoi area
    "11": "Cao Bằng", "12": "Lạng Sơn", "13": "Quảng Ninh", "14": "Hải Phòng",
    "15": "Hải Dương", "16": "Hưng Yên", "17": "Thái Bình", "18": "Nam Định",
    "19": "Ninh Bình",
    
    # Ho Chi Minh City area
    "29": "Hà Nội", "30": "TP.HCM", "31": "TP.HCM", "32": "TP.HCM",
    "33": "TP.HCM", "34": "TP.HCM", "35": "TP.HCM", "36": "TP.HCM",
    "37": "TP.HCM", "38": "TP.HCM",
    
    # Other provinces
    "43": "Đà Nẵng", "44": "Quảng Nam", "47": "Đắk Lắk", "48": "Đắk Nông",
    "49": "Lâm Đồng", "50": "TP.HCM", "51": "Bà Rịa-Vũng Tàu", "52": "Tây Ninh",
    "53": "Long An", "54": "An Giang", "55": "Đồng Tháp", "56": "Tiền Giang",
    "57": "Vĩnh Long", "58": "Bến Tre", "59": "Kiên Giang", "60": "Cà Mau",
    "61": "TP Cần Thơ", "62": "Hậu Giang", "63": "Sóc Trăng", "64": "Trà Vinh",
    "65": "Bạc Liêu", "66": "Cà Mau", "67": "Tây Ninh", "68": "Bến Tre",
    "69": "Cần Thơ", "70": "Lâm Đồng", "71": "Bình Dương", "72": "Bình Phước",
    "73": "Bình Định", "74": "Vĩnh Long", "75": "Đồng Nai", "76": "Bình Thuận",
    "77": "Tây Ninh", "78": "TP.HCM", "79": "TP.HCM", "80": "Long An",
    "81": "Kiên Giang", "82": "Cần Thơ", "83": "Sóc Trăng", "84": "Trà Vinh",
    "85": "Ninh Thuận", "86": "Bình Thuận", "87": "Đồng Nai", "88": "Vĩnh Phúc",
    "89": "Hà Giang", "90": "Tuyên Quang", "91": "Lào Cai", "92": "Điện Biên",
    "93": "Sơn La", "94": "Lai Châu", "95": "Bắc Kạn", "96": "Thái Nguyên",
    "97": "Phú Thọ", "98": "Vĩnh Phúc", "99": "Bắc Giang"
}

# Special license plate codes
SPECIAL_PLATE_CODES = {
    "LD": "Lãnh đạo",
    "QN": "Quân đội",
    "NG": "Ngoại giao",
    "TQ": "Tòa án Quân sự",
    "HC": "Hải quan",
    "CD": "Công an"
}

# OCR correction mapping
OCR_CORRECTIONS = {
    "O": "0", "I": "1", "L": "1", "Z": "2", "S": "5",
    "G": "6", "Q": "0", "D": "0", "T": "7", "B": "8",
    "A": "4", "E": "3", "U": "0"
}

# Color definitions (BGR format for OpenCV)
COLORS = {
    "RED": (0, 0, 255),
    "GREEN": (0, 255, 0),
    "BLUE": (255, 0, 0),
    "YELLOW": (0, 255, 255),
    "ORANGE": (0, 165, 255),
    "PURPLE": (255, 0, 255),
    "CYAN": (255, 255, 0),
    "WHITE": (255, 255, 255),
    "BLACK": (0, 0, 0),
    "GRAY": (128, 128, 128),
    "LIGHT_GREEN": (144, 238, 144),
    "LIGHT_RED": (173, 216, 230),
    "LIGHT_BLUE": (255, 182, 193)
}

# Spot visualization colors
SPOT_COLORS = {
    SpotType.STANDARD: COLORS["GREEN"],
    SpotType.VIP: COLORS["PURPLE"],
    SpotType.ACCESSIBILITY: COLORS["BLUE"],
    SpotType.COMPACT: COLORS["ORANGE"],
    SpotType.TRUCK: COLORS["YELLOW"],
    SpotType.ELECTRIC: COLORS["CYAN"],
    SpotType.RESERVED: COLORS["RED"]
}

# Occupied spot colors by confidence
OCCUPIED_COLORS = {
    "HIGH": COLORS["RED"],        # > 0.8
    "MEDIUM": COLORS["ORANGE"],   # > 0.5
    "LOW": COLORS["LIGHT_RED"],   # > 0.0
    "UNKNOWN": COLORS["YELLOW"]   # = 0.0
}

# Performance thresholds
PERFORMANCE_THRESHOLDS = {
    "MIN_FPS": 10.0,
    "MAX_PROCESSING_TIME": 0.5,  # seconds
    "MAX_MEMORY_USAGE": 80.0,    # percentage
    "MAX_CPU_USAGE": 90.0,       # percentage
    "MAX_GPU_MEMORY": 90.0,      # percentage
    "MIN_DETECTION_RATE": 70.0,  # percentage
    "MIN_PLATE_CONFIDENCE": 0.4,
    "MIN_VEHICLE_CONFIDENCE": 0.6
}

# File paths and directories
DEFAULT_PATHS = {
    "CONFIG_DIR": "config",
    "WEIGHTS_DIR": "weights",
    "LOGS_DIR": "logs",
    "IMAGES_DIR": "images",
    "VEHICLE_IMAGES_DIR": "vehicle_images",
    "PLATE_IMAGES_DIR": "plate_images",
    "ENHANCED_IMAGES_DIR": "enhanced_images",
    "TEMP_DIR": "temp",
    "BACKUP_DIR": "backup"
}

# Model URLs for download
MODEL_URLS = {
    "REAL_ESRGAN": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "YOLO_VEHICLE": "https://github.com/ultralytics/yolov8/releases/download/v8.0.0/yolov8n.pt",
    "YOLO_PLATE": "https://github.com/ultralytics/yolov8/releases/download/v8.0.0/yolov8n.pt"
}

# API endpoints
API_ENDPOINTS = {
    "EVENTS": "/api/events",
    "STATUS": "/api/status",
    "HEALTH": "/api/health",
    "DASHBOARD": "/api/camera-dashboard",
    "STATISTICS": "/api/statistics",
    "CONFIG": "/api/config"
}

# Database collections/tables
DB_COLLECTIONS = {
    "EVENTS": "parking_events",
    "STATUS": "parking_status",
    "STATISTICS": "system_statistics",
    "LOGS": "system_logs",
    "CONFIG": "system_config"
}

# Regular expressions for validation
REGEX_PATTERNS = {
    # Vietnamese license plate patterns
    "PLATE_STANDARD": r"^(\d{2})([A-Z])(\d{4,5})$",  # 30A-12345
    "PLATE_NEW": r"^(\d{2})([A-Z]{2})(\d{4})$",      # 30AA-1234
    "PLATE_OLD": r"^(\d{3})([A-Z])(\d{2,4})$",       # 123A-456
    "PLATE_SPECIAL": r"^([A-Z]{2})(\d{3,5})$",       # LD-12345
    
    # General patterns
    "UUID": r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    "TIMESTAMP_ISO": r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}",
    "IP_ADDRESS": r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$",
    "URL": r"^https?://[^\s/$.?#].[^\s]*$",
    "EMAIL": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
}

# System limits
SYSTEM_LIMITS = {
    "MAX_QUEUE_SIZE": 1000,
    "MAX_CACHE_SIZE": 500,
    "MAX_IMAGE_SIZE": 10 * 1024 * 1024,  # 10MB
    "MAX_VIDEO_SIZE": 100 * 1024 * 1024,  # 100MB
    "MAX_LOG_FILE_SIZE": 50 * 1024 * 1024,  # 50MB
    "MAX_BACKUP_FILES": 10,
    "MAX_CONCURRENT_REQUESTS": 20,
    "MAX_RETRY_ATTEMPTS": 5,
    "MAX_PROCESSING_TIME": 30.0,  # seconds
    "MIN_DISK_SPACE": 1024 * 1024 * 1024,  # 1GB
    "MIN_MEMORY": 2 * 1024 * 1024 * 1024,  # 2GB
}

# Message templates
MESSAGE_TEMPLATES = {
    "VEHICLE_ENTER": "🚗 {spot_name}: Xe vào - {plate_text} ({confidence:.3f})",
    "VEHICLE_EXIT": "🚪 {spot_name}: Xe ra - {plate_text} ({duration}m)",
    "DETECTION_SUCCESS": "✅ Phát hiện thành công: {count} xe",
    "DETECTION_FAILED": "❌ Phát hiện thất bại: {reason}",
    "SERVER_CONNECTED": "🌐 Kết nối server thành công: {url}",
    "SERVER_DISCONNECTED": "📡 Mất kết nối server: {url}",
    "SYNC_SUCCESS": "✅ Đồng bộ thành công: {count} items",
    "SYNC_FAILED": "❌ Đồng bộ thất bại: {reason}",
    "SYSTEM_READY": "🚀 Hệ thống đã sẵn sàng",
    "SYSTEM_SHUTDOWN": "🔚 Hệ thống đã tắt"
}

# Statistics keys
STATS_KEYS = {
    "FRAMES_PROCESSED": "frames_processed",
    "EVENTS_GENERATED": "events_generated", 
    "EVENTS_SENT": "events_sent",
    "EVENTS_QUEUED": "events_queued",
    "PLATES_DETECTED": "plates_detected",
    "PLATES_VALIDATED": "plates_validated",
    "PROCESSING_TIME": "processing_time",
    "DETECTION_TIME": "detection_time",
    "OCR_TIME": "ocr_time",
    "ENHANCEMENT_TIME": "enhancement_time",
    "SYNC_TIME": "sync_time",
    "CACHE_HITS": "cache_hits",
    "CACHE_MISSES": "cache_misses",
    "SERVER_REQUESTS": "server_requests",
    "SERVER_ERRORS": "server_errors"
}

# Feature flags
FEATURE_FLAGS = {
    "ENABLE_GPU_ACCELERATION": True,
    "ENABLE_MULTI_THREADING": True,
    "ENABLE_REAL_TIME_SYNC": True,
    "ENABLE_AUTOMATIC_BACKUP": True,
    "ENABLE_PERFORMANCE_MONITORING": True,
    "ENABLE_ADVANCED_LOGGING": True,
    "ENABLE_WEB_DASHBOARD": True,
    "ENABLE_EMAIL_NOTIFICATIONS": False,
    "ENABLE_SMS_NOTIFICATIONS": False,
    "ENABLE_MOBILE_APP_SYNC": False
}

# Quality thresholds
QUALITY_THRESHOLDS = {
    "MIN_IMAGE_QUALITY": 0.5,
    "MIN_PLATE_AREA": 100,  # pixels
    "MAX_PLATE_AREA": 50000,  # pixels
    "MIN_VEHICLE_AREA": 1000,  # pixels
    "MAX_VEHICLE_AREA": 500000,  # pixels
    "MIN_ASPECT_RATIO": 0.5,
    "MAX_ASPECT_RATIO": 5.0,
    "MIN_DETECTION_SIZE": 32,  # pixels
    "MAX_BLUR_THRESHOLD": 100.0
}

# Timing constants
TIMING = {
    "FRAME_TIMEOUT": 1.0,  # seconds
    "DETECTION_TIMEOUT": 5.0,
    "OCR_TIMEOUT": 3.0,
    "ENHANCEMENT_TIMEOUT": 10.0,
    "SYNC_TIMEOUT": 30.0,
    "HEALTH_CHECK_TIMEOUT": 5.0,
    "STARTUP_TIMEOUT": 60.0,
    "SHUTDOWN_TIMEOUT": 30.0
}

# Memory management
MEMORY = {
    "GC_INTERVAL": 100,  # frames
    "CACHE_CLEANUP_INTERVAL": 1000,  # frames
    "GPU_MEMORY_FRACTION": 0.8,
    "MAX_BATCH_SIZE": 16,
    "PREALLOC_SIZE": 1024 * 1024,  # 1MB
    "BUFFER_SIZE": 4096
}

# Network settings
NETWORK = {
    "MAX_CONNECTIONS": 10,
    "KEEP_ALIVE": True,
    "POOL_CONNECTIONS": 10,
    "POOL_MAXSIZE": 20,
    "USER_AGENT": "ParkingSystem/2.0.0",
    "ACCEPT_ENCODING": "gzip, deflate",
    "CONTENT_TYPE": "application/json",
    "CHARSET": "utf-8"
}

# Security settings
SECURITY = {
    "MAX_LOGIN_ATTEMPTS": 5,
    "SESSION_TIMEOUT": 3600,  # seconds
    "TOKEN_EXPIRY": 86400,  # seconds
    "ALLOWED_ORIGINS": ["localhost", "127.0.0.1"],
    "ENCRYPTION_ALGORITHM": "AES-256-GCM",
    "HASH_ALGORITHM": "SHA-256"
}

# Validation rules
VALIDATION_RULES = {
    "MIN_SPOT_AREA": 10000,  # pixels
    "MAX_SPOT_AREA": 100000,  # pixels
    "MIN_SPOT_DISTANCE": 50,  # pixels
    "MAX_SPOT_OVERLAP": 0.1,  # ratio
    "MIN_POLYGON_POINTS": 3,
    "MAX_POLYGON_POINTS": 20,
    "MIN_CONFIDENCE": 0.0,
    "MAX_CONFIDENCE": 1.0,
    "MIN_TEXT_LENGTH": 3,
    "MAX_TEXT_LENGTH": 15
}

# Helper functions
def get_color_by_confidence(confidence: float) -> Tuple[int, int, int]:
    """Get color based on confidence level"""
    if confidence > 0.8:
        return OCCUPIED_COLORS["HIGH"]
    elif confidence > 0.5:
        return OCCUPIED_COLORS["MEDIUM"]
    elif confidence > 0.0:
        return OCCUPIED_COLORS["LOW"]
    else:
        return OCCUPIED_COLORS["UNKNOWN"]

def get_spot_color(spot_type: str, is_occupied: bool = False) -> Tuple[int, int, int]:
    """Get color for parking spot visualization"""
    if is_occupied:
        return COLORS["RED"]
    
    spot_type_enum = SpotType(spot_type) if spot_type in [e.value for e in SpotType] else SpotType.STANDARD
    return SPOT_COLORS.get(spot_type_enum, COLORS["GREEN"])

def is_valid_province_code(code: str) -> bool:
    """Check if province code is valid"""
    return code in VIETNAMESE_PROVINCE_CODES or code in SPECIAL_PLATE_CODES

def get_province_name(code: str) -> str:
    """Get province name from code"""
    return VIETNAMESE_PROVINCE_CODES.get(code) or SPECIAL_PLATE_CODES.get(code, "Unknown")

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"

def format_duration(seconds: float) -> str:
    """Format duration in human readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def validate_config_value(key: str, value, expected_type=None) -> bool:
    """Validate configuration value"""
    if expected_type and not isinstance(value, expected_type):
        return False
    
    # Specific validations
    if key.endswith("_CONF") and not (0 <= value <= 1):
        return False
    
    if key.endswith("_SIZE") and value <= 0:
        return False
    
    if key.endswith("_TIMEOUT") and value <= 0:
        return False
    
    return True

def get_default_config_value(key: str):
    """Get default configuration value"""
    return DEFAULT_CONFIG.get(key)

def is_supported_image_format(filename: str) -> bool:
    """Check if image format is supported"""
    import os
    _, ext = os.path.splitext(filename.lower())
    return ext in SUPPORTED_IMAGE_FORMATS

def is_supported_video_format(filename: str) -> bool:
    """Check if video format is supported"""
    import os
    _, ext = os.path.splitext(filename.lower())
    return ext in SUPPORTED_VIDEO_FORMATS

def get_error_message(error_code: str) -> str:
    """Get user-friendly error message"""
    error_messages = {
        "MODEL_LOAD_ERROR": "Không thể tải model AI",
        "VEHICLE_DETECTION_ERROR": "Lỗi phát hiện xe",
        "PLATE_DETECTION_ERROR": "Lỗi phát hiện biển số",
        "SERVER_CONNECTION_ERROR": "Lỗi kết nối server",
        "FILE_NOT_FOUND": "Không tìm thấy file",
        "INVALID_CONFIG": "Cấu hình không hợp lệ",
        "INSUFFICIENT_RESOURCES": "Tài nguyên hệ thống không đủ"
    }
    
    return error_messages.get(error_code, "Lỗi không xác định")

# Export commonly used constants
__all__ = [
    'EventType', 'VehicleType', 'SpotType', 'SystemState', 'ConnectionState',
    'ProcessingState', 'Priority', 'ErrorCode', 'DEFAULT_CONFIG',
    'VIETNAMESE_PROVINCE_CODES', 'SPECIAL_PLATE_CODES', 'OCR_CORRECTIONS',
    'COLORS', 'SPOT_COLORS', 'OCCUPIED_COLORS', 'PERFORMANCE_THRESHOLDS',
    'SYSTEM_LIMITS', 'MESSAGE_TEMPLATES', 'STATS_KEYS', 'FEATURE_FLAGS',
    'QUALITY_THRESHOLDS', 'TIMING', 'MEMORY', 'NETWORK', 'SECURITY',
    'VALIDATION_RULES', 'get_color_by_confidence', 'get_spot_color',
    'is_valid_province_code', 'get_province_name', 'format_file_size',
    'format_duration', 'validate_config_value', 'get_default_config_value',
    'is_supported_image_format', 'is_supported_video_format', 'get_error_message'
]

# Example usage
if __name__ == "__main__":
    # Demo các constants và helper functions
    print(f"System Version: {VERSION}")
    print(f"Event Types: {[e.value for e in EventType]}")
    print(f"Vehicle Types: {[e.value for e in VehicleType]}")
    print(f"Spot Colors: {SPOT_COLORS}")
    
    # Test helper functions
    print(f"\nProvince 30: {get_province_name('30')}")
    print(f"File size 1024: {format_file_size(1024)}")
    print(f"Duration 3661s: {format_duration(3661)}")
    print(f"Color for confidence 0.9: {get_color_by_confidence(0.9)}")
    print(f"Image format .jpg supported: {is_supported_image_format('test.jpg')}")
    print(f"Error message for MODEL_LOAD_ERROR: {get_error_message('MODEL_LOAD_ERROR')}")
    
    # Test validation
    print(f"\nConfig validation for VEHICLE_CONF=0.5: {validate_config_value('VEHICLE_CONF', 0.5)}")
    print(f"Config validation for CACHE_SIZE=-1: {validate_config_value('CACHE_SIZE', -1)}")