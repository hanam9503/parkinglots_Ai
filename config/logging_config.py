"""
Enhanced Logging Configuration
Cấu hình logging cho hệ thống parking
"""

import os
import logging
import logging.handlers
from datetime import datetime
import colorlog
from typing import Optional

def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    enable_color: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Setup comprehensive logging with file rotation and colored console output
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        log_format: Custom log format (optional)
        enable_color: Enable colored console output
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup files to keep
        
    Returns:
        Configured logger instance
    """
    
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    # Default log file
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = f"logs/parking_system_{timestamp}.log"
    
    # Default log format
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    
    # Create root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console Handler with colors
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    if enable_color:
        color_formatter = colorlog.ColoredFormatter(
            '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
        console_handler.setFormatter(color_formatter)
    else:
        console_formatter = logging.Formatter(
            log_format,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
    
    logger.addHandler(console_handler)
    
    # File Handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)  # Always DEBUG for file
    
    file_formatter = logging.Formatter(
        log_format,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Error Handler - separate file for errors
    error_log_file = log_file.replace('.log', '_errors.log')
    error_handler = logging.handlers.RotatingFileHandler(
        error_log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)
    logger.addHandler(error_handler)
    
    # Suppress noisy third-party loggers
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("torchvision").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    
    # Log startup info
    logger.info("="*60)
    logger.info("🚀 ENHANCED PARKING SYSTEM - LOGGING INITIALIZED")
    logger.info("="*60)
    logger.info(f"📝 Log Level: {log_level.upper()}")
    logger.info(f"📁 Log File: {log_file}")
    logger.info(f"🎨 Color Output: {'Enabled' if enable_color else 'Disabled'}")
    logger.info(f"🔄 File Rotation: {max_bytes/1024/1024:.1f}MB, {backup_count} backups")
    
    return logger

def get_performance_logger() -> logging.Logger:
    """Get dedicated performance logger"""
    perf_logger = logging.getLogger('performance')
    
    if not perf_logger.handlers:
        # Performance log file
        perf_log_file = f"logs/performance_{datetime.now().strftime('%Y%m%d')}.log"
        
        perf_handler = logging.handlers.RotatingFileHandler(
            perf_log_file,
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3,
            encoding='utf-8'
        )
        perf_handler.setLevel(logging.INFO)
        
        perf_formatter = logging.Formatter(
            '%(asctime)s - PERF - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        perf_handler.setFormatter(perf_formatter)
        perf_logger.addHandler(perf_handler)
        perf_logger.setLevel(logging.INFO)
        perf_logger.propagate = False  # Don't propagate to root logger
    
    return perf_logger

def get_event_logger() -> logging.Logger:
    """Get dedicated event logger for parking events"""
    event_logger = logging.getLogger('events')
    
    if not event_logger.handlers:
        # Event log file
        event_log_file = f"logs/events_{datetime.now().strftime('%Y%m%d')}.log"
        
        event_handler = logging.handlers.RotatingFileHandler(
            event_log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        event_handler.setLevel(logging.INFO)
        
        event_formatter = logging.Formatter(
            '%(asctime)s - EVENT - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        event_handler.setFormatter(event_formatter)
        event_logger.addHandler(event_handler)
        event_logger.setLevel(logging.INFO)
        event_logger.propagate = False
    
    return event_logger

def get_sync_logger() -> logging.Logger:
    """Get dedicated sync logger for server communication"""
    sync_logger = logging.getLogger('sync')
    
    if not sync_logger.handlers:
        # Sync log file
        sync_log_file = f"logs/sync_{datetime.now().strftime('%Y%m%d')}.log"
        
        sync_handler = logging.handlers.RotatingFileHandler(
            sync_log_file,
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3,
            encoding='utf-8'
        )
        sync_handler.setLevel(logging.DEBUG)
        
        sync_formatter = logging.Formatter(
            '%(asctime)s - SYNC - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        sync_handler.setFormatter(sync_formatter)
        sync_logger.addHandler(sync_handler)
        sync_logger.setLevel(logging.DEBUG)
        sync_logger.propagate = False
    
    return sync_logger

class PerformanceLogContext:
    """Context manager for performance logging"""
    
    def __init__(self, operation_name: str, logger: Optional[logging.Logger] = None):
        self.operation_name = operation_name
        self.logger = logger or get_performance_logger()
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"START - {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        if exc_type is None:
            self.logger.info(f"COMPLETE - {self.operation_name} - Duration: {duration:.3f}s")
        else:
            self.logger.error(f"ERROR - {self.operation_name} - Duration: {duration:.3f}s - Error: {exc_val}")

def log_system_info():
    """Log system information at startup"""
    import platform
    import psutil
    import torch
    
    logger = logging.getLogger(__name__)
    
    logger.info("💻 SYSTEM INFORMATION:")
    logger.info(f"   OS: {platform.system()} {platform.release()}")
    logger.info(f"   Python: {platform.python_version()}")
    logger.info(f"   CPU: {platform.processor()}")
    logger.info(f"   CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count()} logical")
    
    # Memory info
    memory = psutil.virtual_memory()
    logger.info(f"   RAM: {memory.total / (1024**3):.1f}GB total, {memory.available / (1024**3):.1f}GB available")
    
    # GPU info
    if torch.cuda.is_available():
        logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"   CUDA Version: {torch.version.cuda}")
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"   GPU Memory: {gpu_memory:.1f}GB")
    else:
        logger.info("   GPU: Not available (using CPU)")
    
    # Disk space
    disk = psutil.disk_usage('.')
    logger.info(f"   Disk Space: {disk.free / (1024**3):.1f}GB free of {disk.total / (1024**3):.1f}GB")

# Example usage
if __name__ == "__main__":
    # Setup logging
    logger = setup_logging(log_level="DEBUG", enable_color=True)
    
    # Log system info
    log_system_info()
    
    # Test different log levels
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    
    # Test performance logging
    with PerformanceLogContext("Test Operation"):
        import time
        time.sleep(1)
    
    # Test specialized loggers
    perf_logger = get_performance_logger()
    perf_logger.info("Performance test completed")
    
    event_logger = get_event_logger()
    event_logger.info("Vehicle entered spot A1")
    
    sync_logger = get_sync_logger()
    sync_logger.info("Server sync successful")