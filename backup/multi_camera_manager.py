"""
multi_camera_manager.py - Main Multi-Camera Management System
=============================================================
Quản lý tổng thể hệ thống đa camera
Bao gồm: model loading, camera management, resource control, event processing
"""

import os
import time
import threading
import queue
import gc
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Any
import torch
from ultralytics import YOLO

# Import các module khác
from .config import CameraConfig, SystemConfig
from .camera_processor import CameraProcessor
from .server_sync import ServerSync
from .state_tracker import VehicleStateTracker

logger = logging.getLogger(__name__)

# Import OCR processor nếu có
try:
    from plate_ocr_processor import PlateOCRProcessor, create_plate_ocr_processor, OCRConfig
except ImportError:
    logger.warning("⚠️ OCR processor not available")
    PlateOCRProcessor = None

class MultiCameraManager:
    """
    Lớp quản lý chính cho hệ thống đa camera
    
    Chức năng chính:
    - Load và quản lý YOLO models (shared across cameras)
    - Quản lý lifecycle của các camera processors
    - Resource management và performance optimization
    - Event processing và server synchronization
    - Statistics và monitoring
    - Thread-safe operations
    """
    
    def __init__(self, system_config: SystemConfig):
        """
        Khởi tạo MultiCameraManager
        
        Args:
            system_config (SystemConfig): Cấu hình hệ thống
        """
        self.config = system_config
        
        # Camera management
        self.cameras = {}  # camera_id -> CameraProcessor
        self.camera_configs = {}  # camera_id -> CameraConfig
        self.active_cameras = set()  # Set of active camera IDs
        self.current_display_camera = None  # Camera hiện tại cho display
        
        # Shared AI models (sử dụng chung cho tất cả cameras)
        self.vehicle_model = None
        self.plate_model = None
        self.ocr_processor = None
        
        # System state
        self.is_running = False
        self.start_time = time.time()
        
        # Resource management
        self.resource_lock = threading.RLock()
        self.max_concurrent = system_config.max_concurrent_cameras
        
        # Server synchronization
        self.server_sync = None
        if system_config.sync_enabled:
            self.server_sync = ServerSync(
                server_url=system_config.sync_server_url,
                connection_timeout=system_config.connection_timeout,
                request_timeout=system_config.request_timeout,
                offline_queue_size=system_config.offline_queue_size
            )
        
        # Event processing
        self.event_queue = queue.Queue(maxsize=1000)
        self.event_thread = None
        self.event_executor = ThreadPoolExecutor(max_workers=system_config.max_workers)
        
        # System statistics
        self.system_stats = {
            'start_time': time.time(),
            'total_events_processed': 0,
            'total_frames_processed': 0,
            'cameras_started': 0,
            'cameras_stopped': 0,
            'system_restarts': 0,
            'memory_cleanups': 0
        }
        
        # Background tasks
        self.cleanup_thread = None
        self.stats_thread = None
        self.stop_event = threading.Event()
        
        logger.info("✅ MultiCameraManager initialized")
    
    def load_models(self, vehicle_model_path: str, plate_model_path: str) -> bool:
        """
        Load shared YOLO models cho tất cả cameras
        
        Args:
            vehicle_model_path (str): Đường dẫn đến vehicle detection model
            plate_model_path (str): Đường dẫn đến plate detection model
            
        Returns:
            bool: True nếu load thành công
        """
        try:
            logger.info("🔄 Loading YOLO models...")
            
            # Validate model files exist
            if not os.path.exists(vehicle_model_path):
                raise FileNotFoundError(f"Vehicle model not found: {vehicle_model_path}")
            
            if not os.path.exists(plate_model_path):
                raise FileNotFoundError(f"Plate model not found: {plate_model_path}")
            
            # Determine device
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"🎯 Using device: {device}")
            
            # Load vehicle detection model
            logger.info("📦 Loading vehicle detection model...")
            self.vehicle_model = YOLO(vehicle_model_path)
            self.vehicle_model.to(device)
            logger.info(f"✅ Vehicle model loaded: {vehicle_model_path}")
            
            # Load plate detection model
            logger.info("📦 Loading plate detection model...")
            self.plate_model = YOLO(plate_model_path)
            self.plate_model.to(device)
            logger.info(f"✅ Plate model loaded: {plate_model_path}")
            
            # Initialize OCR processor if available
            if PlateOCRProcessor:
                try:
                    ocr_config = OCRConfig()
                    self.ocr_processor = create_plate_ocr_processor(ocr_config)
                    logger.info("✅ OCR processor initialized")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to initialize OCR processor: {e}")
                    self.ocr_processor = None
            
            # Run warmup predictions
            self._warmup_models()
            
            logger.info("✅ All models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load models: {e}")
            return False
    
    def _warmup_models(self):
        """Warmup models với dummy predictions để tối ưu hiệu suất"""
        try:
            import numpy as np
            logger.info("🔥 Warming up models...")
            
            # Create dummy image
            dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            # Warmup vehicle model
            if self.vehicle_model:
                self.vehicle_model.predict(dummy_img, verbose=False)
            
            # Warmup plate model
            if self.plate_model:
                self.plate_model.predict(dummy_img, verbose=False)
            
            logger.info("✅ Models warmed up")
            
        except Exception as e:
            logger.warning(f"⚠️ Model warmup failed: {e}")
    
    def add_camera(self, camera_config: CameraConfig) -> bool:
        """
        Thêm camera mới vào hệ thống
        
        Args:
            camera_config (CameraConfig): Cấu hình camera
            
        Returns:
            bool: True nếu thêm thành công
        """
        try:
            camera_id = camera_config.camera_id
            
            with self.resource_lock:
                # Check if camera already exists
                if camera_id in self.cameras:
                    logger.warning(f"⚠️ Camera {camera_id} already exists")
                    return False
                
                # Check concurrent limit
                if len(self.active_cameras) >= self.max_concurrent:
                    logger.warning(f"⚠️ Maximum concurrent cameras reached: {self.max_concurrent}")
                    return False
                
                # Validate models are loaded
                if not self.vehicle_model or not self.plate_model:
                    logger.error("❌ Models not loaded. Call load_models() first")
                    return False
                
                # Create camera processor
                camera_processor = CameraProcessor(
                    config=camera_config,
                    vehicle_model=self.vehicle_model,
                    plate_model=self.plate_model,
                    ocr_processor=self.ocr_processor,
                    event_queue=self.event_queue
                )
                
                # Store camera
                self.cameras[camera_id] = camera_processor
                self.camera_configs[camera_id] = camera_config
                
                logger.info(f"✅ Camera {camera_id} added successfully")
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to add camera {camera_config.camera_id}: {e}")
            return False
    
    def remove_camera(self, camera_id: str) -> bool:
        """
        Xóa camera khỏi hệ thống
        
        Args:
            camera_id (str): ID của camera
            
        Returns:
            bool: True nếu xóa thành công
        """
        try:
            with self.resource_lock:
                if camera_id not in self.cameras:
                    logger.warning(f"⚠️ Camera {camera_id} not found")
                    return False
                
                # Stop camera if running
                if camera_id in self.active_cameras:
                    self.stop_camera(camera_id)
                
                # Remove camera
                camera_processor = self.cameras.pop(camera_id)
                self.camera_configs.pop(camera_id, None)
                
                # Cleanup camera resources
                camera_processor.cleanup()
                
                # Update display camera if needed
                if self.current_display_camera == camera_id:
                    self.current_display_camera = None
                
                logger.info(f"✅ Camera {camera_id} removed successfully")
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to remove camera {camera_id}: {e}")
            return False
    
    def start_camera(self, camera_id: str) -> bool:
        """
        Khởi động camera
        
        Args:
            camera_id (str): ID của camera
            
        Returns:
            bool: True nếu khởi động thành công
        """
        try:
            with self.resource_lock:
                if camera_id not in self.cameras:
                    logger.error(f"❌ Camera {camera_id} not found")
                    return False
                
                if camera_id in self.active_cameras:
                    logger.warning(f"⚠️ Camera {camera_id} already running")
                    return True
                
                # Check concurrent limit
                if len(self.active_cameras) >= self.max_concurrent:
                    logger.warning(f"⚠️ Maximum concurrent cameras reached: {self.max_concurrent}")
                    return False
                
                # Start camera processor
                camera_processor = self.cameras[camera_id]
                if camera_processor.start():
                    self.active_cameras.add(camera_id)
                    self.system_stats['cameras_started'] += 1
                    
                    # Set as display camera if none selected
                    if self.current_display_camera is None:
                        self.current_display_camera = camera_id
                    
                    logger.info(f"✅ Camera {camera_id} started successfully")
                    return True
                else:
                    logger.error(f"❌ Failed to start camera {camera_id}")
                    return False
                
        except Exception as e:
            logger.error(f"❌ Error starting camera {camera_id}: {e}")
            return False
    
    def stop_camera(self, camera_id: str) -> bool:
        """
        Dừng camera
        
        Args:
            camera_id (str): ID của camera
            
        Returns:
            bool: True nếu dừng thành công
        """
        try:
            with self.resource_lock:
                if camera_id not in self.cameras:
                    logger.error(f"❌ Camera {camera_id} not found")
                    return False
                
                if camera_id not in self.active_cameras:
                    logger.warning(f"⚠️ Camera {camera_id} not running")
                    return True
                
                # Stop camera processor
                camera_processor = self.cameras[camera_id]
                if camera_processor.stop():
                    self.active_cameras.discard(camera_id)
                    self.system_stats['cameras_stopped'] += 1
                    
                    # Update display camera if needed
                    if self.current_display_camera == camera_id:
                        if self.active_cameras:
                            self.current_display_camera = next(iter(self.active_cameras))
                        else:
                            self.current_display_camera = None
                    
                    logger.info(f"✅ Camera {camera_id} stopped successfully")
                    return True
                else:
                    logger.error(f"❌ Failed to stop camera {camera_id}")
                    return False
                
        except Exception as e:
            logger.error(f"❌ Error stopping camera {camera_id}: {e}")
            return False
    
    def start_system(self) -> bool:
        """
        Khởi động toàn bộ hệ thống
        
        Returns:
            bool: True nếu khởi động thành công
        """
        try:
            if self.is_running:
                logger.warning("⚠️ System already running")
                return True
            
            logger.info("🚀 Starting multi-camera system...")
            
            # Start event processing thread
            self.event_thread = threading.Thread(
                target=self._event_processor,
                name="EventProcessor",
                daemon=True
            )
            self.event_thread.start()
            
            # Start background cleanup thread
            self.cleanup_thread = threading.Thread(
                target=self._cleanup_worker,
                name="CleanupWorker",
                daemon=True
            )
            self.cleanup_thread.start()
            
            # Start statistics thread
            self.stats_thread = threading.Thread(
                target=self._stats_worker,
                name="StatsWorker",
                daemon=True
            )
            self.stats_thread.start()
            
            # Start server sync if enabled
            if self.server_sync:
                self.server_sync.start()
            
            self.is_running = True
            self.start_time = time.time()
            
            logger.info("✅ Multi-camera system started successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start system: {e}")
            return False
    
    def stop_system(self):
        """Dừng toàn bộ hệ thống"""
        try:
            if not self.is_running:
                logger.warning("⚠️ System not running")
                return
            
            logger.info("🛑 Stopping multi-camera system...")
            
            # Set stop event
            self.stop_event.set()
            
            # Stop all cameras
            camera_ids = list(self.active_cameras)
            for camera_id in camera_ids:
                self.stop_camera(camera_id)
            
            # Stop server sync
            if self.server_sync:
                self.server_sync.stop()
            
            # Stop event executor
            self.event_executor.shutdown(wait=True)
            
            # Wait for threads to finish
            if self.event_thread and self.event_thread.is_alive():
                self.event_thread.join(timeout=5)
            
            if self.cleanup_thread and self.cleanup_thread.is_alive():
                self.cleanup_thread.join(timeout=5)
            
            if self.stats_thread and self.stats_thread.is_alive():
                self.stats_thread.join(timeout=5)
            
            self.is_running = False
            
            logger.info("✅ Multi-camera system stopped successfully")
            
        except Exception as e:
            logger.error(f"❌ Error stopping system: {e}")
    
    def restart_system(self):
        """Khởi động lại toàn bộ hệ thống"""
        logger.info("🔄 Restarting multi-camera system...")
        
        # Store active camera IDs
        active_camera_ids = list(self.active_cameras)
        
        # Stop system
        self.stop_system()
        
        # Wait a moment
        time.sleep(2)
        
        # Reset stop event
        self.stop_event.clear()
        
        # Start system
        if self.start_system():
            # Restart previously active cameras
            for camera_id in active_camera_ids:
                self.start_camera(camera_id)
            
            self.system_stats['system_restarts'] += 1
            logger.info("✅ System restarted successfully")
        else:
            logger.error("❌ Failed to restart system")
    
    def _event_processor(self):
        """Background thread xử lý events"""
        logger.info("🎯 Event processor started")
        
        while not self.stop_event.is_set():
            try:
                # Get event with timeout
                try:
                    event = self.event_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Process event in thread pool
                future = self.event_executor.submit(self._process_event, event)
                
                # Don't wait for completion to avoid blocking
                self.event_queue.task_done()
                
            except Exception as e:
                logger.error(f"❌ Error in event processor: {e}")
        
        logger.info("🎯 Event processor stopped")
    
    def _process_event(self, event: Dict[str, Any]):
        """
        Xử lý một event cụ thể
        
        Args:
            event (Dict[str, Any]): Event data
        """
        try:
            event_type = event.get('type')
            camera_id = event.get('camera_id')
            
            # Update statistics
            self.system_stats['total_events_processed'] += 1
            
            # Process based on event type
            if event_type == 'vehicle_detected':
                self._handle_vehicle_event(event)
            elif event_type == 'plate_detected':
                self._handle_plate_event(event)
            elif event_type == 'frame_processed':
                self._handle_frame_event(event)
            elif event_type == 'error':
                self._handle_error_event(event)
            else:
                logger.warning(f"⚠️ Unknown event type: {event_type}")
            
            # Sync to server if enabled
            if self.server_sync and event_type in ['vehicle_detected', 'plate_detected']:
                self.server_sync.queue_event(event)
            
        except Exception as e:
            logger.error(f"❌ Error processing event: {e}")
    
    def _handle_vehicle_event(self, event: Dict[str, Any]):
        """Xử lý vehicle detection event"""
        camera_id = event.get('camera_id')
        vehicle_data = event.get('vehicle_data', {})
        
        logger.debug(f"🚗 Vehicle detected on camera {camera_id}: {vehicle_data}")
    
    def _handle_plate_event(self, event: Dict[str, Any]):
        """Xử lý plate detection event"""
        camera_id = event.get('camera_id')
        plate_data = event.get('plate_data', {})
        
        logger.debug(f"📋 Plate detected on camera {camera_id}: {plate_data}")
    
    def _handle_frame_event(self, event: Dict[str, Any]):
        """Xử lý frame processing event"""
        camera_id = event.get('camera_id')
        self.system_stats['total_frames_processed'] += 1
    
    def _handle_error_event(self, event: Dict[str, Any]):
        """Xử lý error event"""
        camera_id = event.get('camera_id')
        error_msg = event.get('error', 'Unknown error')
        
        logger.error(f"❌ Camera {camera_id} error: {error_msg}")
    
    def _cleanup_worker(self):
        """Background thread cho memory cleanup"""
        logger.info("🧹 Cleanup worker started")
        
        while not self.stop_event.is_set():
            try:
                # Wait for cleanup interval
                self.stop_event.wait(self.config.cleanup_interval)
                
                if self.stop_event.is_set():
                    break
                
                # Perform cleanup
                self._perform_cleanup()
                
            except Exception as e:
                logger.error(f"❌ Error in cleanup worker: {e}")
        
        logger.info("🧹 Cleanup worker stopped")
    
    def _perform_cleanup(self):
        """Thực hiện memory cleanup"""
        try:
            # Clear cache for inactive cameras
            with self.resource_lock:
                for camera_id, camera_processor in self.cameras.items():
                    if camera_id not in self.active_cameras:
                        camera_processor.clear_cache()
            
            # Clear YOLO model cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force garbage collection
            collected = gc.collect()
            
            self.system_stats['memory_cleanups'] += 1
            logger.debug(f"🧹 Cleanup completed, collected {collected} objects")
            
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")
    
    def _stats_worker(self):
        """Background thread cho statistics logging"""
        logger.info("📊 Statistics worker started")
        
        while not self.stop_event.is_set():
            try:
                # Wait for stats interval
                self.stop_event.wait(self.config.stats_interval)
                
                if self.stop_event.is_set():
                    break
                
                # Log statistics
                self._log_statistics()
                
            except Exception as e:
                logger.error(f"❌ Error in stats worker: {e}")
        
        logger.info("📊 Statistics worker stopped")
    
    def _log_statistics(self):
        """Log system statistics"""
        try:
            uptime = time.time() - self.start_time
            
            stats_msg = (
                f"📊 System Stats - "
                f"Uptime: {uptime:.1f}s, "
                f"Active Cameras: {len(self.active_cameras)}, "
                f"Events: {self.system_stats['total_events_processed']}, "
                f"Frames: {self.system_stats['total_frames_processed']}"
            )
            
            logger.info(stats_msg)
            
        except Exception as e:
            logger.error(f"❌ Stats logging error: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Lấy trạng thái hiện tại của hệ thống
        
        Returns:
            Dict[str, Any]: System status information
        """
        try:
            uptime = time.time() - self.start_time if self.is_running else 0
            
            # Get camera status
            camera_status = {}
            for camera_id, camera_processor in self.cameras.items():
                camera_status[camera_id] = {
                    'active': camera_id in self.active_cameras,
                    'config': self.camera_configs[camera_id].to_dict(),
                    'stats': camera_processor.get_stats() if hasattr(camera_processor, 'get_stats') else {}
                }
            
            return {
                'system_running': self.is_running,
                'uptime': uptime,
                'active_cameras': len(self.active_cameras),
                'total_cameras': len(self.cameras),
                'current_display_camera': self.current_display_camera,
                'models_loaded': {
                    'vehicle_model': self.vehicle_model is not None,
                    'plate_model': self.plate_model is not None,
                    'ocr_processor': self.ocr_processor is not None
                },
                'server_sync_enabled': self.server_sync is not None,
                'server_sync_status': self.server_sync.get_status() if self.server_sync else None,
                'system_stats': self.system_stats.copy(),
                'cameras': camera_status
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting system status: {e}")
            return {'error': str(e)}
    
    def get_current_frame(self, camera_id: Optional[str] = None) -> Optional[Any]:
        """
        Lấy frame hiện tại từ camera
        
        Args:
            camera_id (str, optional): ID của camera. Nếu None thì lấy từ display camera
            
        Returns:
            Optional[Any]: Current frame or None
        """
        try:
            # Use display camera if not specified
            target_camera = camera_id or self.current_display_camera
            
            if not target_camera or target_camera not in self.cameras:
                return None
            
            if target_camera not in self.active_cameras:
                return None
            
            camera_processor = self.cameras[target_camera]
            return camera_processor.get_current_frame()
            
        except Exception as e:
            logger.error(f"❌ Error getting current frame: {e}")
            return None
    
    def set_display_camera(self, camera_id: str) -> bool:
        """
        Đặt camera cho display
        
        Args:
            camera_id (str): ID của camera
            
        Returns:
            bool: True nếu thành công
        """
        try:
            if camera_id not in self.cameras:
                logger.error(f"❌ Camera {camera_id} not found")
                return False
            
            if camera_id not in self.active_cameras:
                logger.error(f"❌ Camera {camera_id} not active")
                return False
            
            self.current_display_camera = camera_id
            logger.info(f"✅ Display camera set to {camera_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error setting display camera: {e}")
            return False
    
    def cleanup(self):
        """Cleanup toàn bộ resources"""
        try:
            logger.info("🧹 Cleaning up MultiCameraManager...")
            
            # Stop system if running
            if self.is_running:
                self.stop_system()
            
            # Cleanup all cameras
            for camera_id in list(self.cameras.keys()):
                self.remove_camera(camera_id)
            
            # Cleanup models
            self.vehicle_model = None
            self.plate_model = None
            self.ocr_processor = None
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
            logger.info("✅ MultiCameraManager cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()