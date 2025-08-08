
"""
Hệ thống Parking Real-time Tối ưu - Đã tách OCR module
- Sử dụng module plate_ocr_processor riêng biệt
- Tối ưu performance và memory management
- Smart caching và error handling
- Real-time monitoring và statistics
- Offline mode với queue system
"""

import cv2
import json
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Point, Polygon, box as create_box
import torch
from collections import defaultdict, deque
import os
import time
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import re
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple, Dict
import gc
import requests
import base64
from datetime import datetime, timedelta
import uuid
from PIL import Image
import io
import logging
import hashlib
from urllib3.exceptions import InsecureRequestWarning
import warnings

# Import module OCR đã tách
from plate_ocr_processor import PlateOCRProcessor, create_plate_ocr_processor, OCRConfig

# Suppress warnings
warnings.filterwarnings("ignore", category=InsecureRequestWarning)
logging.getLogger("ultralytics").setLevel(logging.WARNING)

# Enhanced Configuration
@dataclass
class SystemConfig:
    # System identification
    CAMERA_ID: str = "CAM_001"
    LOCATION_NAME: str = "Bãi đỗ xe tầng 1"
    
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
    INTERSECTION_THRESHOLD: float = 0.3
    
    # Processing mode
    SAVE_IMAGES: bool = True
    IMAGE_QUALITY: int = 85
    IMAGE_CLEANUP_HOURS: int = 24
    
    # Stability settings
    MIN_DETECTION_FRAMES: int = 3
    MAX_MISSED_FRAMES: int = 5
    STATE_TIMEOUT_MINUTES: int = 60
    
    # Offline mode settings
    ENABLE_OFFLINE_MODE: bool = True
    OFFLINE_QUEUE_SIZE: int = 200
    SYNC_INTERVAL: float = 15.0
    BATCH_SYNC_SIZE: int = 10

config = SystemConfig()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('parking_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

print(f"\n🚀 Enhanced Parking System - Modularized Version:")
print(f"   - Server URL: {config.SYNC_SERVER_URL}")
print(f"   - Offline Mode: {config.ENABLE_OFFLINE_MODE}")
print(f"   - OCR Module: Separated")
print(f"   - YOLO Size: {config.YOLO_IMG_SIZE}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Using device: {device}")

# Tạo thư mục
directories = ['vehicle_images', 'plate_images', 'enhanced_images', 'weights', 'logs']
for directory in directories:
    os.makedirs(directory, exist_ok=True)

# Enhanced NodeJS Server Sync với comprehensive error handling
class EnhancedServerSync:
    def __init__(self, server_url):
        self.server_url = server_url
        self.endpoints = {
            'events': f"{server_url}/api/events",
            'status': f"{server_url}/api/status",
            'health': f"{server_url}/api/health",
            'dashboard': f"{server_url}/api/camera-dashboard"
        }
        
        # Enhanced session với connection pooling
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=3
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        
        self.session.headers.update({
            "Content-Type": "application/json",
            "User-Agent": f"ParkingSystem/{config.CAMERA_ID}",
            "Accept": "application/json"
        })
        
        # Connection state
        self.is_connected = False
        self.last_health_check = 0
        self.connection_errors = 0
        self.consecutive_failures = 0
        self.last_successful_sync = time.time()
        
        # Offline queue với persistent storage
        self.offline_queue = deque(maxlen=config.OFFLINE_QUEUE_SIZE)
        self.queue_lock = threading.Lock()
        self.sync_lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'total_events_sent': 0,
            'total_status_sent': 0,
            'total_requests': 0,
            'failed_requests': 0,
            'offline_items_processed': 0,
            'last_sync_time': None,
            'sync_success_rate': 100.0
        }
        
        # Background threads
        self.sync_thread = None
        self.health_thread = None
        self.shutdown_event = threading.Event()
        
        # Start background services
        if config.ENABLE_OFFLINE_MODE:
            self._start_background_services()
        
        # Initial health check
        self.check_server_health()
    
    def _start_background_services(self):
        """Start background threads"""
        self.sync_thread = threading.Thread(target=self._background_sync, daemon=True)
        self.sync_thread.start()
        
        self.health_thread = threading.Thread(target=self._background_health_check, daemon=True)
        self.health_thread.start()
        
        logger.info("✅ Background services started")
    
    def check_server_health(self):
        """Comprehensive server health check"""
        try:
            start_time = time.time()
            response = self.session.get(
                self.endpoints['health'],
                timeout=config.CONNECTION_TIMEOUT
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                
                if not self.is_connected:
                    logger.info(f"✅ Server connection established: {self.server_url}")
                    logger.info(f"   Response time: {response_time:.2f}s")
                    if 'database_stats' in data:
                        logger.info(f"   Database status: {data.get('mongodb_status', 'unknown')}")
                
                self.is_connected = True
                self.connection_errors = 0
                self.consecutive_failures = 0
                return True
            else:
                self._handle_connection_error(f"Health check failed: {response.status_code}")
                return False
                
        except requests.exceptions.Timeout:
            self._handle_connection_error("Health check timeout")
            return False
        except requests.exceptions.ConnectionError:
            self._handle_connection_error("Connection refused")
            return False
        except Exception as e:
            self._handle_connection_error(f"Health check error: {e}")
            return False
        finally:
            self.last_health_check = time.time()
    
    def _handle_connection_error(self, error_msg):
        """Enhanced connection error handling"""
        self.connection_errors += 1
        self.consecutive_failures += 1
        
        if self.is_connected:
            logger.warning(f"⚠️ Server connection lost: {error_msg}")
            self.is_connected = False
        
        # Log periodically to avoid spam
        if self.connection_errors % 10 == 1:
            logger.info(f"📡 Offline mode active. Queue size: {len(self.offline_queue)}")
            logger.info(f"   Consecutive failures: {self.consecutive_failures}")
    
    def send_event(self, event_data):
        """Enhanced event sending with retry and validation"""
        # Validate event data
        required_fields = ['camera_id', 'spot_id', 'event_type']
        for field in required_fields:
            if field not in event_data:
                logger.error(f"Missing required field: {field}")
                return False
        
        # Periodic health check
        if time.time() - self.last_health_check > config.HEALTH_CHECK_INTERVAL:
            self.check_server_health()
        
        # Prepare event data
        event_data_copy = event_data.copy()
        
        # Convert images to base64
        if 'vehicle_image_path' in event_data_copy:
            event_data_copy["vehicle_image"] = self._image_to_base64(event_data_copy.pop('vehicle_image_path'))
        if 'plate_image_path' in event_data_copy:
            event_data_copy["plate_image"] = self._image_to_base64(event_data_copy.pop('plate_image_path'))
        
        # Remove paths that shouldn't be sent to server
        for key in ['enhanced_vehicle_path', 'enhanced_plate_path']:
            event_data_copy.pop(key, None)
        
        self.stats['total_requests'] += 1
        
        if self.is_connected:
            # Try to send immediately
            success = self._send_with_retry(self.endpoints['events'], event_data_copy, 'event')
            if success:
                self.stats['total_events_sent'] += 1
                self.last_successful_sync = time.time()
                return True
            else:
                self.stats['failed_requests'] += 1
        
        # Add to offline queue
        if config.ENABLE_OFFLINE_MODE:
            with self.queue_lock:
                self.offline_queue.append({
                    'type': 'event',
                    'data': event_data_copy,
                    'timestamp': time.time(),
                    'retry_count': 0
                })
            logger.debug("Event added to offline queue")
            return False
        
        return False
    
    def send_status(self, status_data):
        """Enhanced status sending with validation"""
        # Validate status data
        if 'spot_id' not in status_data:
            logger.error("Missing required field: spot_id")
            return False
        
        self.stats['total_requests'] += 1
        
        if self.is_connected:
            success = self._send_with_retry(self.endpoints['status'], status_data, 'status')
            if success:
                self.stats['total_status_sent'] += 1
                self.last_successful_sync = time.time()
                return True
            else:
                self.stats['failed_requests'] += 1
        
        # Add to offline queue
        if config.ENABLE_OFFLINE_MODE:
            with self.queue_lock:
                self.offline_queue.append({
                    'type': 'status',
                    'data': status_data,
                    'timestamp': time.time(),
                    'retry_count': 0
                })
            logger.debug("Status added to offline queue")
            return False
        
        return False
    
    def _send_with_retry(self, url, data, data_type):
        """Send data with retry mechanism"""
        for attempt in range(config.MAX_RETRY_ATTEMPTS):
            try:
                response = self.session.post(
                    url,
                    json=data,
                    timeout=config.REQUEST_TIMEOUT
                )
                
                if response.status_code in [200, 201]:
                    if attempt > 0:  # Log successful retry
                        logger.info(f"✅ {data_type} sent successfully after {attempt + 1} attempts")
                    return True
                elif response.status_code == 409:  # Duplicate
                    logger.warning(f"Duplicate {data_type} detected: {response.text}")
                    return True  # Consider as success
                else:
                    logger.warning(f"⚠️ {data_type} send failed (attempt {attempt + 1}): {response.status_code}")
                    
            except requests.exceptions.Timeout:
                logger.warning(f"⚠️ {data_type} send timeout (attempt {attempt + 1})")
            except requests.exceptions.ConnectionError:
                logger.warning(f"⚠️ {data_type} connection error (attempt {attempt + 1})")
                self._handle_connection_error("Connection error during send")
                break  # Don't retry connection errors
            except Exception as e:
                logger.warning(f"⚠️ {data_type} send error (attempt {attempt + 1}): {e}")
            
            if attempt < config.MAX_RETRY_ATTEMPTS - 1:
                time.sleep(config.RETRY_DELAY * (attempt + 1))  # Exponential backoff
        
        return False
    
    def _background_sync(self):
        """Enhanced background sync with batch processing"""
        while not self.shutdown_event.is_set():
            try:
                time.sleep(config.SYNC_INTERVAL)
                
                if not self.is_connected or len(self.offline_queue) == 0:
                    continue
                
                with self.sync_lock:
                    self._process_offline_queue()
                    
            except Exception as e:
                logger.error(f"Background sync error: {e}")
                time.sleep(config.SYNC_INTERVAL)
    
    def _process_offline_queue(self):
        """Process offline queue with batch processing"""
        if not self.offline_queue:
            return
        
        # Get items to process
        items_to_process = []
        failed_items = []
        
        with self.queue_lock:
            # Process in batches
            batch_size = min(config.BATCH_SYNC_SIZE, len(self.offline_queue))
            for _ in range(batch_size):
                if self.offline_queue:
                    items_to_process.append(self.offline_queue.popleft())
        
        if not items_to_process:
            return
        
        synced_count = 0
        
        for item in items_to_process:
            item_type = item['type']
            item_data = item['data']
            retry_count = item.get('retry_count', 0)
            
            # Skip items that have been retried too many times
            if retry_count >= config.MAX_RETRY_ATTEMPTS:
                logger.warning(f"Dropping {item_type} after {retry_count} failed attempts")
                continue
            
            try:
                if item_type == 'event':
                    url = self.endpoints['events']
                else:
                    url = self.endpoints['status']
                
                response = self.session.post(
                    url,
                    json=item_data,
                    timeout=config.REQUEST_TIMEOUT
                )
                
                if response.status_code in [200, 201, 409]:  # Include 409 as success
                    synced_count += 1
                    self.stats['offline_items_processed'] += 1
                else:
                    # Re-queue with incremented retry count
                    item['retry_count'] = retry_count + 1
                    failed_items.append(item)
                    
            except Exception as e:
                logger.error(f"Sync error for {item_type}: {e}")
                item['retry_count'] = retry_count + 1
                failed_items.append(item)
                
                # If connection error, stop processing
                if isinstance(e, requests.exceptions.ConnectionError):
                    self._handle_connection_error("Connection error during sync")
                    # Re-add all remaining items
                    failed_items.extend(items_to_process[items_to_process.index(item) + 1:])
                    break
        
        # Re-add failed items to queue
        if failed_items:
            with self.queue_lock:
                self.offline_queue.extendleft(reversed(failed_items))
        
        if synced_count > 0:
            logger.info(f"🔄 Synced {synced_count} items, {len(failed_items)} failed, queue: {len(self.offline_queue)}")
            self.stats['last_sync_time'] = datetime.now().isoformat()
            
            # Update success rate
            total_attempts = self.stats['total_requests']
            if total_attempts > 0:
                self.stats['sync_success_rate'] = ((total_attempts - self.stats['failed_requests']) / total_attempts) * 100
    
    def _background_health_check(self):
        """Background health check thread"""
        while not self.shutdown_event.is_set():
            try:
                time.sleep(config.HEALTH_CHECK_INTERVAL)
                
                if not self.is_connected:
                    self.check_server_health()
                    
            except Exception as e:
                logger.error(f"Background health check error: {e}")
    
    def _image_to_base64(self, image_path):
        """Convert image to base64 with error handling"""
        if not image_path or not os.path.exists(image_path):
            return ""
        
        try:
            with open(image_path, "rb") as image_file:
                encoded = base64.b64encode(image_file.read()).decode("utf-8")
                # Validate base64 size (limit to 10MB)
                if len(encoded) > 10 * 1024 * 1024:
                    logger.warning(f"Image too large: {image_path}")
                    return ""
                return encoded
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {e}")
            return ""
    
    def get_connection_status(self):
        """Get comprehensive connection status"""
        return {
            'is_connected': self.is_connected,
            'connection_errors': self.connection_errors,
            'consecutive_failures': self.consecutive_failures,
            'offline_queue_size': len(self.offline_queue),
            'last_health_check': self.last_health_check,
            'last_successful_sync': self.last_successful_sync,
            'stats': self.stats.copy()
        }
    
    def cleanup(self):
        """Enhanced cleanup with graceful shutdown"""
        logger.info("🧹 Shutting down server sync...")
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Final sync attempt
        if config.ENABLE_OFFLINE_MODE and len(self.offline_queue) > 0:
            logger.info(f"🔄 Final sync attempt: {len(self.offline_queue)} items in queue")
            
            sync_timeout = 30  # 30 seconds for final sync
            start_time = time.time()
            
            with self.sync_lock:
                while len(self.offline_queue) > 0 and (time.time() - start_time) < sync_timeout:
                    self._process_offline_queue()
                    if len(self.offline_queue) > 0:
                        time.sleep(2)
            
            if len(self.offline_queue) > 0:
                logger.warning(f"⚠️ {len(self.offline_queue)} items could not be synced")
        
        # Wait for threads to finish
        if self.sync_thread and self.sync_thread.is_alive():
            self.sync_thread.join(timeout=5)
        
        if self.health_thread and self.health_thread.is_alive():
            self.health_thread.join(timeout=5)
        
        # Close session
        self.session.close()
        logger.info("✅ Server sync cleanup completed")

# Enhanced Data Models với validation
@dataclass
class ParkingEvent:
    id: str
    camera_id: str
    spot_id: str
    spot_name: str
    event_type: str
    timestamp: str
    plate_text: Optional[str] = None
    plate_confidence: float = 0.0
    vehicle_image_path: Optional[str] = None
    plate_image_path: Optional[str] = None
    enhanced_vehicle_path: Optional[str] = None
    enhanced_plate_path: Optional[str] = None
    location_name: str = ""
    
    def __post_init__(self):
        # Validate event_type
        if self.event_type not in ['enter', 'exit']:
            raise ValueError(f"Invalid event_type: {self.event_type}")
        
        # Validate confidence
        if not 0 <= self.plate_confidence <= 1:
            self.plate_confidence = max(0, min(1, self.plate_confidence))
    
    def to_dict(self):
        return asdict(self)

@dataclass
class ParkingSpotStatus:
    spot_id: str
    spot_name: str
    camera_id: str
    is_occupied: bool
    enter_time: Optional[str] = None
    plate_text: Optional[str] = None
    plate_confidence: float = 0.0
    last_update: str = ""
    
    def __post_init__(self):
        # Validate confidence
        if not 0 <= self.plate_confidence <= 1:
            self.plate_confidence = max(0, min(1, self.plate_confidence))
    
    def to_dict(self):
        return asdict(self)

# Enhanced Vehicle State Tracking
class VehicleStateTracker:
    def __init__(self):
        self.detection_history = defaultdict(lambda: deque(maxlen=config.MAX_MISSED_FRAMES + config.MIN_DETECTION_FRAMES))
        self.confirmed_states = {}
        self.state_timestamps = {}
    
    def update_detection(self, spot_id, is_occupied, car_id=None):
        """Update detection history for a spot"""
        current_time = time.time()
        self.detection_history[spot_id].append({
            'occupied': is_occupied,
            'timestamp': current_time,
            'car_id': car_id
        })
        
        # Check for state change
        return self._check_state_change(spot_id)
    
    def _check_state_change(self, spot_id):
        """Check if there's a confirmed state change"""
        history = self.detection_history[spot_id]
        if len(history) < config.MIN_DETECTION_FRAMES:
            return None
        
        # Get recent detections
        recent = list(history)[-config.MIN_DETECTION_FRAMES:]
        
        # Check consistency
        occupied_count = sum(1 for d in recent if d['occupied'])
        
        if occupied_count >= config.MIN_DETECTION_FRAMES:
            new_state = True
        elif occupied_count == 0:
            new_state = False
        else:
            return None  # Inconsistent state
        
        # Check if this is a state change
        current_state = self.confirmed_states.get(spot_id, None)
        
        if current_state != new_state:
            self.confirmed_states[spot_id] = new_state
            self.state_timestamps[spot_id] = time.time()
            
            return {
                'spot_id': spot_id,
                'new_state': new_state,
                'previous_state': current_state,
                'car_id': recent[-1]['car_id'] if new_state else None
            }
        
        return None
    
    def get_confirmed_state(self, spot_id):
        """Get confirmed state for a spot"""
        return self.confirmed_states.get(spot_id, False)
    
    def cleanup_old_states(self):
        """Clean up old state data"""
        current_time = time.time()
        timeout = config.STATE_TIMEOUT_MINUTES * 60
        
        expired_spots = []
        for spot_id, timestamp in self.state_timestamps.items():
            if current_time - timestamp > timeout:
                expired_spots.append(spot_id)
        
        for spot_id in expired_spots:
            self.confirmed_states.pop(spot_id, None)
            self.state_timestamps.pop(spot_id, None)
            if spot_id in self.detection_history:
                self.detection_history[spot_id].clear()
        
        if expired_spots:
            logger.info(f"Cleaned up {len(expired_spots)} expired vehicle states")

# MAIN: Enhanced Parking Monitor
class EnhancedParkingMonitor:
    def __init__(self):
        # Core components
        self.parking_state = self._init_parking_state()
        self.server_sync = EnhancedServerSync(config.SYNC_SERVER_URL)
        
        # Initialize OCR processor with optimized config
        ocr_config = OCRConfig()
        ocr_config.USE_REAL_ESRGAN = True
        ocr_config.ENABLE_CACHING = True
        ocr_config.OCR_MIN_CONF = 0.4
        self.ocr_processor = PlateOCRProcessor(ocr_config)
        
        self.state_tracker = VehicleStateTracker()
        
        # Performance tracking
        self.stats = {
            'frames_processed': 0,
            'events_generated': 0,
            'events_sent': 0,
            'events_queued': 0,
            'plates_detected': 0,
            'plates_validated': 0,
            'processing_times': deque(maxlen=100),
            'start_time': time.time()
        }
        
        # Background cleanup thread
        self.cleanup_thread = threading.Thread(target=self._background_cleanup, daemon=True)
        self.cleanup_thread.start()
        
        logger.info("✅ Enhanced Parking Monitor initialized")
    
    def _init_parking_state(self):
        """Initialize parking spot states"""
        states = {}
        for spot in parking_spots:
            states[spot['id']] = {
                'spot_id': spot['id'],
                'spot_name': spot['name'],
                'occupied': False,
                'last_plate': None,
                'last_confidence': 0.0,
                'enter_time': None,
                'last_update': None,
                'last_car_id': None,
                'detection_count': 0,
                'processing_lock': threading.Lock()
            }
        return states
    
    def get_car_id(self, box):
        """Generate car ID from bounding box"""
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        # Create a more stable ID based on position and size
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        size = (x2 - x1) * (y2 - y1)
        return f"car_{center_x//50}_{center_y//50}_{size//1000}"
    
    def process_frame(self, frame, car_boxes):
        """Enhanced frame processing with state tracking"""
        start_time = time.time()
        current_time = datetime.now()
        processed_events = []
        
        self.stats['frames_processed'] += 1
        
        # Process each parking spot
        for spot in parking_spots:
            spot_id = spot['id']
            spot_name = spot['name']
            spot_bbox = spot['bbox']
            
            # Find intersecting cars
            current_car_id = None
            current_car_box = None
            spot_occupied = False
            
            for car_box, car_id in car_boxes:
                x1, y1, x2, y2 = car_box
                
                # Quick bbox intersection check
                if (x2 < spot_bbox[0] - 20 or x1 > spot_bbox[2] + 20 or 
                    y2 < spot_bbox[1] - 20 or y1 > spot_bbox[3] + 20):
                    continue
                
                # Detailed intersection using Shapely
                car_poly = create_box(x1, y1, x2, y2)
                intersection = spot['polygon_shapely'].intersection(car_poly)
                
                if intersection.area / spot['area'] > config.INTERSECTION_THRESHOLD:
                    spot_occupied = True
                    current_car_id = car_id
                    current_car_box = car_box
                    break
            
            # Update state tracker
            state_change = self.state_tracker.update_detection(spot_id, spot_occupied, current_car_id)
            
            if state_change:
                event = self._process_state_change(
                    state_change, spot_name, current_car_box, frame, current_time
                )
                if event:
                    processed_events.append(event)
        
        # Update processing time
        process_time = time.time() - start_time
        self.stats['processing_times'].append(process_time)
        
        return processed_events
    
    def _process_state_change(self, state_change, spot_name, car_box, frame, timestamp):
        """Process confirmed state changes"""
        spot_id = state_change['spot_id']
        new_state = state_change['new_state']
        car_id = state_change['car_id']
        
        with self.parking_state[spot_id]['processing_lock']:
            if new_state:  # Vehicle entered
                return self._process_vehicle_enter(
                    spot_id, spot_name, car_box, frame, timestamp, car_id
                )
            else:  # Vehicle exited
                return self._process_vehicle_exit(
                    spot_id, spot_name, timestamp
                )
    
    def _process_vehicle_enter(self, spot_id, spot_name, car_box, frame, timestamp, car_id):
        """Process vehicle enter event with enhanced plate detection"""
        logger.info(f"🚗 Processing ENTER for {spot_name}")
        
        # Initialize results
        vehicle_image_path = None
        plate_image_path = None
        enhanced_plate_path = None
        plate_text = "Không nhận diện được biển"
        plate_confidence = 0.0
        
        # Process vehicle image and detect plates
        if car_box is not None:
            x1, y1, x2, y2 = car_box
            h, w = frame.shape[:2]
            
            # Ensure coordinates are within frame bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 > x1 and y2 > y1:
                car_crop = frame[y1:y2, x1:x2]
                
                if car_crop.size > 0:
                    # Save vehicle image
                    if config.SAVE_IMAGES:
                        timestamp_str = int(timestamp.timestamp())
                        vehicle_filename = f"vehicle_{spot_id}_{timestamp_str}_{car_id}.jpg"
                        vehicle_image_path = os.path.join('vehicle_images', vehicle_filename)
                        cv2.imwrite(vehicle_image_path, car_crop, [cv2.IMWRITE_JPEG_QUALITY, config.IMAGE_QUALITY])
                        logger.debug(f"Saved vehicle image: {vehicle_filename}")
                    
                    # Detect and process plates using separated OCR module
                    plate_result = self._detect_and_process_plates(
                        car_crop, spot_id, timestamp
                    )
                    
                    if plate_result:
                        plate_text = plate_result['validated_text'] or plate_result['text']
                        plate_confidence = plate_result['confidence']
                        plate_image_path = plate_result.get('save_path')
                        
                        self.stats['plates_detected'] += 1
                        if plate_result['is_valid']:
                            self.stats['plates_validated'] += 1
                        
                        logger.info(f"   ✅ Plate detected: {plate_text} (conf: {plate_confidence:.3f})")
                    else:
                        logger.info(f"   ❌ No plate detected")
        
        # Create event
        event = ParkingEvent(
            id=str(uuid.uuid4()),
            camera_id=config.CAMERA_ID,
            spot_id=spot_id,
            spot_name=spot_name,
            event_type="enter",
            timestamp=timestamp.isoformat(),
            plate_text=plate_text,
            plate_confidence=plate_confidence,
            vehicle_image_path=vehicle_image_path,
            plate_image_path=plate_image_path,
            enhanced_plate_path=enhanced_plate_path,
            location_name=config.LOCATION_NAME
        )
        
        # Create status
        status = ParkingSpotStatus(
            spot_id=spot_id,
            spot_name=spot_name,
            camera_id=config.CAMERA_ID,
            is_occupied=True,
            enter_time=timestamp.isoformat(),
            plate_text=plate_text,
            plate_confidence=plate_confidence,
            last_update=timestamp.isoformat()
        )
        
        # Update internal state
        state = self.parking_state[spot_id]
        state['occupied'] = True
        state['last_plate'] = plate_text
        state['last_confidence'] = plate_confidence
        state['enter_time'] = timestamp.isoformat()
        state['last_update'] = timestamp.isoformat()
        state['last_car_id'] = car_id
        state['detection_count'] += 1
        
        # Send to server
        event_sent = self.server_sync.send_event(event.to_dict())
        status_sent = self.server_sync.send_status(status.to_dict())
        
        # Update statistics
        self.stats['events_generated'] += 1
        if event_sent and status_sent:
            self.stats['events_sent'] += 1
            logger.info("   ✅ Data sent to server successfully")
        else:
            self.stats['events_queued'] += 1
            if config.ENABLE_OFFLINE_MODE:
                logger.info("   📡 Data queued for offline sync")
            else:
                logger.warning("   ⚠️ Failed to send data to server")
        
        return {
            'event': event,
            'status': status,
            'action': 'enter',
            'sent_to_server': event_sent and status_sent,
            'plate_detected': plate_confidence > 0
        }
    
    def _process_vehicle_exit(self, spot_id, spot_name, timestamp):
        """Process vehicle exit event"""
        state = self.parking_state[spot_id]
        
        # Calculate duration
        duration_minutes = 0
        if state['enter_time']:
            try:
                enter_time = datetime.fromisoformat(state['enter_time'])
                duration_minutes = int((timestamp - enter_time).total_seconds() / 60)
            except ValueError:
                logger.error(f"Invalid enter_time format: {state['enter_time']}")
        
        # Create exit event
        event = ParkingEvent(
            id=str(uuid.uuid4()),
            camera_id=config.CAMERA_ID,
            spot_id=spot_id,
            spot_name=spot_name,
            event_type="exit",
            timestamp=timestamp.isoformat(),
            plate_text=state['last_plate'],
            plate_confidence=state['last_confidence'],
            location_name=config.LOCATION_NAME
        )
        
        # Create status update
        status = ParkingSpotStatus(
            spot_id=spot_id,
            spot_name=spot_name,
            camera_id=config.CAMERA_ID,
            is_occupied=False,
            enter_time=None,
            plate_text=None,
            plate_confidence=0.0,
            last_update=timestamp.isoformat()
        )
        
        # Send to server
        event_sent = self.server_sync.send_event(event.to_dict())
        status_sent = self.server_sync.send_status(status.to_dict())
        
        # Update statistics
        self.stats['events_generated'] += 1
        if event_sent and status_sent:
            self.stats['events_sent'] += 1
        else:
            self.stats['events_queued'] += 1
        
        # Reset internal state
        state['occupied'] = False
        state['last_plate'] = None
        state['last_confidence'] = 0.0
        state['enter_time'] = None
        state['last_update'] = timestamp.isoformat()
        state['last_car_id'] = None
        
        logger.info(f"🚪 {spot_name}: Vehicle exited (duration: {duration_minutes}m)")
        
        return {
            'event': event,
            'status': status,
            'action': 'exit',
            'duration_minutes': duration_minutes,
            'sent_to_server': event_sent and status_sent
        }
    
    def _detect_and_process_plates(self, car_image, spot_id, timestamp):
        """Enhanced plate detection using separated OCR module"""
        try:
            # Detect plates with YOLO
            plate_results = plate_model(
                car_image,
                conf=config.PLATE_CONF,
                verbose=False,
                imgsz=config.PLATE_IMG_SIZE
            )[0]
            
            if len(plate_results.boxes) == 0:
                return None
            
            best_result = None
            best_confidence = 0
            
            logger.debug(f"Found {len(plate_results.boxes)} potential plates")
            
            for i, plate_det in enumerate(plate_results.boxes):
                # Extract plate region
                px1, py1, px2, py2 = map(int, plate_det.xyxy[0])
                ch, cw = car_image.shape[:2]
                
                # Ensure coordinates are valid
                px1, py1 = max(0, px1), max(0, py1)
                px2, py2 = min(cw, px2), min(ch, py2)
                
                if px2 <= px1 or py2 <= py1:
                    continue
                
                plate_crop = car_image[py1:py2, px1:px2]
                
                if plate_crop.size == 0 or min(plate_crop.shape[:2]) < 10:
                    continue
                
                # Resize if too small
                if min(plate_crop.shape[:2]) < 32:
                    scale = 32 / min(plate_crop.shape[:2])
                    new_h = int(plate_crop.shape[0] * scale)
                    new_w = int(plate_crop.shape[1] * scale)
                    plate_crop = cv2.resize(plate_crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                
                # Save original plate image path
                timestamp_str = int(timestamp.timestamp())
                plate_filename = f"plate_{spot_id}_{timestamp_str}_{i}.jpg"
                plate_image_path = os.path.join('plate_images', plate_filename)
                
                # Process plate using OCR module
                ocr_result = self.ocr_processor.process_plate(plate_crop, plate_image_path)
                
                if ocr_result['text'] and ocr_result['confidence'] > 0:
                    logger.debug(f"OCR result: '{ocr_result['validated_text'] or ocr_result['text']}' (conf: {ocr_result['confidence']:.3f})")
                    
                    if ocr_result['confidence'] > best_confidence:
                        best_result = ocr_result
                        best_confidence = ocr_result['confidence']
                
                # Early break for high confidence
                if best_confidence > 0.9:
                    logger.debug(f"Early break with high confidence: {best_confidence:.3f}")
                    break
            
            return best_result
            
        except Exception as e:
            logger.error(f"Plate detection error: {e}")
            return None
    
    def _background_cleanup(self):
        """Background cleanup thread"""
        while True:
            try:
                time.sleep(3600)  # Run every hour
                
                # Clean up old images
                self._cleanup_old_images()
                
                # Clean up vehicle states
                self.state_tracker.cleanup_old_states()
                
                # Clean up OCR processor cache
                if hasattr(self.ocr_processor, 'enhancer') and hasattr(self.ocr_processor.enhancer, '_clean_cache'):
                    self.ocr_processor.enhancer._clean_cache()
                
                # Garbage collection
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                logger.info("🧹 Background cleanup completed")
                
            except Exception as e:
                logger.error(f"Background cleanup error: {e}")
    
    def _cleanup_old_images(self):
        """Clean up old image files"""
        try:
            current_time = time.time()
            cleanup_threshold = config.IMAGE_CLEANUP_HOURS * 3600
            
            image_dirs = ['vehicle_images', 'plate_images', 'enhanced_images']
            total_cleaned = 0
            
            for image_dir in image_dirs:
                if not os.path.exists(image_dir):
                    continue
                
                for filename in os.listdir(image_dir):
                    file_path = os.path.join(image_dir, filename)
                    if os.path.isfile(file_path):
                        file_age = current_time - os.path.getmtime(file_path)
                        if file_age > cleanup_threshold:
                            os.remove(file_path)
                            total_cleaned += 1
            
            if total_cleaned > 0:
                logger.info(f"Cleaned up {total_cleaned} old image files")
                
        except Exception as e:
            logger.error(f"Image cleanup error: {e}")
    
    def get_system_status(self):
        """Get comprehensive system status"""
        # Calculate occupancy
        total_spots = len(parking_spots)
        occupied_spots = sum(1 for state in self.parking_state.values() 
                           if self.state_tracker.get_confirmed_state(state['spot_id']))
        
        # Performance metrics
        processing_times = list(self.stats['processing_times'])
        avg_processing_time = np.mean(processing_times) if processing_times else 0
        
        # Server connection status
        connection_status = self.server_sync.get_connection_status()
        
        # OCR processor stats
        ocr_stats = self.ocr_processor.get_stats()
        
        # Calculate uptime
        uptime_seconds = time.time() - self.stats['start_time']
        
        return {
            'camera_id': config.CAMERA_ID,
            'location_name': config.LOCATION_NAME,
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': int(uptime_seconds),
            'parking_summary': {
                'total_spots': total_spots,
                'occupied_spots': occupied_spots,
                'available_spots': total_spots - occupied_spots,
                'occupancy_rate': round((occupied_spots / total_spots) * 100, 1) if total_spots > 0 else 0
            },
            'processing_stats': {
                'frames_processed': self.stats['frames_processed'],
                'events_generated': self.stats['events_generated'],
                'events_sent': self.stats['events_sent'],
                'events_queued': self.stats['events_queued'],
                'plates_detected': self.stats['plates_detected'],
                'plates_validated': self.stats['plates_validated'],
                'avg_processing_time': round(avg_processing_time, 3),
                'detection_rate': round((self.stats['plates_detected'] / max(1, self.stats['events_generated'])) * 100, 1),
                'validation_rate': round((self.stats['plates_validated'] / max(1, self.stats['plates_detected'])) * 100, 1)
            },
            'server_connection': connection_status,
            'ocr_processor': ocr_stats,
            'spots_detail': [
                {
                    'spot_id': spot_id,
                    'spot_name': state['spot_name'],
                    'is_occupied': self.state_tracker.get_confirmed_state(spot_id),
                    'plate_text': state['last_plate'],
                    'plate_confidence': state['last_confidence'],
                    'enter_time': state['enter_time'],
                    'detection_count': state['detection_count']
                }
                for spot_id, state in self.parking_state.items()
            ]
        }
    
    def cleanup(self):
        """Enhanced cleanup with comprehensive resource management"""
        logger.info("🧹 Shutting down Enhanced Parking Monitor...")
        
        # Stop background cleanup thread
        if hasattr(self, 'cleanup_thread') and self.cleanup_thread.is_alive():
            logger.info("Stopping background cleanup thread...")
        
        # Cleanup server sync
        self.server_sync.cleanup()
        
        # Cleanup OCR processor
        self.ocr_processor.cleanup()
        
        # Final statistics
        uptime = time.time() - self.stats['start_time']
        if self.stats['frames_processed'] > 0:
            logger.info(f"📊 Final Performance Summary:")
            logger.info(f"   - Uptime: {uptime:.1f}s ({uptime/3600:.2f}h)")
            logger.info(f"   - Frames processed: {self.stats['frames_processed']}")
            logger.info(f"   - Events generated: {self.stats['events_generated']}")
            logger.info(f"   - Events sent: {self.stats['events_sent']}")
            logger.info(f"   - Events queued: {self.stats['events_queued']}")
            logger.info(f"   - Plates detected: {self.stats['plates_detected']}")
            logger.info(f"   - Plates validated: {self.stats['plates_validated']}")
            
            if self.stats['processing_times']:
                avg_time = np.mean(self.stats['processing_times'])
                logger.info(f"   - Avg processing time: {avg_time:.3f}s")
                logger.info(f"   - Processing FPS: {1/avg_time:.1f}")
        
        # GPU cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Final garbage collection
        gc.collect()
        
        logger.info("✅ Enhanced Parking Monitor cleanup completed")

# Load and optimize models
def load_models():
    """Load and optimize YOLO models"""
    logger.info("🔄 Loading YOLO models...")
    
    try:
        # Load vehicle detection model
        vehicle_model_path = r"C:\Users\nam\runs\detect\train3\weights\best.pt"
        if not os.path.exists(vehicle_model_path):
            raise FileNotFoundError(f"Vehicle model not found: {vehicle_model_path}")
        
        vehicle_model = YOLO(vehicle_model_path).to(device)
        
        # Load plate detection model
        plate_model_path = r"G:\bkstar\parking AI\dataset\license-plate-finetune-v1m.pt"
        if not os.path.exists(plate_model_path):
            raise FileNotFoundError(f"Plate model not found: {plate_model_path}")
        
        plate_model = YOLO(plate_model_path).to(device)
        
        # Model optimization
        if torch.cuda.is_available():
            logger.info("Optimizing models for CUDA...")
            # Use half precision for faster inference
            vehicle_model.model.half()
            plate_model.model.half()
            
            # Warm up models
            dummy_input = torch.randn(1, 3, config.YOLO_IMG_SIZE, config.YOLO_IMG_SIZE).to(device).half()
            with torch.no_grad():
                _ = vehicle_model.model(dummy_input)
                _ = plate_model.model(dummy_input)
        
        logger.info("✅ Models loaded and optimized successfully")
        return vehicle_model, plate_model
        
    except Exception as e:
        logger.error(f"❌ Failed to load models: {e}")
        raise

# Load parking spots configuration
def load_parking_spots():
    """Load and preprocess parking spots configuration"""
    try:
        # Đọc từ file JSON đã upload
        with open('tests/parking_spots.json') as f:
            parking_data = json.load(f)
        
        spots = parking_data['parking_spots']
        
        # Preprocess parking spots
        for i, spot in enumerate(spots):
            if 'id' not in spot:
                spot['id'] = f"{config.CAMERA_ID}_{spot['name'].replace(' ', '_')}"
            
            # Create Shapely polygon
            poly = Polygon(spot['polygon'])
            spot['area'] = poly.area
            spot['polygon_shapely'] = poly
            spot['bbox'] = poly.bounds  # (minx, miny, maxx, maxy)
            
            # Validate polygon
            if not poly.is_valid:
                logger.warning(f"Invalid polygon for spot {spot['name']}")
                # Try to fix
                spot['polygon_shapely'] = poly.buffer(0)
        
        logger.info(f"✅ Loaded {len(spots)} parking spots")
        return spots
        
    except Exception as e:
        logger.error(f"❌ Failed to load parking spots: {e}")
        raise

# Main execution function
def main():
    """Enhanced main execution with comprehensive error handling"""
    
    # Initialize components
    try:
        # Load models
        global vehicle_model, plate_model
        vehicle_model, plate_model = load_models()
        
        # Load parking spots
        global parking_spots
        parking_spots = load_parking_spots()
        
        # Initialize monitor
        monitor = EnhancedParkingMonitor()
        
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        return
    
    # Video processing
    video_path = "videos/0723.mp4"
    
    if not os.path.exists(video_path):
        logger.error(f"❌ Video file not found: {video_path}")
        return
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        logger.error("❌ Cannot open video file")
        return
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    logger.info(f"🎬 Video info: {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s duration")
    
    # Performance tracking
    frame_id = 0
    display_counter = 0
    start_time = time.time()
    last_summary_time = time.time()
    last_status_time = time.time()
    
    SUMMARY_INTERVAL = 30.0  # seconds
    STATUS_INTERVAL = 120.0  # seconds
    
    logger.info("🚀 Starting Enhanced Parking System...")
    logger.info(f"🌐 Server: {config.SYNC_SERVER_URL}")
    logger.info(f"📡 Offline Mode: {'Enabled' if config.ENABLE_OFFLINE_MODE else 'Disabled'}")
    logger.info(f"🔧 OCR Module: Separated and optimized")
    logger.info(f"🎯 Detection: Vehicle={config.VEHICLE_CONF}, Plate={config.PLATE_CONF}")
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logger.info("📹 End of video reached")
                break
            
            frame_id += 1
            
            # Skip frames for performance
            if frame_id % config.FRAME_SKIP != 0:
                continue
            
            display_counter += 1
            current_time = time.time()
            
            # Vehicle detection
            try:
                vehicle_results = vehicle_model(
                    frame,
                    conf=config.VEHICLE_CONF,
                    verbose=False,
                    imgsz=config.YOLO_IMG_SIZE
                )[0]
            except Exception as e:
                logger.error(f"Vehicle detection error: {e}")
                continue
            
            # Process detected vehicles
            car_boxes = []
            for det in vehicle_results.boxes:
                try:
                    x1, y1, x2, y2 = map(int, det.xyxy[0])
                    cls = int(det.cls[0])
                    conf = float(det.conf[0])
                    label = vehicle_model.names[cls]
                    
                    if label.lower() in ['car', 'vehicle', 'truck', 'bus'] and conf >= config.VEHICLE_CONF:
                        car_id = monitor.get_car_id(det)
                        car_boxes.append(((x1, y1, x2, y2), car_id))
                        
                        # Draw vehicle box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                except Exception as e:
                    logger.error(f"Error processing vehicle detection: {e}")
                    continue
            
            # Process frame
            processed_events = monitor.process_frame(frame, car_boxes)
            
            # Handle processed events
            for result in processed_events:
                event = result['event']
                action = result['action']
                sent_status = "✅" if result.get('sent_to_server', False) else "📡"
                
                if action == 'enter':
                    plate_info = f" - {event.plate_text}" if event.plate_text != "Không nhận diện được biển" else ""
                    conf_info = f" (conf: {event.plate_confidence:.3f})" if event.plate_confidence > 0 else ""
                    logger.info(f"{sent_status} 🚗 {event.spot_name}: Vehicle entered{plate_info}{conf_info}")
                
                elif action == 'exit':
                    duration = result.get('duration_minutes', 0)
                    plate_info = f" - {event.plate_text}" if event.plate_text else ""
                    logger.info(f"{sent_status} 🚪 {event.spot_name}: Vehicle exited{plate_info} ({duration}m)")
            
            # Draw parking spots with enhanced visualization
            for spot in parking_spots:
                spot_id = spot['id']
                spot_name = spot['name']
                is_occupied = monitor.state_tracker.get_confirmed_state(spot_id)
                state = monitor.parking_state[spot_id]
                
                pts_array = np.array(spot['polygon'], np.int32).reshape((-1, 1, 2))
                
                # Enhanced color coding
                if is_occupied:
                    conf = state['last_confidence']
                    if conf > 0.8:
                        color = (0, 0, 255)  # Red - high confidence
                        thickness = 3
                    elif conf > 0.5:
                        color = (0, 100, 255)  # Orange - medium confidence
                        thickness = 2
                    elif conf > 0:
                        color = (0, 150, 255)  # Light orange - low confidence
                        thickness = 2
                    else:
                        color = (0, 255, 255)  # Yellow - no plate detected
                        thickness = 2
                    
                    status_text = f"{spot_name}: {state['last_plate'] or 'Occupied'}"
                else:
                    color = (0, 255, 0)  # Green - available
                    thickness = 2
                    status_text = f"{spot_name}: Available"
                
                # Draw polygon
                cv2.polylines(frame, [pts_array], isClosed=True, color=color, thickness=thickness)
                
                # Draw text with background
                text_position = tuple(pts_array[0][0])
                text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame, text_position, 
                             (text_position[0] + text_size[0], text_position[1] - text_size[1] - 5),
                             (0, 0, 0), -1)
                cv2.putText(frame, status_text, text_position,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Performance display
            if display_counter % 30 == 0:  # Update every 30 processed frames
                elapsed = current_time - start_time
                processing_fps = display_counter / elapsed
                
                system_status = monitor.get_system_status()
                connection_status = system_status['server_connection']
                
                # Connection indicator
                conn_indicator = "🟢" if connection_status['is_connected'] else "🔴"
                queue_info = f"Q:{connection_status['offline_queue_size']}" if config.ENABLE_OFFLINE_MODE else ""
                
                # Enhanced display info
                info_lines = [
                    f"FPS: {processing_fps:.1f} | Frame: {frame_id}/{total_frames}",
                    f"Occupied: {system_status['parking_summary']['occupied_spots']}/{system_status['parking_summary']['total_spots']} ({system_status['parking_summary']['occupancy_rate']}%)",
                    f"{conn_indicator} Server | Sent: {system_status['processing_stats']['events_sent']} | {queue_info}",
                    f"Plates: {system_status['processing_stats']['plates_detected']} detected, {system_status['processing_stats']['plates_validated']} valid",
                    f"OCR: {system_status['ocr_processor'].get('ocr_count', 0)} processed"
                ]
                
                # Draw info on frame
                y_offset = 30
                for i, line in enumerate(info_lines):
                    cv2.putText(frame, line, (10, y_offset + i * 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Enhanced console log
                conn_status = "Connected" if connection_status['is_connected'] else "Offline"
                logger.info(f"⚡ FPS: {processing_fps:.1f} | "
                           f"Occupancy: {system_status['parking_summary']['occupancy_rate']}% | "
                           f"Server: {conn_status} | "
                           f"Events: {system_status['processing_stats']['events_sent']}/{system_status['processing_stats']['events_generated']} | "
                           f"Plates: {system_status['processing_stats']['detection_rate']}%")
            
            # Periodic detailed summary
            if current_time - last_summary_time > SUMMARY_INTERVAL:
                system_status = monitor.get_system_status()
                ocr_stats = system_status['ocr_processor']
                
                logger.info(f"📊 System Summary:")
                logger.info(f"   - Occupancy: {system_status['parking_summary']['occupancy_rate']}% "
                           f"({system_status['parking_summary']['occupied_spots']}/{system_status['parking_summary']['total_spots']})")
                logger.info(f"   - Processing: {system_status['processing_stats']['events_generated']} events, "
                           f"{system_status['processing_stats']['plates_detected']} plates detected")
                logger.info(f"   - Server: {'Connected' if system_status['server_connection']['is_connected'] else 'Offline'}, "
                           f"Queue: {system_status['server_connection']['offline_queue_size']}")
                logger.info(f"   - OCR: {ocr_stats.get('ocr_count', 0)} processed, "
                           f"Enhancement: {ocr_stats.get('enhancer_enhancement_count', 0)} images")
                
                last_summary_time = current_time
            
            # Periodic detailed status
            if current_time - last_status_time > STATUS_INTERVAL:
                system_status = monitor.get_system_status()
                
                logger.info("🏁 Detailed Status:")
                for spot_detail in system_status['spots_detail']:
                    if spot_detail['is_occupied']:
                        enter_info = ""
                        if spot_detail['enter_time']:
                            try:
                                enter_time = datetime.fromisoformat(spot_detail['enter_time'])
                                duration = (datetime.now() - enter_time).total_seconds() / 60
                                enter_info = f" ({duration:.0f}m ago)"
                            except:
                                pass
                        
                        plate_info = spot_detail['plate_text'] or "Unknown"
                        conf_info = f" (conf: {spot_detail['plate_confidence']:.3f})" if spot_detail['plate_confidence'] > 0 else ""
                        
                        logger.info(f"   🚗 {spot_detail['spot_name']}: {plate_info}{conf_info}{enter_info}")
                
                last_status_time = current_time
            
            # Memory management
            if display_counter % 200 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Display frame
            cv2.namedWindow("Enhanced Parking Monitor", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Enhanced Parking Monitor", 1280, 720)
            cv2.imshow("Enhanced Parking Monitor", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("🛑 Stopped by user (Q key)")
                break
            elif key == ord('s'):
                # Save current status
                status = monitor.get_system_status()
                with open(f"status_{int(time.time())}.json", 'w') as f:
                    json.dump(status, f, indent=2)
                logger.info("💾 Status saved")
    
    except KeyboardInterrupt:
        logger.info("🛑 Stopped by Ctrl+C")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        logger.info("🧹 Starting cleanup...")
        cap.release()
        cv2.destroyAllWindows()
        monitor.cleanup()
        
        logger.info("🔚 Enhanced Parking System completed")

if __name__ == "__main__":
    main()