"""
camera_processor.py - Individual Camera Processing Module
=========================================================
Xử lý từng camera riêng lẻ
Bao gồm: video capture, vehicle detection, plate recognition, parking spot tracking
"""

import cv2
import json
import numpy as np
import torch
from ultralytics import YOLO
from shapely.geometry import Point, Polygon, box as create_box
from collections import deque
import os
import time
import threading
import queue
import uuid
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any
import logging

# Import các module khác
from .config import CameraConfig
from .state_tracker import VehicleStateTracker

logger = logging.getLogger(__name__)

class CameraProcessor:
    """
    Lớp xử lý cho từng camera riêng lẻ
    
    Chức năng chính:
    - Capture video từ file hoặc RTSP stream
    - Phát hiện xe bằng YOLO
    - Phát hiện biển số và OCR
    - Theo dõi trạng thái các ô đỗ xe
    - Tạo events khi có thay đổi trạng thái
    - Thread-safe processing với queue system
    """
    
    def __init__(self, 
                 camera_config: CameraConfig, 
                 vehicle_model, 
                 plate_model, 
                 ocr_processor=None):
        """
        Khởi tạo CameraProcessor
        
        Args:
            camera_config (CameraConfig): Cấu hình camera
            vehicle_model: YOLO model để phát hiện xe
            plate_model: YOLO model để phát hiện biển số
            ocr_processor: OCR processor để đọc biển số (optional)
        """
        self.config = camera_config
        self.vehicle_model = vehicle_model
        self.plate_model = plate_model
        self.ocr_processor = ocr_processor
        
        # Load parking spots configuration
        self.parking_spots = self._load_parking_spots()
        
        # Initialize state tracking
        self.state_tracker = VehicleStateTracker(
            min_detections=3,
            max_history=8,
            consistency_threshold=0.75
        )
        
        # Parking state storage
        self.parking_state = self._init_parking_state()
        
        # Processing control
        self.is_running = False
        self.is_paused = False
        
        # Thread-safe queues
        self.frame_queue = queue.Queue(maxsize=10)  # Input frames
        self.result_queue = queue.Queue(maxsize=50)  # Output results
        
        # Performance statistics
        self.stats = {
            'frames_processed': 0,
            'events_generated': 0,
            'plates_detected': 0,
            'processing_times': deque(maxlen=100),
            'last_frame_time': 0,
            'start_time': time.time(),
            'errors': 0,
            'fps': 0.0
        }
        
        # Threading components
        self.capture_thread = None
        self.processing_thread = None
        self.stop_event = threading.Event()
        
        # Resource management
        self.processing_lock = threading.RLock()
        
        logger.info(f"✅ Camera processor initialized: {self.config.name}")
    
    def _load_parking_spots(self) -> List[Dict[str, Any]]:
        """
        Load parking spots configuration từ JSON file
        
        Returns:
            List[Dict]: Danh sách parking spots với thông tin geometry
        """
        try:
            with open(self.config.parking_spots_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            spots = data.get('parking_spots', [])
            
            # Preprocess parking spots
            for i, spot in enumerate(spots):
                # Tạo ID nếu chưa có
                if 'id' not in spot:
                    spot['id'] = f"{self.config.camera_id}_{spot['name'].replace(' ', '_')}"
                
                # Tạo Shapely polygon từ coordinates
                try:
                    polygon = Polygon(spot['polygon'])
                    
                    # Validate và fix polygon nếu cần
                    if not polygon.is_valid:
                        polygon = polygon.buffer(0)  # Fix invalid geometry
                    
                    spot['polygon_shapely'] = polygon
                    spot['area'] = polygon.area
                    spot['bbox'] = polygon.bounds  # (minx, miny, maxx, maxy)
                    
                except Exception as e:
                    logger.error(f"❌ Invalid polygon for spot {spot['name']}: {e}")
                    continue
            
            # Filter out invalid spots
            valid_spots = [spot for spot in spots if 'polygon_shapely' in spot]
            
            logger.info(f"📍 Loaded {len(valid_spots)} parking spots for {self.config.camera_id}")
            return valid_spots
            
        except Exception as e:
            logger.error(f"❌ Failed to load parking spots for {self.config.camera_id}: {e}")
            return []
    
    def _init_parking_state(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize parking spot states
        
        Returns:
            Dict: spot_id -> state information
        """
        states = {}
        
        for spot in self.parking_spots:
            spot_id = spot['id']
            states[spot_id] = {
                'spot_id': spot_id,
                'spot_name': spot['name'],
                'occupied': False,
                'last_plate': None,
                'last_confidence': 0.0,
                'enter_time': None,
                'last_update': None,
                'detection_count': 0,
                'lock': threading.Lock()  # Per-spot lock for thread safety
            }
        
        return states
    
    def start(self):
        """Start camera processing threads"""
        if self.is_running:
            logger.warning(f"Camera {self.config.camera_id} is already running")
            return
        
        logger.info(f"🚀 Starting camera processor: {self.config.name}")
        
        self.is_running = True
        self.stop_event.clear()
        
        # Start video capture thread
        self.capture_thread = threading.Thread(
            target=self._capture_loop, 
            name=f"Capture-{self.config.camera_id}",
            daemon=True
        )
        self.capture_thread.start()
        
        # Start frame processing thread
        self.processing_thread = threading.Thread(
            target=self._processing_loop,
            name=f"Process-{self.config.camera_id}",
            daemon=True
        )
        self.processing_thread.start()
        
        logger.info(f"✅ Camera processor started: {self.config.name}")
    
    def stop(self):
        """Stop camera processing"""
        if not self.is_running:
            return
        
        logger.info(f"🛑 Stopping camera processor: {self.config.name}")
        
        self.is_running = False
        self.stop_event.set()
        
        # Wait for threads to finish (with timeout)
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=5)
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5)
        
        logger.info(f"✅ Camera processor stopped: {self.config.name}")
    
    def pause(self):
        """Pause processing (keep capture running)"""
        self.is_paused = True
        logger.info(f"⏸️ Paused processing: {self.config.name}")
    
    def resume(self):
        """Resume processing"""
        self.is_paused = False
        logger.info(f"▶️ Resumed processing: {self.config.name}")
    
    def _capture_loop(self):
        """
        Video capture loop - chạy trong thread riêng
        Đọc frames từ video source và đưa vào queue để xử lý
        """
        cap = cv2.VideoCapture(self.config.stream_url)
        
        if not cap.isOpened():
            logger.error(f"❌ Cannot open video source: {self.config.stream_url}")
            return
        
        # Set capture properties nếu cần
        try:
            # Thử set buffer size để giảm delay
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except:
            pass
        
        frame_interval = 1.0 / self.config.max_fps
        last_frame_time = 0
        frame_count = 0
        
        logger.info(f"📹 Video capture started for {self.config.camera_id}")
        
        try:
            while self.is_running and not self.stop_event.is_set():
                current_time = time.time()
                
                # FPS limiting
                if current_time - last_frame_time < frame_interval:
                    time.sleep(0.01)
                    continue
                
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"⚠️ Failed to read frame from {self.config.camera_id}")
                    
                    # Try to restart video source (useful for video files)
                    cap.release()
                    time.sleep(1)
                    cap = cv2.VideoCapture(self.config.stream_url)
                    continue
                
                frame_count += 1
                
                # Frame skipping for performance
                if frame_count % self.config.frame_skip != 0:
                    continue
                
                # Add frame to queue if not full
                try:
                    frame_data = {
                        'frame': frame.copy(),
                        'timestamp': datetime.now(),
                        'frame_id': frame_count,
                        'capture_time': current_time
                    }
                    
                    self.frame_queue.put_nowait(frame_data)
                    last_frame_time = current_time
                    
                except queue.Full:
                    # Skip frame if queue is full (prevent memory buildup)
                    pass
        
        except Exception as e:
            logger.error(f"❌ Capture error for {self.config.camera_id}: {e}")
        finally:
            cap.release()
            logger.info(f"📹 Video capture stopped for {self.config.camera_id}")
    
    def _processing_loop(self):
        """
        Frame processing loop - chạy trong thread riêng
        Lấy frames từ queue và xử lý detection + tracking
        """
        logger.info(f"🔄 Frame processing started for {self.config.camera_id}")
        
        try:
            while self.is_running and not self.stop_event.is_set():
                try:
                    # Get frame from queue with timeout
                    frame_data = self.frame_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Skip processing if paused
                if self.is_paused:
                    time.sleep(0.1)
                    continue
                
                # Process the frame
                self._process_frame(frame_data)
                
        except Exception as e:
            logger.error(f"❌ Processing error for {self.config.camera_id}: {e}")
        finally:
            logger.info(f"🔄 Frame processing stopped for {self.config.camera_id}")
    
    def _process_frame(self, frame_data: Dict[str, Any]):
        """
        Xử lý một frame để detect vehicles và update parking spots
        
        Args:
            frame_data (Dict): Thông tin frame từ capture thread
                {
                    'frame': numpy array,
                    'timestamp': datetime,
                    'frame_id': int,
                    'capture_time': float
                }
        """
        start_time = time.time()
        
        try:
            frame = frame_data['frame']
            timestamp = frame_data['timestamp']
            
            # Vehicle detection using YOLO
            vehicle_results = self.vehicle_model(
                frame,
                conf=self.config.vehicle_conf,
                verbose=False,
                imgsz=self.config.yolo_img_size
            )[0]
            
            # Extract vehicle bounding boxes
            car_boxes = []
            for det in vehicle_results.boxes:
                x1, y1, x2, y2 = map(int, det.xyxy[0])
                cls = int(det.cls[0])
                conf = float(det.conf[0])
                label = self.vehicle_model.names[cls]
                
                # Filter for relevant vehicle classes
                if label.lower() in ['car', 'vehicle', 'truck', 'bus', 'motorcycle']:
                    car_id = f"car_{x1}_{y1}_{x2}_{y2}_{frame_data['frame_id']}"
                    car_boxes.append({
                        'bbox': (x1, y1, x2, y2),
                        'car_id': car_id,
                        'confidence': conf,
                        'class': label
                    })
            
            # Process parking spots with detected vehicles
            events = self._process_parking_spots(frame, car_boxes, timestamp)
            
            # Create result object
            result = {
                'camera_id': self.config.camera_id,
                'frame': frame,
                'timestamp': timestamp,
                'car_boxes': car_boxes,
                'events': events,
                'processing_time': time.time() - start_time,
                'frame_id': frame_data['frame_id']
            }
            
            # Add result to output queue
            try:
                self.result_queue.put_nowait(result)
            except queue.Full:
                # Remove oldest result and add new one
                try:
                    self.result_queue.get_nowait()
                    self.result_queue.put_nowait(result)
                except queue.Empty:
                    pass
            
            # Update statistics
            with self.processing_lock:
                self.stats['frames_processed'] += 1
                processing_time = time.time() - start_time
                self.stats['processing_times'].append(processing_time)
                self.stats['last_frame_time'] = time.time()
                
                if events:
                    self.stats['events_generated'] += len(events)
                
                # Calculate FPS
                elapsed = time.time() - self.stats['start_time']
                if elapsed > 0:
                    self.stats['fps'] = self.stats['frames_processed'] / elapsed
            
        except Exception as e:
            logger.error(f"❌ Frame processing error for {self.config.camera_id}: {e}")
            with self.processing_lock:
                self.stats['errors'] += 1
    
    def _process_parking_spots(self, frame, car_boxes: List[Dict], timestamp: datetime) -> List[Dict]:
        """
        Xử lý các ô đỗ xe để detect state changes
        
        Args:
            frame: Frame hiện tại
            car_boxes: List các bounding box của xe detect được
            timestamp: Thời gian frame
            
        Returns:
            List[Dict]: Danh sách events được tạo ra
        """
        events = []
        
        for spot in self.parking_spots:
            spot_id = spot['id']
            spot_name = spot['name']
            spot_polygon = spot['polygon_shapely']
            spot_bbox = spot['bbox']
            
            # Find intersecting vehicles
            intersecting_vehicles = []
            
            for car_data in car_boxes:
                car_bbox = car_data['bbox']
                x1, y1, x2, y2 = car_bbox
                
                # Quick bounding box intersection check first
                if (x2 < spot_bbox[0] - 20 or x1 > spot_bbox[2] + 20 or 
                    y2 < spot_bbox[1] - 20 or y1 > spot_bbox[3] + 20):
                    continue
                
                # Detailed polygon intersection
                car_polygon = create_box(x1, y1, x2, y2)
                intersection = spot_polygon.intersection(car_polygon)
                
                # Check if intersection is significant
                intersection_ratio = intersection.area / spot_polygon.area
                if intersection_ratio > self.config.intersection_threshold:
                    intersecting_vehicles.append({
                        'car_data': car_data,
                        'intersection_ratio': intersection_ratio
                    })
            
            # Determine if spot is occupied
            spot_occupied = len(intersecting_vehicles) > 0
            best_vehicle = None
            
            if spot_occupied:
                # Choose vehicle with best intersection ratio
                best_vehicle = max(intersecting_vehicles, key=lambda v: v['intersection_ratio'])
                car_id = best_vehicle['car_data']['car_id']
            else:
                car_id = None
            
            # Update state tracker
            state_change = self.state_tracker.update_detection(spot_id, spot_occupied, car_id)
            
            # Create event if there's a confirmed state change
            if state_change:
                event = self._create_parking_event(
                    state_change, spot_name, best_vehicle, frame, timestamp
                )
                if event:
                    events.append(event)
        
        return events
    
    def _create_parking_event(self, 
                             state_change: Dict[str, Any], 
                             spot_name: str,
                             vehicle_data: Optional[Dict],
                             frame,
                             timestamp: datetime) -> Optional[Dict[str, Any]]:
        """
        Tạo parking event từ state change
        
        Args:
            state_change: Thông tin state change từ VehicleStateTracker
            spot_name: Tên ô đỗ
            vehicle_data: Thông tin vehicle (nếu có)
            frame: Frame hiện tại
            timestamp: Timestamp
            
        Returns:
            Dict hoặc None: Event data
        """
        spot_id = state_change['spot_id']
        new_state = state_change['new_state']
        car_id = state_change.get('car_id')
        
        # Base event data
        event = {
            'id': str(uuid.uuid4()),
            'camera_id': self.config.camera_id,
            'spot_id': spot_id,
            'spot_name': spot_name,
            'event_type': 'enter' if new_state else 'exit',
            'timestamp': timestamp.isoformat(),
            'location_name': self.config.location_name,
            'confidence': state_change.get('confidence', 0.0),
            'plate_text': None,
            'plate_confidence': 0.0,
            'vehicle_image_path': None,
            'plate_image_path': None
        }
        
        # Process plate detection for enter events
        if new_state and vehicle_data:
            car_bbox = vehicle_data['car_data']['bbox']
            plate_result = self._detect_and_process_plate(frame, car_bbox, spot_id, timestamp)
            
            if plate_result:
                event.update({
                    'plate_text': plate_result.get('text', 'Unknown'),
                    'plate_confidence': plate_result.get('confidence', 0.0),
                    'vehicle_image_path': plate_result.get('vehicle_image_path'),
                    'plate_image_path': plate_result.get('plate_image_path')
                })
                
                with self.processing_lock:
                    self.stats['plates_detected'] += 1
        
        # Update parking state
        self._update_parking_state(spot_id, event, timestamp)
        
        return event
    
    def _detect_and_process_plate(self, frame, car_bbox: Tuple[int, int, int, int], 
                                 spot_id: str, timestamp: datetime) -> Optional[Dict[str, Any]]:
        """
        Detect và process license plate từ vehicle crop
        
        Args:
            frame: Frame gốc
            car_bbox: Bounding box của xe (x1, y1, x2, y2)
            spot_id: ID ô đỗ
            timestamp: Timestamp
            
        Returns:
            Dict hoặc None: Plate processing result
        """
        try:
            x1, y1, x2, y2 = car_bbox
            h, w = frame.shape[:2]
            
            # Ensure valid crop coordinates
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                return None
            
            # Crop vehicle region
            car_crop = frame[y1:y2, x1:x2]
            if car_crop.size == 0:
                return None
            
            result = {'confidence': 0.0, 'text': None}
            
            # Save vehicle image if enabled
            if self.config.save_images:
                timestamp_str = int(timestamp.timestamp())
                vehicle_filename = f"vehicle_{self.config.camera_id}_{spot_id}_{timestamp_str}.jpg"
                vehicle_image_path = os.path.join('vehicle_images', vehicle_filename)
                
                os.makedirs('vehicle_images', exist_ok=True)
                cv2.imwrite(vehicle_image_path, car_crop)
                result['vehicle_image_path'] = vehicle_image_path
            
            # Plate detection with YOLO
            plate_results = self.plate_model(
                car_crop,
                conf=self.config.plate_conf,
                verbose=False,
                imgsz=self.config.plate_img_size
            )[0]
            
            if len(plate_results.boxes) == 0:
                return result
            
            # Process best plate detection
            best_plate_crop = None
            best_confidence = 0
            
            for plate_det in plate_results.boxes:
                px1, py1, px2, py2 = map(int, plate_det.xyxy[0])
                ch, cw = car_crop.shape[:2]
                
                # Ensure valid plate crop coordinates
                px1, py1 = max(0, px1), max(0, py1)
                px2, py2 = min(cw, px2), min(ch, py2)
                
                if px2 <= px1 or py2 <= py1:
                    continue
                
                plate_crop = car_crop[py1:py2, px1:px2]
                if plate_crop.size == 0:
                    continue
                
                conf = float(plate_det.conf[0])
                if conf > best_confidence:
                    best_confidence = conf
                    best_plate_crop = plate_crop
            
            # OCR processing if plate found and OCR available
            if best_plate_crop is not None and self.ocr_processor:
                try:
                    # Save plate image
                    if self.config.save_images:
                        plate_filename = f"plate_{self.config.camera_id}_{spot_id}_{timestamp_str}.jpg"
                        plate_image_path = os.path.join('plate_images', plate_filename)
                        os.makedirs('plate_images', exist_ok=True)
                        
                        # Process with OCR
                        ocr_result = self.ocr_processor.process_plate(best_plate_crop, plate_image_path)
                        
                        if ocr_result:
                            result.update({
                                'text': ocr_result.get('validated_text') or ocr_result.get('text', ''),
                                'confidence': ocr_result.get('confidence', 0.0),
                                'plate_image_path': plate_image_path
                            })
                
                except Exception as ocr_error:
                    logger.warning(f"⚠️ OCR processing error: {ocr_error}")
                    # Fallback: just save the plate image
                    if self.config.save_images:
                        result['plate_image_path'] = plate_image_path
            
            return result if result.get('text') or result.get('vehicle_image_path') else None
            
        except Exception as e:
            logger.error(f"❌ Plate detection error: {e}")
            return None
    
    def _update_parking_state(self, spot_id: str, event: Dict[str, Any], timestamp: datetime):
        """
        Cập nhật parking state cho spot
        
        Args:
            spot_id: ID ô đỗ
            event: Event data
            timestamp: Timestamp
        """
        if spot_id not in self.parking_state:
            return
        
        state = self.parking_state[spot_id]
        
        with state['lock']:
            if event['event_type'] == 'enter':
                state.update({
                    'occupied': True,
                    'last_plate': event.get('plate_text'),
                    'last_confidence': event.get('plate_confidence', 0.0),
                    'enter_time': timestamp.isoformat(),
                    'last_update': timestamp.isoformat()
                })
            
            elif event['event_type'] == 'exit':
                # Calculate duration for exit event
                duration_minutes = 0
                if state.get('enter_time'):
                    try:
                        enter_time = datetime.fromisoformat(state['enter_time'])
                        duration_minutes = int((timestamp - enter_time).total_seconds() / 60)
                    except:
                        pass
                
                event['duration_minutes'] = duration_minutes
                event['plate_text'] = state.get('last_plate')  # Use plate from entry
                
                state.update({
                    'occupied': False,
                    'last_plate': None,
                    'last_confidence': 0.0,
                    'enter_time': None,
                    'last_update': timestamp.isoformat()
                })
            
            state['detection_count'] += 1
    
    def get_latest_result(self) -> Optional[Dict[str, Any]]:
        """
        Lấy kết quả processing mới nhất
        
        Returns:
            Dict hoặc None: Latest processing result
        """
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Lấy thống kê performance của processor
        
        Returns:
            Dict: Chi tiết thống kê
        """
        with self.processing_lock:
            elapsed = time.time() - self.stats['start_time']
            processing_times = list(self.stats['processing_times'])
            
            # Get occupied spots count
            occupied_spots = sum(
                1 for spot_id in self.parking_state.keys() 
                if self.state_tracker.get_confirmed_state(spot_id)
            )
            
            return {
                'camera_id': self.config.camera_id,
                'name': self.config.name,
                'uptime_seconds': elapsed,
                'frames_processed': self.stats['frames_processed'],
                'events_generated': self.stats['events_generated'],
                'plates_detected': self.stats['plates_detected'],
                'errors': self.stats['errors'],
                'avg_processing_time': np.mean(processing_times) if processing_times else 0,
                'fps': self.stats['fps'],
                'queue_sizes': {
                    'frame_queue': self.frame_queue.qsize(),
                    'result_queue': self.result_queue.qsize()
                },
                'is_running': self.is_running,
                'is_paused': self.is_paused,
                'parking_spots_count': len(self.parking_spots),
                'occupied_spots': occupied_spots
            }
    
    def get_parking_status(self) -> List[Dict[str, Any]]:
        """
        Lấy trạng thái hiện tại của tất cả parking spots
        
        Returns:
            List[Dict]: Danh sách trạng thái các ô đỗ
        """
        spots_status = []
        
        for spot_id, state in self.parking_state.items():
            with state['lock']:
                is_occupied = self.state_tracker.get_confirmed_state(spot_id)
                detection_stats = self.state_tracker.get_detection_stats(spot_id)
                
                spots_status.append({
                    'spot_id': spot_id,
                    'spot_name': state['spot_name'],
                    'is_occupied': is_occupied,
                    'plate_text': state['last_plate'],
                    'plate_confidence': state['last_confidence'],
                    'enter_time': state['enter_time'],
                    'last_update': state['last_update'],
                    'detection_count': state['detection_count'],
                    'state_confidence': detection_stats['confidence'],
                    'state_duration': self.state_tracker.get_state_duration(spot_id)
                })
        
        return spots_status
    
    def cleanup(self):
        """Dọn dẹp resources"""
        logger.info(f"🧹 Starting cleanup for camera processor: {self.config.name}")
        
        # Stop processing
        self.stop()
        
        # Clear queues
        self._clear_queues()
        
        # Reset state tracker
        self.state_tracker.reset_all_states()
        
        logger.info(f"✅ Camera processor cleanup completed: {self.config.name}")
    
    def _clear_queues(self):
        """Clear all queues"""
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break
    
    def reset_statistics(self):
        """Reset performance statistics"""
        with self.processing_lock:
            self.stats = {
                'frames_processed': 0,
                'events_generated': 0,
                'plates_detected': 0,
                'processing_times': deque(maxlen=100),
                'last_frame_time': 0,
                'start_time': time.time(),
                'errors': 0,
                'fps': 0.0
            }
        
        logger.info(f"🔄 Reset statistics for {self.config.camera_id}")
    
    def update_config(self, new_config: CameraConfig):
        """
        Cập nhật configuration (cần restart để có hiệu lực)
        
        Args:
            new_config (CameraConfig): Configuration mới
        """
        old_config = self.config
        self.config = new_config
        
        # Log changes
        changes = []
        if old_config.vehicle_conf != new_config.vehicle_conf:
            changes.append(f"vehicle_conf: {old_config.vehicle_conf} -> {new_config.vehicle_conf}")
        if old_config.plate_conf != new_config.plate_conf:
            changes.append(f"plate_conf: {old_config.plate_conf} -> {new_config.plate_conf}")
        if old_config.max_fps != new_config.max_fps:
            changes.append(f"max_fps: {old_config.max_fps} -> {new_config.max_fps}")
        
        if changes:
            logger.info(f"🔧 Config updated for {self.config.camera_id}: {', '.join(changes)}")
        
        # Reload parking spots if file changed
        if old_config.parking_spots_file != new_config.parking_spots_file:
            logger.info(f"🔄 Reloading parking spots for {self.config.camera_id}")
            self.parking_spots = self._load_parking_spots()
            self.parking_state = self._init_parking_state()
            self.state_tracker.reset_all_states()