"""
Vehicle Detector - Enhanced YOLO Vehicle Detection
Phát hiện xe với YOLO và tracking nâng cao
"""

import cv2
import numpy as np
import logging
import time
from typing import List, Tuple, Optional, Dict, Any
from collections import defaultdict, deque
import threading
from dataclasses import dataclass

from config.settings import config
from core.models import DetectionResult
from core.exceptions import VehicleDetectionException, ModelInferenceException
from core.constants import VehicleType, PERFORMANCE_THRESHOLDS

logger = logging.getLogger(__name__)

@dataclass
class VehicleTrack:
    """Vehicle tracking information"""
    track_id: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    vehicle_type: str
    first_seen: float
    last_seen: float
    track_history: deque
    stable: bool = False
    
    def update(self, bbox: Tuple[int, int, int, int], confidence: float, timestamp: float):
        """Update track information"""
        self.bbox = bbox
        self.confidence = confidence
        self.last_seen = timestamp
        self.track_history.append((bbox, confidence, timestamp))
        
        # Mark as stable if tracked for sufficient time
        if not self.stable and (timestamp - self.first_seen) > 1.0:  # 1 second
            self.stable = True

class EnhancedVehicleDetector:
    """Enhanced vehicle detector with tracking and optimization"""
    
    def __init__(self, model):
        self.model = model
        self.device = next(model.model.parameters()).device
        
        # Detection settings
        self.conf_threshold = config.VEHICLE_CONF
        self.img_size = config.YOLO_IMG_SIZE
        self.max_det = 100
        
        # Tracking
        self.tracks = {}
        self.next_track_id = 1
        self.track_timeout = 2.0  # seconds
        self.iou_threshold = 0.3
        
        # Performance monitoring
        self.detection_times = deque(maxlen=100)
        self.frame_count = 0
        self.total_detections = 0
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Supported vehicle classes (COCO format)
        self.vehicle_classes = {
            2: VehicleType.CAR.value,      # car
            3: VehicleType.MOTORCYCLE.value,  # motorcycle  
            5: VehicleType.BUS.value,      # bus
            7: VehicleType.TRUCK.value,    # truck
        }
        
        logger.info(f"🚗 Enhanced Vehicle Detector initialized")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Confidence threshold: {self.conf_threshold}")
        logger.info(f"   Image size: {self.img_size}")
        logger.info(f"   Supported classes: {list(self.vehicle_classes.values())}")
    
    def detect_vehicles(self, frame: np.ndarray, frame_id: int = 0) -> List[DetectionResult]:
        """
        Detect vehicles in frame with enhanced processing
        
        Args:
            frame: Input image frame
            frame_id: Frame identifier for tracking
            
        Returns:
            List of DetectionResult objects
        """
        start_time = time.time()
        
        try:
            with self.lock:
                self.frame_count += 1
            
            # Validate input
            if frame is None or frame.size == 0:
                raise VehicleDetectionException("Invalid input frame", {"frame_id": frame_id})
            
            # Preprocess frame
            processed_frame = self._preprocess_frame(frame)
            
            # Run YOLO inference
            results = self._run_inference(processed_frame)
            
            # Process detections
            detections = self._process_detections(results, frame.shape, frame_id)
            
            # Update tracking
            tracked_detections = self._update_tracking(detections, time.time())
            
            # Update statistics
            detection_time = time.time() - start_time
            self.detection_times.append(detection_time)
            
            with self.lock:
                self.total_detections += len(tracked_detections)
            
            # Log performance periodically
            if self.frame_count % 100 == 0:
                self._log_performance_stats()
            
            return tracked_detections
            
        except Exception as e:
            logger.error(f"Vehicle detection failed for frame {frame_id}: {e}")
            raise VehicleDetectionException(str(e), {"frame_id": frame_id})
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for optimal detection"""
        try:
            # Ensure frame is in BGR format
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # Already BGR
                processed = frame.copy()
            else:
                raise ValueError("Frame must be BGR color image")
            
            # Optional: Apply preprocessing for better detection
            if config.ENABLE_SMART_ENHANCEMENT:
                processed = self._enhance_frame_for_detection(processed)
            
            return processed
            
        except Exception as e:
            logger.error(f"Frame preprocessing failed: {e}")
            return frame
    
    def _enhance_frame_for_detection(self, frame: np.ndarray) -> np.ndarray:
        """Apply enhancements to improve detection accuracy"""
        try:
            # Improve contrast and brightness slightly
            enhanced = cv2.convertScaleAbs(frame, alpha=1.1, beta=10)
            
            # Reduce noise while preserving edges
            enhanced = cv2.bilateralFilter(enhanced, 5, 75, 75)
            
            return enhanced
            
        except Exception as e:
            logger.warning(f"Frame enhancement failed: {e}")
            return frame
    
    def _run_inference(self, frame: np.ndarray):
        """Run YOLO inference with error handling"""
        try:
            # Run inference
            results = self.model(
                frame,
                conf=self.conf_threshold,
                verbose=False,
                imgsz=self.img_size,
                max_det=self.max_det,
                device=self.device
            )
            
            return results[0] if results else None
            
        except Exception as e:
            logger.error(f"YOLO inference failed: {e}")
            raise ModelInferenceException("vehicle_detector", str(e))
    
    def _process_detections(self, results, frame_shape: Tuple[int, int, int], frame_id: int) -> List[DetectionResult]:
        """Process YOLO detection results"""
        detections = []
        
        if results is None or len(results.boxes) == 0:
            return detections
        
        frame_height, frame_width = frame_shape[:2]
        
        for det in results.boxes:
            try:
                # Extract detection info
                bbox = det.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                conf = float(det.conf[0])
                cls = int(det.cls[0])
                
                # Filter by class (only vehicles)
                if cls not in self.vehicle_classes:
                    continue
                
                # Validate and clip bounding box
                x1, y1, x2, y2 = bbox
                x1 = max(0, min(int(x1), frame_width - 1))
                y1 = max(0, min(int(y1), frame_height - 1))
                x2 = max(x1 + 1, min(int(x2), frame_width))
                y2 = max(y1 + 1, min(int(y2), frame_height))
                
                # Validate detection quality
                if not self._is_valid_detection(x1, y1, x2, y2, conf):
                    continue
                
                # Create detection result
                detection = DetectionResult(
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    class_name=self.vehicle_classes[cls],
                    vehicle_type=self.vehicle_classes[cls],
                    additional_data={
                        'frame_id': frame_id,
                        'class_id': cls,
                        'area': (x2 - x1) * (y2 - y1)
                    }
                )
                
                detections.append(detection)
                
            except Exception as e:
                logger.warning(f"Error processing detection: {e}")
                continue
        
        # Sort by confidence (highest first)
        detections.sort(key=lambda d: d.confidence, reverse=True)
        
        logger.debug(f"Processed {len(detections)} valid vehicle detections")
        return detections
    
    def _is_valid_detection(self, x1: int, y1: int, x2: int, y2: int, conf: float) -> bool:
        """Validate detection quality"""
        # Check minimum confidence
        if conf < self.conf_threshold:
            return False
        
        # Check minimum size
        width, height = x2 - x1, y2 - y1
        area = width * height
        
        if area < PERFORMANCE_THRESHOLDS.get("MIN_DETECTION_SIZE", 32) ** 2:
            return False
        
        # Check aspect ratio (reasonable vehicle proportions)
        aspect_ratio = width / height if height > 0 else 0
        if aspect_ratio < 0.3 or aspect_ratio > 5.0:
            return False
        
        return True
    
    def _update_tracking(self, detections: List[DetectionResult], timestamp: float) -> List[DetectionResult]:
        """Update vehicle tracking with new detections"""
        try:
            # Clean up old tracks
            self._cleanup_old_tracks(timestamp)
            
            # Match detections to existing tracks
            matched_pairs, unmatched_detections, unmatched_tracks = self._match_detections_to_tracks(detections)
            
            # Update matched tracks
            for det_idx, track_id in matched_pairs:
                detection = detections[det_idx]
                track = self.tracks[track_id]
                track.update(detection.bbox, detection.confidence, timestamp)
                detection.car_id = track_id
            
            # Create new tracks for unmatched detections
            for det_idx in unmatched_detections:
                detection = detections[det_idx]
                track_id = self._create_new_track(detection, timestamp)
                detection.car_id = track_id
            
            # Mark unmatched tracks as potentially lost
            for track_id in unmatched_tracks:
                # Keep track but don't update (will be cleaned up later if timeout)
                pass
            
            return detections
            
        except Exception as e:
            logger.error(f"Tracking update failed: {e}")
            # Return detections without tracking IDs
            return detections
    
    def _cleanup_old_tracks(self, current_time: float):
        """Remove old tracks that haven't been updated"""
        expired_tracks = []
        
        for track_id, track in self.tracks.items():
            if current_time - track.last_seen > self.track_timeout:
                expired_tracks.append(track_id)
        
        for track_id in expired_tracks:
            del self.tracks[track_id]
        
        if expired_tracks:
            logger.debug(f"Cleaned up {len(expired_tracks)} expired tracks")
    
    def _match_detections_to_tracks(self, detections: List[DetectionResult]) -> Tuple[List[Tuple[int, str]], List[int], List[str]]:
        """Match current detections to existing tracks using IoU"""
        if not self.tracks:
            return [], list(range(len(detections))), []
        
        # Compute IoU matrix
        detection_boxes = [det.bbox for det in detections]
        track_boxes = {track_id: track.bbox for track_id, track in self.tracks.items()}
        
        iou_matrix = {}
        for det_idx, det_box in enumerate(detection_boxes):
            for track_id, track_box in track_boxes.items():
                iou = self._compute_iou(det_box, track_box)
                if iou > self.iou_threshold:
                    iou_matrix[(det_idx, track_id)] = iou
        
        # Simple greedy matching (could use Hungarian algorithm for optimal matching)
        matched_pairs = []
        used_detections = set()
        used_tracks = set()
        
        # Sort by IoU (highest first)
        sorted_matches = sorted(iou_matrix.items(), key=lambda x: x[1], reverse=True)
        
        for (det_idx, track_id), iou in sorted_matches:
            if det_idx not in used_detections and track_id not in used_tracks:
                matched_pairs.append((det_idx, track_id))
                used_detections.add(det_idx)
                used_tracks.add(track_id)
        
        # Find unmatched detections and tracks
        unmatched_detections = [i for i in range(len(detections)) if i not in used_detections]
        unmatched_tracks = [track_id for track_id in self.tracks.keys() if track_id not in used_tracks]
        
        return matched_pairs, unmatched_detections, unmatched_tracks
    
    def _compute_iou(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """Compute Intersection over Union (IoU) of two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Compute intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Compute union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - intersection_area
        
        if union_area <= 0:
            return 0.0
        
        return intersection_area / union_area
    
    def _create_new_track(self, detection: DetectionResult, timestamp: float) -> str:
        """Create new vehicle track"""
        track_id = f"vehicle_{self.next_track_id:06d}"
        self.next_track_id += 1
        
        track = VehicleTrack(
            track_id=track_id,
            bbox=detection.bbox,
            confidence=detection.confidence,
            vehicle_type=detection.vehicle_type,
            first_seen=timestamp,
            last_seen=timestamp,
            track_history=deque(maxlen=50)
        )
        
        track.track_history.append((detection.bbox, detection.confidence, timestamp))
        self.tracks[track_id] = track
        
        logger.debug(f"Created new track: {track_id}")
        return track_id
    
    def _log_performance_stats(self):
        """Log performance statistics"""
        if not self.detection_times:
            return
        
        avg_time = np.mean(self.detection_times)
        max_time = np.max(self.detection_times)
        min_time = np.min(self.detection_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        with self.lock:
            total_detections = self.total_detections
            frame_count = self.frame_count
        
        avg_detections_per_frame = total_detections / frame_count if frame_count > 0 else 0
        
        logger.info(f"🚗 Vehicle Detection Performance (last 100 frames):")
        logger.info(f"   Average time: {avg_time*1000:.1f}ms")
        logger.info(f"   Min/Max time: {min_time*1000:.1f}/{max_time*1000:.1f}ms")
        logger.info(f"   FPS: {fps:.1f}")
        logger.info(f"   Active tracks: {len(self.tracks)}")
        logger.info(f"   Avg detections/frame: {avg_detections_per_frame:.1f}")
    
    def get_active_tracks(self) -> Dict[str, VehicleTrack]:
        """Get currently active vehicle tracks"""
        return self.tracks.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive detection statistics"""
        with self.lock:
            stats = {
                'total_frames': self.frame_count,
                'total_detections': self.total_detections,
                'active_tracks': len(self.tracks),
                'stable_tracks': sum(1 for track in self.tracks.values() if track.stable),
                'vehicle_types': {}
            }
        
        # Count vehicles by type
        for track in self.tracks.values():
            vehicle_type = track.vehicle_type
            stats['vehicle_types'][vehicle_type] = stats['vehicle_types'].get(vehicle_type, 0) + 1
        
        # Performance stats
        if self.detection_times:
            detection_times_array = np.array(self.detection_times)
            stats.update({
                'avg_detection_time': float(np.mean(detection_times_array)),
                'max_detection_time': float(np.max(detection_times_array)),
                'min_detection_time': float(np.min(detection_times_array)),
                'detection_fps': float(1.0 / np.mean(detection_times_array)) if np.mean(detection_times_array) > 0 else 0
            })
        
        return stats
    
    def reset_tracking(self):
        """Reset all tracking data"""
        with self.lock:
            self.tracks.clear()
            self.next_track_id = 1
            logger.info("🔄 Vehicle tracking reset")
    
    def cleanup(self):
        """Clean up detector resources"""
        logger.info("🧹 Cleaning up vehicle detector...")
        
        with self.lock:
            self.tracks.clear()
            self.detection_times.clear()
        
        logger.info("✅ Vehicle detector cleanup completed")

# Factory function
def create_vehicle_detector(model):
    """Create enhanced vehicle detector instance"""
    return EnhancedVehicleDetector(model)

# Example usage
if __name__ == "__main__":
    # This would normally be run as part of the main system
    print("Vehicle Detector module - use within main parking system")