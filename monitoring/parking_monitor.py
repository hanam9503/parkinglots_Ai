"""
Enhanced Parking Monitor - Giám sát bãi đỗ xe nâng cao
Hệ thống giám sát bãi đỗ xe với AI detection và tracking
"""

import cv2
import numpy as np
import logging
import time
import threading
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
import json
import uuid

from detection.vehicle_detector import EnhancedVehicleDetector
from detection.plate_detector import EnhancedPlateDetector
from core.models import DetectionResult, ParkingEvent, ParkingSpotStatus as SpotStatus
from core.exceptions import ParkingSystemException, SystemException, DetectionException
from config.settings import config

logger = logging.getLogger(__name__)

@dataclass
class ParkingSpot:
    """Parking spot configuration"""
    id: str
    polygon: List[Tuple[int, int]]
    name: str
    zone: str
    capacity: int = 1
    priority: int = 0
    
    def contains_point(self, point: Tuple[int, int]) -> bool:
        """Check if point is inside parking spot polygon"""
        return cv2.pointPolygonTest(np.array(self.polygon, np.int32), point, False) >= 0
    
    def intersects_bbox(self, bbox: Tuple[int, int, int, int]) -> bool:
        """Check if bounding box intersects with parking spot"""
        x1, y1, x2, y2 = bbox
        
        # Check if any corner of bbox is inside polygon
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        for corner in corners:
            if self.contains_point(corner):
                return True
        
        # Check if any polygon point is inside bbox
        for px, py in self.polygon:
            if x1 <= px <= x2 and y1 <= py <= y2:
                return True
        
        return False
    
    def get_center(self) -> Tuple[int, int]:
        """Get center point of parking spot"""
        points = np.array(self.polygon)
        center_x = int(np.mean(points[:, 0]))
        center_y = int(np.mean(points[:, 1]))
        return (center_x, center_y)

@dataclass
class SpotState:
    """Current state of a parking spot"""
    spot_id: str
    is_occupied: bool
    last_change: datetime
    vehicle_id: Optional[str] = None
    license_plate: Optional[str] = None
    confidence: float = 0.0
    detection_count: int = 0
    stable_count: int = 0
    
    def is_stable(self, threshold: int = 3) -> bool:
        """Check if state is stable (confirmed)"""
        return self.stable_count >= threshold

class EnhancedParkingMonitor:
    """Enhanced parking monitor with comprehensive detection and tracking"""
    
    def __init__(self, parking_spots_config: List[Dict[str, Any]]):
        """
        Initialize parking monitor
        
        Args:
            parking_spots_config: List of parking spot configurations
        """
        # Initialize parking spots
        self.parking_spots = self._init_parking_spots(parking_spots_config)
        self.spot_states = {spot.id: SpotState(
            spot_id=spot.id,
            is_occupied=False,
            last_change=datetime.now()
        ) for spot in self.parking_spots}
        
        # Detectors (will be initialized with models)
        self.vehicle_detector = None
        self.plate_detector = None
        
        # Frame processing
        self.frame_count = 0
        self.last_frame_time = time.time()
        
        # Event tracking
        self.events_buffer = deque(maxlen=1000)
        self.last_event_id = 0
        
        # Performance monitoring
        self.processing_times = deque(maxlen=100)
        self.detection_stats = defaultdict(int)
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Status
        self.is_running = False
        self.start_time = datetime.now()
        
        logger.info(f"🏁 Enhanced Parking Monitor initialized")
        logger.info(f"   Parking spots: {len(self.parking_spots)}")
        logger.info(f"   Detection stability threshold: {getattr(config, 'DETECTION_STABILITY_THRESHOLD', 3)}")
    
    def _init_parking_spots(self, spots_config: List[Dict[str, Any]]) -> List[ParkingSpot]:
        """Initialize parking spots from configuration"""
        parking_spots = []
        
        for spot_config in spots_config:
            try:
                spot = ParkingSpot(
                    id=spot_config['id'],
                    polygon=spot_config['polygon'],
                    name=spot_config.get('name', f"Spot {spot_config['id']}"),
                    zone=spot_config.get('zone', 'default'),
                    capacity=spot_config.get('capacity', 1),
                    priority=spot_config.get('priority', 0)
                )
                parking_spots.append(spot)
                
            except KeyError as e:
                logger.error(f"Invalid parking spot config: missing {e}")
                continue
            except Exception as e:
                logger.error(f"Error creating parking spot: {e}")
                continue
        
        return parking_spots
    
    def initialize_models(self, vehicle_model, plate_model):
        """Initialize detection models"""
        try:
            self.vehicle_detector = EnhancedVehicleDetector(vehicle_model)
            self.plate_detector = EnhancedPlateDetector(plate_model)
            
            logger.info("✅ Detection models initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize models: {e}")
            raise SystemException(f"Model initialization failed: {e}", "MODEL_INIT_ERROR")
    
    def process_frame(self, frame: np.ndarray, timestamp: Optional[datetime] = None) -> List[ParkingEvent]:
        """
        Process a single frame and detect parking changes
        
        Args:
            frame: Input video frame
            timestamp: Frame timestamp (optional)
            
        Returns:
            List of parking events detected in this frame
        """
        start_time = time.time()
        
        if timestamp is None:
            timestamp = datetime.now()
        
        try:
            with self.lock:
                self.frame_count += 1
            
            # Validate models
            if not self.vehicle_detector or not self.plate_detector:
                raise SystemException("Detection models not initialized", "MODELS_NOT_INITIALIZED")
            
            # Detect vehicles
            vehicle_detections = self.vehicle_detector.detect_vehicles(frame, self.frame_count)
            
            # Process parking spots
            events = self._process_parking_spots(vehicle_detections, frame, timestamp)
            
            # Update statistics
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            self.last_frame_time = time.time()
            
            # Log performance periodically
            if self.frame_count % 100 == 0:
                self._log_performance_stats()
            
            return events
            
        except Exception as e:
            logger.error(f"Frame processing failed: {e}")
            raise SystemException(f"Frame processing error: {e}", "FRAME_PROCESSING_ERROR")
    
    def _process_parking_spots(self, vehicle_detections: List[DetectionResult], 
                              frame: np.ndarray, timestamp: datetime) -> List[ParkingEvent]:
        """Process all parking spots for occupancy changes"""
        events = []
        
        for spot in self.parking_spots:
            try:
                # Find vehicles in this parking spot
                vehicles_in_spot = self._find_vehicles_in_spot(spot, vehicle_detections)
                
                # Determine occupancy
                is_occupied = len(vehicles_in_spot) > 0
                best_vehicle = vehicles_in_spot[0] if vehicles_in_spot else None
                
                # Update spot state
                event = self._update_spot_state(spot, is_occupied, best_vehicle, frame, timestamp)
                
                if event:
                    events.append(event)
                    
            except Exception as e:
                logger.error(f"Error processing spot {spot.id}: {e}")
                continue
        
        return events
    
    def _find_vehicles_in_spot(self, spot: ParkingSpot, 
                              vehicle_detections: List[DetectionResult]) -> List[DetectionResult]:
        """Find vehicles that intersect with a parking spot"""
        vehicles_in_spot = []
        
        for detection in vehicle_detections:
            # Check if vehicle bounding box intersects with parking spot
            if spot.intersects_bbox(detection.bbox):
                vehicles_in_spot.append(detection)
        
        # Sort by confidence (best first)
        vehicles_in_spot.sort(key=lambda v: v.confidence, reverse=True)
        
        return vehicles_in_spot
    
    def _update_spot_state(self, spot: ParkingSpot, is_occupied: bool, 
                          best_vehicle: Optional[DetectionResult], 
                          frame: np.ndarray, timestamp: datetime) -> Optional[ParkingEvent]:
        """Update parking spot state and generate events if needed"""
        
        current_state = self.spot_states[spot.id]
        
        # Update detection count
        current_state.detection_count += 1
        
        # Check for state change
        if current_state.is_occupied != is_occupied:
            # Reset stability counter on state change
            current_state.stable_count = 1
        else:
            # Increment stability counter for consistent state
            current_state.stable_count += 1
        
        # Update current values
        current_state.is_occupied = is_occupied
        if best_vehicle:
            current_state.vehicle_id = getattr(best_vehicle, 'car_id', None)
            current_state.confidence = best_vehicle.confidence
        else:
            current_state.vehicle_id = None
            current_state.confidence = 0.0
        
        # Generate event if state is stable and changed
        stability_threshold = getattr(config, 'DETECTION_STABILITY_THRESHOLD', 3)
        if (current_state.is_stable(stability_threshold) and 
            current_state.stable_count == stability_threshold):
            
            # State just became stable, generate event
            event = self._create_parking_event(spot, current_state, best_vehicle, frame, timestamp)
            current_state.last_change = timestamp
            
            # Try to detect license plate if vehicle present
            if is_occupied and best_vehicle:
                self._detect_license_plate(event, best_vehicle, frame)
            
            return event
        
        return None
    
    def _create_parking_event(self, spot: ParkingSpot, state: SpotState, 
                             vehicle: Optional[DetectionResult], 
                             frame: np.ndarray, timestamp: datetime) -> ParkingEvent:
        """Create a parking event"""
        
        with self.lock:
            self.last_event_id += 1
            event_id = f"event_{self.last_event_id:08d}"
        
        # Determine event type
        if state.is_occupied:
            event_type = "VEHICLE_ENTERED"
            status = "occupied"
        else:
            event_type = "VEHICLE_EXITED"
            status = "empty"
        
        # Create event
        event = ParkingEvent(
            event_id=event_id,
            spot_id=spot.id,
            event_type=event_type,
            timestamp=timestamp,
            vehicle_type=vehicle.vehicle_type if vehicle else None,
            confidence=state.confidence,
            license_plate=state.license_plate,
            additional_data={
                'spot_name': spot.name,
                'spot_zone': spot.zone,
                'vehicle_id': state.vehicle_id,
                'detection_count': state.detection_count,
                'frame_number': self.frame_count
            }
        )
        
        # Store event
        self.events_buffer.append(event)
        
        # Update statistics
        self.detection_stats[event_type] += 1
        self.detection_stats['total_events'] += 1
        
        logger.info(f"🚗 Parking Event: {event_type} at {spot.name} (confidence: {state.confidence:.2f})")
        
        return event
    
    def _detect_license_plate(self, event: ParkingEvent, vehicle: DetectionResult, frame: np.ndarray):
        """Attempt to detect license plate for vehicle"""
        try:
            # Extract vehicle crop
            x1, y1, x2, y2 = vehicle.bbox
            vehicle_crop = frame[y1:y2, x1:x2]
            
            if vehicle_crop.size > 0:
                # Detect plates in vehicle crop
                plate_detections = self.plate_detector.detect_plates_in_vehicle(
                    vehicle_crop, vehicle.bbox
                )
                
                if plate_detections:
                    # Get best plate
                    best_plate = self.plate_detector.get_best_plate(plate_detections)
                    if best_plate and best_plate.quality_score > 0.6:
                        # Note: In a real system, you would run OCR here
                        # For now, we just mark that a plate was detected
                        event.license_plate = f"DETECTED_{event.event_id[-4:]}"
                        event.additional_data['plate_confidence'] = best_plate.confidence
                        event.additional_data['plate_quality'] = best_plate.quality_score
                        
                        logger.debug(f"License plate detected for event {event.event_id}")
        
        except Exception as e:
            logger.warning(f"License plate detection failed for event {event.event_id}: {e}")
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current parking status"""
        occupied_spots = sum(1 for state in self.spot_states.values() 
                           if state.is_occupied and state.is_stable())
        
        total_spots = len(self.parking_spots)
        occupancy_rate = (occupied_spots / total_spots * 100) if total_spots > 0 else 0
        
        # Get zone statistics
        zone_stats = defaultdict(lambda: {'total': 0, 'occupied': 0})
        for spot in self.parking_spots:
            state = self.spot_states[spot.id]
            zone_stats[spot.zone]['total'] += 1
            if state.is_occupied and state.is_stable():
                zone_stats[spot.zone]['occupied'] += 1
        
        return {
            'timestamp': datetime.now().isoformat(),
            'camera_id': getattr(config, 'CAMERA_ID', 'default'),
            'system_status': 'running' if self.is_running else 'stopped',
            'parking_summary': {
                'total_spots': total_spots,
                'occupied_spots': occupied_spots,
                'available_spots': total_spots - occupied_spots,
                'occupancy_rate': round(occupancy_rate, 1)
            },
            'zone_statistics': dict(zone_stats),
            'processing_stats': self._get_processing_stats(),
            'recent_events': [asdict(event) for event in list(self.events_buffer)[-10:]]
        }
    
    def get_spot_details(self) -> List[Dict[str, Any]]:
        """Get detailed information for each parking spot"""
        spot_details = []
        
        for spot in self.parking_spots:
            state = self.spot_states[spot.id]
            
            spot_info = {
                'id': spot.id,
                'name': spot.name,
                'zone': spot.zone,
                'capacity': spot.capacity,
                'priority': spot.priority,
                'center': spot.get_center(),
                'polygon': spot.polygon,
                'status': {
                    'is_occupied': state.is_occupied,
                    'is_stable': state.is_stable(),
                    'vehicle_id': state.vehicle_id,
                    'license_plate': state.license_plate,
                    'confidence': state.confidence,
                    'last_change': state.last_change.isoformat(),
                    'detection_count': state.detection_count,
                    'stable_count': state.stable_count
                }
            }
            
            spot_details.append(spot_info)
        
        return spot_details
    
    def _get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        stats = {
            'frames_processed': self.frame_count,
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds(),
            'events_generated': self.detection_stats.get('total_events', 0)
        }
        
        # Performance metrics
        if self.processing_times:
            processing_times_array = np.array(self.processing_times)
            stats.update({
                'avg_processing_time': float(np.mean(processing_times_array)),
                'max_processing_time': float(np.max(processing_times_array)),
                'processing_fps': float(1.0 / np.mean(processing_times_array)) if np.mean(processing_times_array) > 0 else 0
            })
        
        # Detection statistics
        stats['detection_events'] = dict(self.detection_stats)
        
        return stats
    
    def _log_performance_stats(self):
        """Log performance statistics"""
        if not self.processing_times:
            return
        
        avg_time = np.mean(self.processing_times)
        max_time = np.max(self.processing_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        occupied_spots = sum(1 for state in self.spot_states.values() 
                           if state.is_occupied and state.is_stable())
        
        logger.info(f"🏁 Parking Monitor Performance (last 100 frames):")
        logger.info(f"   Processing time: {avg_time*1000:.1f}ms avg, {max_time*1000:.1f}ms max")
        logger.info(f"   Processing FPS: {fps:.1f}")
        logger.info(f"   Occupied spots: {occupied_spots}/{len(self.parking_spots)}")
        logger.info(f"   Events generated: {self.detection_stats.get('total_events', 0)}")
    
    def export_events(self, start_time: Optional[datetime] = None, 
                     end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Export events within time range"""
        events = list(self.events_buffer)
        
        if start_time or end_time:
            filtered_events = []
            for event in events:
                event_time = event.timestamp
                if start_time and event_time < start_time:
                    continue
                if end_time and event_time > end_time:
                    continue
                filtered_events.append(event)
            events = filtered_events
        
        return [asdict(event) for event in events]
    
    def reset_statistics(self):
        """Reset monitoring statistics"""
        with self.lock:
            self.frame_count = 0
            self.last_event_id = 0
            self.processing_times.clear()
            self.detection_stats.clear()
            self.events_buffer.clear()
            self.start_time = datetime.now()
        
        logger.info("🔄 Parking monitor statistics reset")
    
    def start_monitoring(self):
        """Start the monitoring system"""
        self.is_running = True
        logger.info("🚀 Parking monitoring started")
    
    def stop_monitoring(self):
        """Stop the monitoring system"""
        self.is_running = False
        logger.info("⏹️ Parking monitoring stopped")
    
    def cleanup(self):
        """Cleanup monitoring resources"""
        logger.info("🧹 Cleaning up parking monitor...")
        
        self.stop_monitoring()
        
        if self.vehicle_detector:
            self.vehicle_detector.cleanup()
        
        if self.plate_detector:
            self.plate_detector.cleanup()
        
        with self.lock:
            self.events_buffer.clear()
            self.processing_times.clear()
            self.detection_stats.clear()
        
        logger.info("✅ Parking monitor cleanup completed")

# Factory functions
def create_parking_monitor(spots_config: List[Dict[str, Any]]) -> EnhancedParkingMonitor:
    """Create parking monitor instance"""
    return EnhancedParkingMonitor(spots_config)

# Example parking spots configuration
EXAMPLE_PARKING_SPOTS = [
    {
        'id': 'A1',
        'name': 'Spot A1',
        'zone': 'Zone A',
        'polygon': [(100, 100), (200, 100), (200, 200), (100, 200)],
        'capacity': 1,
        'priority': 1
    },
    {
        'id': 'A2', 
        'name': 'Spot A2',
        'zone': 'Zone A',
        'polygon': [(220, 100), (320, 100), (320, 200), (220, 200)],
        'capacity': 1,
        'priority': 1
    },
    {
        'id': 'B1',
        'name': 'Spot B1', 
        'zone': 'Zone B',
        'polygon': [(100, 220), (200, 220), (200, 320), (100, 320)],
        'capacity': 1,
        'priority': 0
    }
]

# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create parking monitor
    monitor = create_parking_monitor(EXAMPLE_PARKING_SPOTS)
    
    print("Parking Monitor created successfully!")
    print(f"Monitoring {len(monitor.parking_spots)} parking spots")
    
    # Display current status
    status = monitor.get_current_status()
    print(f"\nCurrent Status:")
    print(json.dumps(status, indent=2, default=str))