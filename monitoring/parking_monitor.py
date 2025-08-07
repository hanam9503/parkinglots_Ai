import cv2
import numpy as np
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import threading

logger = logging.getLogger(__name__)

@dataclass
class ParkingSpot:
    """Simplified parking spot"""
    id: str
    polygon: List[Tuple[int, int]]
    name: str = ""
    zone: str = "default"
    
    def contains_point(self, point: Tuple[int, int]) -> bool:
        """Check if point is inside spot polygon"""
        return cv2.pointPolygonTest(np.array(self.polygon, np.int32), point, False) >= 0
    
    def intersects_bbox(self, bbox: Tuple[int, int, int, int]) -> bool:
        """Check if bounding box intersects with spot"""
        x1, y1, x2, y2 = bbox
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        return any(self.contains_point(corner) for corner in corners)
    
    def get_overlap_score(self, bbox: Tuple[int, int, int, int]) -> float:
        """Calculate overlap score for vehicle assignment"""
        if not self.intersects_bbox(bbox):
            return 0.0
        
        # Simple overlap calculation based on center distance
        spot_center = self._get_center()
        bbox_center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
        
        # Calculate distance
        dx = spot_center[0] - bbox_center[0]
        dy = spot_center[1] - bbox_center[1]
        distance = np.sqrt(dx*dx + dy*dy)
        
        # Convert distance to score (closer = higher score)
        max_distance = 200
        score = max(0, (max_distance - distance) / max_distance)
        return score
    
    def _get_center(self) -> Tuple[int, int]:
        """Get center point of spot"""
        points = np.array(self.polygon)
        return (int(np.mean(points[:, 0])), int(np.mean(points[:, 1])))

@dataclass
class SpotState:
    """Current state of parking spot"""
    spot_id: str
    is_occupied: bool = False
    confidence: float = 0.0
    stable_count: int = 0
    last_change: datetime = None
    vehicle_id: Optional[str] = None
    
    def __post_init__(self):
        if self.last_change is None:
            self.last_change = datetime.now()
    
    def is_stable(self, threshold: int = 3) -> bool:
        """Check if state is stable"""
        return self.stable_count >= threshold

@dataclass
class ParkingEvent:
    """Simplified parking event"""
    event_id: str
    spot_id: str
    event_type: str  # "VEHICLE_ENTERED" or "VEHICLE_EXITED"
    timestamp: datetime
    confidence: float = 0.0
    vehicle_id: Optional[str] = None

class ParkingMonitor:
    """Optimized parking monitor"""
    
    def __init__(self, spots_config: List[Dict]):
        # Initialize spots
        self.spots = [ParkingSpot(**config) for config in spots_config]
        self.spot_states = {spot.id: SpotState(spot.id) for spot in self.spots}
        
        # Core tracking
        self.frame_count = 0
        self.event_count = 0
        self.events = []  # Simple event list
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Configuration
        self.stability_threshold = 3
        self.max_events = 100  # Keep last 100 events
        
        logger.info(f"🏁 Parking Monitor initialized: {len(self.spots)} spots")
    
    def process_frame(self, vehicle_detections: List[Any], timestamp: datetime = None) -> List[ParkingEvent]:
        """
        Process frame with vehicle detections
        
        Args:
            vehicle_detections: List of vehicle detection objects with .bbox and .confidence
            timestamp: Optional timestamp
            
        Returns:
            List of parking events
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        self.frame_count += 1
        
        # Step 1: Assign vehicles to spots (1 vehicle = 1 spot)
        spot_assignments = self._assign_vehicles_to_spots(vehicle_detections)
        
        # Step 2: Update spot states and generate events
        events = []
        for spot in self.spots:
            assigned_vehicle = spot_assignments.get(spot.id)
            is_occupied = assigned_vehicle is not None
            
            event = self._update_spot_state(spot.id, is_occupied, assigned_vehicle, timestamp)
            if event:
                events.append(event)
        
        return events
    
    def _assign_vehicles_to_spots(self, vehicles: List[Any]) -> Dict[str, Any]:
        """Assign each vehicle to best matching spot"""
        # Calculate all possible assignments with scores
        assignments = []
        for vehicle in vehicles:
            for spot in self.spots:
                score = spot.get_overlap_score(vehicle.bbox)
                if score > 0:
                    assignments.append((vehicle, spot, score))
        
        # Sort by score (best first)
        assignments.sort(key=lambda x: x[2], reverse=True)
        
        # Greedy assignment: best score first, avoid conflicts
        spot_assignments = {}
        used_vehicles = set()
        
        for vehicle, spot, score in assignments:
            vehicle_id = id(vehicle)  # Simple vehicle ID
            
            # Skip if spot or vehicle already assigned
            if spot.id in spot_assignments or vehicle_id in used_vehicles:
                continue
            
            # Make assignment
            spot_assignments[spot.id] = vehicle
            used_vehicles.add(vehicle_id)
        
        return spot_assignments
    
    def _update_spot_state(self, spot_id: str, is_occupied: bool, 
                          vehicle: Any, timestamp: datetime) -> Optional[ParkingEvent]:
        """Update spot state and generate event if needed"""
        state = self.spot_states[spot_id]
        
        # Check for state change
        if state.is_occupied != is_occupied:
            state.stable_count = 1  # Reset stability counter
        else:
            state.stable_count += 1  # Increment for consistent state
        
        # Update state
        state.is_occupied = is_occupied
        state.confidence = vehicle.confidence if vehicle else 0.0
        state.vehicle_id = str(id(vehicle)) if vehicle else None
        
        # Generate event if state just became stable
        if state.is_stable(self.stability_threshold) and state.stable_count == self.stability_threshold:
            event = self._create_event(spot_id, is_occupied, state, timestamp)
            state.last_change = timestamp
            return event
        
        return None
    
    def _create_event(self, spot_id: str, is_occupied: bool, 
                     state: SpotState, timestamp: datetime) -> ParkingEvent:
        """Create parking event"""
        with self.lock:
            self.event_count += 1
            event_id = f"evt_{self.event_count:06d}"
        
        event_type = "VEHICLE_ENTERED" if is_occupied else "VEHICLE_EXITED"
        
        event = ParkingEvent(
            event_id=event_id,
            spot_id=spot_id,
            event_type=event_type,
            timestamp=timestamp,
            confidence=state.confidence,
            vehicle_id=state.vehicle_id
        )
        
        # Store event (keep only recent ones)
        self.events.append(event)
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]
        
        logger.info(f"🚗 {event_type} at spot {spot_id} (conf: {state.confidence:.2f})")
        return event
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current parking status"""
        with self.lock:
            occupied = sum(1 for state in self.spot_states.values() 
                          if state.is_occupied and state.is_stable())
            total = len(self.spots)
            
            return {
                'timestamp': datetime.now().isoformat(),
                'total_spots': total,
                'occupied_spots': occupied,
                'available_spots': total - occupied,
                'occupancy_rate': round((occupied / total * 100) if total > 0 else 0, 1),
                'frames_processed': self.frame_count,
                'events_generated': self.event_count
            }
    
    def get_spot_details(self) -> List[Dict[str, Any]]:
        """Get detailed spot information"""
        details = []
        for spot in self.spots:
            state = self.spot_states[spot.id]
            details.append({
                'id': spot.id,
                'name': spot.name,
                'zone': spot.zone,
                'is_occupied': state.is_occupied,
                'is_stable': state.is_stable(),
                'confidence': state.confidence,
                'stable_count': state.stable_count,
                'last_change': state.last_change.isoformat()
            })
        return details
    
    def get_recent_events(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent events"""
        with self.lock:
            recent = self.events[-count:] if count <= len(self.events) else self.events
            return [{
                'event_id': event.event_id,
                'spot_id': event.spot_id,
                'event_type': event.event_type,
                'timestamp': event.timestamp.isoformat(),
                'confidence': event.confidence,
                'vehicle_id': event.vehicle_id
            } for event in recent]
    
    def reset_statistics(self):
        """Reset monitoring statistics"""
        with self.lock:
            self.frame_count = 0
            self.event_count = 0
            self.events.clear()
        logger.info("🔄 Statistics reset")