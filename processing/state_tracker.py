"""
Vehicle State Tracker - Optimized
Theo dõi trạng thái xe tối ưu
"""

import time
from typing import Dict, Optional, List
from collections import deque, defaultdict
from dataclasses import dataclass
import threading

@dataclass
class StateChange:
    spot_id: str
    new_state: bool
    previous_state: Optional[bool]
    confidence: float
    timestamp: float

class StateTracker:
    """Optimized vehicle state tracker"""
    
    def __init__(self, min_frames: int = 3, max_missed: int = 5):
        self.min_frames = min_frames
        self.max_missed = max_missed
        
        # Core state data
        self.detections: Dict[str, deque] = {}
        self.confirmed_states: Dict[str, bool] = {}
        self.state_confidences: Dict[str, float] = {}
        self.last_updates: Dict[str, float] = {}
        
        # Thread safety
        self.lock = threading.Lock()
    
    def update(self, spot_id: str, is_occupied: bool, confidence: float = 0.0) -> Optional[StateChange]:
        """Update detection and check for state change"""
        current_time = time.time()
        
        with self.lock:
            # Initialize detection history
            if spot_id not in self.detections:
                self.detections[spot_id] = deque(maxlen=self.min_frames + self.max_missed)
            
            # Add new detection
            self.detections[spot_id].append({
                'occupied': is_occupied,
                'confidence': confidence,
                'timestamp': current_time
            })
            
            self.last_updates[spot_id] = current_time
            
            # Check for state change
            return self._check_state_change(spot_id)
    
    def _check_state_change(self, spot_id: str) -> Optional[StateChange]:
        """Check if state should change based on recent detections"""
        detections = self.detections[spot_id]
        
        if len(detections) < self.min_frames:
            return None
        
        # Analyze recent detections
        recent = list(detections)[-self.min_frames:]
        occupied_count = sum(1 for d in recent if d['occupied'])
        
        # Determine dominant state
        new_state = occupied_count > (len(recent) / 2)
        consistency = max(occupied_count, len(recent) - occupied_count) / len(recent)
        
        # Must be consistent enough
        if consistency < 0.7:
            return None
        
        # Calculate average confidence
        avg_confidence = sum(d['confidence'] for d in recent) / len(recent)
        
        # Check for state change
        current_state = self.confirmed_states.get(spot_id)
        
        if current_state != new_state:
            # Update confirmed state
            self.confirmed_states[spot_id] = new_state
            self.state_confidences[spot_id] = avg_confidence
            
            return StateChange(
                spot_id=spot_id,
                new_state=new_state,
                previous_state=current_state,
                confidence=avg_confidence,
                timestamp=time.time()
            )
        
        # Update confidence for existing state
        self.state_confidences[spot_id] = avg_confidence
        return None
    
    def get_state(self, spot_id: str) -> bool:
        """Get confirmed state for spot"""
        with self.lock:
            return self.confirmed_states.get(spot_id, False)
    
    def get_all_states(self) -> Dict[str, bool]:
        """Get all confirmed states"""
        with self.lock:
            return self.confirmed_states.copy()
    
    def cleanup_old_states(self, timeout_seconds: int = 300):
        """Remove old states"""
        current_time = time.time()
        expired_spots = []
        
        with self.lock:
            for spot_id, last_update in self.last_updates.items():
                if current_time - last_update > timeout_seconds:
                    expired_spots.append(spot_id)
            
            for spot_id in expired_spots:
                self.detections.pop(spot_id, None)
                self.confirmed_states.pop(spot_id, None)
                self.state_confidences.pop(spot_id, None)
                self.last_updates.pop(spot_id, None)
    
    def get_occupancy_summary(self) -> Dict[str, int]:
        """Get occupancy summary"""
        with self.lock:
            total = len(self.confirmed_states)
            occupied = sum(self.confirmed_states.values())
            
            return {
                'total': total,
                'occupied': occupied,
                'available': total - occupied
            }

# Factory functions
# Make sure to import ImageProcessor if it's defined elsewhere, or define a stub here
class ImageProcessor:
    def __init__(self, config):
        self.config = config

def create_image_processor(config):
    return ImageProcessor(config)

class PlateValidator:
    def __init__(self):
        pass
    # Add validation methods as needed

def create_plate_validator():
    return PlateValidator()

def create_state_tracker(min_frames=3, max_missed=5):
    return StateTracker(min_frames, max_missed)