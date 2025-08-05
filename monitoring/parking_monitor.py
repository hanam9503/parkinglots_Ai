from typing import List, Dict, Any, Optional
from datetime import datetime
import threading
import uuid
import logging

from detection.vehicle_detector import VehicleDetector
from detection.plate_detector import PlateDetector
from processing.image_processor import ImageProcessor
from processing.state_tracker import VehicleStateTracker
from sync.server_sync import ServerSync
from core.models import ParkingEvent, ParkingSpotStatus

logger = logging.getLogger(__name__)

class ParkingMonitor:
    def __init__(self, parking_spots: List[Dict]):
        self.parking_spots = parking_spots
        self.parking_state = self._init_parking_state()
        
        # Initialize components
        self.vehicle_detector = VehicleDetector("path/to/vehicle/model.pt")
        self.plate_detector = PlateDetector("path/to/plate/model.pt")
        self.image_processor = ImageProcessor()
        self.state_tracker = VehicleStateTracker()
        self.server_sync = ServerSync()
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'events_generated': 0,
            'start_time': time.time()
        }
    
    def process_frame(self, frame) -> List[Dict[str, Any]]:
        """Process a single frame"""
        self.stats['frames_processed'] += 1
        
        # Detect vehicles
        vehicle_detections = self.vehicle_detector.detect(frame)
        
        # Process each parking spot
        processed_events = []
        for spot in self.parking_spots:
            event = self._process_spot(spot, vehicle_detections, frame)
            if event:
                processed_events.append(event)
        
        return processed_events
    
    def _process_spot(self, spot: Dict, detections: List, frame) -> Optional[Dict]:
        """Process individual parking spot"""
        spot_id = spot['id']
        
        # Find intersecting vehicles
        intersecting_vehicle = self._find_intersecting_vehicle(spot, detections)
        
        # Update state tracker
        is_occupied = intersecting_vehicle is not None
        state_change = self.state_tracker.update_detection(spot_id, is_occupied)
        
        if state_change:
            return self._handle_state_change(state_change, spot, frame, intersecting_vehicle)
        
        return None
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        occupied_spots = sum(1 for state in self.parking_state.values() 
                           if self.state_tracker.get_confirmed_state(state['spot_id']))
        
        return {
            'camera_id': config.system.CAMERA_ID,
            'timestamp': datetime.now().isoformat(),
            'parking_summary': {
                'total_spots': len(self.parking_spots),
                'occupied_spots': occupied_spots,
                'occupancy_rate': (occupied_spots / len(self.parking_spots)) * 100
            },
            'processing_stats': self.stats,
            'server_connection': self.server_sync.get_connection_status()
        }