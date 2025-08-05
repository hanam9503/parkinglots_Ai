"""
Core Data Models
Enhanced data models cho hệ thống parking
"""

from dataclasses import dataclass, asdict, field
from typing import Optional, List, Tuple, Dict, Any
from datetime import datetime
import uuid
from shapely.geometry import Polygon
import json

@dataclass
class ParkingEvent:
    """Enhanced parking event model with comprehensive validation"""
    
    id: str
    camera_id: str
    spot_id: str
    spot_name: str
    event_type: str  # 'enter' or 'exit'
    timestamp: str
    plate_text: Optional[str] = None
    plate_confidence: float = 0.0
    vehicle_image_path: Optional[str] = None
    plate_image_path: Optional[str] = None
    enhanced_vehicle_path: Optional[str] = None
    enhanced_plate_path: Optional[str] = None
    location_name: str = ""
    vehicle_type: Optional[str] = None
    detection_confidence: float = 0.0
    processing_time: Optional[float] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization validation"""
        # Validate event_type
        if self.event_type not in ['enter', 'exit']:
            raise ValueError(f"Invalid event_type: {self.event_type}. Must be 'enter' or 'exit'")
        
        # Validate and normalize confidence values
        self.plate_confidence = max(0.0, min(1.0, self.plate_confidence))
        self.detection_confidence = max(0.0, min(1.0, self.detection_confidence))
        
        # Validate required fields
        if not self.id:
            self.id = str(uuid.uuid4())
        
        if not self.camera_id:
            raise ValueError("camera_id is required")
        
        if not self.spot_id:
            raise ValueError("spot_id is required")
        
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        
        # Validate timestamp format
        try:
            datetime.fromisoformat(self.timestamp.replace('Z', '+00:00'))
        except ValueError:
            raise ValueError(f"Invalid timestamp format: {self.timestamp}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper serialization"""
        data = asdict(self)
        
        # Remove None values and empty paths
        cleaned_data = {}
        for key, value in data.items():
            if value is not None and value != "":
                cleaned_data[key] = value
        
        return cleaned_data
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ParkingEvent':
        """Create instance from dictionary"""
        return cls(**data)
    
    def is_valid(self) -> bool:
        """Check if event data is valid"""
        try:
            # Check required fields
            if not all([self.id, self.camera_id, self.spot_id, self.event_type]):
                return False
            
            # Check timestamp
            datetime.fromisoformat(self.timestamp.replace('Z', '+00:00'))
            
            # Check confidence ranges
            if not (0 <= self.plate_confidence <= 1):
                return False
            
            if not (0 <= self.detection_confidence <= 1):
                return False
            
            return True
        except Exception:
            return False
    
    def get_duration_from_enter_event(self, enter_event: 'ParkingEvent') -> Optional[float]:
        """Calculate duration in minutes from enter event"""
        if self.event_type != 'exit' or enter_event.event_type != 'enter':
            return None
        
        try:
            exit_time = datetime.fromisoformat(self.timestamp.replace('Z', '+00:00'))
            enter_time = datetime.fromisoformat(enter_event.timestamp.replace('Z', '+00:00'))
            
            duration = (exit_time - enter_time).total_seconds() / 60
            return max(0, duration)  # Ensure non-negative duration
        except Exception:
            return None

@dataclass
class ParkingSpotStatus:
    """Enhanced parking spot status model"""
    
    spot_id: str
    spot_name: str
    camera_id: str
    is_occupied: bool
    enter_time: Optional[str] = None
    plate_text: Optional[str] = None
    plate_confidence: float = 0.0
    last_update: str = ""
    vehicle_type: Optional[str] = None
    spot_type: str = "standard"  # standard, vip, accessibility, compact, truck
    priority: int = 1
    reserved: bool = False
    accessibility: bool = False
    additional_info: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization validation"""
        # Validate confidence
        self.plate_confidence = max(0.0, min(1.0, self.plate_confidence))
        
        # Validate required fields
        if not self.spot_id:
            raise ValueError("spot_id is required")
        
        if not self.camera_id:
            raise ValueError("camera_id is required")
        
        if not self.last_update:
            self.last_update = datetime.now().isoformat()
        
        # Validate spot_type
        valid_types = ['standard', 'vip', 'accessibility', 'compact', 'truck']
        if self.spot_type not in valid_types:
            raise ValueError(f"Invalid spot_type: {self.spot_type}. Must be one of {valid_types}")
        
        # Set accessibility flag based on spot_type
        if self.spot_type == 'accessibility':
            self.accessibility = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        
        # Remove None values
        cleaned_data = {}
        for key, value in data.items():
            if value is not None and value != "":
                cleaned_data[key] = value
        
        return cleaned_data
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    def get_occupancy_duration(self) -> Optional[float]:
        """Get occupancy duration in minutes"""
        if not self.is_occupied or not self.enter_time:
            return None
        
        try:
            enter_time = datetime.fromisoformat(self.enter_time.replace('Z', '+00:00'))
            current_time = datetime.now()
            
            # Handle timezone-aware datetime
            if enter_time.tzinfo is None:
                enter_time = enter_time.replace(tzinfo=current_time.tzinfo)
            elif current_time.tzinfo is None:
                current_time = current_time.replace(tzinfo=enter_time.tzinfo)
            
            duration = (current_time - enter_time).total_seconds() / 60
            return max(0, duration)
        except Exception:
            return None
    
    def is_available(self) -> bool:
        """Check if spot is available for parking"""
        return not self.is_occupied and not self.reserved
    
    def can_accommodate_vehicle(self, vehicle_type: str = "car") -> bool:
        """Check if spot can accommodate specific vehicle type"""
        vehicle_spot_compatibility = {
            'car': ['standard', 'vip', 'accessibility', 'compact'],
            'truck': ['truck', 'standard'],
            'bus': ['truck'],
            'motorcycle': ['compact', 'standard', 'vip', 'accessibility']
        }
        
        return self.spot_type in vehicle_spot_compatibility.get(vehicle_type.lower(), ['standard'])

@dataclass
class ParkingSpot:
    """Enhanced parking spot configuration model"""
    
    id: str
    name: str
    polygon: List[List[int]]  # List of [x, y] coordinates
    spot_type: str = "standard"
    priority: int = 1
    accessibility: bool = False
    reserved: bool = False
    description: str = ""
    
    # Computed properties (set in __post_init__)
    area: float = field(init=False)
    polygon_shapely: Polygon = field(init=False)
    bbox: Tuple[float, float, float, float] = field(init=False)  # (minx, miny, maxx, maxy)
    
    def __post_init__(self):
        """Post-initialization processing"""
        # Validate polygon
        if len(self.polygon) < 3:
            raise ValueError(f"Polygon must have at least 3 points, got {len(self.polygon)}")
        
        # Create Shapely polygon
        try:
            self.polygon_shapely = Polygon(self.polygon)
            
            # Fix invalid polygons
            if not self.polygon_shapely.is_valid:
                self.polygon_shapely = self.polygon_shapely.buffer(0)
                
            self.area = self.polygon_shapely.area
            self.bbox = self.polygon_shapely.bounds
            
        except Exception as e:
            raise ValueError(f"Invalid polygon for spot {self.id}: {e}")
        
        # Validate spot_type
        valid_types = ['standard', 'vip', 'accessibility', 'compact', 'truck']
        if self.spot_type not in valid_types:
            raise ValueError(f"Invalid spot_type: {self.spot_type}")
        
        # Set accessibility flag
        if self.spot_type == 'accessibility':
            self.accessibility = True
        
        # Generate description if empty
        if not self.description:
            self.description = f"Vị trí đỗ xe {self.spot_type} {self.name}"
    
    def contains_point(self, x: float, y: float) -> bool:
        """Check if point is inside the parking spot"""
        from shapely.geometry import Point
        point = Point(x, y)
        return self.polygon_shapely.contains(point)
    
    def intersects_with(self, other_polygon: Polygon, threshold: float = 0.1) -> bool:
        """Check if spot intersects with another polygon"""
        intersection = self.polygon_shapely.intersection(other_polygon)
        intersection_ratio = intersection.area / self.area
        return intersection_ratio > threshold
    
    def get_center(self) -> Tuple[float, float]:
        """Get center point of the parking spot"""
        centroid = self.polygon_shapely.centroid
        return (centroid.x, centroid.y)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding computed properties)"""
        return {
            'id': self.id,
            'name': self.name,
            'polygon': self.polygon,
            'spot_type': self.spot_type,
            'priority': self.priority,
            'accessibility': self.accessibility,
            'reserved': self.reserved,
            'description': self.description,
            'area': self.area,
            'bbox': list(self.bbox)
        }
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

@dataclass
class DetectionResult:
    """Vehicle detection result model"""
    
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    class_name: str
    car_id: Optional[str] = None
    vehicle_type: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    additional_data: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate detection result"""
        self.confidence = max(0.0, min(1.0, self.confidence))
        
        if len(self.bbox) != 4:
            raise ValueError("bbox must contain exactly 4 values (x1, y1, x2, y2)")
        
        x1, y1, x2, y2 = self.bbox
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Invalid bbox coordinates")
    
    def get_area(self) -> int:
        """Get bounding box area"""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)
    
    def get_center(self) -> Tuple[int, int]:
        """Get center point of bounding box"""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

@dataclass
class SystemStatus:
    """System status model"""
    
    camera_id: str
    location_name: str
    timestamp: str
    uptime_seconds: int
    parking_summary: Dict[str, Any]
    processing_stats: Dict[str, Any]
    server_connection: Dict[str, Any]
    image_processor: Dict[str, Any]
    spots_detail: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    def get_occupancy_rate(self) -> float:
        """Get current occupancy rate"""
        summary = self.parking_summary
        total = summary.get('total_spots', 0)
        occupied = summary.get('occupied_spots', 0)
        
        if total == 0:
            return 0.0
        
        return (occupied / total) * 100

# Validation functions
def validate_event_data(data: Dict[str, Any]) -> bool:
    """Validate event data dictionary"""
    required_fields = ['camera_id', 'spot_id', 'event_type']
    
    for field in required_fields:
        if field not in data or not data[field]:
            return False
    
    if data['event_type'] not in ['enter', 'exit']:
        return False
    
    return True

def validate_status_data(data: Dict[str, Any]) -> bool:
    """Validate status data dictionary"""
    required_fields = ['spot_id', 'camera_id']
    
    for field in required_fields:
        if field not in data or not data[field]:
            return False
    
    if 'is_occupied' not in data:
        return False
    
    return True

# Factory functions
def create_parking_event(
    camera_id: str,
    spot_id: str,
    spot_name: str,
    event_type: str,
    **kwargs
) -> ParkingEvent:
    """Factory function to create parking event"""
    return ParkingEvent(
        id=str(uuid.uuid4()),
        camera_id=camera_id,
        spot_id=spot_id,
        spot_name=spot_name,
        event_type=event_type,
        timestamp=datetime.now().isoformat(),
        **kwargs
    )

def create_parking_status(
    spot_id: str,
    spot_name: str,
    camera_id: str,
    is_occupied: bool,
    **kwargs
) -> ParkingSpotStatus:
    """Factory function to create parking status"""
    return ParkingSpotStatus(
        spot_id=spot_id,
        spot_name=spot_name,
        camera_id=camera_id,
        is_occupied=is_occupied,
        last_update=datetime.now().isoformat(),
        **kwargs
    )

# Example usage and testing
if __name__ == "__main__":
    # Test ParkingEvent
    event = create_parking_event(
        camera_id="CAM_001",
        spot_id="SPOT_001",
        spot_name="Vị trí A1",
        event_type="enter",
        plate_text="30A-12345",
        plate_confidence=0.95
    )
    
    print("Parking Event:")
    print(event.to_json())
    print(f"Valid: {event.is_valid()}")
    
    # Test ParkingSpotStatus
    status = create_parking_status(
        spot_id="SPOT_001",
        spot_name="Vị trí A1",
        camera_id="CAM_001",
        is_occupied=True,
        plate_text="30A-12345",
        plate_confidence=0.95
    )
    
    print("\nParking Status:")
    print(status.to_json())
    
    # Test ParkingSpot
    spot = ParkingSpot(
        id="SPOT_001",
        name="Vị trí A1",
        polygon=[[100, 200], [300, 200], [300, 400], [100, 400]],
        spot_type="standard",
        priority=1
    )
    
    print("\nParking Spot:")
    print(spot.to_json())
    print(f"Area: {spot.area}")
    print(f"Center: {spot.get_center()}")
    print(f"Contains point (200, 300): {spot.contains_point(200, 300)}")