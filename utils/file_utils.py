"""
File Utilities for Enhanced Parking System
Các tiện ích xử lý file và thư mục
"""

import os
import json
import yaml
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from core.models import ParkingSpot
from core.exceptions import FileNotFoundException, FileReadException, FileWriteException
from config.settings import config

logger = logging.getLogger(__name__)

def ensure_directories():
    """Ensure all required directories exist"""
    required_dirs = [
        "logs",
        "images",
        "vehicle_images", 
        "plate_images",
        "enhanced_images",
        "temp",
        "backup",
        "weights",
        "config"
    ]
    
    try:
        for dir_name in required_dirs:
            dir_path = Path(dir_name)
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created directory: {dir_path}")
        
        logger.info(f"✅ Ensured {len(required_dirs)} directories exist")
        
    except Exception as e:
        logger.error(f"❌ Failed to create directories: {e}")
        raise FileWriteException("directory_creation", str(e))

def load_parking_spots(config_path: str = None) -> List[Dict[str, Any]]:
    """
    Load parking spots configuration from file
    
    Args:
        config_path: Path to parking spots config file
        
    Returns:
        List of parking spot configurations
    """
    if config_path is None:
        # Try multiple possible locations
        possible_paths = [
            "config/parking_spots.json",
            "config/parking_spots.yaml", 
            "parking_spots.json",
            "spots_config.json"
        ]
        
        config_path = None
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break
    
    # If no config file found, create default configuration
    if not config_path or not os.path.exists(config_path):
        logger.warning("⚠️ No parking spots config found, creating default configuration")
        return create_default_parking_spots()
    
    try:
        logger.info(f"📖 Loading parking spots from: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        # Extract spots data (handle different formats)
        if isinstance(data, dict):
            spots_data = data.get('parking_spots', data.get('spots', data))
        else:
            spots_data = data
        
        # Validate spots data
        if not isinstance(spots_data, list):
            raise ValueError("Parking spots data must be a list")
        
        # Validate each spot
        validated_spots = []
        for i, spot_data in enumerate(spots_data):
            try:
                validated_spot = validate_parking_spot_config(spot_data)
                validated_spots.append(validated_spot)
            except Exception as e:
                logger.warning(f"⚠️ Invalid spot config at index {i}: {e}")
                continue
        
        logger.info(f"✅ Loaded {len(validated_spots)} valid parking spots")
        return validated_spots
        
    except Exception as e:
        logger.error(f"❌ Failed to load parking spots config: {e}")
        raise FileReadException(config_path or "parking_spots_config", str(e))

def validate_parking_spot_config(spot_data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize parking spot configuration"""
    
    # Required fields
    required_fields = ['id', 'polygon']
    for field in required_fields:
        if field not in spot_data:
            raise ValueError(f"Missing required field: {field}")
    
    # Normalize data
    validated_spot = {
        'id': str(spot_data['id']),
        'name': spot_data.get('name', f"Spot {spot_data['id']}"),
        'polygon': spot_data['polygon'],
        'zone': spot_data.get('zone', 'default'),
        'capacity': spot_data.get('capacity', 1),
        'priority': spot_data.get('priority', 0),
        'spot_type': spot_data.get('spot_type', 'standard'),
        'accessibility': spot_data.get('accessibility', False),
        'reserved': spot_data.get('reserved', False)
    }
    
    # Validate polygon
    polygon = validated_spot['polygon']
    if not isinstance(polygon, list) or len(polygon) < 3:
        raise ValueError("Polygon must be a list of at least 3 points")
    
    # Ensure each point has x,y coordinates
    for i, point in enumerate(polygon):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(f"Polygon point {i} must be [x, y] coordinates")
        
        # Convert to integers
        try:
            validated_spot['polygon'][i] = [int(point[0]), int(point[1])]
        except (ValueError, TypeError):
            raise ValueError(f"Polygon point {i} coordinates must be numeric")
    
    # Validate spot type
    valid_spot_types = ['standard', 'vip', 'accessibility', 'compact', 'truck', 'electric', 'reserved']
    if validated_spot['spot_type'] not in valid_spot_types:
        logger.warning(f"Invalid spot_type '{validated_spot['spot_type']}', using 'standard'")
        validated_spot['spot_type'] = 'standard'
    
    return validated_spot

def create_default_parking_spots() -> List[Dict[str, Any]]:
    """Create default parking spots configuration"""
    
    default_spots = [
        {
            'id': 'A1',
            'name': 'Vị trí A1',
            'polygon': [[100, 100], [300, 100], [300, 250], [100, 250]],
            'zone': 'Zone A',
            'capacity': 1,
            'priority': 1,
            'spot_type': 'standard'
        },
        {
            'id': 'A2', 
            'name': 'Vị trí A2',
            'polygon': [[320, 100], [520, 100], [520, 250], [320, 250]],
            'zone': 'Zone A',
            'capacity': 1,
            'priority': 1,
            'spot_type': 'standard'
        },
        {
            'id': 'A3',
            'name': 'Vị trí A3',
            'polygon': [[540, 100], [740, 100], [740, 250], [540, 250]],
            'zone': 'Zone A', 
            'capacity': 1,
            'priority': 1,
            'spot_type': 'standard'
        },
        {
            'id': 'B1',
            'name': 'Vị trí B1',
            'polygon': [[100, 270], [300, 270], [300, 420], [100, 420]],
            'zone': 'Zone B',
            'capacity': 1,
            'priority': 2,
            'spot_type': 'vip'
        },
        {
            'id': 'B2',
            'name': 'Vị trí B2', 
            'polygon': [[320, 270], [520, 270], [520, 420], [320, 420]],
            'zone': 'Zone B',
            'capacity': 1,
            'priority': 2,
            'spot_type': 'accessibility',
            'accessibility': True
        }
    ]
    
    # Save default configuration
    try:
        save_parking_spots_config(default_spots, "config/parking_spots.json")
        logger.info("💾 Created default parking spots configuration")
    except Exception as e:
        logger.warning(f"⚠️ Could not save default config: {e}")
    
    return default_spots

def save_parking_spots_config(spots: List[Dict[str, Any]], file_path: str):
    """Save parking spots configuration to file"""
    
    try:
        # Ensure directory exists
        dir_path = Path(file_path).parent
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Prepare data
        from datetime import datetime
        config_data = {
            'version': '1.0',
            'created_at': datetime.now().isoformat(),
            'total_spots': len(spots),
            'parking_spots': spots
        }
        
        # Save to file
        with open(file_path, 'w', encoding='utf-8') as f:
            if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
            else:
                json.dump(config_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 Saved parking spots config to: {file_path}")
        
    except Exception as e:
        logger.error(f"❌ Failed to save parking spots config: {e}")
        raise FileWriteException(file_path, str(e))

def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load JSON file with error handling"""
    
    if not os.path.exists(file_path):
        raise FileNotFoundException(file_path, "JSON")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        raise FileReadException(file_path, str(e))

def save_json_file(data: Dict[str, Any], file_path: str):
    """Save data to JSON file with error handling"""
    
    try:
        # Ensure directory exists
        dir_path = Path(file_path).parent
        dir_path.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.debug(f"💾 Saved JSON file: {file_path}")
        
    except Exception as e:
        raise FileWriteException(file_path, str(e))

def load_yaml_file(file_path: str) -> Dict[str, Any]:
    """Load YAML file with error handling"""
    
    if not os.path.exists(file_path):
        raise FileNotFoundException(file_path, "YAML")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise FileReadException(file_path, str(e))

def save_yaml_file(data: Dict[str, Any], file_path: str):
    """Save data to YAML file with error handling"""
    
    try:
        # Ensure directory exists
        dir_path = Path(file_path).parent
        dir_path.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        
        logger.debug(f"💾 Saved YAML file: {file_path}")
        
    except Exception as e:
        raise FileWriteException(file_path, str(e))

def get_file_size(file_path: str) -> int:
    """Get file size in bytes"""
    
    if not os.path.exists(file_path):
        return 0
    
    try:
        return os.path.getsize(file_path)
    except Exception as e:
        logger.warning(f"⚠️ Could not get size of {file_path}: {e}")
        return 0

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"

def cleanup_old_files(directory: str, max_age_hours: int = 24):
    """Clean up old files in directory"""
    
    if not os.path.exists(directory):
        return
    
    try:
        import time
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        removed_count = 0
        total_size = 0
        
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            
            if os.path.isfile(file_path):
                file_age = current_time - os.path.getctime(file_path)
                
                if file_age > max_age_seconds:
                    file_size = get_file_size(file_path)
                    try:
                        os.remove(file_path)
                        removed_count += 1
                        total_size += file_size
                        logger.debug(f"🗑️ Removed old file: {filename}")
                    except Exception as e:
                        logger.warning(f"⚠️ Could not remove {filename}: {e}")
        
        if removed_count > 0:
            logger.info(f"🗑️ Cleaned up {removed_count} old files, freed {format_file_size(total_size)}")
        
    except Exception as e:
        logger.warning(f"⚠️ Cleanup failed for {directory}: {e}")

def create_backup(source_file: str, backup_dir: str = "backup") -> bool:
    """Create backup of file"""
    
    if not os.path.exists(source_file):
        return False
    
    try:
        from datetime import datetime
        import shutil
        
        # Ensure backup directory exists
        Path(backup_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate backup filename
        source_name = Path(source_file).name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{source_name}_{timestamp}"
        backup_path = os.path.join(backup_dir, backup_name)
        
        # Copy file
        shutil.copy2(source_file, backup_path)
        
        logger.debug(f"💾 Created backup: {backup_path}")
        return True
        
    except Exception as e:
        logger.warning(f"⚠️ Backup failed for {source_file}: {e}")
        return False

def get_disk_usage(path: str = ".") -> Dict[str, int]:
    """Get disk usage information"""
    
    try:
        import shutil
        total, used, free = shutil.disk_usage(path)
        
        return {
            'total': total,
            'used': used,
            'free': free,
            'percent_used': (used / total) * 100 if total > 0 else 0
        }
        
    except Exception as e:
        logger.warning(f"⚠️ Could not get disk usage for {path}: {e}")
        return {'total': 0, 'used': 0, 'free': 0, 'percent_used': 0}

def list_files_by_extension(directory: str, extension: str) -> List[str]:
    """List all files with specific extension in directory"""
    
    if not os.path.exists(directory):
        return []
    
    try:
        files = []
        for filename in os.listdir(directory):
            if filename.lower().endswith(extension.lower()):
                files.append(os.path.join(directory, filename))
        
        return sorted(files)
        
    except Exception as e:
        logger.warning(f"⚠️ Could not list files in {directory}: {e}")
        return []

def find_latest_file(directory: str, pattern: str = "*") -> Optional[str]:
    """Find the most recently modified file matching pattern"""
    
    if not os.path.exists(directory):
        return None
    
    try:
        import glob
        
        search_pattern = os.path.join(directory, pattern)
        files = glob.glob(search_pattern)
        
        if not files:
            return None
        
        # Sort by modification time (newest first)
        latest_file = max(files, key=os.path.getmtime)
        return latest_file
        
    except Exception as e:
        logger.warning(f"⚠️ Could not find latest file in {directory}: {e}")
        return None

# Export commonly used functions
__all__ = [
    'ensure_directories',
    'load_parking_spots', 
    'save_parking_spots_config',
    'validate_parking_spot_config',
    'create_default_parking_spots',
    'load_json_file',
    'save_json_file',
    'load_yaml_file',
    'save_yaml_file',
    'get_file_size',
    'format_file_size',
    'cleanup_old_files',
    'create_backup',
    'get_disk_usage',
    'list_files_by_extension',
    'find_latest_file'
]

# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Test functions
    print("Testing file utilities...")
    
    # Ensure directories
    ensure_directories()
    
    # Load parking spots (will create default if not found)
    try:
        spots = load_parking_spots()
        print(f"✅ Loaded {len(spots)} parking spots")
        
        # Display first spot as example
        if spots:
            print(f"Example spot: {spots[0]['name']} - {len(spots[0]['polygon'])} points")
    
    except Exception as e:
        print(f"❌ Error loading spots: {e}")
    
    # Test disk usage
    disk_info = get_disk_usage()
    print(f"💾 Disk usage: {disk_info['percent_used']:.1f}% used "
          f"({format_file_size(disk_info['free'])} free)")
    
    # Test file operations
    test_data = {"test": "data", "timestamp": "2024-01-01"}
    
    try:
        save_json_file(test_data, "temp/test.json")
        loaded_data = load_json_file("temp/test.json")
        print(f"✅ JSON test passed: {loaded_data}")
    except Exception as e:
        print(f"❌ JSON test failed: {e}")
    
    print("File utilities test completed!")