
"""
Enhanced Parking System - Main Entry Point
Hệ thống Parking Real-time Tối ưu với Node.js Integration
"""

import os
import sys
import cv2
import time
import threading
import gc
import torch
import numpy as np
from datetime import datetime
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.settings import config
from config.logging_config import setup_logging
from detection.model_loader import load_models
from core.models import ParkingSpot
from monitoring.parking_monitor import EnhancedParkingMonitor
from utils.file_utils import load_parking_spots, ensure_directories

# Setup logging
logger = setup_logging()

def display_system_info():
    """Display system information and configuration"""
    print("\n" + "="*60)
    print("🚀 ENHANCED PARKING SYSTEM - NODE.JS INTEGRATION")
    print("="*60)
    print(f"📍 Location: {getattr(config, 'LOCATION_NAME', 'Default Location')}")
    print(f"📹 Camera ID: {getattr(config, 'CAMERA_ID', 'CAM_001')}")
    print(f"🌐 Server URL: {getattr(config, 'SYNC_SERVER_URL', 'http://localhost:5000')}")
    print(f"📡 Offline Mode: {'Enabled' if getattr(config, 'ENABLE_OFFLINE_MODE', True) else 'Disabled'}")
    print(f"🔧 Enhancement: {'Enabled' if getattr(config, 'USE_REAL_ESRGAN', False) else 'Disabled'}")
    print(f"🎯 Detection Thresholds: Vehicle={config.VEHICLE_CONF}, Plate={config.PLATE_CONF}")
    print(f"💾 Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"🔥 CUDA Available: {torch.cuda.is_available()}")
    print("="*60)

def main():
    """Enhanced main execution with comprehensive error handling"""
    display_system_info()
    
    # Initialize components
    monitor = None
    cap = None
    
    try:
        # Ensure required directories exist
        ensure_directories()
        
        # Load models
        logger.info("🔄 Loading YOLO models...")
        vehicle_model, plate_model = load_models()
        logger.info("✅ Models loaded successfully")
        
        # Load parking spots
        logger.info("🔄 Loading parking spots configuration...")
        parking_spots_config = load_parking_spots()
        logger.info(f"✅ Loaded {len(parking_spots_config)} parking spots")
        
        # Initialize monitor
        logger.info("🔄 Initializing Enhanced Parking Monitor...")
        monitor = EnhancedParkingMonitor(parking_spots_config)
        monitor.initialize_models(vehicle_model, plate_model)
        monitor.start_monitoring()
        logger.info("✅ Monitor initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        return 1
    
    # Video processing
    video_path = "videos/0723.mp4"
    
    if not os.path.exists(video_path):
        logger.error(f"❌ Video file not found: {video_path}")
        logger.info("💡 Please ensure video file exists or use camera input")
        return 1
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        logger.error("❌ Cannot open video file")
        return 1
    
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
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logger.info("📹 End of video reached")
                break
            
            frame_id += 1
            
            # Skip frames for performance
            if frame_id % getattr(config, 'FRAME_SKIP', 2) != 0:
                continue
            
            display_counter += 1
            current_time = time.time()
            
            # Process frame
            events = monitor.process_frame(frame, datetime.now())
            
            # Handle events
            for event in events:
                action = "entered" if event.event_type == "VEHICLE_ENTERED" else "exited"
                plate_info = f" - {event.license_plate}" if event.license_plate else ""
                conf_info = f" (conf: {event.confidence:.3f})" if event.confidence > 0 else ""
                
                logger.info(f"🚗 {event.additional_data.get('spot_name', event.spot_id)}: "
                           f"Vehicle {action}{plate_info}{conf_info}")
            
            # Performance display
            if display_counter % 30 == 0:
                elapsed = current_time - start_time
                processing_fps = display_counter / elapsed if elapsed > 0 else 0
                
                system_status = monitor.get_current_status()
                parking_summary = system_status['parking_summary']
                
                logger.info(f"⚡ FPS: {processing_fps:.1f} | "
                           f"Occupancy: {parking_summary['occupancy_rate']}% | "
                           f"Occupied: {parking_summary['occupied_spots']}/{parking_summary['total_spots']} | "
                           f"Events: {len(list(monitor.events_buffer))}")
            
            # Periodic detailed summary
            if current_time - last_summary_time > SUMMARY_INTERVAL:
                system_status = monitor.get_current_status()
                processing_stats = system_status['processing_stats']
                
                logger.info(f"📊 System Summary:")
                logger.info(f"   - Occupancy: {system_status['parking_summary']['occupancy_rate']:.1f}% "
                           f"({system_status['parking_summary']['occupied_spots']}/{system_status['parking_summary']['total_spots']})")
                logger.info(f"   - Processing: Frames={processing_stats['frames_processed']}, "
                           f"Events={processing_stats['events_generated']}")
                logger.info(f"   - Performance: {processing_stats.get('processing_fps', 0):.1f} FPS, "
                           f"{processing_stats.get('avg_processing_time', 0)*1000:.1f}ms avg")
                
                last_summary_time = current_time
            
            # Memory management
            if display_counter % 200 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Create visualization frame
            display_frame = create_display_frame(frame, monitor)
            
            # Display frame
            cv2.namedWindow("Enhanced Parking Monitor", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Enhanced Parking Monitor", 1280, 720)
            cv2.imshow("Enhanced Parking Monitor", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("🛑 Stopped by user (Q key)")
                break
            elif key == ord('s'):
                # Save current status
                status = monitor.get_current_status()
                import json
                filename = f"status_{int(time.time())}.json"
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(status, f, indent=2, ensure_ascii=False, default=str)
                logger.info(f"💾 Status saved to {filename}")
            elif key == ord('r'):
                # Reset statistics
                monitor.reset_statistics()
                logger.info("🔄 Statistics reset")
    
    except KeyboardInterrupt:
        logger.info("🛑 Stopped by Ctrl+C")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Cleanup
        logger.info("🧹 Starting cleanup...")
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        if monitor:
            monitor.cleanup()
        
        logger.info("🔚 Enhanced Parking System completed")
        return 0

def create_display_frame(frame: np.ndarray, monitor: EnhancedParkingMonitor) -> np.ndarray:
    """Create display frame with parking spot overlays"""
    
    display_frame = frame.copy()
    
    try:
        # Draw parking spots
        for spot in monitor.parking_spots:
            spot_state = monitor.spot_states[spot.id]
            
            # Choose color based on occupancy
            if spot_state.is_occupied and spot_state.is_stable():
                color = (0, 0, 255)  # Red for occupied
                thickness = 3
            elif spot_state.is_occupied:
                color = (0, 165, 255)  # Orange for detecting
                thickness = 2
            else:
                color = (0, 255, 0)  # Green for empty
                thickness = 2
            
            # Draw polygon
            points = np.array(spot.polygon, np.int32)
            cv2.polylines(display_frame, [points], True, color, thickness)
            
            # Draw spot label
            center = spot.get_center()
            label = f"{spot.name}"
            if spot_state.is_occupied and spot_state.license_plate:
                label += f"\n{spot_state.license_plate}"
            
            # Draw text background
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(display_frame, 
                         (center[0] - text_width//2 - 5, center[1] - text_height - 5),
                         (center[0] + text_width//2 + 5, center[1] + 5),
                         (0, 0, 0), -1)
            
            # Draw text
            cv2.putText(display_frame, label, 
                       (center[0] - text_width//2, center[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw system info
        draw_system_info(display_frame, monitor)
        
    except Exception as e:
        logger.warning(f"⚠️ Display frame creation failed: {e}")
    
    return display_frame

def draw_system_info(frame: np.ndarray, monitor: EnhancedParkingMonitor):
    """Draw system information on frame"""
    
    try:
        status = monitor.get_current_status()
        
        # System info text
        info_lines = [
            f"System: Enhanced Parking Monitor",
            f"Occupancy: {status['parking_summary']['occupied_spots']}/{status['parking_summary']['total_spots']} ({status['parking_summary']['occupancy_rate']:.1f}%)",
            f"Events: {status['processing_stats']['events_generated']}",
            f"FPS: {status['processing_stats'].get('processing_fps', 0):.1f}",
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        ]
        
        # Draw info box background
        box_height = len(info_lines) * 25 + 20
        cv2.rectangle(frame, (10, 10), (400, box_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, box_height), (255, 255, 255), 2)
        
        # Draw text
        for i, line in enumerate(info_lines):
            y = 35 + i * 25
            cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw legend
        legend_y = box_height + 30
        cv2.putText(frame, "Legend:", (20, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        legend_items = [
            ("Green: Empty", (0, 255, 0)),
            ("Red: Occupied", (0, 0, 255)), 
            ("Orange: Detecting", (0, 165, 255))
        ]
        
        for i, (text, color) in enumerate(legend_items):
            y = legend_y + 25 + i * 20
            cv2.rectangle(frame, (20, y-10), (35, y+5), color, -1)
            cv2.putText(frame, text, (45, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    except Exception as e:
        logger.warning(f"⚠️ System info drawing failed: {e}")

if __name__ == "__main__":
    sys.exit(main())