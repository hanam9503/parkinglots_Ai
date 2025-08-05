#!/usr/bin/env python3
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
    print(f"📍 Location: {config.LOCATION_NAME}")
    print(f"📹 Camera ID: {config.CAMERA_ID}")
    print(f"🌐 Server URL: {config.SYNC_SERVER_URL}")
    print(f"📡 Offline Mode: {'Enabled' if config.ENABLE_OFFLINE_MODE else 'Disabled'}")
    print(f"🔧 Enhancement: {'Enabled' if config.USE_REAL_ESRGAN else 'Disabled'}")
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
        parking_spots = load_parking_spots()
        logger.info(f"✅ Loaded {len(parking_spots)} parking spots")
        
        # Initialize monitor
        logger.info("🔄 Initializing Enhanced Parking Monitor...")
        monitor = EnhancedParkingMonitor(vehicle_model, plate_model, parking_spots)
        logger.info("✅ Monitor initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        return 1
    
    # Video processing
    video_path = "videos/0723.mp4"
    
    if not os.path.exists(video_path):
        logger.error(f"❌ Video file not found: {video_path}")
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
            if frame_id % config.FRAME_SKIP != 0:
                continue
            
            display_counter += 1
            current_time = time.time()
            
            # Process frame
            processed_frame, processed_events = monitor.process_frame(frame, frame_id)
            
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
            
            # Performance display
            if display_counter % 30 == 0:
                elapsed = current_time - start_time
                processing_fps = display_counter / elapsed
                
                system_status = monitor.get_system_status()
                connection_status = system_status['server_connection']
                
                conn_status = "Connected" if connection_status['is_connected'] else "Offline"
                logger.info(f"⚡ FPS: {processing_fps:.1f} | "
                           f"Occupancy: {system_status['parking_summary']['occupancy_rate']}% | "
                           f"Server: {conn_status} | "
                           f"Events: {system_status['processing_stats']['events_sent']}/{system_status['processing_stats']['events_generated']} | "
                           f"Plates: {system_status['processing_stats']['detection_rate']}%")
            
            # Periodic detailed summary
            if current_time - last_summary_time > SUMMARY_INTERVAL:
                system_status = monitor.get_system_status()
                processor_stats = system_status['image_processor']
                
                logger.info(f"📊 System Summary:")
                logger.info(f"   - Occupancy: {system_status['parking_summary']['occupancy_rate']}% "
                           f"({system_status['parking_summary']['occupied_spots']}/{system_status['parking_summary']['total_spots']})")
                logger.info(f"   - Processing: {system_status['processing_stats']['events_generated']} events, "
                           f"{system_status['processing_stats']['plates_detected']} plates detected")
                logger.info(f"   - Server: {'Connected' if system_status['server_connection']['is_connected'] else 'Offline'}, "
                           f"Queue: {system_status['server_connection']['offline_queue_size']}")
                logger.info(f"   - Enhancement: {processor_stats.get('enhancement_count', 0)} processed, "
                           f"Cache: {processor_stats.get('cache_hit_rate', 0):.1f}% hit rate")
                
                last_summary_time = current_time
            
            # Memory management
            if display_counter % 200 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Display frame
            cv2.namedWindow("Enhanced Parking Monitor", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Enhanced Parking Monitor", 1280, 720)
            cv2.imshow("Enhanced Parking Monitor", processed_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("🛑 Stopped by user (Q key)")
                break
            elif key == ord('s'):
                # Save current status
                status = monitor.get_system_status()
                import json
                with open(f"status_{int(time.time())}.json", 'w') as f:
                    json.dump(status, f, indent=2)
                logger.info("💾 Status saved")
    
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

if __name__ == "__main__":
    sys.exit(main())