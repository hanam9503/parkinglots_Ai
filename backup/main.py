"""
main.py - Main Application Entry Point
======================================
Entry point cho Multi-Camera Parking Management System
Handles initialization, configuration loading, và application startup
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Optional
import traceback

# Import system modules
from config import CameraConfig, SystemConfig, create_default_configs, load_camera_configs_from_json
from multi_camera_manager import MultiCameraManager
from gui import MultiCameraParkingGUI, MultiCameraCLI

# Setup logging
def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Setup logging configuration
    
    Args:
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file (str, optional): Log file path
    """
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Configure logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Setup handlers
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    else:
        # Default log file
        handlers.append(logging.FileHandler("logs/parking_system.log"))
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )
    
    # Set specific logger levels
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

def create_required_directories():
    """Tạo các thư mục cần thiết cho hệ thống"""
    directories = [
        'vehicle_images',      # Lưu ảnh xe
        'plate_images',        # Lưu ảnh biển số
        'enhanced_images',     # Ảnh đã enhance
        'config',              # File cấu hình
        'logs',                # Log files
        'reports',             # System reports
        'models',              # AI models (optional)
        'temp'                 # Temporary files
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        
    logging.info(f"✅ Created {len(directories)} required directories")

def create_sample_parking_spots():
    """Tạo file cấu hình mẫu cho parking spots"""
    
    # Sample parking spots configurations
    parking_configs = {
        "config/camera1_spots.json": {
            "camera_info": {
                "camera_id": "CAM_001",
                "name": "Tầng 1 - Khu A",
                "description": "Khu đỗ xe tầng 1, khu vực A"
            },
            "parking_spots": [
                {
                    "name": "A1",
                    "description": "Ô đỗ A1 - gần lối vào",
                    "polygon": [[100, 200], [200, 200], [200, 300], [100, 300]],
                    "type": "normal",
                    "reserved": False
                },
                {
                    "name": "A2", 
                    "description": "Ô đỗ A2",
                    "polygon": [[220, 200], [320, 200], [320, 300], [220, 300]],
                    "type": "normal",
                    "reserved": False
                },
                {
                    "name": "A3",
                    "description": "Ô đỗ A3",
                    "polygon": [[340, 200], [440, 200], [440, 300], [340, 300]],
                    "type": "normal",
                    "reserved": False
                },
                {
                    "name": "A4",
                    "description": "Ô đỗ A4 - gần thang máy",
                    "polygon": [[460, 200], [560, 200], [560, 300], [460, 300]],
                    "type": "vip",
                    "reserved": True
                },
                {
                    "name": "B1",
                    "description": "Ô đỗ B1",
                    "polygon": [[100, 320], [200, 320], [200, 420], [100, 420]],
                    "type": "normal",
                    "reserved": False
                },
                {
                    "name": "B2",
                    "description": "Ô đỗ B2",
                    "polygon": [[220, 320], [320, 320], [320, 420], [220, 420]],
                    "type": "normal",
                    "reserved": False
                }
            ]
        },
        
        "config/camera2_spots.json": {
            "camera_info": {
                "camera_id": "CAM_002",
                "name": "Tầng 1 - Khu B",
                "description": "Khu đỗ xe tầng 1, khu vực B"
            },
            "parking_spots": [
                {
                    "name": "C1",
                    "description": "Ô đỗ C1",
                    "polygon": [[150, 150], [250, 150], [250, 250], [150, 250]],
                    "type": "normal",
                    "reserved": False
                },
                {
                    "name": "C2",
                    "description": "Ô đỗ C2",
                    "polygon": [[270, 150], [370, 150], [370, 250], [270, 250]],
                    "type": "normal",
                    "reserved": False
                },
                {
                    "name": "C3",
                    "description": "Ô đỗ C3 - handicap",
                    "polygon": [[390, 150], [490, 150], [490, 250], [390, 250]],
                    "type": "handicap",
                    "reserved": True
                },
                {
                    "name": "D1",
                    "description": "Ô đỗ D1",
                    "polygon": [[150, 270], [250, 270], [250, 370], [150, 370]],
                    "type": "normal",
                    "reserved": False
                },
                {
                    "name": "D2",
                    "description": "Ô đỗ D2",
                    "polygon": [[270, 270], [370, 270], [370, 370], [270, 370]],
                    "type": "normal",
                    "reserved": False
                }
            ]
        },
        
        "config/camera3_spots.json": {
            "camera_info": {
                "camera_id": "CAM_003",
                "name": "Tầng 2 - Khu A",
                "description": "Khu đỗ xe tầng 2, khu vực A"
            },
            "parking_spots": [
                {
                    "name": "E1",
                    "description": "Ô đỗ E1",
                    "polygon": [[80, 180], [180, 180], [180, 280], [80, 280]],
                    "type": "normal",
                    "reserved": False
                },
                {
                    "name": "E2",
                    "description": "Ô đỗ E2 - electric car",
                    "polygon": [[200, 180], [300, 180], [300, 280], [200, 280]],
                    "type": "electric",
                    "reserved": True
                },
                {
                    "name": "E3",
                    "description": "Ô đỗ E3",
                    "polygon": [[320, 180], [420, 180], [420, 280], [320, 280]],
                    "type": "normal",
                    "reserved": False
                },
                {
                    "name": "F1",
                    "description": "Ô đỗ F1",
                    "polygon": [[80, 300], [180, 300], [180, 400], [80, 400]],
                    "type": "normal",
                    "reserved": False
                }
            ]
        },
        
        "config/entrance_spots.json": {
            "camera_info": {
                "camera_id": "CAM_004",
                "name": "Lối vào chính",
                "description": "Camera giám sát lối vào chính"
            },
            "parking_spots": [
                {
                    "name": "GATE_IN",
                    "description": "Khu vực check-in",
                    "polygon": [[200, 250], [400, 250], [400, 350], [200, 350]],
                    "type": "checkpoint",
                    "reserved": False
                },
                {
                    "name": "GATE_OUT", 
                    "description": "Khu vực check-out",
                    "polygon": [[420, 250], [620, 250], [620, 350], [420, 350]],
                    "type": "checkpoint",
                    "reserved": False
                }
            ]
        }
    }
    
    # Write configuration files
    created_count = 0
    for filepath, config_data in parking_configs.items():
        if not os.path.exists(filepath):
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            created_count += 1
            logging.info(f"📄 Created parking spots config: {filepath}")
    
    if created_count > 0:
        logging.info(f"✅ Created {created_count} parking spots configuration files")
    else:
        logging.info("📄 Parking spots configurations already exist")

def validate_model_paths(vehicle_model_path: str, plate_model_path: str) -> bool:
    """
    Validate AI model file paths
    
    Args:
        vehicle_model_path (str): Path to vehicle detection model
        plate_model_path (str): Path to plate detection model
        
    Returns:
        bool: True if both models exist
    """
    if not os.path.exists(vehicle_model_path):
        logging.error(f"❌ Vehicle model not found: {vehicle_model_path}")
        print(f"\n❌ Vehicle detection model not found!")
        print(f"Expected path: {vehicle_model_path}")
        print("\n💡 Please:")
        print("1. Download or train a vehicle detection YOLO model")
        print("2. Update the model path in main.py")
        print("3. Or use --vehicle-model argument")
        return False
    
    if not os.path.exists(plate_model_path):
        logging.error(f"❌ Plate model not found: {plate_model_path}")
        print(f"\n❌ License plate detection model not found!")
        print(f"Expected path: {plate_model_path}")
        print("\n💡 Please:")
        print("1. Download or train a license plate detection YOLO model")
        print("2. Update the model path in main.py")
        print("3. Or use --plate-model argument")
        return False
    
    logging.info(f"✅ Model validation passed")
    return True

def create_system_config(args) -> SystemConfig:
    """
    Tạo system configuration từ arguments
    
    Args:
        args: Command line arguments từ argparse
        
    Returns:
        SystemConfig: System configuration object
    """
    return SystemConfig(
        # Server settings
        sync_enabled=not args.no_sync,
        sync_server_url=args.server_url,
        connection_timeout=args.connection_timeout,
        request_timeout=args.request_timeout,
        
        # Performance settings
        max_concurrent_cameras=args.max_cameras,
        max_workers=args.max_workers,
        memory_limit_mb=args.memory_limit,
        
        # Processing settings
        save_images=not args.no_save_images,
        image_quality=args.image_quality,
        image_cleanup_hours=args.cleanup_hours,
        
        # Offline mode
        enable_offline_mode=not args.no_offline,
        offline_queue_size=args.offline_queue_size,
        sync_interval=args.sync_interval
    )

def create_camera_configs(args) -> List[CameraConfig]:
    """
    Tạo camera configurations
    
    Args:
        args: Command line arguments
        
    Returns:
        List[CameraConfig]: Danh sách camera configurations
    """
    if args.config_file and os.path.exists(args.config_file):
        # Load from JSON config file
        logging.info(f"📄 Loading camera configs from: {args.config_file}")
        return load_camera_configs_from_json(args.config_file)
    else:
        # Use default configurations
        logging.info("📄 Using default camera configurations")
        _, camera_configs = create_default_configs()
        
        # Update video paths if provided
        if args.video_sources:
            for i, video_path in enumerate(args.video_sources):
                if i < len(camera_configs):
                    camera_configs[i].stream_url = video_path
                    logging.info(f"🎥 Updated camera {camera_configs[i].camera_id} video source: {video_path}")
        
        return camera_configs

def print_system_info(manager: MultiCameraManager):
    """
    In thông tin hệ thống ra console
    
    Args:
        manager (MultiCameraManager): Manager instance
    """
    print("\n" + "="*60)
    print("🚀 MULTI-CAMERA PARKING MANAGEMENT SYSTEM")
    print("="*60)
    
    # System info
    config = manager.config
    print(f"🖥️  Max Concurrent Cameras: {config.max_concurrent_cameras}")
    print(f"🧵 Max Workers: {config.max_workers}")
    print(f"💾 Memory Limit: {config.memory_limit_mb} MB")
    print(f"🌐 Server Sync: {'Enabled' if config.sync_enabled else 'Disabled'}")
    if config.sync_enabled:
        print(f"📡 Server URL: {config.sync_server_url}")
    print(f"📱 Offline Mode: {'Enabled' if config.enable_offline_mode else 'Disabled'}")
    print(f"💾 Save Images: {'Enabled' if config.save_images else 'Disabled'}")
    
    # Camera info
    print(f"\n📹 Configured Cameras: {len(manager.camera_configs)}")
    for camera_id, camera_config in manager.camera_configs.items():
        status = "✅ Enabled" if camera_config.enabled else "❌ Disabled"
        priority = "⭐" * camera_config.priority
        print(f"  {camera_id}: {camera_config.name} {status} {priority}")
        print(f"    📹 Source: {camera_config.stream_url}")
        print(f"    📍 Location: {camera_config.location_name}")
        print(f"    🎯 Vehicle Conf: {camera_config.vehicle_conf}")
        print(f"    🚗 Plate Conf: {camera_config.plate_conf}")
        print(f"    ⚡ Max FPS: {camera_config.max_fps}")

def run_gui_mode(manager: MultiCameraManager):
    """
    Chạy ở GUI mode
    
    Args:
        manager (MultiCameraManager): Manager instance
    """
    try:
        print("\n🎮 Starting GUI interface...")
        print("📖 GUI Controls:")
        print("  • Left panel: Camera controls và system status")
        print("  • Right panel: Real-time video display")
        print("  • Bottom panel: Parking spots status")
        print("  • Use mouse to interact with controls")
        print("  • Close window to shutdown system")
        
        # Create and run GUI
        gui = MultiCameraParkingGUI(manager)
        gui.run()
        
    except Exception as e:
        logging.error(f"❌ GUI error: {e}")
        print(f"❌ GUI failed to start: {e}")
        traceback.print_exc()

def run_cli_mode(manager: MultiCameraManager):
    """
    Chạy ở CLI mode
    
    Args:
        manager (MultiCameraManager): Manager instance
    """
    try:
        print("\n🖥️ Starting CLI interface...")
        print("📖 CLI Commands Available:")
        print("  • help - Show all commands")
        print("  • list - List cameras")
        print("  • start/stop - Control cameras")
        print("  • status - System status")
        print("  • parking - Parking spots status")
        print("  • quit - Exit application")
        
        # Create and run CLI
        cli = MultiCameraCLI(manager)
        cli.run()
        
    except Exception as e:
        logging.error(f"❌ CLI error: {e}")
        print(f"❌ CLI failed to start: {e}")
        traceback.print_exc()

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Multi-Camera Parking Management System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Run with GUI (default)
  python main.py --cli                              # Run with CLI
  python main.py --no-gui --auto-start             # Headless mode with auto-start
  python main.py --config cameras.json             # Load camera config from file
  python main.py --vehicle-model model.pt          # Custom vehicle model
  python main.py --max-cameras 2                   # Limit concurrent cameras
  python main.py --no-sync                         # Disable server sync
        """
    )
    
    # Interface options
    interface_group = parser.add_mutually_exclusive_group()
    interface_group.add_argument('--cli', action='store_true', 
                                help='Use CLI interface instead of GUI')
    interface_group.add_argument('--no-gui', action='store_true',
                                help='Run without GUI (headless mode)')
    
    # Model paths
    parser.add_argument('--vehicle-model', type=str,
                       default=r"C:\Users\nam\runs\detect\train3\weights\best.pt",
                       help='Path to vehicle detection YOLO model')
    parser.add_argument('--plate-model', type=str,
                       default=r"G:\bkstar\parking AI\dataset\license-plate-finetune-v1m.pt",
                       help='Path to plate detection YOLO model')
    
    # Configuration
    parser.add_argument('--config', '--config-file', type=str,
                       help='Path to camera configuration JSON file')
    parser.add_argument('--video-sources', nargs='+', type=str,
                       help='Video sources for cameras (space separated)')
    
    # System settings
    parser.add_argument('--max-cameras', type=int, default=4,
                       help='Maximum concurrent cameras (default: 4)')
    parser.add_argument('--max-workers', type=int, default=8,
                       help='Maximum worker threads (default: 8)')
    parser.add_argument('--memory-limit', type=int, default=4096,
                       help='Memory limit in MB (default: 4096)')
    
    # Server settings
    parser.add_argument('--server-url', type=str, default="http://localhost:5000",
                       help='Server URL for synchronization')
    parser.add_argument('--no-sync', action='store_true',
                       help='Disable server synchronization')
    parser.add_argument('--connection-timeout', type=float, default=5.0,
                       help='Server connection timeout (seconds)')
    parser.add_argument('--request-timeout', type=float, default=10.0,
                       help='Server request timeout (seconds)')
    
    # Processing settings
    parser.add_argument('--no-save-images', action='store_true',
                       help='Disable saving vehicle/plate images')
    parser.add_argument('--image-quality', type=int, default=85,
                       help='JPEG image quality (1-100)')
    parser.add_argument('--cleanup-hours', type=int, default=24,
                       help='Hours after which to cleanup old images')
    
    # Offline settings
    parser.add_argument('--no-offline', action='store_true',
                       help='Disable offline mode')
    parser.add_argument('--offline-queue-size', type=int, default=500,
                       help='Offline queue size')
    parser.add_argument('--sync-interval', type=float, default=15.0,
                       help='Sync interval for offline data (seconds)')
    
    # Control options
    parser.add_argument('--auto-start', action='store_true',
                       help='Automatically start all cameras on startup')
    parser.add_argument('--start-cameras', nargs='+', type=str,
                       help='Specific cameras to start (space separated)')
    
    # Logging
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')
    parser.add_argument('--log-file', type=str,
                       help='Custom log file path')
    
    # Debug options
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode (sets log level to DEBUG)')
    parser.add_argument('--profile', action='store_true',
                       help='Enable performance profiling')
    
    return parser.parse_args()

def main():
    """Main application entry point"""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Set debug mode
        if args.debug:
            args.log_level = 'DEBUG'
        
        # Setup logging
        setup_logging(args.log_level, args.log_file)
        
        # Log startup
        logging.info("🚀 Starting Multi-Camera Parking Management System")
        logging.info(f"📋 Arguments: {vars(args)}")
        
        # Create required directories
        create_required_directories()
        
        # Create sample configurations if needed
        create_sample_parking_spots()
        
        # Validate model paths
        if not validate_model_paths(args.vehicle_model, args.plate_model):
            sys.exit(1)
        
        # Create system configuration
        system_config = create_system_config(args)
        
        # Initialize multi-camera manager
        logging.info("🔧 Initializing Multi-Camera Manager...")
        manager = MultiCameraManager(system_config)
        
        # Load AI models
        logging.info("📚 Loading AI models...")
        if not manager.load_models(args.vehicle_model, args.plate_model):
            logging.error("❌ Failed to load AI models")
            sys.exit(1)
        
        # Load camera configurations
        camera_configs = create_camera_configs(args)
        
        # Add cameras to manager
        for config in camera_configs:
            manager.add_camera(config)
        
        # Print system information
        print_system_info(manager)
        
        # Auto-start cameras if requested
        if args.auto_start:
            logging.info("🚀 Auto-starting all cameras...")
            results = manager.start_all_cameras()
            success_count = sum(results.values())
            print(f"\n✅ Auto-started {success_count}/{len(results)} cameras")
        
        elif args.start_cameras:
            logging.info(f"🚀 Starting specific cameras: {args.start_cameras}")
            for camera_id in args.start_cameras:
                if not camera_id.startswith('CAM_'):
                    camera_id = f"CAM_{camera_id}"
                success = manager.start_camera(camera_id)
                status = "✅" if success else "❌"
                print(f"{status} Camera {camera_id}")
        
        # Determine interface mode
        if args.no_gui:
            # Headless mode
            print("\n🔧 Running in headless mode...")
            print("📖 System is running without GUI. Check logs for status.")
            print("Press Ctrl+C to shutdown.")
            
            try:
                while True:
                    import time
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Shutdown initiated by user")
        
        elif args.cli:
            # CLI mode
            run_cli_mode(manager)
        
        else:
            # GUI mode (default)
            run_gui_mode(manager)
    
    except KeyboardInterrupt:
        print("\n🛑 Application interrupted by user")
        logging.info("🛑 Application interrupted by user")
    
    except Exception as e:
        print(f"\n❌ Application error: {e}")
        logging.error(f"❌ Application error: {e}")
        if args.debug if 'args' in locals() else False:
            traceback.print_exc()
    
    finally:
        # Cleanup
        if 'manager' in locals():
            try:
                print("\n🧹 Cleaning up resources...")
                logging.info("🧹 Starting application cleanup...")
                manager.cleanup()
                logging.info("✅ Application cleanup completed")
            except Exception as e:
                print(f"⚠️ Cleanup error: {e}")
                logging.error(f"⚠️ Cleanup error: {e}")
        
        print("👋 Application shutdown complete")
        logging.info("👋 Application shutdown complete")

# Performance profiling (optional)
def run_with_profiling():
    """Run application with performance profiling"""
    try:
        import cProfile
        import pstats
        from pstats import SortKey
        
        print("🔍 Running with performance profiling...")
        
        # Create profiler
        profiler = cProfile.Profile()
        profiler.enable()
        
        # Run main application
        main()
        
        # Stop profiler and save results
        profiler.disable()
        
        # Save profiling results
        profile_file = f"reports/profile_{int(time.time())}.prof"
        profiler.dump_stats(profile_file)
        
        # Print top functions
        stats = pstats.Stats(profiler)
        stats.sort_stats(SortKey.CUMULATIVE)
        
        print("\n📊 Top 20 functions by cumulative time:")
        stats.print_stats(20)
        
        print(f"\n📄 Full profiling report saved to: {profile_file}")
        
    except ImportError:
        print("⚠️ Profiling not available (cProfile not found)")
        main()
    except Exception as e:
        print(f"❌ Profiling error: {e}")
        main()

if __name__ == "__main__":
    # Check for profiling
    if len(sys.argv) > 1 and '--profile' in sys.argv:
        run_with_profiling()
    else:
        main()

# Helper functions for development and testing

def create_sample_config_file():
    """Tạo file cấu hình mẫu cho cameras"""
    sample_config = {
        "cameras": [
            {
                "camera_id": "CAM_001",
                "name": "Tầng 1 - Khu A",
                "stream_url": "videos/camera1.mp4",
                "location_name": "Tầng 1 - Khu A",
                "parking_spots_file": "config/camera1_spots.json",
                "vehicle_conf": 0.6,
                "plate_conf": 0.4,
                "intersection_threshold": 0.3,
                "frame_skip": 2,
                "priority": 1,
                "max_fps": 15.0,
                "enabled": True
            },
            {
                "camera_id": "CAM_002",
                "name": "Tầng 1 - Khu B",
                "stream_url": "videos/camera2.mp4",
                "location_name": "Tầng 1 - Khu B",
                "parking_spots_file": "config/camera2_spots.json",
                "vehicle_conf": 0.6,
                "plate_conf": 0.4,
                "intersection_threshold": 0.3,
                "frame_skip": 2,
                "priority": 2,
                "max_fps": 12.0,
                "enabled": True
            },
            {
                "camera_id": "CAM_003",
                "name": "Tầng 2 - Khu A",
                "stream_url": "videos/camera3.mp4",
                "location_name": "Tầng 2 - Khu A",
                "parking_spots_file": "config/camera3_spots.json",
                "vehicle_conf": 0.5,
                "plate_conf": 0.4,
                "intersection_threshold": 0.3,
                "frame_skip": 3,
                "priority": 2,
                "max_fps": 10.0,
                "enabled": True
            }
        ]
    }
    
    config_file = "config/cameras.json"
    if not os.path.exists(config_file):
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(sample_config, f, indent=2, ensure_ascii=False)
        print(f"📄 Created sample camera config: {config_file}")
    
    return config_file

def test_system():
    """Function để test hệ thống với dữ liệu mẫu"""
    print("🧪 Testing system with sample data...")
    
    # Create sample configs
    create_sample_config_file()
    
    # Set test arguments
    class TestArgs:
        def __init__(self):
            self.vehicle_model = "models/vehicle_model.pt"  # Update path as needed
            self.plate_model = "models/plate_model.pt"      # Update path as needed
            self.config_file = "config/cameras.json"
            self.max_cameras = 2
            self.cli = True
            self.debug = True
            self.log_level = "DEBUG"
            self.no_sync = True  # Disable server sync for testing
            
    # Override sys.argv for testing
    original_argv = sys.argv
    sys.argv = ['main.py', '--cli', '--debug', '--no-sync']
    
    try:
        main()
    finally:
        sys.argv = original_argv

# Development utilities
def check_dependencies():
    """Kiểm tra các dependencies cần thiết"""
    required_packages = [
        'opencv-python',
        'ultralytics', 
        'torch',
        'numpy',
        'pillow',
        'shapely',
        'requests',
        'psutil'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install " + " ".join(missing_packages))
        return False
    
    print("\n✅ All dependencies satisfied")
    return True

# Run dependency check if called directly with --check-deps
if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == '--check-deps':
    check_dependencies()
elif __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == '--test':
    test_system()
elif __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == '--create-config':
    create_sample_config_file()
    create_sample_parking_spots()