"""
Model Loader - Enhanced YOLO Model Loading and Optimization
Tải và tối ưu hóa các YOLO models cho detection
"""

import os
import logging
import torch
import requests
from pathlib import Path
from typing import Tuple, Optional
from ultralytics import YOLO

from config.settings import config
from core.exceptions import ModelLoadException, ModelInferenceException
from core.constants import MODEL_URLS

logger = logging.getLogger(__name__)

class ModelManager:
    """Enhanced model manager with optimization and caching"""
    
    def __init__(self):
        self.device = self._get_optimal_device()
        self.vehicle_model = None
        self.plate_model = None
        self.models_loaded = False
        
        # Model optimization flags
        self.half_precision = torch.cuda.is_available()
        self.compile_models = hasattr(torch, 'compile') and torch.cuda.is_available()
        
        logger.info(f"🔧 ModelManager initialized - Device: {self.device}")
        logger.info(f"   Half precision: {self.half_precision}")
        logger.info(f"   Model compilation: {self.compile_models}")
    
    def _get_optimal_device(self) -> str:
        """Determine optimal device for inference"""
        if torch.cuda.is_available():
            # Check GPU memory
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"🔥 CUDA available - GPU: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB)")
            
            # Ensure sufficient GPU memory
            if gpu_memory >= 4.0:  # Minimum 4GB for reliable inference
                return 'cuda'
            else:
                logger.warning(f"⚠️ GPU memory insufficient ({gpu_memory:.1f}GB), using CPU")
                return 'cpu'
        else:
            logger.info("💻 Using CPU for inference")
            return 'cpu'
    
    def download_model(self, url: str, model_path: str, model_name: str) -> bool:
        """Download model with progress tracking"""
        try:
            logger.info(f"📥 Downloading {model_name} model...")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\r📥 Download progress: {percent:.1f}%", end='', flush=True)
            
            print(f"\n✅ {model_name} model downloaded successfully")
            logger.info(f"Model saved to: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to download {model_name} model: {e}")
            # Clean up partial download
            if os.path.exists(model_path):
                os.remove(model_path)
            return False
    
    def load_vehicle_model(self) -> YOLO:
        """Load and optimize vehicle detection model"""
        model_path = config.VEHICLE_MODEL_PATH
        
        try:
            # Check if model exists
            if not os.path.exists(model_path):
                logger.warning(f"Vehicle model not found at {model_path}")
                
                # Try to download default model
                default_url = MODEL_URLS.get("YOLO_VEHICLE")
                if default_url and self.download_model(default_url, model_path, "vehicle"):
                    logger.info("Using downloaded default vehicle model")
                else:
                    raise ModelLoadException("vehicle", model_path, "File not found and download failed")
            
            # Load model
            logger.info(f"🔄 Loading vehicle detection model from {model_path}")
            model = YOLO(model_path)
            
            # Move to device
            model.to(self.device)
            
            # Optimize model
            self._optimize_model(model, "vehicle")
            
            # Warm up model
            self._warmup_model(model, "vehicle")
            
            logger.info("✅ Vehicle detection model loaded and optimized")
            return model
            
        except Exception as e:
            logger.error(f"❌ Failed to load vehicle model: {e}")
            raise ModelLoadException("vehicle", model_path, str(e))
    
    def load_plate_model(self) -> YOLO:
        """Load and optimize license plate detection model"""
        model_path = config.PLATE_MODEL_PATH
        
        try:
            # Check if model exists
            if not os.path.exists(model_path):
                logger.warning(f"Plate model not found at {model_path}")
                
                # Try to download default model
                default_url = MODEL_URLS.get("YOLO_PLATE")
                if default_url and self.download_model(default_url, model_path, "plate"):
                    logger.info("Using downloaded default plate model")
                else:
                    raise ModelLoadException("plate", model_path, "File not found and download failed")
            
            # Load model
            logger.info(f"🔄 Loading license plate detection model from {model_path}")
            model = YOLO(model_path)
            
            # Move to device
            model.to(self.device)
            
            # Optimize model
            self._optimize_model(model, "plate")
            
            # Warm up model
            self._warmup_model(model, "plate")
            
            logger.info("✅ License plate detection model loaded and optimized")
            return model
            
        except Exception as e:
            logger.error(f"❌ Failed to load plate model: {e}")
            raise ModelLoadException("plate", model_path, str(e))
    
    def _optimize_model(self, model: YOLO, model_name: str):
        """Apply various optimizations to the model"""
        try:
            # Half precision optimization
            if self.half_precision and self.device == 'cuda':
                logger.info(f"🔧 Applying half precision to {model_name} model")
                model.model.half()
            
            # Model compilation (PyTorch 2.0+)
            if self.compile_models:
                logger.info(f"🔧 Compiling {model_name} model for optimization")
                try:
                    model.model = torch.compile(model.model, mode='reduce-overhead')
                except Exception as e:
                    logger.warning(f"Model compilation failed for {model_name}: {e}")
            
            # Set model to evaluation mode
            model.model.eval()
            
            # Disable gradient computation for inference
            for param in model.model.parameters():
                param.requires_grad = False
            
            logger.debug(f"✅ {model_name} model optimization completed")
            
        except Exception as e:
            logger.warning(f"⚠️ Model optimization failed for {model_name}: {e}")
    
    def _warmup_model(self, model: YOLO, model_name: str):
        """Warm up model with dummy input"""
        try:
            logger.info(f"🔥 Warming up {model_name} model...")
            
            # Determine input size based on model type
            if model_name == "vehicle":
                img_size = config.YOLO_IMG_SIZE
            else:
                img_size = config.PLATE_IMG_SIZE
            
            # Create dummy input
            dummy_input = torch.randn(1, 3, img_size, img_size)
            
            if self.device == 'cuda':
                dummy_input = dummy_input.cuda()
                if self.half_precision:
                    dummy_input = dummy_input.half()
            
            # Run warmup inference
            with torch.no_grad():
                _ = model.model(dummy_input)
            
            # Additional warmups for stability
            for _ in range(3):
                with torch.no_grad():
                    _ = model.model(dummy_input)
            
            logger.info(f"✅ {model_name} model warmup completed")
            
        except Exception as e:
            logger.warning(f"⚠️ Model warmup failed for {model_name}: {e}")
    
    def load_all_models(self) -> Tuple[YOLO, YOLO]:
        """Load and optimize all models"""
        try:
            logger.info("🚀 Loading all detection models...")
            
            # Load vehicle model
            vehicle_model = self.load_vehicle_model()
            
            # Load plate model  
            plate_model = self.load_plate_model()
            
            # Store references
            self.vehicle_model = vehicle_model
            self.plate_model = plate_model
            self.models_loaded = True
            
            # Log model information
            self._log_model_info(vehicle_model, "vehicle")
            self._log_model_info(plate_model, "plate")
            
            logger.info("✅ All models loaded successfully")
            return vehicle_model, plate_model
            
        except Exception as e:
            logger.error(f"❌ Failed to load models: {e}")
            raise
    
    def _log_model_info(self, model: YOLO, model_name: str):
        """Log detailed model information"""
        try:
            # Get model info
            model_info = {
                'name': model_name,
                'device': next(model.model.parameters()).device,
                'dtype': next(model.model.parameters()).dtype,
                'num_classes': getattr(model.model, 'nc', 'unknown'),
                'input_size': getattr(model.model, 'imgsz', 'unknown')
            }
            
            logger.info(f"📊 {model_name.title()} Model Info:")
            for key, value in model_info.items():
                logger.info(f"   {key}: {value}")
            
            # Memory usage
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / (1024**2)  # MB
                logger.info(f"   GPU Memory: {memory_allocated:.1f}MB allocated")
                
        except Exception as e:
            logger.warning(f"Could not retrieve model info for {model_name}: {e}")
    
    def validate_models(self) -> bool:
        """Validate that models are working correctly"""
        if not self.models_loaded:
            logger.error("Models not loaded")
            return False
        
        try:
            import numpy as np
            
            # Create test image
            test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            # Test vehicle model
            logger.info("🧪 Testing vehicle detection model...")
            vehicle_results = self.vehicle_model(
                test_image,
                conf=config.VEHICLE_CONF,
                verbose=False,
                imgsz=config.YOLO_IMG_SIZE
            )
            
            if len(vehicle_results) == 0:
                logger.warning("No vehicle detection results from test")
            else:
                logger.info(f"✅ Vehicle model test passed - {len(vehicle_results[0].boxes)} detections")
            
            # Test plate model
            logger.info("🧪 Testing license plate detection model...")
            plate_results = self.plate_model(
                test_image,
                conf=config.PLATE_CONF,
                verbose=False,
                imgsz=config.PLATE_IMG_SIZE
            )
            
            if len(plate_results) == 0:
                logger.warning("No plate detection results from test")
            else:
                logger.info(f"✅ Plate model test passed - {len(plate_results[0].boxes)} detections")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Model validation failed: {e}")
            return False
    
    def get_model_stats(self) -> dict:
        """Get comprehensive model statistics"""
        stats = {
            'models_loaded': self.models_loaded,
            'device': self.device,
            'half_precision': self.half_precision,
            'compile_models': self.compile_models,
            'vehicle_model_path': config.VEHICLE_MODEL_PATH,
            'plate_model_path': config.PLATE_MODEL_PATH
        }
        
        if torch.cuda.is_available():
            stats.update({
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_memory_total': torch.cuda.get_device_properties(0).total_memory / (1024**3),
                'gpu_memory_allocated': torch.cuda.memory_allocated() / (1024**2),
                'gpu_memory_cached': torch.cuda.memory_reserved() / (1024**2)
            })
        
        return stats
    
    def cleanup(self):
        """Clean up model resources"""
        logger.info("🧹 Cleaning up model resources...")
        
        # Clear model references
        self.vehicle_model = None
        self.plate_model = None
        self.models_loaded = False
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        logger.info("✅ Model cleanup completed")

# Global model manager instance
_model_manager = None

def get_model_manager() -> ModelManager:
    """Get global model manager instance"""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager

def load_models() -> Tuple[YOLO, YOLO]:
    """Load and return optimized YOLO models"""
    model_manager = get_model_manager()
    return model_manager.load_all_models()

def validate_models() -> bool:
    """Validate that models are working correctly"""
    model_manager = get_model_manager()
    return model_manager.validate_models()

def get_model_stats() -> dict:
    """Get model statistics"""
    model_manager = get_model_manager()
    return model_manager.get_model_stats()

def cleanup_models():
    """Clean up model resources"""
    global _model_manager
    if _model_manager:
        _model_manager.cleanup()
        _model_manager = None

# Example usage and testing
if __name__ == "__main__":
    import time
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Load models
        start_time = time.time()
        vehicle_model, plate_model = load_models()
        load_time = time.time() - start_time
        
        print(f"\n🎯 Models loaded in {load_time:.2f} seconds")
        
        # Validate models
        if validate_models():
            print("✅ Model validation passed")
        else:
            print("❌ Model validation failed")
        
        # Get model stats
        stats = get_model_stats()
        print(f"\n📊 Model Statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        # Test inference speed
        import numpy as np
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Time vehicle detection
        start_time = time.time()
        for _ in range(10):
            results = vehicle_model(test_image, verbose=False)
        vehicle_time = (time.time() - start_time) / 10
        
        # Time plate detection
        start_time = time.time()
        for _ in range(10):
            results = plate_model(test_image, verbose=False)
        plate_time = (time.time() - start_time) / 10
        
        print(f"\n⚡ Inference Speed:")
        print(f"   Vehicle detection: {vehicle_time*1000:.1f}ms avg")
        print(f"   Plate detection: {plate_time*1000:.1f}ms avg")
        print(f"   Combined FPS: {1/(vehicle_time + plate_time):.1f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Cleanup
        cleanup_models()
        print("🧹 Cleanup completed")