"""
Enhanced Image Processor - Real-ESRGAN và PaddleOCR Integration
Xử lý và nâng cao chất lượng hình ảnh cho OCR
"""

import cv2
import numpy as np
import logging
import time
import hashlib
import os
from datetime import datetime, timedelta
from collections import deque
from typing import Optional, Tuple, Dict, Any
import threading

from config.settings import config
from core.exceptions import ImageProcessingException, OCRException, EnhancementException
from core.constants import TIMING, MEMORY, QUALITY_THRESHOLDS

logger = logging.getLogger(__name__)

# Import enhancement libraries with error handling
try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    ESRGAN_AVAILABLE = True
    logger.info("✅ Real-ESRGAN imported successfully")
except ImportError as e:
    ESRGAN_AVAILABLE = False
    RealESRGANer = None
    logger.warning(f"❌ Real-ESRGAN not available: {e}")

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
    logger.info("✅ PaddleOCR imported successfully")
except ImportError as e:
    PADDLEOCR_AVAILABLE = False
    PaddleOCR = None
    logger.warning(f"❌ PaddleOCR not available: {e}")

class EnhancedImageProcessor:
    """Enhanced image processor với Real-ESRGAN và PaddleOCR"""
    
    def __init__(self):
        # Core components
        self.upsampler = None
        self.paddle_ocr = None
        
        # Caching system
        self.enhancement_cache = {} if config.ENABLE_CACHING else None
        self.cache_timestamps = {} if config.ENABLE_CACHING else None
        self.ocr_cache = {} if config.ENABLE_CACHING else None
        
        # Performance tracking
        self.processing_stats = {
            'enhancement_count': 0,
            'ocr_count': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'enhancement_time': deque(maxlen=100),
            'ocr_time': deque(maxlen=100),
            'preprocessing_time': deque(maxlen=100)
        }
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Initialize models
        self._init_components()
        
        logger.info("🎨 Enhanced Image Processor initialized")
        logger.info(f"   Real-ESRGAN: {'Available' if self.upsampler else 'Not available'}")
        logger.info(f"   PaddleOCR: {'Available' if self.paddle_ocr else 'Not available'}")
        logger.info(f"   Caching: {'Enabled' if config.ENABLE_CACHING else 'Disabled'}")
    
    def _init_components(self):
        """Initialize enhancement and OCR components"""
        try:
            # Initialize Real-ESRGAN
            if config.USE_REAL_ESRGAN and ESRGAN_AVAILABLE:
                self._init_esrgan()
            
            # Initialize PaddleOCR
            if PADDLEOCR_AVAILABLE:
                self._init_paddleocr()
                
        except Exception as e:
            logger.error(f"Component initialization error: {e}")
            raise ImageProcessingException("initialization", str(e))
    
    def _init_esrgan(self):
        """Initialize Real-ESRGAN model"""
        try:
            model_path = config.ESRGAN_MODEL_PATH
            
            # Create weights directory
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Download model if not exists
            if not os.path.exists(model_path):
                logger.info("📥 Downloading Real-ESRGAN model...")
                self._download_esrgan_model(model_path)
            
            # Initialize model architecture
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=config.ESRGAN_SCALE
            )
            
            # Create upsampler
            device = 'cuda' if config.USE_REAL_ESRGAN and hasattr(config, 'device') else None
            
            self.upsampler = RealESRGANer(
                scale=config.ESRGAN_SCALE,
                model_path=model_path,
                model=model,
                tile=config.ESRGAN_TILE_SIZE,
                tile_pad=4,
                pre_pad=0,
                half=True if device == 'cuda' else False,
                gpu_id=0 if device == 'cuda' else None
            )
            
            # Test the model
            test_img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            try:
                _, _ = self.upsampler.enhance(test_img, outscale=config.ESRGAN_SCALE)
                logger.info("✅ Real-ESRGAN initialized and tested successfully")
            except Exception as e:
                logger.error(f"Real-ESRGAN test failed: {e}")
                self.upsampler = None
                
        except Exception as e:
            logger.error(f"Real-ESRGAN initialization failed: {e}")
            self.upsampler = None
    
    def _init_paddleocr(self):
        """Initialize PaddleOCR"""
        try:
            import torch
            use_gpu = torch.cuda.is_available()
            
            self.paddle_ocr = PaddleOCR(
                use_angle_cls=True,
                lang='en',  # English for license plates
                use_gpu=use_gpu,
                show_log=False,
                det_limit_side_len=640,
                rec_batch_num=1,
                det_algorithm='DB',
                rec_algorithm='CRNN',
                drop_score=0.3,
                max_text_length=25,
                rec_image_shape="3, 48, 320"
            )
            
            # Test OCR
            test_img = np.ones((50, 200, 3), dtype=np.uint8) * 255
            cv2.putText(test_img, "TEST123", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
            try:
                result = self.paddle_ocr.ocr(test_img, cls=True)
                logger.info("✅ PaddleOCR initialized and tested successfully")
            except Exception as e:
                logger.error(f"PaddleOCR test failed: {e}")
                self.paddle_ocr = None
                
        except Exception as e:
            logger.error(f"PaddleOCR initialization failed: {e}")
            self.paddle_ocr = None
    
    def _download_esrgan_model(self, model_path: str):
        """Download Real-ESRGAN model with progress tracking"""
        try:
            import requests
            from core.constants import MODEL_URLS
            
            url = MODEL_URLS.get("REAL_ESRGAN")
            if not url:
                raise ValueError("Real-ESRGAN model URL not found")
            
            logger.info(f"Downloading from: {url}")
            
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
            
            print(f"\n✅ Real-ESRGAN model downloaded to {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to download Real-ESRGAN model: {e}")
            raise EnhancementException("download", str(e))
    
    def _get_cache_key(self, image: np.ndarray) -> str:
        """Generate cache key for image"""
        # Use image hash and shape for caching
        image_bytes = image.tobytes()
        image_hash = hashlib.md5(image_bytes).hexdigest()[:16]
        return f"{image.shape}_{image_hash}"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is still valid"""
        if not self.cache_timestamps or cache_key not in self.cache_timestamps:
            return False
        
        cache_time = self.cache_timestamps[cache_key]
        expire_time = cache_time + timedelta(minutes=config.CACHE_EXPIRE_MINUTES)
        return datetime.now() < expire_time
    
    def should_enhance(self, image: np.ndarray) -> bool:
        """Smart decision for image enhancement"""
        if not config.ENABLE_SMART_ENHANCEMENT or self.upsampler is None:
            return False
        
        h, w = image.shape[:2]
        
        # Skip if image is too large or too small
        if max(h, w) > 400 or min(h, w) < 15:
            return False
        
        # Enhance small to medium sized images
        if 15 <= min(h, w) <= 150 and max(h, w) <= 400:
            # Additional quality check
            return self._needs_enhancement(image)
        
        return False
    
    def _needs_enhancement(self, image: np.ndarray) -> bool:
        """Check if image needs enhancement based on quality metrics"""
        try:
            # Convert to grayscale for analysis
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Check sharpness
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Check contrast
            contrast = gray.std()
            
            # Enhance if low quality
            return sharpness < 50 or contrast < 30
            
        except Exception:
            return False
    
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Enhanced image processing with caching"""
        if not self.should_enhance(image):
            return image
        
        start_time = time.time()
        
        # Check cache
        cache_key = None
        if self.enhancement_cache is not None:
            cache_key = self._get_cache_key(image)
            if cache_key in self.enhancement_cache and self._is_cache_valid(cache_key):
                with self.lock:
                    self.processing_stats['cache_hits'] += 1
                return self.enhancement_cache[cache_key]
            
            with self.lock:
                self.processing_stats['cache_misses'] += 1
        
        try:
            # Preprocessing
            processed_image = self._preprocess_for_enhancement(image)
            
            # Enhancement with Real-ESRGAN
            if self.upsampler:
                enhanced, _ = self.upsampler.enhance(processed_image, outscale=config.ESRGAN_SCALE)
            else:
                # Fallback enhancement
                enhanced = self._fallback_enhancement(processed_image)
            
            # Post-processing
            enhanced = self._postprocess_enhanced(enhanced, image.shape[:2])
            
            # Cache result
            if cache_key and self.enhancement_cache is not None:
                # Clean old cache entries if cache is full
                if len(self.enhancement_cache) >= config.CACHE_SIZE:
                    self._clean_cache()
                
                self.enhancement_cache[cache_key] = enhanced
                self.cache_timestamps[cache_key] = datetime.now()
            
            # Update statistics
            process_time = time.time() - start_time
            with self.lock:
                self.processing_stats['enhancement_time'].append(process_time)
                self.processing_stats['enhancement_count'] += 1
            
            logger.debug(f"Image enhanced in {process_time:.3f}s")
            return enhanced
            
        except Exception as e:
            logger.error(f"Enhancement error: {e}")
            return image
    
    def _preprocess_for_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image before enhancement"""
        try:
            # Resize if too large for efficient processing
            h, w = image.shape[:2]
            if max(h, w) > 300:
                scale = 300 / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Ensure minimum size
            h, w = image.shape[:2]
            if min(h, w) < 32:
                scale = 32 / min(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            return image
            
        except Exception as e:
            logger.warning(f"Preprocessing failed: {e}")
            return image
    
    def _fallback_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Fallback enhancement when Real-ESRGAN is not available"""
        try:
            # Upscale using INTER_CUBIC
            h, w = image.shape[:2]
            enhanced = cv2.resize(image, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
            
            # Apply sharpening
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)
            
            # Enhance contrast
            enhanced = cv2.convertScaleAbs(enhanced, alpha=1.1, beta=5)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Fallback enhancement failed: {e}")
            return image
    
    def _postprocess_enhanced(self, enhanced: np.ndarray, original_shape: Tuple[int, int]) -> np.ndarray:
        """Post-process enhanced image"""
        try:
            # Resize to reasonable size if too large
            h, w = enhanced.shape[:2]
            if max(h, w) > 800:  # Limit maximum size
                scale = 800 / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                enhanced = cv2.resize(enhanced, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Ensure values are in valid range
            enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
            
            return enhanced
            
        except Exception as e:
            logger.warning(f"Post-processing failed: {e}")
            return enhanced
    
    def preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """Advanced preprocessing for OCR"""
        start_time = time.time()
        
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply CLAHE for contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Noise reduction with bilateral filter
            enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
            
            # Morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
            
            # Resize if too small for OCR
            h, w = enhanced.shape
            if min(h, w) < 32:
                scale = 48 / min(h, w)
                enhanced = cv2.resize(enhanced, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            
            # Ensure minimum size for reliable OCR
            h, w = enhanced.shape
            if h < 32 or w < 100:
                target_h = max(32, h)
                target_w = max(100, w)
                enhanced = cv2.resize(enhanced, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            
            # Convert back to BGR for PaddleOCR
            result = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            
            # Update statistics
            process_time = time.time() - start_time
            with self.lock:
                self.processing_stats['preprocessing_time'].append(process_time)
            
            return result
            
        except Exception as e:
            logger.error(f"OCR preprocessing error: {e}")
            return image
    
    def ocr_plate(self, plate_image: np.ndarray, use_cache: bool = True) -> Tuple[Optional[str], float]:
        """Enhanced OCR with multiple strategies and caching"""
        if self.paddle_ocr is None:
            logger.warning("PaddleOCR not available")
            return None, 0.0
        
        start_time = time.time()
        
        # Check cache
        cache_key = None
        if use_cache and self.ocr_cache is not None:
            cache_key = self._get_cache_key(plate_image)
            if cache_key in self.ocr_cache and self._is_cache_valid(cache_key):
                with self.lock:
                    self.processing_stats['cache_hits'] += 1
                return self.ocr_cache[cache_key]
        
        results = []
        
        try:
            # Strategy 1: Direct OCR
            ocr_result = self._run_ocr(plate_image)
            if ocr_result:
                results.extend(ocr_result)
            
            # Strategy 2: Preprocessed OCR
            if not results or (results and max(r[1] for r in results) < 0.7):
                preprocessed = self.preprocess_for_ocr(plate_image)
                ocr_result = self._run_ocr(preprocessed)
                if ocr_result:
                    # Apply slight penalty for preprocessing
                    preprocessed_results = [(text, conf * 0.95) for text, conf in ocr_result]
                    results.extend(preprocessed_results)
            
            # Strategy 3: Enhanced OCR (if available and low confidence)
            if ((not results or (results and max(r[1] for r in results) < 0.6)) and 
                self.upsampler and self.should_enhance(plate_image)):
                
                enhanced = self.enhance_image(plate_image)
                if not np.array_equal(enhanced, plate_image):
                    ocr_result = self._run_ocr(enhanced)
                    if ocr_result:
                        # Apply bonus for enhancement
                        enhanced_results = [(text, min(conf * 1.05, 1.0)) for text, conf in ocr_result]
                        results.extend(enhanced_results)
            
            # Select best result
            best_result = self._select_best_ocr_result(results)
            
            # Cache result
            if cache_key and self.ocr_cache is not None and best_result[0]:
                if len(self.ocr_cache) >= config.CACHE_SIZE:
                    self._clean_ocr_cache()
                
                self.ocr_cache[cache_key] = best_result
                self.cache_timestamps[cache_key] = datetime.now()
            
            # Update statistics
            process_time = time.time() - start_time
            with self.lock:
                self.processing_stats['ocr_time'].append(process_time)
                self.processing_stats['ocr_count'] += 1
            
            return best_result
            
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return None, 0.0
    
    def _run_ocr(self, image: np.ndarray) -> List[Tuple[str, float]]:
        """Run PaddleOCR on image"""
        try:
            results = []
            ocr_result = self.paddle_ocr.ocr(image, cls=True)
            
            if ocr_result and ocr_result[0]:
                for line in ocr_result[0]:
                    if len(line) >= 2:
                        bbox, (text, conf) = line[0], line[1]
                        if conf >= config.OCR_MIN_CONF and text.strip():
                            clean_text = text.upper().strip()
                            results.append((clean_text, conf))
            
            return results
            
        except Exception as e:
            logger.error(f"PaddleOCR execution failed: {e}")
            return []
    
    def _select_best_ocr_result(self, results: List[Tuple[str, float]]) -> Tuple[Optional[str], float]:
        """Select the best OCR result from multiple attempts"""
        if not results:
            return None, 0.0
        
        # Filter out invalid results
        valid_results = []
        for text, conf in results:
            if self._is_valid_plate_text(text):
                valid_results.append((text, conf))
        
        if not valid_results:
            return None, 0.0
        
        # Sort by confidence
        valid_results.sort(key=lambda x: x[1], reverse=True)
        
        # Return best result
        return valid_results[0]
    
    def _is_valid_plate_text(self, text: str) -> bool:
        """Basic validation for plate text"""
        if not text or len(text) < 3:
            return False
        
        # Must contain both letters and numbers
        has_letter = any(c.isalpha() for c in text)
        has_number = any(c.isdigit() for c in text)
        
        if not (has_letter and has_number):
            return False
        
        # Reasonable length
        if len(text) > 15:
            return False
        
        return True
    
    def batch_process_plates(self, plate_images: List[np.ndarray]) -> List[Tuple[Optional[str], float]]:
        """Process multiple plate images efficiently"""
        results = []
        
        for plate_image in plate_images:
            try:
                text, conf = self.ocr_plate(plate_image)
                results.append((text, conf))
            except Exception as e:
                logger.error(f"Batch OCR failed for image: {e}")
                results.append((None, 0.0))
        
        return results
    
    def _clean_cache(self):
        """Clean expired enhancement cache entries"""
        if not self.enhancement_cache or not self.cache_timestamps:
            return
        
        current_time = datetime.now()
        expired_keys = []
        
        for key, timestamp in self.cache_timestamps.items():
            if current_time - timestamp > timedelta(minutes=config.CACHE_EXPIRE_MINUTES):
                expired_keys.append(key)
        
        for key in expired_keys:
            self.enhancement_cache.pop(key, None)
            self.cache_timestamps.pop(key, None)
        
        logger.debug(f"Cleaned {len(expired_keys)} expired enhancement cache entries")
    
    def _clean_ocr_cache(self):
        """Clean expired OCR cache entries"""
        if not self.ocr_cache or not self.cache_timestamps:
            return
        
        current_time = datetime.now()
        expired_keys = []
        
        for key, timestamp in self.cache_timestamps.items():
            if (key in self.ocr_cache and 
                current_time - timestamp > timedelta(minutes=config.CACHE_EXPIRE_MINUTES)):
                expired_keys.append(key)
        
        for key in expired_keys:
            self.ocr_cache.pop(key, None)
            self.cache_timestamps.pop(key, None)
        
        logger.debug(f"Cleaned {len(expired_keys)} expired OCR cache entries")
    
    def save_processed_image(self, image: np.ndarray, filepath: str, 
                           enhance: bool = False, preprocess_ocr: bool = False) -> bool:
        """Save processed image with optional enhancements"""
        try:
            # Apply requested processing
            processed_image = image.copy()
            
            if enhance and self.should_enhance(image):
                processed_image = self.enhance_image(processed_image)
            
            if preprocess_ocr:
                processed_image = self.preprocess_for_ocr(processed_image)
            
            # Create directory if needed
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save with high quality
            success = cv2.imwrite(filepath, processed_image, 
                                [cv2.IMWRITE_JPEG_QUALITY, config.IMAGE_QUALITY])
            
            if success:
                logger.debug(f"Saved processed image to {filepath}")
            else:
                logger.error(f"Failed to save image to {filepath}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error saving processed image: {e}")
            return False
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics"""
        stats = {}
        
        with self.lock:
            stats.update(self.processing_stats)
        
        # Calculate averages
        for time_key in ['enhancement_time', 'ocr_time', 'preprocessing_time']:
            times = stats.get(time_key, [])
            if times:
                stats[f'avg_{time_key}'] = np.mean(times)
                stats[f'max_{time_key}'] = np.max(times)
                stats[f'min_{time_key}'] = np.min(times)
            else:
                stats[f'avg_{time_key}'] = 0
                stats[f'max_{time_key}'] = 0
                stats[f'min_{time_key}'] = 0
        
        # Cache statistics
        if config.ENABLE_CACHING:
            stats['enhancement_cache_size'] = len(self.enhancement_cache) if self.enhancement_cache else 0
            stats['ocr_cache_size'] = len(self.ocr_cache) if self.ocr_cache else 0
            
            total_requests = stats['cache_hits'] + stats['cache_misses']
            stats['cache_hit_rate'] = (stats['cache_hits'] / total_requests * 100) if total_requests > 0 else 0
        
        # Component availability
        stats['esrgan_available'] = self.upsampler is not None
        stats['paddleocr_available'] = self.paddle_ocr is not None
        
        return stats
    
    def reset_statistics(self):
        """Reset processing statistics"""
        with self.lock:
            self.processing_stats = {
                'enhancement_count': 0,
                'ocr_count': 0,  
                'cache_hits': 0,
                'cache_misses': 0,
                'enhancement_time': deque(maxlen=100),
                'ocr_time': deque(maxlen=100),
                'preprocessing_time': deque(maxlen=100)
            }
        logger.info("🔄 Image processor statistics reset")
    
    def cleanup(self):
        """Clean up processor resources"""
        logger.info("🧹 Cleaning up image processor...")
        
        # Clear caches
        if self.enhancement_cache:
            self.enhancement_cache.clear()
        if self.ocr_cache:
            self.ocr_cache.clear()
        if self.cache_timestamps:
            self.cache_timestamps.clear()
        
        # Clear statistics
        with self.lock:
            for key in self.processing_stats:
                if isinstance(self.processing_stats[key], deque):
                    self.processing_stats[key].clear()
                else:
                    self.processing_stats[key] = 0
        
        # Cleanup models if needed
        self.upsampler = None
        self.paddle_ocr = None
        
        logger.info("✅ Image processor cleanup completed")

# Utility functions
def create_test_plate_image(text: str = "30A12345", size: Tuple[int, int] = (200, 50)) -> np.ndarray:
    """Create a test license plate image for testing"""
    image = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255
    
    # Add some noise and texture
    noise = np.random.randint(0, 30, (size[1], size[0], 3))
    image = cv2.subtract(image, noise)
    
    # Add text
    font_scale = size[0] / 200  # Scale font based on image width
    thickness = max(1, int(font_scale * 2))
    
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    text_x = (size[0] - text_size[0]) // 2
    text_y = (size[1] + text_size[1]) // 2
    
    cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, (0, 0, 0), thickness)
    
    return image

def assess_enhancement_quality(original: np.ndarray, enhanced: np.ndarray) -> Dict[str, float]:
    """Assess the quality improvement from enhancement"""
    try:
        # Convert to grayscale for analysis
        if len(original.shape) == 3:
            orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        else:
            orig_gray = original
            
        if len(enhanced.shape) == 3:
            enh_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        else:
            enh_gray = enhanced
        
        # Resize enhanced to match original size for fair comparison
        if orig_gray.shape != enh_gray.shape:
            enh_gray = cv2.resize(enh_gray, (orig_gray.shape[1], orig_gray.shape[0]))
        
        # Calculate metrics
        metrics = {}
        
        # Sharpness (Laplacian variance)
        orig_sharpness = cv2.Laplacian(orig_gray, cv2.CV_64F).var()
        enh_sharpness = cv2.Laplacian(enh_gray, cv2.CV_64F).var()
        metrics['sharpness_improvement'] = (enh_sharpness - orig_sharpness) / max(orig_sharpness, 1)
        
        # Contrast (standard deviation)
        orig_contrast = orig_gray.std()
        enh_contrast = enh_gray.std()
        metrics['contrast_improvement'] = (enh_contrast - orig_contrast) / max(orig_contrast, 1)
        
        # Overall quality score
        metrics['quality_score'] = (metrics['sharpness_improvement'] + metrics['contrast_improvement']) / 2
        
        return metrics
        
    except Exception as e:
        logger.error(f"Quality assessment failed: {e}")
        return {'quality_score': 0.0, 'sharpness_improvement': 0.0, 'contrast_improvement': 0.0}

# Factory function
def create_image_processor():
    """Create enhanced image processor instance"""
    return EnhancedImageProcessor()

# Example usage
if __name__ == "__main__":
    # Test the image processor
    processor = create_image_processor()
    
    # Create test image
    test_image = create_test_plate_image("30A12345", (200, 50))
    
    # Test enhancement
    if processor.should_enhance(test_image):
        enhanced = processor.enhance_image(test_image)
        quality_metrics = assess_enhancement_quality(test_image, enhanced)
        print(f"Enhancement quality: {quality_metrics}")
    
    # Test OCR
    text, confidence = processor.ocr_plate(test_image)
    print(f"OCR Result: '{text}' (confidence: {confidence:.3f})")
    
    # Get statistics
    stats = processor.get_processing_stats()
    print(f"Processing stats: {stats}")
    
    # Cleanup
    processor.cleanup()