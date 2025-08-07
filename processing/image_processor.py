"""
Optimized Image Processor - Enhanced OCR processing
Xử lý hình ảnh tối ưu cho OCR biển số xe
"""

import cv2
import numpy as np
import logging
import time
import hashlib
from typing import Optional, Tuple, Dict, Any
import threading
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Import enhancement libraries with fallbacks
try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    ESRGAN_AVAILABLE = True
except ImportError:
    ESRGAN_AVAILABLE = False
    RealESRGANer = None

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    PaddleOCR = None

class ImageProcessor:
    """Optimized image processor for license plate OCR"""
    
    def __init__(self, config):
        self.config = config
        self.upsampler = None
        self.paddle_ocr = None
        self.cache = {} if config.ENABLE_CACHING else None
        self.lock = threading.Lock()
        
        self._init_components()
        logger.info(f"🎨 Image Processor initialized - ESRGAN: {'✓' if self.upsampler else '✗'}, OCR: {'✓' if self.paddle_ocr else '✗'}")
    
    def _init_components(self):
        """Initialize components with error handling"""
        try:
            if ESRGAN_AVAILABLE and self.config.USE_REAL_ESRGAN:
                self._init_esrgan()
            if PADDLEOCR_AVAILABLE:
                self._init_paddleocr()
        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
    
    def _init_esrgan(self):
        """Initialize Real-ESRGAN model"""
        try:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            self.upsampler = RealESRGANer(
                scale=2,
                model_path=self.config.ESRGAN_MODEL_PATH,
                model=model,
                tile=256,
                tile_pad=4,
                half=True
            )
        except Exception as e:
            logger.error(f"ESRGAN init failed: {e}")
            self.upsampler = None
    
    def _init_paddleocr(self):
        """Initialize PaddleOCR"""
        try:
            self.paddle_ocr = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                show_log=False,
                det_limit_side_len=640,
                drop_score=0.3
            )
        except Exception as e:
            logger.error(f"PaddleOCR init failed: {e}")
            self.paddle_ocr = None
    
    def should_enhance(self, image: np.ndarray) -> bool:
        """Smart decision for image enhancement"""
        if not self.upsampler:
            return False
        
        h, w = image.shape[:2]
        if max(h, w) > 400 or min(h, w) < 15:
            return False
        
        # Check image quality
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        return sharpness < 50
    
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Enhanced image processing"""
        if not self.should_enhance(image):
            return image
        
        cache_key = hashlib.md5(image.tobytes()).hexdigest()[:16]
        
        # Check cache
        if self.cache and cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            if self.upsampler:
                enhanced, _ = self.upsampler.enhance(image, outscale=2)
            else:
                # Fallback enhancement
                enhanced = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                enhanced = cv2.filter2D(enhanced, -1, kernel)
            
            # Cache result
            if self.cache and len(self.cache) < 50:
                self.cache[cache_key] = enhanced
            
            return enhanced
        except Exception as e:
            logger.error(f"Enhancement error: {e}")
            return image
    
    def preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """Optimized preprocessing for OCR"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Apply CLAHE for contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Noise reduction
            enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
            
            # Ensure minimum size
            h, w = enhanced.shape
            if min(h, w) < 32:
                scale = 32 / min(h, w)
                enhanced = cv2.resize(enhanced, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            
            return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        except Exception as e:
            logger.error(f"OCR preprocessing error: {e}")
            return image
    
    def ocr_plate(self, plate_image: np.ndarray) -> Tuple[Optional[str], float]:
        """Enhanced OCR with multiple strategies"""
        if not self.paddle_ocr:
            return None, 0.0
        
        results = []
        
        # Strategy 1: Direct OCR
        result = self._run_ocr(plate_image)
        if result:
            results.extend(result)
        
        # Strategy 2: Preprocessed OCR (if low confidence)
        if not results or max(r[1] for r in results) < 0.7:
            preprocessed = self.preprocess_for_ocr(plate_image)
            result = self._run_ocr(preprocessed)
            if result:
                results.extend([(text, conf * 0.95) for text, conf in result])
        
        # Strategy 3: Enhanced OCR (if available and needed)
        if not results or max(r[1] for r in results) < 0.6:
            if self.should_enhance(plate_image):
                enhanced = self.enhance_image(plate_image)
                result = self._run_ocr(enhanced)
                if result:
                    results.extend([(text, min(conf * 1.05, 1.0)) for text, conf in result])
        
        return self._select_best_result(results)
    
    def _run_ocr(self, image: np.ndarray) -> list:
        """Run PaddleOCR on image"""
        try:
            ocr_result = self.paddle_ocr.ocr(image, cls=True)
            results = []
            
            if ocr_result and ocr_result[0]:
                for line in ocr_result[0]:
                    if len(line) >= 2:
                        text, conf = line[1][0], line[1][1]
                        if conf >= 0.3 and text.strip():
                            results.append((text.upper().strip(), conf))
            
            return results
        except Exception as e:
            logger.error(f"OCR execution failed: {e}")
            return []
    
    def _select_best_result(self, results: list) -> Tuple[Optional[str], float]:
        """Select best OCR result"""
        if not results:
            return None, 0.0
        
        # Filter valid results
        valid = [(text, conf) for text, conf in results if self._is_valid_plate(text)]
        if not valid:
            return None, 0.0
        
        # Return highest confidence
        return max(valid, key=lambda x: x[1])
    
    def _is_valid_plate(self, text: str) -> bool:
        """Basic plate validation"""
        if not text or len(text) < 3 or len(text) > 15:
            return False
        return any(c.isalpha() for c in text) and any(c.isdigit() for c in text)