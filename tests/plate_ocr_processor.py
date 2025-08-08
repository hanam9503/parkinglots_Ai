"""
Module xử lý OCR biển số xe Việt Nam tối ưu
- Hỗ trợ biển số 1 dòng và 2 dòng
- Enhanced OCR với PaddleOCR và Real-ESRGAN
- Smart validation và formatting
- Tối ưu hóa performance và code
"""

import cv2
import numpy as np
import torch
import time
import re
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, List, Any
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
logging.getLogger("ultralytics").setLevel(logging.WARNING)

# Import modules
try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    ESRGAN_AVAILABLE = True
except ImportError:
    print("❌ Real-ESRGAN not available. Install: pip install realesrgan")
    ESRGAN_AVAILABLE = False

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    print("❌ PaddleOCR not available. Install: pip install paddleocr")
    PADDLEOCR_AVAILABLE = False

logger = logging.getLogger(__name__)

class PlateOCRConfig:
    """Configuration for OCR processing"""
    USE_ESRGAN = True
    ESRGAN_SCALE = 4
    OCR_MIN_CONF = 0.4
    ENABLE_CACHING = True
    CACHE_SIZE = 50
    CACHE_EXPIRE_MIN = 30

class VietnamesePlateProcessor:
    """Unified Vietnamese license plate processor"""
    
    def __init__(self, config: PlateOCRConfig = None):
        self.config = config or PlateOCRConfig()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize components
        self.paddle_ocr = self._init_paddleocr()
        self.upsampler = self._init_esrgan()
        
        # Caching
        self.cache = {} if self.config.ENABLE_CACHING else None
        self.cache_times = {} if self.config.ENABLE_CACHING else None
        
        # Vietnamese patterns and corrections
        self._init_vietnamese_data()
        
        # Stats
        self.stats = {'processed': 0, 'valid': 0, 'cache_hits': 0}
    
    def _init_paddleocr(self):
        """Initialize PaddleOCR"""
        if not PADDLEOCR_AVAILABLE:
            return None
        try:
            return PaddleOCR(
                use_angle_cls=True,
                lang='en',
                use_gpu=torch.cuda.is_available(),
                show_log=False,
                det_limit_side_len=320,
                drop_score=0.3
            )
        except Exception as e:
            logger.error(f"PaddleOCR init error: {e}")
            return None
    
    def _init_esrgan(self):
        """Initialize Real-ESRGAN"""
        if not self.config.USE_ESRGAN or not ESRGAN_AVAILABLE:
            return None
        
        try:
            model_path = "weights/RealESRGAN_x4plus.pth"
            if not os.path.exists(model_path):
                os.makedirs("weights", exist_ok=True)
                self._download_model(model_path)
            
            model = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=23, num_grow_ch=32, scale=self.config.ESRGAN_SCALE
            )
            
            return RealESRGANer(
                scale=self.config.ESRGAN_SCALE,
                model_path=model_path,
                model=model,
                tile=128,
                tile_pad=4,
                half=True if self.device == 'cuda' else False,
                gpu_id=0 if self.device == 'cuda' else None
            )
        except Exception as e:
            logger.error(f"ESRGAN init error: {e}")
            return None
    
    def _download_model(self, model_path: str):
        """Download ESRGAN model"""
        try:
            import requests
            url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
            
            print("📥 Downloading Real-ESRGAN model...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print("✅ Model downloaded successfully")
        except Exception as e:
            logger.error(f"Download failed: {e}")
            raise
    
    def _init_vietnamese_data(self):
        """Initialize Vietnamese license plate data"""
        # Province codes
        self.province_codes = {
            '11', '12', '13', '14', '15', '16', '17', '18', '19',  # Hanoi
            '29', '30', '31', '32', '33', '34', '35', '36', '37', '38',  # HCMC
            '43', '44', '47', '48', '49', '50', '51', '52', '53', '54',
            '55', '56', '57', '58', '59', '60', '61', '62', '63', '64',
            '65', '66', '67', '68', '69', '70', '71', '72', '73', '74',
            '75', '76', '77', '78', '79', '80', '81', '82', '83', '84',
            '85', '86', '87', '88', '89', '90', '91', '92', '93', '94',
            '95', '96', '97', '98', '99'
        }
        
        # Special codes
        self.special_codes = ['LD', 'QN', 'NG', 'TQ', 'HC', 'CD', 'TV']
        
        # OCR corrections
        self.corrections = {
            'O': '0', 'I': '1', 'L': '1', 'Z': '2', 'S': '5',
            'G': '6', 'Q': '0', 'D': '0', 'T': '7', 'B': '8'
        }
        
        # Plate patterns (single and double line)
        self.patterns = [
            # Single line patterns
            r'^(\d{2})([A-Z])(\d{4,5})$',      # 30A-12345
            r'^(\d{2})([A-Z]{2})(\d{4})$',     # 30AA-1234
            r'^([A-Z]{2})(\d{4,5})$',          # LD-12345
            
            # Double line patterns (combined)
            r'^(\d{2})([A-Z]{1,2})(\d{4,5})$', # Combined: 30A12345 or 30AA1234
            r'^([A-Z]{2})(\d{4,5})$'           # Combined: LD12345
        ]
    
    def _get_cache_key(self, image: np.ndarray) -> str:
        """Generate cache key"""
        return hashlib.md5(image.tobytes()).hexdigest()[:12]
    
    def _check_cache(self, cache_key: str) -> Optional[Dict]:
        """Check cache for result"""
        if not self.cache or cache_key not in self.cache:
            return None
        
        cache_time = self.cache_times.get(cache_key)
        if cache_time and (datetime.now() - cache_time).seconds > self.config.CACHE_EXPIRE_MIN * 60:
            # Expired
            self.cache.pop(cache_key, None)
            self.cache_times.pop(cache_key, None)
            return None
        
        self.stats['cache_hits'] += 1
        return self.cache[cache_key]
    
    def _save_cache(self, cache_key: str, result: Dict):
        """Save to cache"""
        if not self.cache:
            return
        
        # Clean old entries if cache is full
        if len(self.cache) >= self.config.CACHE_SIZE:
            oldest_key = min(self.cache_times.keys(), key=lambda k: self.cache_times[k])
            self.cache.pop(oldest_key, None)
            self.cache_times.pop(oldest_key, None)
        
        self.cache[cache_key] = result
        self.cache_times[cache_key] = datetime.now()
    
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Enhance image quality"""
        if self.upsampler is None:
            return image
        
        h, w = image.shape[:2]
        if min(h, w) < 20 or max(h, w) > 400:
            return image
        
        try:
            enhanced, _ = self.upsampler.enhance(image)
            return enhanced
        except Exception as e:
            logger.error(f"Enhancement error: {e}")
            return image
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for OCR"""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Noise reduction
            enhanced = cv2.fastNlMeansDenoising(enhanced, h=10)
            
            # Ensure minimum size
            h, w = enhanced.shape
            if h < 32 or w < 100:
                scale = max(32/h, 100/w)
                enhanced = cv2.resize(enhanced, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            
            return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            
        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            return image
    
    def extract_text_from_ocr(self, ocr_result: List) -> List[Tuple[str, float]]:
        """Extract text and confidence from OCR result"""
        results = []
        
        if not ocr_result or not ocr_result[0]:
            return results
        
        for line in ocr_result[0]:
            if len(line) >= 2:
                bbox, (text, conf) = line[0], line[1]
                if conf >= self.config.OCR_MIN_CONF:
                    clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
                    if len(clean_text) >= 3:
                        results.append((clean_text, conf))
        
        return results
    
    def detect_plate_lines(self, ocr_result: List) -> Tuple[List[str], bool]:
        """Detect if plate is single or double line and extract text"""
        if not ocr_result or not ocr_result[0]:
            return [], False
        
        # Extract text with positions
        text_boxes = []
        for line in ocr_result[0]:
            if len(line) >= 2:
                bbox, (text, conf) = line[0], line[1]
                if conf >= self.config.OCR_MIN_CONF:
                    # Get center Y coordinate
                    y_coords = [point[1] for point in bbox]
                    center_y = sum(y_coords) / len(y_coords)
                    
                    clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
                    if len(clean_text) >= 2:
                        text_boxes.append((clean_text, center_y, conf))
        
        if not text_boxes:
            return [], False
        
        # Sort by Y coordinate (top to bottom)
        text_boxes.sort(key=lambda x: x[1])
        
        # Check if double line (significant Y difference)
        if len(text_boxes) >= 2:
            y_diff = text_boxes[-1][1] - text_boxes[0][1]
            image_height = 100  # Approximate
            
            if y_diff > image_height * 0.3:  # 30% height difference indicates double line
                # Double line: combine texts from top and bottom
                texts = [box[0] for box in text_boxes]
                combined_text = ''.join(texts)
                return [combined_text], True
        
        # Single line or close together - treat as single
        best_text = max(text_boxes, key=lambda x: x[2])[0]  # Highest confidence
        return [best_text], False
    
    def correct_ocr_text(self, text: str) -> str:
        """Apply OCR corrections"""
        if not text:
            return ""
        
        result = ""
        for i, char in enumerate(text):
            if char in self.corrections:
                # Context-aware correction
                next_char = text[i + 1] if i < len(text) - 1 else None
                if next_char and next_char.isdigit() and char in ['O', 'Q']:
                    result += '0'
                elif next_char and next_char.isalpha() and char == '0':
                    result += 'O'
                else:
                    result += self.corrections.get(char, char)
            else:
                result += char
        
        return result
    
    def validate_and_format_plate(self, text: str, is_double_line: bool = False) -> str:
        """Validate and format Vietnamese license plate"""
        if not text or len(text) < 4:
            return ""
        
        corrected = self.correct_ocr_text(text)
        
        # Try different patterns
        for pattern in self.patterns:
            match = re.match(pattern, corrected)
            if match:
                groups = match.groups()
                
                # Format based on pattern
                if len(groups) == 3:
                    prefix, middle, suffix = groups
                    
                    # Check province code for standard format
                    if prefix.isdigit() and len(prefix) == 2:
                        if prefix in self.province_codes:
                            if is_double_line:
                                return f"{prefix}{middle}\n{suffix}"
                            else:
                                return f"{prefix}{middle}-{suffix}"
                    
                    # Check special codes
                    elif prefix in self.special_codes:
                        if is_double_line:
                            return f"{prefix}\n{middle}"
                        else:
                            return f"{prefix}-{middle}"
                    
                    # Default formatting
                    if is_double_line:
                        return f"{prefix}{middle}\n{suffix}"
                    else:
                        return f"{prefix}{middle}-{suffix}"
                
                elif len(groups) == 2:
                    prefix, suffix = groups
                    if is_double_line:
                        return f"{prefix}\n{suffix}"
                    else:
                        return f"{prefix}-{suffix}"
        
        # If no pattern matches but looks valid
        if len(corrected) >= 5:
            has_letter = any(c.isalpha() for c in corrected)
            has_number = any(c.isdigit() for c in corrected)
            if has_letter and has_number:
                return corrected
        
        return ""
    
    def is_valid_plate(self, text: str) -> bool:
        """Check if plate text is valid"""
        if not text or len(text) < 4:
            return False
        
        # Remove newlines for validation
        clean_text = text.replace('\n', '').replace('-', '')
        
        has_letter = any(c.isalpha() for c in clean_text)
        has_number = any(c.isdigit() for c in clean_text)
        
        if not (has_letter and has_number):
            return False
        
        if len(clean_text) > 12:
            return False
        
        # Check for obvious errors
        if re.search(r'^(.)\1{4,}', clean_text):  # Too many repeated chars
            return False
        
        return True
    
    def ocr_with_strategies(self, image: np.ndarray) -> Tuple[Optional[str], float, bool]:
        """OCR with multiple strategies"""
        if self.paddle_ocr is None:
            return None, 0.0, False
        
        best_result = None
        best_conf = 0.0
        is_double_line = False
        
        strategies = [
            ("direct", image),
            ("preprocessed", self.preprocess_image(image)),
        ]
        
        # Add enhancement if available
        if self.upsampler:
            enhanced = self.enhance_image(image)
            if not np.array_equal(enhanced, image):
                strategies.append(("enhanced", enhanced))
        
        for strategy_name, processed_image in strategies:
            try:
                ocr_result = self.paddle_ocr.ocr(processed_image, cls=True)
                texts, double_line = self.detect_plate_lines(ocr_result)
                
                for text in texts:
                    if text:
                        # Calculate average confidence (simplified)
                        text_results = self.extract_text_from_ocr(ocr_result)
                        if text_results:
                            avg_conf = sum(conf for _, conf in text_results) / len(text_results)
                            
                            # Apply strategy penalty/bonus
                            if strategy_name == "enhanced":
                                avg_conf *= 1.05  # Bonus for enhancement
                            elif strategy_name == "preprocessed":
                                avg_conf *= 0.95  # Slight penalty
                            
                            if avg_conf > best_conf:
                                best_result = text
                                best_conf = avg_conf
                                is_double_line = double_line
                
            except Exception as e:
                logger.error(f"OCR error in {strategy_name}: {e}")
                continue
        
        return best_result, best_conf, is_double_line
    
    def process_plate(self, plate_image: np.ndarray, save_path: str = None) -> Dict[str, Any]:
        """Process license plate image"""
        self.stats['processed'] += 1
        
        # Initialize result
        result = {
            'text': None,
            'confidence': 0.0,
            'formatted_text': None,
            'is_valid': False,
            'is_double_line': False,
            'processing_time': 0.0,
            'save_path': save_path,
            'from_cache': False
        }
        
        if plate_image is None or plate_image.size == 0:
            return result
        
        start_time = time.time()
        
        # Check cache
        cache_key = self._get_cache_key(plate_image)
        cached_result = self._check_cache(cache_key)
        if cached_result:
            cached_result['from_cache'] = True
            return cached_result
        
        # Save image if requested
        if save_path:
            try:
                cv2.imwrite(save_path, plate_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            except Exception as e:
                logger.error(f"Save error: {e}")
        
        # Perform OCR
        raw_text, confidence, is_double_line = self.ocr_with_strategies(plate_image)
        
        if raw_text and confidence > 0:
            result['text'] = raw_text
            result['confidence'] = confidence
            result['is_double_line'] = is_double_line
            
            # Format and validate
            formatted_text = self.validate_and_format_plate(raw_text, is_double_line)
            
            if formatted_text:
                result['formatted_text'] = formatted_text
                result['is_valid'] = self.is_valid_plate(formatted_text)
                
                if result['is_valid']:
                    self.stats['valid'] += 1
        
        result['processing_time'] = time.time() - start_time
        
        # Save to cache
        self._save_cache(cache_key, result.copy())
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        stats = self.stats.copy()
        
        if stats['processed'] > 0:
            stats['success_rate'] = (stats['valid'] / stats['processed']) * 100
        else:
            stats['success_rate'] = 0.0
        
        if self.cache:
            stats['cache_size'] = len(self.cache)
            total_requests = stats['processed']
            stats['cache_hit_rate'] = (stats['cache_hits'] / max(1, total_requests)) * 100
        
        return stats
    
    def cleanup(self):
        """Cleanup resources"""
        if self.cache:
            self.cache.clear()
        if self.cache_times:
            self.cache_times.clear()
        logger.info("Cleanup completed")

# Factory function
def create_plate_processor(use_esrgan: bool = True, enable_caching: bool = True) -> VietnamesePlateProcessor:
    """Create optimized plate processor"""
    config = PlateOCRConfig()
    config.USE_ESRGAN = use_esrgan
    config.ENABLE_CACHING = enable_caching
    
    return VietnamesePlateProcessor(config)

# Example usage
if __name__ == "__main__":
    processor = create_plate_processor()
    
    # Test with sample image
    test_image = "test_plate.jpg"
    if os.path.exists(test_image):
        image = cv2.imread(test_image)
        result = processor.process_plate(image)
        
        print("=== OCR Result ===")
        print(f"Raw text: {result['text']}")
        print(f"Formatted: {result['formatted_text']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Valid: {result['is_valid']}")
        print(f"Double line: {result['is_double_line']}")
        print(f"Time: {result['processing_time']:.3f}s")
        print(f"From cache: {result['from_cache']}")
        
        # Display formatted plate
        if result['formatted_text']:
            print(f"\nFormatted plate:")
            print(result['formatted_text'])
        
        # Stats
        stats = processor.get_stats()
        print(f"\nStats: {stats}")
        
        processor.cleanup()
    else:
        print("Test image not found. Place a license plate image as 'test_plate.jpg'")