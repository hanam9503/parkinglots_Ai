"""
Module xử lý OCR biển số xe Việt Nam tối ưu - Enhanced Version (FIXED)
- Chuẩn hóa biển số Việt Nam theo quy định
- Xử lý biển số 1 dòng và 2 dòng
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

class OCRConfig:
    """Configuration for OCR processing"""
    USE_REAL_ESRGAN = True
    ESRGAN_SCALE = 4
    OCR_MIN_CONF = 0.4
    ENABLE_CACHING = True
    CACHE_SIZE = 50
    CACHE_EXPIRE_MIN = 30

class VietnamesePlateNormalizer:
    """Vietnamese License Plate Normalizer - FIXED VERSION"""
    
    def __init__(self):
        self._init_vietnamese_data()
    
    def _init_vietnamese_data(self):
        """Initialize Vietnamese license plate data"""
        
        # Province codes (2 digits) - COMPLETE LIST INCLUDING 10
        self.province_codes = {
            # Hanoi region  
            '10',  # Added Hanoi code 10
            '11', '12', '13', '14', '15', '16', '17', '18', '19',
            # HCMC region  
            '29', '30', '31', '32', '33', '34', '35', '36', '37', '38',
            # Other provinces
            '20', '21', '22', '23', '24', '25', '26', '27', '28',  # Added missing codes
            '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', 
            '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', 
            '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', 
            '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', 
            '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', 
            '90', '91', '92', '93', '94', '95', '96', '97', '98', '99'
        }
        
        # Special administrative codes
        self.special_codes = ['LD', 'QN', 'NG', 'TQ', 'HC', 'CD', 'TV']
        
        # Valid letter patterns for Vietnamese plates
        self.valid_letters = set('ABCDEFGHIKLMNPRSTUVXYZ')
        
        # Fixed plate format patterns
        self.plate_patterns = [
            # Standard formats (XX-YYY.ZZ or XX-YYYY.Z)
            r'^(\d{2})([A-Z])(\d{4,5})$',      # 30F54504 -> 30F-545.04
            r'^(\d{2})([A-Z]{2})(\d{4})$',     # 30AB1234 -> 30AB-12.34
            
            # Special administrative formats
            r'^([A-Z]{2})(\d{4,5})$',          # LD12345 -> LD-123.45
            
            # Diplomatic/Military formats  
            r'^(\d{3})([A-Z])(\d{2})$',        # 123A12
            r'^([A-Z]{2})(\d{3})([A-Z])(\d{2})$', # AB123C12
        ]
    
    def clean_ocr_text(self, text: str) -> str:
        """Clean OCR text by removing unwanted characters"""
        if not text:
            return ""
        
        # Remove all non-alphanumeric characters including newlines
        cleaned = re.sub(r'[^A-Za-z0-9]', '', text.upper())
        
        # Remove common OCR artifacts
        cleaned = re.sub(r'[.,;:|!@#$%^&*()_+=\[\]{}"\'/\\<>?`~]', '', cleaned)
        
        return cleaned
    
    def apply_ocr_corrections(self, text: str) -> str:
        """Apply smart OCR corrections - FIXED VERSION"""
        if not text or len(text) < 3:
            return text
        
        result = list(text)
        
        # Handle the special case JOH79007 -> 30H79007
        if text.startswith('JO') and len(text) >= 8:
            result[0] = '3'  # J -> 3
            result[1] = '0'  # O -> 0
            print(f"Fixed common OCR error: {text[:2]} -> {result[0]}{result[1]}")
        elif text.startswith('J') and len(text) >= 7:
            result[0] = '3'  # J -> 3 at beginning
            print(f"Fixed J at start: {text[0]} -> {result[0]}")
        
        # Apply standard corrections
        for i, char in enumerate(text):
            if i < 2:  # First 2 positions - province code (must be numbers)
                if char in ['O', 'Q', 'D']:
                    result[i] = '0'
                elif char in ['I', 'L', 'T']:
                    result[i] = '1'
                elif char == 'S':
                    result[i] = '5'
                elif char == 'G':
                    result[i] = '6'
                elif char == 'B':
                    result[i] = '8'
                elif char == 'Z':
                    result[i] = '2'
                elif char == 'J':
                    result[i] = '3'
                elif char == 'A' and i == 0:
                    result[i] = '3'
            elif i == 2:  # 3rd position - letter position
                if char == '0':
                    result[i] = 'O'
                elif char == '1':
                    result[i] = 'I'
                elif char == '2':
                    result[i] = 'Z'
                elif char == '5':
                    result[i] = 'S'
                elif char == '6':
                    result[i] = 'G'
                elif char == '8':
                    result[i] = 'B'
            else:  # Later positions - likely numbers
                if char in ['O', 'Q', 'D']:
                    result[i] = '0'
                elif char in ['I', 'L', 'T']:
                    result[i] = '1'
                elif char == 'S':
                    result[i] = '5'
                elif char == 'G':
                    result[i] = '6'
                elif char == 'B':
                    result[i] = '8'
                elif char == 'Z':
                    result[i] = '2'
        
        return ''.join(result)
    
    def detect_plate_format(self, text: str) -> Dict[str, Any]:
        """Detect Vietnamese plate format and extract components"""
        if not text or len(text) < 4:
            return {'valid': False, 'format': None, 'components': None}
        
        # Try each pattern
        for i, pattern in enumerate(self.plate_patterns):
            match = re.match(pattern, text)
            if match:
                groups = match.groups()
                
                # Validate components based on format
                if i == 0:  # Standard format: 30F54504
                    province, letter, numbers = groups
                    
                    if (province in self.province_codes and 
                        letter in self.valid_letters and
                        len(numbers) in [4, 5] and
                        numbers.isdigit()):
                        return {
                            'valid': True,
                            'format': 'standard',
                            'components': {
                                'province': province,
                                'letter': letter,
                                'numbers': numbers
                            }
                        }
                
                elif i == 1:  # Double letter format: 30AB1234
                    province, letters, numbers = groups
                    if (province in self.province_codes and 
                        all(l in self.valid_letters for l in letters) and
                        len(numbers) == 4 and
                        numbers.isdigit()):
                        return {
                            'valid': True,
                            'format': 'double_letter',
                            'components': {
                                'province': province,
                                'letters': letters,
                                'numbers': numbers
                            }
                        }
                
                elif i == 2:  # Special format: LD12345
                    letters, numbers = groups
                    if (letters in self.special_codes and 
                        len(numbers) in [4, 5] and
                        numbers.isdigit()):
                        return {
                            'valid': True,
                            'format': 'special',
                            'components': {
                                'letters': letters,
                                'numbers': numbers
                            }
                        }
        
        return {'valid': False, 'format': None, 'components': None}
    
    def format_plate(self, components: Dict[str, str], format_type: str, is_double_line: bool = False) -> str:
        """Format plate according to Vietnamese standards"""
        
        if format_type == 'standard':
            province = components['province']
            letter = components['letter']
            numbers = components['numbers']
            
            if is_double_line:
                return f"{province}{letter}\n{numbers}"
            else:
                # Split numbers: XXXX -> XX.XX or XXXXX -> XXX.XX
                if len(numbers) == 4:
                    return f"{province}{letter}-{numbers[:2]}.{numbers[2:]}"
                else:  # len == 5
                    return f"{province}{letter}-{numbers[:3]}.{numbers[3:]}"
        
        elif format_type == 'double_letter':
            province = components['province']
            letters = components['letters']
            numbers = components['numbers']
            
            if is_double_line:
                return f"{province}{letters}\n{numbers}"
            else:
                return f"{province}{letters}-{numbers[:2]}.{numbers[2:]}"
        
        elif format_type == 'special':
            letters = components['letters']
            numbers = components['numbers']
            
            if is_double_line:
                return f"{letters}\n{numbers}"
            else:
                if len(numbers) == 4:
                    return f"{letters}-{numbers[:2]}.{numbers[2:]}"
                else:  # len == 5
                    return f"{letters}-{numbers[:3]}.{numbers[3:]}"
        
        return ""
    
    def normalize_plate(self, raw_text: str, is_double_line: bool = False) -> Dict[str, Any]:
        """Normalize Vietnamese license plate - FIXED"""
        
        result = {
            'original': raw_text,
            'normalized': "",
            'is_valid': False,
            'format_type': None,
            'confidence_boost': 0.0,
            'corrections_applied': []
        }
        
        if not raw_text:
            return result
        
        # Step 1: Clean text
        cleaned = self.clean_ocr_text(raw_text)
        if cleaned != raw_text:
            result['corrections_applied'].append('cleaning')
        
        # Step 2: Apply OCR corrections
        corrected = self.apply_ocr_corrections(cleaned)
        if corrected != cleaned:
            result['corrections_applied'].append('ocr_correction')
            print(f"OCR correction applied: '{cleaned}' -> '{corrected}'")
        
        # Step 3: Detect format
        format_info = self.detect_plate_format(corrected)
        
        if format_info['valid']:
            result['is_valid'] = True
            result['format_type'] = format_info['format']
            
            # Step 4: Format according to standards
            normalized = self.format_plate(
                format_info['components'], 
                format_info['format'], 
                is_double_line
            )
            result['normalized'] = normalized
            
            # Confidence boost based on corrections and format matching
            if result['corrections_applied']:
                result['confidence_boost'] = 0.1 * len(result['corrections_applied'])
            if format_info['format'] in ['standard', 'double_letter']:
                result['confidence_boost'] += 0.2
        
        return result

class PlateOCRProcessor:
    """Enhanced Vietnamese license plate OCR processor"""
    
    def __init__(self, config: OCRConfig = None):
        self.config = config or OCRConfig()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize components
        self.paddle_ocr = self._init_paddleocr()
        self.upsampler = self._init_esrgan()
        self.enhancer = self.upsampler  # Alias for compatibility
        self.normalizer = VietnamesePlateNormalizer()
        
        # Caching
        self.cache = {} if self.config.ENABLE_CACHING else None
        self.cache_times = {} if self.config.ENABLE_CACHING else None
        
        # Stats
        self.stats = {
            'processed': 0, 
            'valid': 0, 
            'cache_hits': 0,
            'ocr_count': 0,
            'enhancer_enhancement_count': 0,
            'normalization_improved': 0
        }
        
        logger.info("✅ Enhanced Vietnamese Plate OCR Processor initialized")
    
    def _init_paddleocr(self):
        """Initialize PaddleOCR"""
        if not PADDLEOCR_AVAILABLE:
            logger.warning("PaddleOCR not available")
            return None
        try:
            return PaddleOCR(
                use_angle_cls=True,
                lang='en',  # Use English for alphanumeric
                use_gpu=torch.cuda.is_available(),
                show_log=False,
                det_limit_side_len=320,
                drop_score=0.2,  # Lower threshold for better detection
                det_db_thresh=0.3,
                det_db_box_thresh=0.5,
            )
        except Exception as e:
            logger.error(f"PaddleOCR init error: {e}")
            return None
    
    def _init_esrgan(self):
        """Initialize Real-ESRGAN"""
        if not self.config.USE_REAL_ESRGAN or not ESRGAN_AVAILABLE:
            return None
        
        try:
            model_path = "weights/RealESRGAN_x4plus.pth"
            if not os.path.exists(model_path):
                os.makedirs("weights", exist_ok=True)
                logger.info("Real-ESRGAN model not found, using without enhancement")
                return None
            
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
            self.stats['enhancer_enhancement_count'] += 1
            return enhanced
        except Exception as e:
            logger.error(f"Enhancement error: {e}")
            return image
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Advanced image preprocessing for Vietnamese plates"""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Adaptive histogram equalization
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Denoising
            enhanced = cv2.fastNlMeansDenoising(enhanced, h=10)
            
            # Sharpening kernel
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)
            
            # Ensure minimum size for OCR
            h, w = enhanced.shape
            min_height, min_width = 32, 120
            if h < min_height or w < min_width:
                scale_h = min_height / h if h < min_height else 1
                scale_w = min_width / w if w < min_width else 1
                scale = max(scale_h, scale_w)
                
                new_h, new_w = int(h * scale), int(w * scale)
                enhanced = cv2.resize(enhanced, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            # Convert back to BGR for PaddleOCR
            return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            
        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            return image
    
    def extract_text_with_confidence(self, ocr_result: List) -> List[Tuple[str, float, List]]:
        """Extract text, confidence and bbox from OCR result"""
        results = []
        
        if not ocr_result or not ocr_result[0]:
            return results
        
        for line in ocr_result[0]:
            if len(line) >= 2:
                bbox, (text, conf) = line[0], line[1]
                if conf >= self.config.OCR_MIN_CONF:
                    # Clean text
                    clean_text = self.normalizer.clean_ocr_text(text)
                    if len(clean_text) >= 2:  # Minimum length
                        results.append((clean_text, conf, bbox))
        
        return results
    
    def detect_plate_layout(self, ocr_results: List) -> Tuple[str, bool]:
        """Detect if plate is single or double line and combine text"""
        text_boxes = self.extract_text_with_confidence(ocr_results)
        
        if not text_boxes:
            return "", False
        
        if len(text_boxes) == 1:
            return text_boxes[0][0], False
        
        # Sort by Y coordinate (top to bottom)
        text_boxes.sort(key=lambda x: sum(point[1] for point in x[2]) / len(x[2]))
        
        # Check Y coordinate difference to determine if double line
        if len(text_boxes) >= 2:
            first_y = sum(point[1] for point in text_boxes[0][2]) / len(text_boxes[0][2])
            last_y = sum(point[1] for point in text_boxes[-1][2]) / len(text_boxes[-1][2])
            y_diff = abs(last_y - first_y)
            
            # Estimate image height from bounding boxes
            all_y_coords = []
            for _, _, bbox in text_boxes:
                all_y_coords.extend([point[1] for point in bbox])
            
            if all_y_coords:
                image_height = max(all_y_coords) - min(all_y_coords)
                
                # If Y difference is > 25% of image height, consider as double line
                if y_diff > image_height * 0.25:
                    combined_text = ''.join([box[0] for box in text_boxes])
                    return combined_text, True
        
        # Single line - use highest confidence text
        best_text = max(text_boxes, key=lambda x: x[1])[0]
        return best_text, False
    
    def ocr_with_multiple_strategies(self, image: np.ndarray) -> Tuple[str, float, bool, Dict]:
        """Perform OCR with multiple strategies and return best result"""
        if self.paddle_ocr is None:
            return "", 0.0, False, {}
        
        self.stats['ocr_count'] += 1
        
        strategies = [
            ("original", image),
            ("preprocessed", self.preprocess_image(image))
        ]
        
        # Add enhancement if available
        if self.upsampler:
            try:
                enhanced = self.enhance_image(image)
                if not np.array_equal(enhanced, image):
                    strategies.append(("enhanced", enhanced))
            except Exception as e:
                logger.error(f"Enhancement failed: {e}")
        
        best_result = ""
        best_confidence = 0.0
        best_is_double_line = False
        best_details = {}
        
        for strategy_name, processed_image in strategies:
            try:
                # Perform OCR
                ocr_result = self.paddle_ocr.ocr(processed_image, cls=True)
                
                # Extract and combine text
                combined_text, is_double_line = self.detect_plate_layout(ocr_result)
                
                if combined_text:
                    # Calculate average confidence
                    text_boxes = self.extract_text_with_confidence(ocr_result)
                    if text_boxes:
                        avg_conf = sum(conf for _, conf, _ in text_boxes) / len(text_boxes)
                        
                        # Apply strategy bonuses/penalties
                        strategy_multiplier = {
                            "original": 1.0,
                            "preprocessed": 1.05,  # Slight bonus for preprocessing
                            "enhanced": 1.1       # Bonus for enhancement
                        }.get(strategy_name, 1.0)
                        
                        adjusted_conf = avg_conf * strategy_multiplier
                        
                        # Bonus for reasonable text length
                        if 5 <= len(combined_text) <= 10:
                            adjusted_conf *= 1.05
                        
                        if adjusted_conf > best_confidence:
                            best_result = combined_text
                            best_confidence = adjusted_conf
                            best_is_double_line = is_double_line
                            best_details = {
                                'strategy': strategy_name,
                                'raw_confidence': avg_conf,
                                'adjusted_confidence': adjusted_conf,
                                'text_boxes_count': len(text_boxes)
                            }
                
            except Exception as e:
                logger.error(f"OCR error in strategy '{strategy_name}': {e}")
                continue
        
        return best_result, best_confidence, best_is_double_line, best_details
    
    def process_plate(self, plate_image: np.ndarray, save_path: str = None) -> Dict[str, Any]:
        """Enhanced plate processing with Vietnamese normalization"""
        self.stats['processed'] += 1
        
        # Initialize result
        result = {
            'text': None,
            'confidence': 0.0,
            'validated_text': None,
            'is_valid': False,
            'is_double_line': False,
            'processing_time': 0.0,
            'save_path': save_path,
            'from_cache': False,
            'normalization_details': {},
            'ocr_details': {}
        }
        
        if plate_image is None or plate_image.size == 0:
            return result
        
        start_time = time.time()
        
        # Check cache first
        cache_key = self._get_cache_key(plate_image)
        cached_result = self._check_cache(cache_key)
        if cached_result:
            cached_result['from_cache'] = True
            return cached_result
        
        # Save original image if requested
        if save_path:
            try:
                cv2.imwrite(save_path, plate_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                result['save_path'] = save_path
            except Exception as e:
                logger.error(f"Failed to save image: {e}")
        
        # Perform OCR with multiple strategies
        raw_text, confidence, is_double_line, ocr_details = self.ocr_with_multiple_strategies(plate_image)
        
        result['ocr_details'] = ocr_details
        
        if raw_text and confidence > 0:
            result['text'] = raw_text
            result['confidence'] = confidence
            result['is_double_line'] = is_double_line
            
            # Normalize the plate text
            normalization_result = self.normalizer.normalize_plate(raw_text, is_double_line)
            result['normalization_details'] = normalization_result
            
            if normalization_result['is_valid']:
                result['validated_text'] = normalization_result['normalized']
                result['is_valid'] = True
                
                # Apply confidence boost from normalization
                boost = normalization_result['confidence_boost']
                result['confidence'] = min(1.0, confidence + boost)
                
                self.stats['valid'] += 1
                
                if normalization_result['corrections_applied']:
                    self.stats['normalization_improved'] += 1
                    logger.debug(f"Normalization improved: {raw_text} → {result['validated_text']}")
        
        result['processing_time'] = time.time() - start_time
        
        # Cache the result
        self._save_cache(cache_key, result.copy())
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics"""
        stats = self.stats.copy()
        
        if stats['processed'] > 0:
            stats['success_rate'] = (stats['valid'] / stats['processed']) * 100
            stats['normalization_improvement_rate'] = (stats['normalization_improved'] / stats['processed']) * 100
        else:
            stats['success_rate'] = 0.0
            stats['normalization_improvement_rate'] = 0.0
        
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
        logger.info("✅ Enhanced OCR Processor cleanup completed")

# Factory function
def create_plate_ocr_processor(use_esrgan: bool = True, enable_caching: bool = True) -> PlateOCRProcessor:
    """Create enhanced Vietnamese plate OCR processor"""
    config = OCRConfig()
    config.USE_REAL_ESRGAN = use_esrgan
    config.ENABLE_CACHING = enable_caching
    
    return PlateOCRProcessor(config)

# Backward compatibility
VietnamesePlateProcessor = PlateOCRProcessor
PlateOCRConfig = OCRConfig

# Example usage and testing
if __name__ == "__main__":
    processor = create_plate_ocr_processor()
    
    # Test normalization with problematic cases from server logs
    test_cases = [
        "30F54504",    # Should become 30F-545.04
        "30H67885",    # Should become 30H-678.85
        "30L12206",    # Should become 30L-122.06
        "J0H79007",    # Should be corrected to 30H-790.07
        "30F\n54504",  # Double line should become 30F-545.04
        "51A12345",    # Standard 5-digit format
        "29AB1234",    # Double letter format
        "LD12345",     # Special administrative format
    ]
    
    normalizer = VietnamesePlateNormalizer()
    
    print("=== Testing Vietnamese Plate Normalization (FIXED) ===")
    for test_text in test_cases:
        print(f"\nInput: '{test_text}'")
        result = normalizer.normalize_plate(test_text, is_double_line=('\n' in test_text))
        print(f"Normalized: '{result['normalized']}'")
        print(f"Valid: {result['is_valid']}")
        print(f"Format: {result['format_type']}")
        print(f"Corrections: {result['corrections_applied']}")
        print(f"Confidence boost: +{result['confidence_boost']:.2f}")
    
    # Test with actual image if available
    test_image = "test_plate.jpg"
    if os.path.exists(test_image):
        print(f"\n=== Testing with image: {test_image} ===")
        image = cv2.imread(test_image)
        result = processor.process_plate(image)
        
        print(f"Raw OCR: '{result['text']}'")
        print(f"Validated: '{result['validated_text']}'")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Valid: {result['is_valid']}")
        print(f"Double line: {result['is_double_line']}")
        print(f"Processing time: {result['processing_time']:.3f}s")
        
        if result['normalization_details']:
            norm = result['normalization_details']
            print(f"Normalization corrections: {norm['corrections_applied']}")
            print(f"Format detected: {norm['format_type']}")
        
        # Display stats
        stats = processor.get_stats()
        print(f"\n=== OCR Statistics ===")
        print(f"Processed: {stats['processed']}")
        print(f"Valid: {stats['valid']}")
        print(f"Success rate: {stats['success_rate']:.1f}%")
        print(f"Normalization improvements: {stats['normalization_improvement_rate']:.1f}%")
        
        processor.cleanup()
    else:
        print(f"\nTest image '{test_image}' not found.")
        print("Place a license plate image to test full OCR pipeline.")
        
    print("\n✅ Testing completed!")