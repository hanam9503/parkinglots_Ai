"""
License Plate Detector - Enhanced YOLO Plate Detection
Phát hiện biển số xe với YOLO và xử lý nâng cao
"""

import cv2
import numpy as np
import logging
import time
from typing import List, Tuple, Optional, Dict, Any
from collections import deque
import threading
from dataclasses import dataclass

from config.settings import config
from core.models import DetectionResult
from core.exceptions import PlateDetectionException, ModelInferenceException
from core.constants import QUALITY_THRESHOLDS, PERFORMANCE_THRESHOLDS

logger = logging.getLogger(__name__)

@dataclass
class PlateDetection:
    """License plate detection result"""
    bbox: Tuple[int, int, int, int]
    confidence: float
    plate_crop: np.ndarray
    relative_bbox: Tuple[int, int, int, int]  # Relative to vehicle crop
    quality_score: float
    area: int
    aspect_ratio: float
    
    def is_valid(self) -> bool:
        """Check if plate detection is valid"""
        return (
            self.confidence >= config.PLATE_CONF and
            self.quality_score >= QUALITY_THRESHOLDS["MIN_IMAGE_QUALITY"] and
            QUALITY_THRESHOLDS["MIN_PLATE_AREA"] <= self.area <= QUALITY_THRESHOLDS["MAX_PLATE_AREA"] and
            QUALITY_THRESHOLDS["MIN_ASPECT_RATIO"] <= self.aspect_ratio <= QUALITY_THRESHOLDS["MAX_ASPECT_RATIO"]
        )

class EnhancedPlateDetector:
    """Enhanced license plate detector with quality assessment"""
    
    def __init__(self, model):
        self.model = model
        self.device = next(model.model.parameters()).device
        
        # Detection settings
        self.conf_threshold = config.PLATE_CONF
        self.img_size = config.PLATE_IMG_SIZE
        self.max_det = 20  # Maximum plates per vehicle
        
        # Quality assessment
        self.min_plate_area = QUALITY_THRESHOLDS["MIN_PLATE_AREA"]
        self.max_plate_area = QUALITY_THRESHOLDS["MAX_PLATE_AREA"]
        self.min_aspect_ratio = QUALITY_THRESHOLDS["MIN_ASPECT_RATIO"]
        self.max_aspect_ratio = QUALITY_THRESHOLDS["MAX_ASPECT_RATIO"]
        
        # Performance monitoring
        self.detection_times = deque(maxlen=100)
        self.total_detections = 0
        self.valid_detections = 0
        self.frame_count = 0
        
        # Thread safety
        self.lock = threading.Lock()
        
        logger.info(f"🔢 Enhanced Plate Detector initialized")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Confidence threshold: {self.conf_threshold}")
        logger.info(f"   Image size: {self.img_size}")
        logger.info(f"   Plate area range: {self.min_plate_area}-{self.max_plate_area} pixels")
        logger.info(f"   Aspect ratio range: {self.min_aspect_ratio:.1f}-{self.max_aspect_ratio:.1f}")
    
    def detect_plates_in_vehicle(self, vehicle_crop: np.ndarray, vehicle_bbox: Tuple[int, int, int, int] = None) -> List[PlateDetection]:
        """
        Detect license plates within a vehicle crop
        
        Args:
            vehicle_crop: Cropped vehicle image
            vehicle_bbox: Original vehicle bounding box (for absolute coordinates)
            
        Returns:
            List of PlateDetection objects
        """
        start_time = time.time()
        
        try:
            with self.lock:
                self.frame_count += 1
            
            # Validate input
            if vehicle_crop is None or vehicle_crop.size == 0:
                raise PlateDetectionException("Invalid vehicle crop")
            
            # Preprocess vehicle crop for plate detection
            processed_crop = self._preprocess_vehicle_crop(vehicle_crop)
            
            # Run YOLO inference
            results = self._run_inference(processed_crop)
            
            # Process detections
            plate_detections = self._process_plate_detections(results, vehicle_crop, vehicle_bbox)
            
            # Assess and filter by quality
            valid_plates = self._filter_by_quality(plate_detections)
            
            # Update statistics
            detection_time = time.time() - start_time
            self.detection_times.append(detection_time)
            
            with self.lock:
                self.total_detections += len(plate_detections)
                self.valid_detections += len(valid_plates)
            
            # Log performance periodically
            if self.frame_count % 50 == 0:
                self._log_performance_stats()
            
            logger.debug(f"Detected {len(valid_plates)}/{len(plate_detections)} valid plates")
            return valid_plates
            
        except Exception as e:
            logger.error(f"Plate detection failed: {e}")
            raise PlateDetectionException(str(e), {"vehicle_bbox": vehicle_bbox})
    
    def _preprocess_vehicle_crop(self, vehicle_crop: np.ndarray) -> np.ndarray:
        """Preprocess vehicle crop for better plate detection"""
        try:
            processed = vehicle_crop.copy()
            
            # Resize if too small for effective detection
            height, width = processed.shape[:2]
            min_size = max(self.img_size // 2, 320)
            
            if min(height, width) < min_size:
                scale = min_size / min(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                processed = cv2.resize(processed, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                logger.debug(f"Upscaled vehicle crop: {width}x{height} -> {new_width}x{new_height}")
            
            # Enhance for better plate visibility
            if config.ENABLE_SMART_ENHANCEMENT:
                processed = self._enhance_for_plate_detection(processed)
            
            return processed
            
        except Exception as e:
            logger.error(f"Vehicle crop preprocessing failed: {e}")
            return vehicle_crop
    
    def _enhance_for_plate_detection(self, image: np.ndarray) -> np.ndarray:
        """Apply enhancements specifically for plate detection"""
        try:
            # Convert to LAB color space for better enhancement
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel to improve contrast
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge channels and convert back to BGR
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # Slight sharpening to improve text clarity
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)
            
            # Ensure values are in valid range
            enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
            
            return enhanced
            
        except Exception as e:
            logger.warning(f"Plate detection enhancement failed: {e}")
            return image
    
    def _run_inference(self, image: np.ndarray):
        """Run YOLO inference for plate detection"""
        try:
            results = self.model(
                image,
                conf=self.conf_threshold,
                verbose=False,
                imgsz=self.img_size,
                max_det=self.max_det,
                device=self.device
            )
            
            return results[0] if results else None
            
        except Exception as e:
            logger.error(f"Plate YOLO inference failed: {e}")
            raise ModelInferenceException("plate_detector", str(e))
    
    def _process_plate_detections(self, results, original_crop: np.ndarray, vehicle_bbox: Tuple[int, int, int, int] = None) -> List[PlateDetection]:
        """Process YOLO plate detection results"""
        detections = []
        
        if results is None or len(results.boxes) == 0:
            return detections
        
        original_height, original_width = original_crop.shape[:2]
        
        for det in results.boxes:
            try:
                # Extract detection info
                bbox = det.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                conf = float(det.conf[0])
                
                # Convert to integer coordinates
                x1, y1, x2, y2 = map(int, bbox)
                
                # Clip to image bounds
                x1 = max(0, min(x1, original_width - 1))
                y1 = max(0, min(y1, original_height - 1))
                x2 = max(x1 + 1, min(x2, original_width))
                y2 = max(y1 + 1, min(y2, original_height))
                
                # Extract plate crop
                plate_crop = original_crop[y1:y2, x1:x2]
                
                if plate_crop.size == 0:
                    continue
                
                # Calculate quality metrics
                area = (x2 - x1) * (y2 - y1)
                aspect_ratio = (x2 - x1) / (y2 - y1) if y2 > y1 else 0
                quality_score = self._assess_plate_quality(plate_crop, conf)
                
                # Convert to absolute coordinates if vehicle bbox provided
                if vehicle_bbox:
                    vx1, vy1, vx2, vy2 = vehicle_bbox
                    abs_x1 = vx1 + x1
                    abs_y1 = vy1 + y1
                    abs_x2 = vx1 + x2
                    abs_y2 = vy1 + y2
                    abs_bbox = (abs_x1, abs_y1, abs_x2, abs_y2)
                else:
                    abs_bbox = (x1, y1, x2, y2)
                
                # Create plate detection
                plate_detection = PlateDetection(
                    bbox=abs_bbox,
                    confidence=conf,
                    plate_crop=plate_crop,
                    relative_bbox=(x1, y1, x2, y2),
                    quality_score=quality_score,
                    area=area,
                    aspect_ratio=aspect_ratio
                )
                
                detections.append(plate_detection)
                
            except Exception as e:
                logger.warning(f"Error processing plate detection: {e}")
                continue
        
        # Sort by confidence * quality score
        detections.sort(key=lambda d: d.confidence * d.quality_score, reverse=True)
        
        return detections
    
    def _assess_plate_quality(self, plate_crop: np.ndarray, confidence: float) -> float:
        """Assess the quality of a detected plate crop"""
        try:
            if plate_crop.size == 0:
                return 0.0
            
            quality_factors = []
            
            # 1. Confidence factor (30% weight)
            conf_factor = min(confidence / 0.9, 1.0)  # Normalize to 0.9 as max
            quality_factors.append(conf_factor * 0.3)
            
            # 2. Sharpness/blur assessment (25% weight)
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_factor = min(laplacian_var / 100.0, 1.0)  # Normalize
            quality_factors.append(sharpness_factor * 0.25)
            
            # 3. Contrast assessment (20% weight)
            contrast = gray.std()
            contrast_factor = min(contrast / 50.0, 1.0)  # Normalize
            quality_factors.append(contrast_factor * 0.20)
            
            # 4. Size adequacy (15% weight)
            height, width = plate_crop.shape[:2]
            size_factor = 1.0
            if min(height, width) < 20:
                size_factor = 0.3
            elif min(height, width) < 32:
                size_factor = 0.6
            elif min(height, width) < 48:
                size_factor = 0.8
            quality_factors.append(size_factor * 0.15)
            
            # 5. Aspect ratio appropriateness (10% weight)
            aspect_ratio = width / height if height > 0 else 0
            if 2.0 <= aspect_ratio <= 5.0:  # Typical plate aspect ratios
                aspect_factor = 1.0
            elif 1.5 <= aspect_ratio < 2.0 or 5.0 < aspect_ratio <= 6.0:
                aspect_factor = 0.7
            else:
                aspect_factor = 0.3
            quality_factors.append(aspect_factor * 0.10)
            
            # Combine all factors
            total_quality = sum(quality_factors)
            
            return min(total_quality, 1.0)
            
        except Exception as e:
            logger.warning(f"Quality assessment failed: {e}")
            return confidence * 0.5  # Fallback to half confidence
    
    def _filter_by_quality(self, plate_detections: List[PlateDetection]) -> List[PlateDetection]:
        """Filter plate detections by quality thresholds"""
        valid_plates = []
        
        for plate in plate_detections:
            # Basic validation
            if not plate.is_valid():
                logger.debug(f"Plate rejected - basic validation failed: conf={plate.confidence:.3f}, "
                           f"quality={plate.quality_score:.3f}, area={plate.area}, ratio={plate.aspect_ratio:.2f}")
                continue
            
            # Additional quality checks
            if self._passes_advanced_quality_checks(plate):
                valid_plates.append(plate)
            else:
                logger.debug(f"Plate rejected - advanced quality checks failed")
        
        return valid_plates
    
    def _passes_advanced_quality_checks(self, plate: PlateDetection) -> bool:
        """Perform advanced quality checks on plate detection"""
        try:
            # Check for minimum text-like content
            gray = cv2.cvtColor(plate.plate_crop, cv2.COLOR_BGR2GRAY)
            
            # Edge density check (plates should have good edge content)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            if edge_density < 0.05:  # Too few edges, probably not a plate
                return False
            
            # Horizontal line detection (plates typically have horizontal structure)
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
            horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
            horizontal_density = np.sum(horizontal_lines > 0) / horizontal_lines.size
            
            if horizontal_density < 0.01:  # No horizontal structure
                return False
            
            # Color uniformity check (license plates typically have uniform background)
            color_std = np.std(gray)
            if color_std < 10:  # Too uniform, might be a solid surface
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Advanced quality check failed: {e}")
            return True  # Default to accepting if check fails
    
    def detect_multiple_vehicles(self, vehicle_crops: List[Tuple[np.ndarray, Tuple[int, int, int, int]]]) -> Dict[int, List[PlateDetection]]:
        """Detect plates in multiple vehicle crops efficiently"""
        results = {}
        
        for idx, (vehicle_crop, vehicle_bbox) in enumerate(vehicle_crops):
            try:
                plates = self.detect_plates_in_vehicle(vehicle_crop, vehicle_bbox)
                if plates:
                    results[idx] = plates
            except Exception as e:
                logger.error(f"Failed to detect plates in vehicle {idx}: {e}")
                results[idx] = []
        
        return results
    
    def get_best_plate(self, plate_detections: List[PlateDetection]) -> Optional[PlateDetection]:
        """Get the best plate detection from a list"""
        if not plate_detections:
            return None
        
        # Score plates by confidence * quality * size factor
        def score_plate(plate: PlateDetection) -> float:
            size_factor = min(plate.area / 5000.0, 1.0)  # Normalize area
            return plate.confidence * plate.quality_score * (0.7 + 0.3 * size_factor)
        
        best_plate = max(plate_detections, key=score_plate)
        return best_plate
    
    def _log_performance_stats(self):
        """Log performance statistics"""
        if not self.detection_times:
            return
        
        avg_time = np.mean(self.detection_times)
        max_time = np.max(self.detection_times)
        min_time = np.min(self.detection_times)
        
        with self.lock:
            total_detections = self.total_detections
            valid_detections = self.valid_detections
            frame_count = self.frame_count
        
        detection_rate = (valid_detections / max(1, total_detections)) * 100
        avg_plates_per_vehicle = total_detections / max(1, frame_count)
        
        logger.info(f"🔢 Plate Detection Performance (last 50 vehicles):")
        logger.info(f"   Average time: {avg_time*1000:.1f}ms")
        logger.info(f"   Min/Max time: {min_time*1000:.1f}/{max_time*1000:.1f}ms")
        logger.info(f"   Detection rate: {detection_rate:.1f}% valid")
        logger.info(f"   Avg plates/vehicle: {avg_plates_per_vehicle:.1f}")
    
    def save_plate_crop(self, plate_crop: np.ndarray, filename: str, enhance: bool = True) -> bool:
        """Save plate crop with optional enhancement"""
        try:
            # Create directory if it doesn't exist
            import os
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            # Apply enhancement if requested
            if enhance:
                # Resize to standard size if too small
                height, width = plate_crop.shape[:2]
                if min(height, width) < 48:
                    scale = 48 / min(height, width)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    plate_crop = cv2.resize(plate_crop, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                
                # Enhance contrast
                plate_crop = cv2.convertScaleAbs(plate_crop, alpha=1.2, beta=10)
            
            # Save with high quality
            cv2.imwrite(filename, plate_crop, [cv2.IMWRITE_JPEG_QUALITY, config.IMAGE_QUALITY])
            return True
            
        except Exception as e:
            logger.error(f"Failed to save plate crop to {filename}: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive detection statistics"""
        with self.lock:
            stats = {
                'total_vehicles_processed': self.frame_count,
                'total_plate_detections': self.total_detections,
                'valid_plate_detections': self.valid_detections,
                'detection_rate': (self.valid_detections / max(1, self.total_detections)) * 100,
                'avg_plates_per_vehicle': self.total_detections / max(1, self.frame_count)
            }
        
        # Performance stats
        if self.detection_times:
            detection_times_array = np.array(self.detection_times)
            stats.update({
                'avg_detection_time': float(np.mean(detection_times_array)),
                'max_detection_time': float(np.max(detection_times_array)),
                'min_detection_time': float(np.min(detection_times_array)),
                'detection_fps': float(1.0 / np.mean(detection_times_array)) if np.mean(detection_times_array) > 0 else 0
            })
        
        return stats
    
    def reset_statistics(self):
        """Reset detection statistics"""
        with self.lock:
            self.detection_times.clear()
            self.total_detections = 0
            self.valid_detections = 0
            self.frame_count = 0
            logger.info("🔄 Plate detector statistics reset")
    
    def cleanup(self):
        """Clean up detector resources"""
        logger.info("🧹 Cleaning up plate detector...")
        
        with self.lock:
            self.detection_times.clear()
            self.total_detections = 0
            self.valid_detections = 0
            self.frame_count = 0
        
        logger.info("✅ Plate detector cleanup completed")

# Utility functions
def extract_plate_regions(vehicle_crop: np.ndarray, min_area: int = 100) -> List[np.ndarray]:
    """Extract potential plate regions using image processing techniques"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Find edges
        edges = cv2.Canny(filtered, 30, 200)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        plate_regions = []
        
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size and aspect ratio
            area = w * h
            aspect_ratio = w / h if h > 0 else 0
            
            if (area >= min_area and 
                2.0 <= aspect_ratio <= 5.0 and
                w >= 50 and h >= 15):
                
                # Extract region
                plate_region = vehicle_crop[y:y+h, x:x+w]
                if plate_region.size > 0:
                    plate_regions.append(plate_region)
        
        return plate_regions
        
    except Exception as e:
        logger.error(f"Failed to extract plate regions: {e}")
        return []

def assess_plate_readability(plate_crop: np.ndarray) -> float:
    """Assess how readable a plate crop is for OCR"""
    try:
        if plate_crop is None or plate_crop.size == 0:
            return 0.0
        
        # Convert to grayscale
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        
        # Calculate various quality metrics
        metrics = []
        
        # 1. Sharpness (Laplacian variance)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        metrics.append(min(sharpness / 100.0, 1.0) * 0.3)
        
        # 2. Contrast (standard deviation)
        contrast = gray.std()
        metrics.append(min(contrast / 50.0, 1.0) * 0.25)
        
        # 3. Text-like edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        metrics.append(min(edge_density / 0.1, 1.0) * 0.2)
        
        # 4. Size adequacy
        height, width = gray.shape
        size_score = 1.0 if min(height, width) >= 32 else min(height, width) / 32.0
        metrics.append(size_score * 0.15)
        
        # 5. Aspect ratio
        aspect_ratio = width / height if height > 0 else 0
        if 2.0 <= aspect_ratio <= 5.0:
            aspect_score = 1.0
        else:
            aspect_score = max(0.3, 1.0 - abs(aspect_ratio - 3.5) / 3.5)
        metrics.append(aspect_score * 0.1)
        
        return sum(metrics)
        
    except Exception as e:
        logger.error(f"Readability assessment failed: {e}")
        return 0.0

# Factory function
def create_plate_detector(model):
    """Create enhanced plate detector instance"""
    return EnhancedPlateDetector(model)

# Example usage
if __name__ == "__main__":
    # This would normally be run as part of the main system
    print("Plate Detector module - use within main parking system")