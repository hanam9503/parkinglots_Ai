"""
Vietnamese License Plate Validator
Xác thực và chuẩn hóa biển số xe Việt Nam
"""

import re
import logging
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass

from core.constants import (
    VIETNAMESE_PROVINCE_CODES, 
    SPECIAL_PLATE_CODES, 
    OCR_CORRECTIONS,
    REGEX_PATTERNS
)

logger = logging.getLogger(__name__)

@dataclass
class PlateValidationResult:
    """Kết quả xác thực biển số"""
    original_text: str
    cleaned_text: str
    formatted_text: str
    is_valid: bool
    confidence_score: float
    province_code: Optional[str] = None
    province_name: Optional[str] = None
    plate_type: str = "unknown"
    validation_details: Dict = None

class VietnamesePlateValidator:
    """Enhanced Vietnamese license plate validator với pattern matching nâng cao"""
    
    def __init__(self):
        # Vietnamese province codes
        self.province_codes = set(VIETNAMESE_PROVINCE_CODES.keys())
        self.special_codes = set(SPECIAL_PLATE_CODES.keys())
        
        # OCR correction mapping
        self.ocr_corrections = OCR_CORRECTIONS.copy()
        
        # Plate type patterns
        self.plate_patterns = {
            'standard_new': {
                'pattern': REGEX_PATTERNS['PLATE_STANDARD'],
                'description': 'Standard format: 30A-12345',
                'example': '30A-12345'
            },
            'standard_new2': {
                'pattern': REGEX_PATTERNS['PLATE_NEW'], 
                'description': 'New format: 30AA-1234',
                'example': '30AA-1234'
            },
            'standard_old': {
                'pattern': REGEX_PATTERNS['PLATE_OLD'],
                'description': 'Old format: 123A-456',
                'example': '123A-456'
            },
            'special': {
                'pattern': REGEX_PATTERNS['PLATE_SPECIAL'],
                'description': 'Special format: LD-12345',
                'example': 'LD-12345'
            }
        }
        
        # Common OCR error patterns
        self.error_patterns = [
            (r'[IL1]{3,}', 'Too many similar characters'),
            (r'^[^A-Z0-9]*$', 'Only special characters'),
            (r'^(.)\1{4,}', 'Too many repeated characters'),
            (r'[OILSZBGQD]{5,}', 'Too many ambiguous characters')
        ]
        
        # Statistics
        self.validation_stats = {
            'total_validations': 0,
            'valid_plates': 0,
            'by_type': {},
            'common_errors': {}
        }
        
        logger.info("🔍 Vietnamese Plate Validator initialized")
        logger.info(f"   Province codes: {len(self.province_codes)}")
        logger.info(f"   Special codes: {len(self.special_codes)}")
        logger.info(f"   Plate patterns: {len(self.plate_patterns)}")
    
    def validate_plate(self, text: str, apply_corrections: bool = True) -> PlateValidationResult:
        """
        Comprehensive validation of Vietnamese license plate
        
        Args:
            text: Raw plate text from OCR
            apply_corrections: Whether to apply OCR corrections
            
        Returns:
            PlateValidationResult object with validation details
        """
        self.validation_stats['total_validations'] += 1
        
        if not text:
            return PlateValidationResult(
                original_text="",
                cleaned_text="",
                formatted_text="",
                is_valid=False,
                confidence_score=0.0,
                validation_details={'error': 'Empty text'}
            )
        
        original_text = text
        
        # Step 1: Clean OCR text
        cleaned_text = self.clean_ocr_text(text, apply_corrections)
        
        # Step 2: Validate cleaned text
        validation_result = self._validate_cleaned_text(cleaned_text)
        
        # Step 3: Format if valid
        formatted_text = ""
        if validation_result['is_valid']:
            formatted_text = self._format_plate(cleaned_text, validation_result['plate_type'])
        
        # Step 4: Calculate confidence score
        confidence_score = self._calculate_confidence_score(
            original_text, cleaned_text, validation_result
        )
        
        # Update statistics
        if validation_result['is_valid']:
            self.validation_stats['valid_plates'] += 1
            plate_type = validation_result['plate_type']
            self.validation_stats['by_type'][plate_type] = \
                self.validation_stats['by_type'].get(plate_type, 0) + 1
        
        return PlateValidationResult(
            original_text=original_text,
            cleaned_text=cleaned_text,
            formatted_text=formatted_text,
            is_valid=validation_result['is_valid'],
            confidence_score=confidence_score,
            province_code=validation_result.get('province_code'),
            province_name=validation_result.get('province_name'),
            plate_type=validation_result.get('plate_type', 'unknown'),
            validation_details=validation_result
        )
    
    def clean_ocr_text(self, text: str, apply_corrections: bool = True) -> str:
        """Advanced OCR text cleaning với context-aware corrections"""
        if not text:
            return ""
        
        # Remove special characters and spaces
        cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        if not apply_corrections:
            return cleaned
        
        # Apply intelligent OCR corrections
        result = ""
        for i, char in enumerate(cleaned):
            if char in self.ocr_corrections:
                # Context-aware correction
                corrected_char = self._apply_contextual_correction(
                    char, cleaned, i
                )
                result += corrected_char
            else:
                result += char
        
        return result
    
    def _apply_contextual_correction(self, char: str, text: str, position: int) -> str:
        """Apply contextual OCR corrections based on surrounding characters"""
        correction = self.ocr_corrections.get(char, char)
        
        # Look at surrounding context
        prev_char = text[position - 1] if position > 0 else None
        next_char = text[position + 1] if position < len(text) - 1 else None
        
        # Context-based rules
        if char in ['O', 'Q', 'D']:
            # Likely to be '0' if surrounded by digits
            if (prev_char and prev_char.isdigit()) or (next_char and next_char.isdigit()):
                return '0'
            # Likely to be 'O' if at start of likely letter position
            elif position <= 2:  # Province code area
                return 'O' if char == 'O' else '0'
        
        elif char in ['I', 'L']:
            # Likely to be '1' if surrounded by digits
            if (prev_char and prev_char.isdigit()) or (next_char and next_char.isdigit()):
                return '1'
            # Likely to be 'I' in letter positions
            elif 2 <= position <= 4:  # Letter area in standard plates
                return 'I'
        
        elif char == 'S':
            # Usually '5' in number contexts
            if (prev_char and prev_char.isdigit()) or (next_char and next_char.isdigit()):
                return '5'
        
        return correction
    
    def _validate_cleaned_text(self, cleaned_text: str) -> Dict:
        """Validate cleaned text against Vietnamese plate patterns"""
        validation_result = {
            'is_valid': False,
            'plate_type': 'unknown',
            'province_code': None,
            'province_name': None,
            'errors': []
        }
        
        if not cleaned_text:
            validation_result['errors'].append('Empty cleaned text')
            return validation_result
        
        # Check basic requirements
        if len(cleaned_text) < 4:
            validation_result['errors'].append('Text too short')
            return validation_result
        
        if len(cleaned_text) > 12:
            validation_result['errors'].append('Text too long')
            return validation_result
        
        # Check for obvious OCR errors
        for pattern, error_msg in self.error_patterns:
            if re.search(pattern, cleaned_text):
                validation_result['errors'].append(error_msg)
                return validation_result
        
        # Must have both letters and numbers
        has_letter = any(c.isalpha() for c in cleaned_text)
        has_number = any(c.isdigit() for c in cleaned_text)
        
        if not (has_letter and has_number):
            validation_result['errors'].append('Must contain both letters and numbers')
            return validation_result
        
        # Try to match against known patterns
        for plate_type, pattern_info in self.plate_patterns.items():
            match = re.match(pattern_info['pattern'], cleaned_text)
            if match:
                validation_result.update(
                    self._validate_pattern_match(match, plate_type, cleaned_text)
                )
                if validation_result['is_valid']:
                    break
        
        return validation_result
    
    def _validate_pattern_match(self, match, plate_type: str, text: str) -> Dict:
        """Validate a successful pattern match"""
        groups = match.groups()
        result = {
            'is_valid': False,
            'plate_type': plate_type,
            'province_code': None,
            'province_name': None,
            'errors': []
        }
        
        if plate_type in ['standard_new', 'standard_new2', 'standard_old']:
            # Standard plates - validate province code
            province_code = groups[0]
            
            if len(province_code) == 2 and province_code.isdigit():
                if province_code in self.province_codes:
                    result['is_valid'] = True
                    result['province_code'] = province_code
                    result['province_name'] = VIETNAMESE_PROVINCE_CODES[province_code]
                else:
                    result['errors'].append(f'Invalid province code: {province_code}')
            else:
                result['errors'].append(f'Invalid province code format: {province_code}')
        
        elif plate_type == 'special':
            # Special plates - validate special code
            special_code = groups[0]
            
            if special_code in self.special_codes:
                result['is_valid'] = True
                result['province_code'] = special_code
                result['province_name'] = SPECIAL_PLATE_CODES[special_code]
            else:
                result['errors'].append(f'Invalid special code: {special_code}')
        
        return result
    
    def _format_plate(self, text: str, plate_type: str) -> str:
        """Format validated plate text according to Vietnamese standards"""
        try:
            for pattern_info in self.plate_patterns.values():
                match = re.match(pattern_info['pattern'], text)
                if match:
                    groups = match.groups()
                    
                    if len(groups) == 3:
                        # Standard format: XX[X]-XXXXX
                        return f"{groups[0]}{groups[1]}-{groups[2]}"
                    elif len(groups) == 2:
                        # Special format: XX-XXXXX
                        return f"{groups[0]}-{groups[1]}"
            
            # Fallback: try to add dash intelligently
            return self._add_dash_intelligently(text)
            
        except Exception as e:
            logger.warning(f"Formatting failed for {text}: {e}")
            return text
    
    def _add_dash_intelligently(self, text: str) -> str:
        """Add dash to plate text intelligently"""
        # Find the boundary between letters and the final number sequence
        for i in range(len(text) - 1, 0, -1):
            if text[i].isdigit() and not text[i-1].isdigit():
                return f"{text[:i]}-{text[i:]}"
        
        # Fallback: add dash after first 3-4 characters
        split_pos = 3 if len(text) > 6 else len(text) // 2
        return f"{text[:split_pos]}-{text[split_pos:]}"
    
    def _calculate_confidence_score(self, original: str, cleaned: str, validation: Dict) -> float:
        """Calculate confidence score for plate validation"""
        if not validation['is_valid']:
            return 0.0
        
        confidence_factors = []
        
        # 1. Text similarity (how much cleaning was needed)
        if original and cleaned:
            similarity = len(cleaned) / len(original) if original else 0
            similarity_score = min(similarity, 1.0) * 0.25
            confidence_factors.append(similarity_score)
        
        # 2. Pattern match strength
        plate_type = validation.get('plate_type', 'unknown')
        if plate_type in ['standard_new', 'standard_new2']:
            confidence_factors.append(0.3)  # High confidence for standard patterns
        elif plate_type == 'standard_old':
            confidence_factors.append(0.25)  # Medium confidence for old patterns
        elif plate_type == 'special':
            confidence_factors.append(0.2)   # Lower confidence for special plates
        else:
            confidence_factors.append(0.1)   # Low confidence for unknown patterns
        
        # 3. Province/special code validity
        if validation.get('province_code'):
            confidence_factors.append(0.2)
        
        # 4. Text quality factors
        quality_score = self._assess_text_quality(cleaned)
        confidence_factors.append(quality_score * 0.25)
        
        # Combine factors
        total_confidence = sum(confidence_factors)
        return min(total_confidence, 1.0)
    
    def _assess_text_quality(self, text: str) -> float:
        """Assess the quality of the plate text"""
        if not text:
            return 0.0
        
        quality_factors = []
        
        # Length appropriateness
        if 5 <= len(text) <= 9:
            quality_factors.append(1.0)
        elif 4 <= len(text) <= 10:
            quality_factors.append(0.8)
        else:
            quality_factors.append(0.5)
        
        # Character diversity
        unique_chars = len(set(text))
        diversity_score = min(unique_chars / len(text), 1.0)
        quality_factors.append(diversity_score)
        
        # Letter-number balance
        letters = sum(1 for c in text if c.isalpha())
        numbers = sum(1 for c in text if c.isdigit())
        
        if letters > 0 and numbers > 0:
            balance = min(letters, numbers) / max(letters, numbers)
            quality_factors.append(balance)
        else:
            quality_factors.append(0.0)
        
        return sum(quality_factors) / len(quality_factors)
    
    def batch_validate(self, texts: List[str]) -> List[PlateValidationResult]:
        """Batch validate multiple plate texts"""
        results = []
        for text in texts:
            try:
                result = self.validate_plate(text)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch validation failed for '{text}': {e}")
                results.append(PlateValidationResult(
                    original_text=text,
                    cleaned_text="",
                    formatted_text="",
                    is_valid=False,
                    confidence_score=0.0,
                    validation_details={'error': str(e)}
                ))
        
        return results
    
    def get_validation_suggestions(self, text: str) -> List[str]:
        """Get suggestions for invalid plate text"""
        if not text:
            return []
        
        suggestions = []
        cleaned = self.clean_ocr_text(text)
        
        # Try common corrections
        corrections = [
            self._try_length_corrections(cleaned),
            self._try_character_corrections(cleaned),
            self._try_pattern_fixes(cleaned)
        ]
        
        for correction_list in corrections:
            for suggestion in correction_list:
                if suggestion and suggestion != cleaned:
                    validation = self.validate_plate(suggestion)
                    if validation.is_valid:
                        suggestions.append(suggestion)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_suggestions = []
        for suggestion in suggestions:
            if suggestion not in seen:
                seen.add(suggestion)
                unique_suggestions.append(suggestion)
        
        return unique_suggestions[:5]  # Return top 5 suggestions
    
    def _try_length_corrections(self, text: str) -> List[str]:
        """Try to fix length-related issues"""
        suggestions = []
        
        if len(text) < 5:
            # Try adding missing characters
            suggestions.extend([
                text + '0',
                text + '1',
                '0' + text,
                '1' + text
            ])
        
        elif len(text) > 10:
            # Try removing extra characters
            suggestions.extend([
                text[1:],   # Remove first
                text[:-1],  # Remove last
                text[1:-1]  # Remove first and last
            ])
        
        return suggestions
    
    def _try_character_corrections(self, text: str) -> List[str]:
        """Try character-level corrections"""
        suggestions = []
        
        # Try common OCR error corrections
        common_mistakes = {
            '0': ['O', 'D', 'Q'],
            'O': ['0', 'Q', 'D'],
            '1': ['I', 'L', '|'],
            'I': ['1', 'L', '|'],
            '5': ['S'],
            'S': ['5'],
            '8': ['B'],
            'B': ['8']
        }
        
        for i, char in enumerate(text):
            if char in common_mistakes:
                for replacement in common_mistakes[char]:
                    suggestion = text[:i] + replacement + text[i+1:]
                    suggestions.append(suggestion)
        
        return suggestions
    
    def _try_pattern_fixes(self, text: str) -> List[str]:
        """Try to fix pattern-related issues"""
        suggestions = []
        
        # Try different dash positions
        for i in range(2, min(len(text), 6)):
            if i < len(text):
                suggestion = text[:i] + text[i:]  # Remove any existing formatting first
                suggestions.append(suggestion)
        
        # Try province code fixes
        if len(text) >= 2:
            first_two = text[:2]
            if first_two.isdigit():
                # Try nearby province codes
                code_num = int(first_two)
                for offset in [-1, 1, -2, 2]:
                    new_code = str(code_num + offset).zfill(2)
                    if new_code in self.province_codes:
                        suggestion = new_code + text[2:]
                        suggestions.append(suggestion)
        
        return suggestions
    
    def is_valid_plate(self, text: str, min_confidence: float = 0.5) -> bool:
        """Quick validation check with minimum confidence threshold"""
        result = self.validate_plate(text)
        return result.is_valid and result.confidence_score >= min_confidence
    
    def get_plate_info(self, text: str) -> Dict[str, str]:
        """Get detailed information about a plate"""
        result = self.validate_plate(text)
        
        info = {
            'original': result.original_text,
            'cleaned': result.cleaned_text,
            'formatted': result.formatted_text,
            'valid': str(result.is_valid),
            'confidence': f"{result.confidence_score:.3f}",
            'type': result.plate_type
        }
        
        if result.province_code:
            info['province_code'] = result.province_code
            info['province_name'] = result.province_name or 'Unknown'
        
        if result.validation_details and 'errors' in result.validation_details:
            info['errors'] = ', '.join(result.validation_details['errors'])
        
        return info
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get validation statistics"""
        stats = self.validation_stats.copy()
        
        if stats['total_validations'] > 0:
            stats['success_rate'] = (stats['valid_plates'] / stats['total_validations']) * 100
        else:
            stats['success_rate'] = 0.0
        
        return stats
    
    def reset_statistics(self):
        """Reset validation statistics"""
        self.validation_stats = {
            'total_validations': 0,
            'valid_plates': 0,
            'by_type': {},
            'common_errors': {}
        }
        logger.info("🔄 Plate validator statistics reset")
    
    def export_patterns(self) -> Dict[str, str]:
        """Export plate patterns for reference"""
        return {
            name: info['description'] 
            for name, info in self.plate_patterns.items()
        }

# Utility functions
def quick_validate(text: str) -> bool:
    """Quick validation without detailed analysis"""
    if not text or len(text) < 4 or len(text) > 12:
        return False
    
    # Basic character check
    clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
    has_letter = any(c.isalpha() for c in clean_text)
    has_number = any(c.isdigit() for c in clean_text)
    
    return has_letter and has_number

def extract_province_info(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract province information from plate text"""
    clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
    
    if len(clean_text) >= 2:
        first_two = clean_text[:2]
        
        # Check numeric province codes
        if first_two.isdigit() and first_two in VIETNAMESE_PROVINCE_CODES:
            return first_two, VIETNAMESE_PROVINCE_CODES[first_two]
        
        # Check special codes
        if first_two in SPECIAL_PLATE_CODES:
            return first_two, SPECIAL_PLATE_CODES[first_two]
    
    return None, None

def format_plate_display(text: str) -> str:
    """Format plate for display purposes"""
    validator = VietnamesePlateValidator()
    result = validator.validate_plate(text)
    
    if result.is_valid and result.formatted_text:
        return result.formatted_text
    
    # Fallback formatting
    clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
    if len(clean_text) >= 5:
        # Try to add dash intelligently
        for i in range(2, min(6, len(clean_text))):
            if i < len(clean_text) and clean_text[i].isdigit():
                return f"{clean_text[:i]}-{clean_text[i:]}"
    
    return clean_text

def is_likely_vietnamese_plate(text: str) -> bool:
    """Check if text is likely a Vietnamese license plate"""
    if not text:
        return False
    
    clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
    
    # Length check
    if not (4 <= len(clean_text) <= 10):
        return False
    
    # Must have both letters and numbers
    has_letter = any(c.isalpha() for c in clean_text)
    has_number = any(c.isdigit() for c in clean_text)
    
    if not (has_letter and has_number):
        return False
    
    # Check for Vietnamese patterns
    province_patterns = [
        r'^\d{2}[A-Z]',      # Standard format start
        r'^[A-Z]{2}\d',      # Special format start
        r'^\d{3}[A-Z]'       # Old format start
    ]
    
    for pattern in province_patterns:
        if re.match(pattern, clean_text):
            return True
    
    return False

# Factory function
def create_plate_validator():
    """Create Vietnamese plate validator instance"""
    return VietnamesePlateValidator()

# Example usage and testing
if __name__ == "__main__":
    # Test the validator
    validator = VietnamesePlateValidator()
    
    # Test cases
    test_plates = [
        "30A12345",      # Valid standard
        "30AA1234",      # Valid new format
        "LD12345",       # Valid special
        "123A456",       # Valid old format
        "99Z99999",      # Invalid province
        "ABCD1234",      # Invalid format
        "30A123456789",  # Too long
        "30",            # Too short
        "3OA12345",      # OCR error (O instead of 0)
        "30AI2345",      # OCR error (I instead of 1)
    ]
    
    print("🔍 Vietnamese License Plate Validation Tests")
    print("=" * 60)
    
    for plate in test_plates:
        result = validator.validate_plate(plate)
        
        print(f"\nInput: '{plate}'")
        print(f"  Valid: {result.is_valid}")
        print(f"  Formatted: '{result.formatted_text}'")
        print(f"  Confidence: {result.confidence_score:.3f}")
        print(f"  Type: {result.plate_type}")
        
        if result.province_name:
            print(f"  Province: {result.province_name}")
        
        if not result.is_valid:
            suggestions = validator.get_validation_suggestions(plate)
            if suggestions:
                print(f"  Suggestions: {', '.join(suggestions)}")
    
    # Statistics
    print(f"\n📊 Statistics:")
    stats = validator.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Export patterns
    print(f"\n📋 Supported Patterns:")
    patterns = validator.export_patterns()
    for name, description in patterns.items():
        print(f"  {name}: {description}")