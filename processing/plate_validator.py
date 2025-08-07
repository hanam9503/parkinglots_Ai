"""
Vietnamese License Plate Validator - Optimized
Xác thực biển số xe Việt Nam (phiên bản tối ưu)
"""

import re
from typing import Optional, Dict, Tuple
from dataclasses import dataclass

# Vietnamese province codes (main ones)
PROVINCE_CODES = {
    '11': 'TP.HCM', '12': 'Tây Ninh', '13': 'Long An', '14': 'Bình Dương',
    '15': 'Vũng Tàu', '16': 'Đồng Nai', '17': 'Bình Thuận', '18': 'Bình Phước',
    '19': 'An Giang', '20': 'Cần Thơ', '21': 'Tiền Giang', '22': 'Bến Tre',
    '30': 'Hà Nội', '31': 'Hà Giang', '32': 'Cao Bằng', '33': 'Lào Cai',
    '34': 'Điện Biên', '35': 'Lai Châu', '36': 'Sơn La', '37': 'Yên Bái',
    '38': 'Tuyên Quang', '43': 'Đà Nẵng', '47': 'Đắk Lắk', '49': 'Lâm Đồng',
    '50': 'TP.HCM', '51': 'Hà Nội', '77': 'Ba Ria-Vung Tau', '85': 'Ninh Bình',
    '90': 'Hà Nội', '92': 'Hà Nội'
}

SPECIAL_CODES = {'LD': 'Lãnh đạo', 'HC': 'Hải quan', 'CD': 'Công an'}

OCR_CORRECTIONS = {'0': 'O', 'O': '0', 'I': '1', 'L': '1', '1': 'I', 'S': '5', '5': 'S'}

@dataclass
class ValidationResult:
    original: str
    cleaned: str
    formatted: str
    is_valid: bool
    confidence: float
    province: Optional[str] = None

class PlateValidator:
    """Optimized Vietnamese plate validator"""
    
    def __init__(self):
        self.patterns = {
            'standard': r'^(\d{2})([A-Z])(\d{4,5})$',        # 30A12345
            'new': r'^(\d{2})([A-Z]{2})(\d{3,4})$',          # 30AA1234
            'old': r'^(\d{3})([A-Z])(\d{3})$',               # 123A456
            'special': r'^([A-Z]{2})(\d{4,5})$'              # LD12345
        }
    
    def validate(self, text: str, apply_corrections: bool = True) -> ValidationResult:
        """Main validation function"""
        if not text:
            return ValidationResult("", "", "", False, 0.0)
        
        original = text
        cleaned = self._clean_text(text, apply_corrections)
        
        # Try each pattern
        for pattern_type, pattern in self.patterns.items():
            match = re.match(pattern, cleaned)
            if match:
                if self._validate_codes(match, pattern_type):
                    formatted = self._format_plate(match, pattern_type)
                    confidence = self._calculate_confidence(original, cleaned, match)
                    province = self._get_province_name(match, pattern_type)
                    
                    return ValidationResult(
                        original=original,
                        cleaned=cleaned, 
                        formatted=formatted,
                        is_valid=True,
                        confidence=confidence,
                        province=province
                    )
        
        return ValidationResult(original, cleaned, "", False, 0.0)
    
    def _clean_text(self, text: str, apply_corrections: bool) -> str:
        """Clean and correct OCR text"""
        cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        if not apply_corrections:
            return cleaned
        
        # Apply context-aware corrections
        result = ""
        for i, char in enumerate(cleaned):
            if char in OCR_CORRECTIONS:
                # Simple context rules
                if char in ['O', '0']:
                    # '0' if surrounded by digits, 'O' otherwise
                    prev_digit = i > 0 and cleaned[i-1].isdigit()
                    next_digit = i < len(cleaned)-1 and cleaned[i+1].isdigit()
                    result += '0' if (prev_digit or next_digit) else 'O'
                elif char in ['I', 'L', '1']:
                    # Similar logic for I/L/1
                    prev_digit = i > 0 and cleaned[i-1].isdigit()
                    next_digit = i < len(cleaned)-1 and cleaned[i+1].isdigit()
                    result += '1' if (prev_digit or next_digit) else 'I'
                else:
                    result += OCR_CORRECTIONS.get(char, char)
            else:
                result += char
        
        return result
    
    def _validate_codes(self, match, pattern_type: str) -> bool:
        """Validate province/special codes"""
        groups = match.groups()
        
        if pattern_type in ['standard', 'new', 'old']:
            return groups[0] in PROVINCE_CODES
        elif pattern_type == 'special':
            return groups[0] in SPECIAL_CODES
        
        return False
    
    def _format_plate(self, match, pattern_type: str) -> str:
        """Format validated plate"""
        groups = match.groups()
        
        if pattern_type in ['standard', 'new']:
            return f"{groups[0]}{groups[1]}-{groups[2]}"
        elif pattern_type == 'old':
            return f"{groups[0]}{groups[1]}-{groups[2]}"
        elif pattern_type == 'special':
            return f"{groups[0]}-{groups[1]}"
        
        return "".join(groups)
    
    def _calculate_confidence(self, original: str, cleaned: str, match) -> float:
        """Calculate validation confidence"""
        # Basic confidence based on text similarity and pattern match
        similarity = len(cleaned) / max(len(original), 1)
        pattern_conf = 0.8  # High confidence for pattern match
        quality = self._assess_quality(cleaned)
        
        return min((similarity * 0.3 + pattern_conf * 0.4 + quality * 0.3), 1.0)
    
    def _assess_quality(self, text: str) -> float:
        """Assess text quality"""
        if not text:
            return 0.0
        
        # Length appropriateness
        length_score = 1.0 if 5 <= len(text) <= 9 else 0.5
        
        # Character diversity
        unique_ratio = len(set(text)) / len(text)
        
        # Letter-number balance
        letters = sum(1 for c in text if c.isalpha())
        numbers = sum(1 for c in text if c.isdigit())
        balance = min(letters, numbers) / max(letters, numbers) if min(letters, numbers) > 0 else 0
        
        return (length_score + unique_ratio + balance) / 3
    
    def _get_province_name(self, match, pattern_type: str) -> Optional[str]:
        """Get province name from validated plate"""
        groups = match.groups()
        
        if pattern_type in ['standard', 'new', 'old']:
            return PROVINCE_CODES.get(groups[0])
        elif pattern_type == 'special':
            return SPECIAL_CODES.get(groups[0])
        
        return None
    
    def quick_validate(self, text: str) -> bool:
        """Quick validation check"""
        result = self.validate(text)
        return result.is_valid and result.confidence > 0.5