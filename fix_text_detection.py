#!/usr/bin/env python3
"""
Fix Text Detection Issues
Create a comprehensive solution for accurate text extraction and element matching
"""

import sys
import os
import cv2
import numpy as np
import tempfile
import logging
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TextElement:
    """Represents detected text with its properties"""
    text: str
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]
    confidence: float
    font_size: float
    is_button: bool
    is_clickable: bool

class ImprovedTextDetector:
    """Improved text detection that properly extracts and merges text"""
    
    def __init__(self):
        try:
            import easyocr
            self.ocr_reader = easyocr.Reader(['en'])
            self.ocr_available = True
        except ImportError:
            self.ocr_available = False
            logger.warning("OCR not available")
    
    def detect_text_elements(self, image: np.ndarray) -> List[TextElement]:
        """Detect and properly extract text elements from image"""
        if not self.ocr_available:
            return []
        
        text_elements = []
        
        # Run OCR with lower threshold to catch more text
        ocr_results = self.ocr_reader.readtext(image, detail=1)
        
        # Group nearby text elements
        grouped_results = self._group_nearby_text(ocr_results)
        
        for group in grouped_results:
            merged_text = self._merge_text_group(group)
            if merged_text:
                text_elements.append(merged_text)
        
        return text_elements
    
    def _group_nearby_text(self, ocr_results: List) -> List[List]:
        """Group nearby OCR results that likely belong to the same UI element"""
        if not ocr_results:
            return []
        
        # Convert OCR results to a more workable format
        text_items = []
        for (bbox, text, confidence) in ocr_results:
            if confidence < 0.3 or not text.strip():
                continue
            
            x1, y1 = int(min([p[0] for p in bbox])), int(min([p[1] for p in bbox]))
            x2, y2 = int(max([p[0] for p in bbox])), int(max([p[1] for p in bbox]))
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            
            text_items.append({
                'text': text.strip(),
                'bbox': (x1, y1, x2, y2),
                'center': (center_x, center_y),
                'confidence': confidence,
                'grouped': False
            })
        
        # Group items that are close to each other
        groups = []
        threshold_distance = 50  # pixels
        
        for item in text_items:
            if item['grouped']:
                continue
            
            # Start a new group
            current_group = [item]
            item['grouped'] = True
            
            # Find nearby items to add to this group
            for other_item in text_items:
                if other_item['grouped']:
                    continue
                
                # Calculate distance between centers
                dx = abs(item['center'][0] - other_item['center'][0])
                dy = abs(item['center'][1] - other_item['center'][1])
                distance = np.sqrt(dx*dx + dy*dy)
                
                # If close enough and on similar horizontal line, group them
                if distance < threshold_distance and dy < 20:
                    current_group.append(other_item)
                    other_item['grouped'] = True
            
            groups.append(current_group)
        
        return groups
    
    def _merge_text_group(self, group: List[Dict]) -> TextElement:
        """Merge a group of text items into a single text element"""
        if not group:
            return None
        
        # Sort by horizontal position (left to right)
        group.sort(key=lambda x: x['center'][0])
        
        # Merge text
        merged_text = ' '.join([item['text'] for item in group])
        
        # Calculate combined bounding box
        x1 = min([item['bbox'][0] for item in group])
        y1 = min([item['bbox'][1] for item in group])
        x2 = max([item['bbox'][2] for item in group])
        y2 = max([item['bbox'][3] for item in group])
        
        # Calculate average confidence
        avg_confidence = sum([item['confidence'] for item in group]) / len(group)
        
        # Calculate center
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        
        # Estimate font size based on height
        font_size = y2 - y1
        
        # Determine if this looks like a button or clickable element
        is_button = self._is_likely_button(merged_text, font_size, x2-x1, y2-y1)
        is_clickable = self._is_likely_clickable(merged_text)
        
        return TextElement(
            text=merged_text,
            bbox=(x1, y1, x2, y2),
            center=(center_x, center_y),
            confidence=avg_confidence,
            font_size=font_size,
            is_button=is_button,
            is_clickable=is_clickable
        )
    
    def _is_likely_button(self, text: str, font_size: float, width: int, height: int) -> bool:
        """Determine if text element is likely a button"""
        text_lower = text.lower()
        
        # Check for button keywords
        button_words = [
            'sign in', 'log in', 'login', 'signin',
            'sign up', 'signup', 'register',
            'submit', 'send', 'save', 'continue', 'next',
            'buy', 'purchase', 'add to cart', 'checkout',
            'cancel', 'close', 'ok', 'yes', 'no',
            'apply', 'confirm', 'accept', 'agree'
        ]
        
        for word in button_words:
            if word in text_lower:
                return True
        
        # Check button-like dimensions (reasonable aspect ratio)
        if width > 50 and height > 25 and width / height < 6:
            return True
        
        return False
    
    def _is_likely_clickable(self, text: str) -> bool:
        """Determine if text element is likely clickable"""
        text_lower = text.lower()
        
        clickable_indicators = [
            'click', 'tap', 'press', 'select',
            'learn more', 'read more', 'see more',
            'forgot', 'forgot password', 'help',
            'terms', 'privacy', 'policy'
        ]
        
        for indicator in clickable_indicators:
            if indicator in text_lower:
                return True
        
        # Check if it ends with common clickable suffixes
        if text_lower.endswith('?') or text_lower.endswith('!'):
            return True
        
        return False

def test_improved_text_detection():
    """Test the improved text detection system"""
    print("🔧 TESTING IMPROVED TEXT DETECTION")
    print("=" * 50)
    
    detector = ImprovedTextDetector()
    
    if not detector.ocr_available:
        print("❌ OCR not available, cannot test")
        return
    
    # Create test image with common UI elements
    img = np.ones((400, 600, 3), dtype=np.uint8) * 245
    
    # Add some text elements
    cv2.putText(img, 'Sign In', (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 2)
    cv2.putText(img, 'Forgot Password?', (180, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 200), 1)
    cv2.putText(img, 'Add to Cart', (400, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Test detection
    text_elements = detector.detect_text_elements(img)
    
    print(f"Found {len(text_elements)} text elements:")
    for i, elem in enumerate(text_elements):
        print(f"  {i+1}. '{elem.text}' (confidence: {elem.confidence:.2f})")
        print(f"      Button: {elem.is_button}, Clickable: {elem.is_clickable}")
        print(f"      Bbox: {elem.bbox}")

def create_comprehensive_fix():
    """Create a comprehensive fix for the screenshot processor"""
    
    fix_code = '''
# Add this to screenshot_processor.py to improve text detection

def _improved_extract_text_features(self, image: np.ndarray, elements: List[UIElement]) -> List[UIElement]:
    """Enhanced text extraction with better OCR integration"""
    if not self.ocr_reader:
        return elements
    
    # First, run full-image OCR to get all text
    try:
        ocr_results = self.ocr_reader.readtext(image, detail=1)
        all_text_elements = self._process_ocr_results(ocr_results)
    except Exception as e:
        logging.error(f"OCR failed: {e}")
        return elements
    
    enhanced_elements = []
    
    for element in elements:
        if element.text and element.text.strip():
            # Element already has text, keep it
            enhanced_elements.append(element)
        else:
            # Try to find matching text from OCR results
            matched_text = self._find_matching_text(element, all_text_elements)
            
            if matched_text:
                # Create enhanced element with matched text
                enhanced_element = UIElement(
                    element_id=element.element_id,
                    element_type=self._classify_element_type(matched_text),
                    text=matched_text,
                    bbox=element.bbox,
                    center=element.center,
                    size=element.size,
                    prominence=element.prominence,
                    visibility=element.visibility,
                    color_features=element.color_features,
                    position_features=element.position_features
                )
                enhanced_elements.append(enhanced_element)
    
    # Also add pure text elements that weren't matched to visual elements
    for text_elem in all_text_elements:
        if not self._is_text_already_used(text_elem, enhanced_elements):
            # Create new UI element for this text
            ui_element = UIElement(
                element_id=f"text_only_{uuid.uuid4().hex[:8]}",
                element_type=self._classify_element_type(text_elem['text']),
                text=text_elem['text'],
                bbox=text_elem['bbox'],
                center=text_elem['center'],
                size=(text_elem['bbox'][2] - text_elem['bbox'][0], 
                      text_elem['bbox'][3] - text_elem['bbox'][1]),
                prominence=text_elem['confidence'],
                visibility=True,
                color_features={},
                position_features={}
            )
            enhanced_elements.append(ui_element)
    
    return enhanced_elements

def _process_ocr_results(self, ocr_results: List) -> List[Dict]:
    """Process OCR results into usable text elements"""
    text_elements = []
    
    for (bbox, text, confidence) in ocr_results:
        if confidence < 0.3 or not text.strip():
            continue
        
        x1, y1 = int(min([p[0] for p in bbox])), int(min([p[1] for p in bbox]))
        x2, y2 = int(max([p[0] for p in bbox])), int(max([p[1] for p in bbox]))
        
        text_elements.append({
            'text': text.strip(),
            'bbox': (x1, y1, x2, y2),
            'center': ((x1 + x2) // 2, (y1 + y2) // 2),
            'confidence': confidence
        })
    
    return text_elements

def _find_matching_text(self, element: UIElement, text_elements: List[Dict]) -> str:
    """Find text that overlaps with a visual element"""
    elem_x1, elem_y1, elem_x2, elem_y2 = element.bbox
    
    best_match = None
    best_overlap = 0
    
    for text_elem in text_elements:
        text_x1, text_y1, text_x2, text_y2 = text_elem['bbox']
        
        # Calculate overlap
        overlap_x1 = max(elem_x1, text_x1)
        overlap_y1 = max(elem_y1, text_y1)
        overlap_x2 = min(elem_x2, text_x2)
        overlap_y2 = min(elem_y2, text_y2)
        
        if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
            overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
            text_area = (text_x2 - text_x1) * (text_y2 - text_y1)
            
            if text_area > 0:
                overlap_ratio = overlap_area / text_area
                
                if overlap_ratio > best_overlap and overlap_ratio > 0.3:
                    best_overlap = overlap_ratio
                    best_match = text_elem['text']
    
    return best_match
    '''
    
    print("📋 COMPREHENSIVE FIX CREATED")
    print("=" * 50)
    print("The fix above should be integrated into screenshot_processor.py")
    print("Key improvements:")
    print("- Better OCR text grouping and merging")
    print("- Overlay matching between visual elements and text")
    print("- Preservation of pure text elements")
    print("- Better button/clickable detection")

if __name__ == "__main__":
    test_improved_text_detection()
    create_comprehensive_fix()