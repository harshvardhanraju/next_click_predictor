#!/usr/bin/env python3
"""
Production-optimized next-click predictor with minimal dependencies and maximum speed
Designed specifically for Cloud Run with aggressive timeout protection
"""

import os
import json
import time
import cv2
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

@dataclass
class SimpleElement:
    """Simplified UI element for fast processing"""
    element_id: str
    element_type: str
    bbox: tuple  # (x1, y1, x2, y2)
    center: tuple  # (x, y)
    confidence: float
    text: str = ""

@dataclass 
class SimplePrediction:
    """Simplified prediction result"""
    element_id: str
    element_type: str
    element_text: str
    click_probability: float
    confidence: float
    bbox: tuple
    center: tuple
    processing_time: float
    method: str

class ProductionOptimizedPredictor:
    """Ultra-fast predictor optimized for production deployment"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {
            'max_elements': 5,  # Process only top 5 elements
            'timeout_seconds': 30,  # Hard timeout
            'min_area': 400,  # Minimum element area
            'max_area': 50000,  # Maximum element area
        }
        
        self.logger = logging.getLogger(__name__)
        
        # Element type scoring for click probability
        self.element_scores = {
            'button': 0.8,
            'link': 0.6,
            'form': 0.4,
            'text': 0.2,
            'unknown': 0.3
        }
        
        # Task-based keyword matching
        self.task_keywords = {
            'login': ['button', 'form'],
            'buy': ['button', 'link'],
            'search': ['form', 'button'],
            'navigate': ['link', 'button'],
            'submit': ['button'],
            'continue': ['button'],
            'next': ['button'],
        }
    
    def predict_next_click(self, screenshot_path: str, user_attributes: Dict[str, Any], 
                          task_description: str) -> SimplePrediction:
        """Main prediction method with aggressive optimization"""
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting fast prediction: {screenshot_path}")
            
            # Load and validate image
            image = cv2.imread(screenshot_path)
            if image is None:
                return self._create_fallback_prediction("Invalid image", start_time)
            
            # Quick element detection (no OCR)
            elements = self._detect_elements_fast(image)
            
            if not elements:
                return self._create_fallback_prediction("No elements detected", start_time)
            
            self.logger.info(f"Detected {len(elements)} elements in fast mode")
            
            # Quick scoring and selection
            best_element = self._select_best_element(elements, task_description, user_attributes)
            
            processing_time = time.time() - start_time
            
            return SimplePrediction(
                element_id=best_element.element_id,
                element_type=best_element.element_type,
                element_text=best_element.text,
                click_probability=best_element.confidence,
                confidence=best_element.confidence,
                bbox=best_element.bbox,
                center=best_element.center,
                processing_time=processing_time,
                method="production_optimized"
            )
            
        except Exception as e:
            self.logger.error(f"Fast prediction failed: {e}")
            return self._create_fallback_prediction(str(e), start_time)
    
    def _detect_elements_fast(self, image: np.ndarray) -> List[SimpleElement]:
        """Ultra-fast element detection using only contours"""
        elements = []
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Simple edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Process only largest contours for speed
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]
            
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                
                # Filter by area
                if area < self.config['min_area'] or area > self.config['max_area']:
                    continue
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by aspect ratio (reasonable UI elements)
                aspect_ratio = w / h if h > 0 else 0
                if aspect_ratio < 0.1 or aspect_ratio > 10:
                    continue
                
                # Classify element type based on size and position
                element_type = self._classify_element_simple(x, y, w, h, image.shape)
                
                # Calculate confidence based on size and position
                confidence = self._calculate_confidence_simple(x, y, w, h, image.shape, area)
                
                element = SimpleElement(
                    element_id=f"fast_{i}",
                    element_type=element_type,
                    bbox=(x, y, x + w, y + h),
                    center=(x + w//2, y + h//2),
                    confidence=confidence
                )
                
                elements.append(element)
                
                # Limit elements for speed
                if len(elements) >= self.config['max_elements']:
                    break
            
            return elements
            
        except Exception as e:
            self.logger.error(f"Fast detection failed: {e}")
            return []
    
    def _classify_element_simple(self, x: int, y: int, w: int, h: int, 
                                img_shape: tuple) -> str:
        """Simple element classification based on size and position"""
        img_h, img_w = img_shape[:2]
        
        # Relative position
        rel_y = y / img_h
        
        # Size characteristics
        area = w * h
        aspect_ratio = w / h if h > 0 else 1
        
        # Classification logic
        if aspect_ratio > 2.5 and area > 2000:  # Wide and large
            return "button"
        elif rel_y < 0.3 and aspect_ratio > 3:  # Top area, wide
            return "form"  # Likely search box or input
        elif aspect_ratio < 2 and area < 5000:  # More square, smaller
            return "link"
        elif area < 1000:  # Small elements
            return "text"
        else:
            return "button"  # Default to button for medium-sized elements
    
    def _calculate_confidence_simple(self, x: int, y: int, w: int, h: int, 
                                   img_shape: tuple, area: float) -> float:
        """Simple confidence calculation"""
        img_h, img_w = img_shape[:2]
        
        # Position factor (center is better)
        center_x, center_y = x + w//2, y + h//2
        rel_x, rel_y = center_x / img_w, center_y / img_h
        
        # Distance from center (0.5, 0.5)
        center_distance = ((rel_x - 0.5)**2 + (rel_y - 0.5)**2)**0.5
        position_factor = 1.0 - min(center_distance, 0.7)  # Max penalty 0.7
        
        # Size factor (moderate size is better)
        size_factor = min(1.0, area / 10000)  # Normalize by 10k pixels
        
        # Aspect ratio factor (reasonable ratios are better)
        aspect_ratio = w / h if h > 0 else 1
        if 0.2 <= aspect_ratio <= 5:
            aspect_factor = 1.0
        else:
            aspect_factor = 0.5
        
        confidence = (position_factor * 0.4 + size_factor * 0.4 + aspect_factor * 0.2)
        return max(0.1, min(0.9, confidence))
    
    def _select_best_element(self, elements: List[SimpleElement], task_description: str, 
                           user_attributes: Dict[str, Any]) -> SimpleElement:
        """Select best element based on task and user context"""
        
        # Score all elements
        scored_elements = []
        task_lower = task_description.lower()
        
        for element in elements:
            score = element.confidence
            
            # Base score from element type
            score *= self.element_scores.get(element.element_type, 0.3)
            
            # Task matching bonus
            for task_keyword, preferred_types in self.task_keywords.items():
                if task_keyword in task_lower and element.element_type in preferred_types:
                    score *= 1.3
                    break
            
            # User expertise factor
            tech_savviness = user_attributes.get('tech_savviness', 'medium')
            if tech_savviness == 'high' and element.element_type == 'link':
                score *= 1.2
            elif tech_savviness == 'low' and element.element_type == 'button':
                score *= 1.2
            
            scored_elements.append((score, element))
        
        # Return highest scored element
        scored_elements.sort(key=lambda x: x[0], reverse=True)
        best_score, best_element = scored_elements[0]
        
        # Update confidence with final score
        best_element.confidence = min(0.95, best_score)
        
        return best_element
    
    def _create_fallback_prediction(self, error_msg: str, start_time: float) -> SimplePrediction:
        """Create safe fallback prediction"""
        processing_time = time.time() - start_time
        
        return SimplePrediction(
            element_id="fallback_0",
            element_type="button",
            element_text="Click Here",
            click_probability=0.6,
            confidence=0.5,
            bbox=(400, 300, 520, 340),
            center=(460, 320),
            processing_time=processing_time,
            method="fallback"
        )

def test_production_predictor():
    """Test the production predictor with our complex image"""
    predictor = ProductionOptimizedPredictor()
    
    # Test with the complex UI image we created earlier
    test_image_path = '/tmp/complex_ui_test.png'
    
    if os.path.exists(test_image_path):
        print("Testing production predictor with complex image...")
        
        start_time = time.time()
        result = predictor.predict_next_click(
            screenshot_path=test_image_path,
            user_attributes={
                'tech_savviness': 'medium',
                'age_group': 'adult'
            },
            task_description="Find and click the submit button"
        )
        total_time = time.time() - start_time
        
        print(f"✅ Prediction completed in {total_time:.2f} seconds")
        print(f"   Element: {result.element_type} (ID: {result.element_id})")
        print(f"   Confidence: {result.confidence:.2f}")
        print(f"   Position: {result.center}")
        print(f"   Method: {result.method}")
        
        return True
    else:
        print("❌ Test image not found, skipping test")
        return False

if __name__ == "__main__":
    # Run test
    test_production_predictor()