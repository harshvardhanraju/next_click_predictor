import cv2
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
import logging
from advanced_ui_detector import DetectedElement
import uuid

@dataclass
class UIPattern:
    """Represents a modern UI pattern with its characteristics"""
    name: str
    description: str
    detection_method: str
    confidence_weight: float
    visual_characteristics: Dict[str, Any]

class ModernUIPatterns:
    """
    Recognition system for contemporary UI patterns found in modern web/mobile applications
    """
    
    def __init__(self):
        self.patterns = self._initialize_patterns()
        self.logger = logging.getLogger(__name__)
    
    def _initialize_patterns(self) -> Dict[str, UIPattern]:
        """Initialize modern UI pattern definitions"""
        patterns = {}
        
        # Material Design Button
        patterns['material_button'] = UIPattern(
            name="material_button",
            description="Material Design elevated/contained button",
            detection_method="material_design",
            confidence_weight=1.2,
            visual_characteristics={
                'rounded_corners': True,
                'shadow': True,
                'min_padding': 16,
                'typical_height': (36, 56),
                'aspect_ratio_range': (1.5, 6.0),
                'color_saturation': 'high'
            }
        )
        
        # Flat Design Button
        patterns['flat_button'] = UIPattern(
            name="flat_button",
            description="Flat design button without elevation",
            detection_method="flat_design",
            confidence_weight=1.1,
            visual_characteristics={
                'border': False,
                'gradient': False,
                'solid_color': True,
                'typical_height': (32, 48),
                'aspect_ratio_range': (1.2, 8.0),
                'color_contrast': 'medium'
            }
        )
        
        # Icon Button (FAB - Floating Action Button)
        patterns['icon_button'] = UIPattern(
            name="icon_button",
            description="Icon-based button or floating action button",
            detection_method="icon_detection",
            confidence_weight=1.3,
            visual_characteristics={
                'aspect_ratio': (0.8, 1.2),
                'contains_symbol': True,
                'typically_circular': True,
                'size_range': (40, 80),
                'high_contrast': True
            }
        )
        
        # Card Element
        patterns['card_element'] = UIPattern(
            name="card_element",
            description="Card-based UI component",
            detection_method="card_detection",
            confidence_weight=1.0,
            visual_characteristics={
                'elevation': True,
                'padding': True,
                'rounded_corners': True,
                'background_distinct': True,
                'size_range': (200, 600),
                'aspect_ratio_range': (0.7, 3.0)
            }
        )
        
        # Toggle Switch
        patterns['toggle_switch'] = UIPattern(
            name="toggle_switch",
            description="Toggle switch or slider control",
            detection_method="toggle_detection",
            confidence_weight=1.4,
            visual_characteristics={
                'pill_shape': True,
                'contains_circle': True,
                'dual_state_colors': True,
                'aspect_ratio': (1.8, 2.5),
                'size_range': (30, 60)
            }
        )
        
        # Input Field with Floating Label
        patterns['floating_label_input'] = UIPattern(
            name="floating_label_input",
            description="Modern input field with floating label",
            detection_method="floating_label_detection",
            confidence_weight=1.1,
            visual_characteristics={
                'underline_focus': True,
                'label_animation_space': True,
                'minimal_border': True,
                'aspect_ratio_range': (3.0, 12.0),
                'height_range': (40, 70)
            }
        )
        
        # Chip/Tag Element
        patterns['chip_tag'] = UIPattern(
            name="chip_tag",
            description="Chip or tag element for categories/filters",
            detection_method="chip_detection",
            confidence_weight=1.0,
            visual_characteristics={
                'rounded_pill': True,
                'compact_size': True,
                'aspect_ratio_range': (1.5, 4.0),
                'height_range': (24, 40),
                'background_light': True
            }
        )
        
        return patterns
    
    def detect_modern_patterns(self, image: np.ndarray, 
                             existing_elements: List[DetectedElement]) -> List[DetectedElement]:
        """
        Enhance existing element detection with modern UI pattern recognition
        
        Args:
            image: Input image
            existing_elements: Elements already detected by basic methods
            
        Returns:
            Enhanced list of detected elements with modern pattern recognition
        """
        enhanced_elements = list(existing_elements)  # Copy existing elements
        
        # Detect each modern pattern type
        enhanced_elements.extend(self._detect_material_buttons(image))
        enhanced_elements.extend(self._detect_flat_buttons(image))
        enhanced_elements.extend(self._detect_icon_buttons(image))
        enhanced_elements.extend(self._detect_card_elements(image))
        enhanced_elements.extend(self._detect_toggle_switches(image))
        enhanced_elements.extend(self._detect_floating_label_inputs(image))
        enhanced_elements.extend(self._detect_chips_tags(image))
        
        # Apply pattern-based confidence boosting to existing elements
        enhanced_elements = self._boost_pattern_matches(image, enhanced_elements)
        
        return enhanced_elements
    
    def _detect_material_buttons(self, image: np.ndarray) -> List[DetectedElement]:
        """Detect Material Design buttons with elevation/shadow"""
        elements = []
        
        # Convert to different color spaces for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Detect shadow-like effects (common in material design)
        # Apply Gaussian blur to detect subtle shadows
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        shadow_diff = cv2.absdiff(gray, blurred)
        
        # Threshold to find potential shadow areas
        _, shadow_mask = cv2.threshold(shadow_diff, 10, 255, cv2.THRESH_BINARY)
        
        # Find contours in shadow areas
        contours, _ = cv2.findContours(shadow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1000 or area > 20000:  # Filter by reasonable button size
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            # Check if it matches material button characteristics
            if (1.5 <= aspect_ratio <= 6.0 and 
                36 <= h <= 56 and 
                w >= 64):  # Minimum material button width
                
                # Check for rounded corners by analyzing corner regions
                corner_roundness = self._check_rounded_corners(gray, (x, y, x+w, y+h))
                
                if corner_roundness > 0.3:  # Has rounded corners
                    confidence = min(0.85, (corner_roundness + (area / 5000)) / 2)
                    
                    elements.append(DetectedElement(
                        element_id=f"material_btn_{uuid.uuid4().hex[:8]}",
                        element_type="button",
                        bbox=(x, y, x + w, y + h),
                        center=(x + w//2, y + h//2),
                        size=(w, h),
                        confidence=confidence,
                        detection_method="material_design",
                        visual_features={
                            "pattern": "material_button",
                            "corner_roundness": corner_roundness,
                            "has_shadow": True,
                            "aspect_ratio": aspect_ratio
                        }
                    ))
        
        return elements
    
    def _detect_flat_buttons(self, image: np.ndarray) -> List[DetectedElement]:
        """Detect flat design buttons with solid colors"""
        elements = []
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for common button colors
        button_color_ranges = [
            # Blue range
            (np.array([100, 100, 100]), np.array([130, 255, 255])),
            # Green range
            (np.array([40, 100, 100]), np.array([80, 255, 255])),
            # Red range
            (np.array([0, 100, 100]), np.array([20, 255, 255])),
            # Orange range
            (np.array([10, 100, 100]), np.array([25, 255, 255]))
        ]
        
        for i, (lower, upper) in enumerate(button_color_ranges):
            # Create mask for this color range
            mask = cv2.inRange(hsv, lower, upper)
            
            # Morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 800 or area > 25000:
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                # Check flat button characteristics
                if 1.2 <= aspect_ratio <= 8.0 and 32 <= h <= 48:
                    # Check color uniformity (flat buttons have solid colors)
                    roi = image[y:y+h, x:x+w]
                    color_variance = self._calculate_color_variance(roi)
                    
                    if color_variance < 0.3:  # Low variance indicates flat color
                        confidence = min(0.8, (1.0 - color_variance) * 0.9)
                        
                        elements.append(DetectedElement(
                            element_id=f"flat_btn_{uuid.uuid4().hex[:8]}",
                            element_type="button",
                            bbox=(x, y, x + w, y + h),
                            center=(x + w//2, y + h//2),
                            size=(w, h),
                            confidence=confidence,
                            detection_method="flat_design",
                            visual_features={
                                "pattern": "flat_button",
                                "color_variance": color_variance,
                                "color_range": i,
                                "aspect_ratio": aspect_ratio
                            }
                        ))
        
        return elements
    
    def _detect_icon_buttons(self, image: np.ndarray) -> List[DetectedElement]:
        """Detect icon buttons and floating action buttons"""
        elements = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use HoughCircles to detect circular buttons (common for FABs)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
            param1=50, param2=30, minRadius=20, maxRadius=50
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in circles:
                # Check if the circle contains high-contrast content (likely an icon)
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.circle(mask, (x, y), r, 255, -1)
                
                roi = cv2.bitwise_and(gray, mask)
                non_zero_pixels = cv2.countNonZero(roi)
                
                if non_zero_pixels > 0:
                    # Calculate contrast within the circle
                    roi_values = roi[roi > 0]
                    if len(roi_values) > 10:
                        contrast = np.std(roi_values) / 255.0
                        
                        if contrast > 0.3:  # High contrast suggests icon content
                            confidence = min(0.85, contrast)
                            
                            elements.append(DetectedElement(
                                element_id=f"icon_btn_{uuid.uuid4().hex[:8]}",
                                element_type="button",
                                bbox=(x-r, y-r, x+r, y+r),
                                center=(x, y),
                                size=(2*r, 2*r),
                                confidence=confidence,
                                detection_method="icon_detection",
                                visual_features={
                                    "pattern": "icon_button",
                                    "is_circular": True,
                                    "radius": r,
                                    "contrast": contrast
                                }
                            ))
        
        return elements
    
    def _detect_card_elements(self, image: np.ndarray) -> List[DetectedElement]:
        """Detect card-based UI components"""
        elements = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect rectangular regions with subtle shadows (card elevation)
        # Apply morphological operations to find rectangular structures
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        
        # Threshold to find elevated regions
        _, thresh = cv2.threshold(tophat, 10, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 5000 or area > 100000:  # Cards are medium to large
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            # Check card characteristics
            if 0.7 <= aspect_ratio <= 3.0 and w > 150 and h > 100:
                # Check for background distinctness
                roi = image[y:y+h, x:x+w]
                background_uniformity = self._check_background_uniformity(roi)
                
                if background_uniformity > 0.4:  # Relatively uniform background
                    confidence = min(0.75, (background_uniformity + area/50000) / 2)
                    
                    elements.append(DetectedElement(
                        element_id=f"card_{uuid.uuid4().hex[:8]}",
                        element_type="card",
                        bbox=(x, y, x + w, y + h),
                        center=(x + w//2, y + h//2),
                        size=(w, h),
                        confidence=confidence,
                        detection_method="card_detection",
                        visual_features={
                            "pattern": "card_element",
                            "background_uniformity": background_uniformity,
                            "aspect_ratio": aspect_ratio,
                            "area": area
                        }
                    ))
        
        return elements
    
    def _detect_toggle_switches(self, image: np.ndarray) -> List[DetectedElement]:
        """Detect toggle switches and sliders"""
        elements = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Look for pill-shaped elements (elongated rounded rectangles)
        # Use morphological operations with elongated kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 20))
        opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 300 or area > 3000:  # Toggle switches are small to medium
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            # Check toggle switch characteristics
            if 1.8 <= aspect_ratio <= 2.5 and 15 <= h <= 35:
                # Look for circular element within (the toggle knob)
                roi = gray[y:y+h, x:x+w]
                circles = cv2.HoughCircles(
                    roi, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                    param1=30, param2=15, minRadius=h//4, maxRadius=h//2
                )
                
                if circles is not None and len(circles[0]) > 0:
                    confidence = 0.7
                    
                    elements.append(DetectedElement(
                        element_id=f"toggle_{uuid.uuid4().hex[:8]}",
                        element_type="toggle",
                        bbox=(x, y, x + w, y + h),
                        center=(x + w//2, y + h//2),
                        size=(w, h),
                        confidence=confidence,
                        detection_method="toggle_detection",
                        visual_features={
                            "pattern": "toggle_switch",
                            "aspect_ratio": aspect_ratio,
                            "has_knob": True
                        }
                    ))
        
        return elements
    
    def _detect_floating_label_inputs(self, image: np.ndarray) -> List[DetectedElement]:
        """Detect modern input fields with floating labels"""
        elements = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Look for horizontal lines (underline style inputs)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
        horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Find contours of horizontal lines
        contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check if it's a reasonable input field underline
            if w > 100 and h < 5:  # Long and thin line
                # Expand upward to include the input area
                input_y = max(0, y - 40)
                input_h = 50
                
                # Check if there's space above for floating label
                if input_y >= 20:  # Space for label
                    confidence = min(0.7, w / 300.0)
                    
                    elements.append(DetectedElement(
                        element_id=f"float_input_{uuid.uuid4().hex[:8]}",
                        element_type="form",
                        bbox=(x, input_y, x + w, input_y + input_h),
                        center=(x + w//2, input_y + input_h//2),
                        size=(w, input_h),
                        confidence=confidence,
                        detection_method="floating_label_detection",
                        visual_features={
                            "pattern": "floating_label_input",
                            "underline_width": w,
                            "has_label_space": True
                        }
                    ))
        
        return elements
    
    def _detect_chips_tags(self, image: np.ndarray) -> List[DetectedElement]:
        """Detect chip/tag elements"""
        elements = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Look for small rounded rectangular elements
        # Apply morphological operations with rounded kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 15))
        opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 200 or area > 2000:  # Chips are small
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            # Check chip characteristics
            if 1.5 <= aspect_ratio <= 4.0 and 20 <= h <= 35:
                # Check for light background (common for chips)
                roi = gray[y:y+h, x:x+w]
                avg_brightness = np.mean(roi)
                
                if avg_brightness > 180:  # Light background
                    confidence = min(0.6, (avg_brightness - 180) / 75.0 + 0.3)
                    
                    elements.append(DetectedElement(
                        element_id=f"chip_{uuid.uuid4().hex[:8]}",
                        element_type="chip",
                        bbox=(x, y, x + w, y + h),
                        center=(x + w//2, y + h//2),
                        size=(w, h),
                        confidence=confidence,
                        detection_method="chip_detection",
                        visual_features={
                            "pattern": "chip_tag",
                            "aspect_ratio": aspect_ratio,
                            "brightness": avg_brightness
                        }
                    ))
        
        return elements
    
    def _boost_pattern_matches(self, image: np.ndarray, 
                              elements: List[DetectedElement]) -> List[DetectedElement]:
        """Boost confidence of elements that match modern UI patterns"""
        enhanced_elements = []
        
        for element in elements:
            enhanced_element = element
            
            # Check if element matches any modern pattern
            pattern_match = self._match_element_to_pattern(image, element)
            
            if pattern_match:
                pattern_name, boost_factor = pattern_match
                
                # Boost confidence
                new_confidence = min(0.95, element.confidence * boost_factor)
                
                # Update visual features
                visual_features = element.visual_features.copy()
                visual_features.update({
                    "matched_pattern": pattern_name,
                    "confidence_boost": boost_factor,
                    "original_confidence": element.confidence
                })
                
                # Create updated element
                enhanced_element = DetectedElement(
                    element_id=element.element_id,
                    element_type=element.element_type,
                    bbox=element.bbox,
                    center=element.center,
                    size=element.size,
                    confidence=new_confidence,
                    detection_method=element.detection_method,
                    visual_features=visual_features
                )
            
            enhanced_elements.append(enhanced_element)
        
        return enhanced_elements
    
    def _match_element_to_pattern(self, image: np.ndarray, 
                                 element: DetectedElement) -> Optional[Tuple[str, float]]:
        """Check if an element matches any modern UI pattern"""
        
        # Extract element region
        x1, y1, x2, y2 = element.bbox
        roi = image[y1:y2, x1:x2]
        
        if roi.size == 0:
            return None
        
        w, h = element.size
        aspect_ratio = w / h if h > 0 else 1.0
        
        # Check for material button pattern
        if (element.element_type == "button" and 
            1.5 <= aspect_ratio <= 6.0 and 
            30 <= h <= 60):
            
            corner_roundness = self._check_rounded_corners(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (0, 0, w, h))
            if corner_roundness > 0.3:
                return ("material_button", 1.2)
        
        # Check for flat button pattern
        if (element.element_type == "button" and 
            1.2 <= aspect_ratio <= 8.0 and 
            25 <= h <= 50):
            
            color_variance = self._calculate_color_variance(roi)
            if color_variance < 0.3:
                return ("flat_button", 1.1)
        
        # Check for icon button pattern
        if (element.element_type == "button" and 
            0.8 <= aspect_ratio <= 1.2 and 
            30 <= w <= 80 and 30 <= h <= 80):
            
            return ("icon_button", 1.3)
        
        return None
    
    def _check_rounded_corners(self, gray_roi: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        """Check how rounded the corners of a region are"""
        if gray_roi.size == 0:
            return 0.0
        
        x1, y1, x2, y2 = bbox
        h, w = gray_roi.shape
        
        # Sample corner regions
        corner_size = min(10, w//4, h//4)
        if corner_size < 3:
            return 0.0
        
        corners = [
            gray_roi[0:corner_size, 0:corner_size],  # Top-left
            gray_roi[0:corner_size, w-corner_size:w],  # Top-right
            gray_roi[h-corner_size:h, 0:corner_size],  # Bottom-left
            gray_roi[h-corner_size:h, w-corner_size:w]  # Bottom-right
        ]
        
        roundness_scores = []
        for corner in corners:
            if corner.size > 0:
                # Calculate gradient magnitude in corner
                grad_x = cv2.Sobel(corner, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(corner, cv2.CV_64F, 0, 1, ksize=3)
                gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
                
                # Rounded corners have curved gradients
                gradient_variation = np.std(gradient_mag)
                roundness_scores.append(min(1.0, gradient_variation / 50.0))
        
        return np.mean(roundness_scores) if roundness_scores else 0.0
    
    def _calculate_color_variance(self, roi: np.ndarray) -> float:
        """Calculate color variance in a region (lower = more uniform)"""
        if roi.size == 0:
            return 1.0
        
        # Convert to LAB color space for better color analysis
        try:
            lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
            
            # Calculate variance in each channel
            l_var = np.var(lab[:,:,0]) / (255.0**2)
            a_var = np.var(lab[:,:,1]) / (255.0**2)
            b_var = np.var(lab[:,:,2]) / (255.0**2)
            
            # Weighted average (L channel is most important)
            total_variance = (l_var * 0.6 + a_var * 0.2 + b_var * 0.2)
            return min(1.0, total_variance)
            
        except Exception:
            # Fallback to grayscale variance
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            return min(1.0, np.var(gray) / (255.0**2))
    
    def _check_background_uniformity(self, roi: np.ndarray) -> float:
        """Check how uniform the background of a region is"""
        if roi.size == 0:
            return 0.0
        
        # Sample center region (avoid edges which might have borders)
        h, w = roi.shape[:2]
        center_h, center_w = h//3, w//3
        
        if center_h < 10 or center_w < 10:
            return 0.0
        
        center_roi = roi[center_h:h-center_h, center_w:w-center_w]
        
        # Calculate color uniformity in center region
        color_variance = self._calculate_color_variance(center_roi)
        uniformity = 1.0 - color_variance
        
        return max(0.0, uniformity)