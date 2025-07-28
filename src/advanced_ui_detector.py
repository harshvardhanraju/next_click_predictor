import cv2
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
import logging
from sklearn.cluster import DBSCAN
import uuid

# Import modern UI patterns if available
try:
    from modern_ui_patterns import ModernUIPatterns
    MODERN_PATTERNS_AVAILABLE = True
except ImportError:
    MODERN_PATTERNS_AVAILABLE = False
    logging.warning("Modern UI patterns module not available")

@dataclass
class DetectedElement:
    """Enhanced detected UI element with confidence scores"""
    element_id: str
    element_type: str
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]
    size: Tuple[int, int]
    confidence: float
    detection_method: str
    visual_features: Dict[str, Any]


class ButtonDetector:
    """Specialized detector for button elements"""
    
    def __init__(self):
        self.button_templates = self._load_button_templates()
        
    def detect(self, image: np.ndarray) -> List[DetectedElement]:
        """Detect button elements using multiple techniques"""
        elements = []
        
        # Method 1: Template matching for common button shapes
        elements.extend(self._template_matching(image))
        
        # Method 2: Color-based button detection
        elements.extend(self._color_based_detection(image))
        
        # Method 3: Morphological button detection
        elements.extend(self._morphological_detection(image))
        
        return elements
    
    def _template_matching(self, image: np.ndarray) -> List[DetectedElement]:
        """Detect buttons using template matching"""
        elements = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        for template_name, template in self.button_templates.items():
            # Multi-scale template matching
            for scale in [0.5, 0.75, 1.0, 1.25, 1.5]:
                scaled_template = cv2.resize(template, None, fx=scale, fy=scale)
                if scaled_template.shape[0] > gray.shape[0] or scaled_template.shape[1] > gray.shape[1]:
                    continue
                
                result = cv2.matchTemplate(gray, scaled_template, cv2.TM_CCOEFF_NORMED)
                locations = np.where(result >= 0.6)  # Confidence threshold
                
                for pt in zip(*locations[::-1]):
                    h, w = scaled_template.shape
                    x1, y1 = pt
                    x2, y2 = x1 + w, y1 + h
                    
                    elements.append(DetectedElement(
                        element_id=f"btn_template_{uuid.uuid4().hex[:8]}",
                        element_type="button",
                        bbox=(x1, y1, x2, y2),
                        center=(x1 + w//2, y1 + h//2),
                        size=(w, h),
                        confidence=float(result[y1, x1]),
                        detection_method="template_matching",
                        visual_features={"template_name": template_name, "scale": scale}
                    ))
        
        return elements
    
    def _color_based_detection(self, image: np.ndarray) -> List[DetectedElement]:
        """Detect buttons based on common button colors"""
        elements = []
        
        # Common button color ranges in HSV
        button_colors = [
            # Blue buttons
            (np.array([100, 50, 50]), np.array([130, 255, 255])),
            # Green buttons (success/continue)
            (np.array([40, 50, 50]), np.array([80, 255, 255])),
            # Red buttons (danger/delete)
            (np.array([0, 50, 50]), np.array([20, 255, 255])),
            # Orange buttons (warning/action)
            (np.array([10, 50, 50]), np.array([25, 255, 255]))
        ]
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        for i, (lower, upper) in enumerate(button_colors):
            mask = cv2.inRange(hsv, lower, upper)
            
            # Morphological operations to clean up the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 200 or area > 10000:  # Filter by reasonable button size
                    continue
                
                # Check if contour is roughly rectangular
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) >= 4:  # Roughly rectangular
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Check aspect ratio (buttons are typically not too tall or wide)
                    aspect_ratio = w / h
                    if 0.2 <= aspect_ratio <= 8.0:
                        confidence = min(0.8, area / 1000.0)  # Higher area = higher confidence
                        
                        elements.append(DetectedElement(
                            element_id=f"btn_color_{uuid.uuid4().hex[:8]}",
                            element_type="button",
                            bbox=(x, y, x + w, y + h),
                            center=(x + w//2, y + h//2),
                            size=(w, h),
                            confidence=confidence,
                            detection_method="color_based",
                            visual_features={"color_range": i, "area": area}
                        ))
        
        return elements
    
    def _morphological_detection(self, image: np.ndarray) -> List[DetectedElement]:
        """Detect buttons using morphological operations"""
        elements = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive threshold to handle varying lighting
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Morphological operations to find button-like shapes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 300 or area > 15000:
                continue
            
            # Check if contour is convex (buttons are typically convex)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            
            if hull_area > 0:
                solidity = area / hull_area
                if solidity > 0.7:  # Reasonably solid shape
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    
                    if 0.3 <= aspect_ratio <= 6.0:  # Reasonable button aspect ratio
                        confidence = min(0.7, solidity * 0.8)
                        
                        elements.append(DetectedElement(
                            element_id=f"btn_morph_{uuid.uuid4().hex[:8]}",
                            element_type="button",
                            bbox=(x, y, x + w, y + h),
                            center=(x + w//2, y + h//2),
                            size=(w, h),
                            confidence=confidence,
                            detection_method="morphological",
                            visual_features={"solidity": solidity, "area": area}
                        ))
        
        return elements
    
    def _load_button_templates(self) -> Dict[str, np.ndarray]:
        """Create synthetic button templates for matching"""
        templates = {}
        
        # Rectangular button template
        rect_template = np.zeros((40, 120), dtype=np.uint8)
        cv2.rectangle(rect_template, (5, 5), (115, 35), 255, 2)
        templates["rectangular"] = rect_template
        
        # Rounded button template  
        rounded_template = np.zeros((40, 120), dtype=np.uint8)
        cv2.ellipse(rounded_template, (60, 20), (55, 15), 0, 0, 360, 255, 2)
        templates["rounded"] = rounded_template
        
        # Small square button template
        square_template = np.zeros((30, 30), dtype=np.uint8)
        cv2.rectangle(square_template, (3, 3), (27, 27), 255, 2)
        templates["square"] = square_template
        
        return templates


class FormDetector:
    """Specialized detector for form elements"""
    
    def detect(self, image: np.ndarray) -> List[DetectedElement]:
        """Detect form input fields"""
        elements = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Detect rectangular input fields with borders
        elements.extend(self._detect_bordered_inputs(gray))
        
        # Method 2: Detect text input areas by texture
        elements.extend(self._detect_text_areas(image))
        
        return elements
    
    def _detect_bordered_inputs(self, gray: np.ndarray) -> List[DetectedElement]:
        """Detect input fields with visible borders"""
        elements = []
        
        # Use edge detection to find rectangular borders
        edges = cv2.Canny(gray, 30, 100)
        
        # Dilate to connect broken lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500 or area > 20000:  # Form fields are medium sized
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            # Form fields are typically wide and short
            if aspect_ratio > 2.0 and h < 50:
                confidence = min(0.8, aspect_ratio / 10.0)
                
                elements.append(DetectedElement(
                    element_id=f"form_border_{uuid.uuid4().hex[:8]}",
                    element_type="form",
                    bbox=(x, y, x + w, y + h),
                    center=(x + w//2, y + h//2),
                    size=(w, h),
                    confidence=confidence,
                    detection_method="border_detection",
                    visual_features={"aspect_ratio": aspect_ratio}
                ))
        
        return elements
    
    def _detect_text_areas(self, image: np.ndarray) -> List[DetectedElement]:
        """Detect text input areas by analyzing texture"""
        elements = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Look for areas with consistent light background (typical of input fields)
        # Apply Gaussian blur to smooth the image
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Threshold to find light areas
        _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to find rectangular regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 800 or area > 25000:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            # Check if it looks like a text input field
            if aspect_ratio > 1.5 and 15 < h < 60:
                confidence = min(0.7, (aspect_ratio - 1.5) / 5.0)
                
                elements.append(DetectedElement(
                    element_id=f"form_text_{uuid.uuid4().hex[:8]}",
                    element_type="form",
                    bbox=(x, y, x + w, y + h),
                    center=(x + w//2, y + h//2),
                    size=(w, h),
                    confidence=confidence,
                    detection_method="texture_analysis",
                    visual_features={"aspect_ratio": aspect_ratio, "background_light": True}
                ))
        
        return elements


class LinkDetector:
    """Specialized detector for link elements"""
    
    def detect(self, image: np.ndarray) -> List[DetectedElement]:
        """Detect clickable links (typically underlined text)"""
        elements = []
        
        # Method 1: Detect underlined text
        elements.extend(self._detect_underlined_text(image))
        
        # Method 2: Detect blue text (common link color)
        elements.extend(self._detect_blue_text(image))
        
        return elements
    
    def _detect_underlined_text(self, image: np.ndarray) -> List[DetectedElement]:
        """Detect underlined text which often indicates links"""
        elements = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Create horizontal line detection kernel
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        
        # Apply morphological operations to detect horizontal lines
        detected_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Threshold to get clear lines
        _, thresh = cv2.threshold(detected_lines, 50, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Look for thin horizontal lines
            if w > 30 and h < 5:
                # Expand vertically to include the text above the underline
                text_y = max(0, y - 20)
                text_h = 25
                
                confidence = min(0.6, w / 100.0)
                
                elements.append(DetectedElement(
                    element_id=f"link_underline_{uuid.uuid4().hex[:8]}",
                    element_type="link",
                    bbox=(x, text_y, x + w, text_y + text_h),
                    center=(x + w//2, text_y + text_h//2),
                    size=(w, text_h),
                    confidence=confidence,
                    detection_method="underline_detection",
                    visual_features={"underline_width": w}
                ))
        
        return elements
    
    def _detect_blue_text(self, image: np.ndarray) -> List[DetectedElement]:
        """Detect blue text which commonly indicates links"""
        elements = []
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Blue color range for links
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
        
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Morphological operations to group nearby blue pixels
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 50 or area > 5000:  # Links are typically small to medium
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            # Links are typically wider than tall
            if aspect_ratio > 1.2:
                confidence = min(0.7, area / 1000.0)
                
                elements.append(DetectedElement(
                    element_id=f"link_blue_{uuid.uuid4().hex[:8]}",
                    element_type="link",
                    bbox=(x, y, x + w, y + h),
                    center=(x + w//2, y + h//2),
                    size=(w, h),
                    confidence=confidence,
                    detection_method="blue_text_detection",
                    visual_features={"area": area, "aspect_ratio": aspect_ratio}
                ))
        
        return elements


class TextDetector:
    """Specialized detector for readable text elements"""
    
    def detect(self, image: np.ndarray) -> List[DetectedElement]:
        """Detect text regions that might be clickable"""
        elements = []
        
        # Use MSER (Maximally Stable Extremal Regions) for text detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Create MSER detector
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)
        
        for region in regions:
            if len(region) < 50:  # Skip very small regions
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(region)
            
            # Filter by reasonable text dimensions
            if w < 20 or h < 8 or w > 300 or h > 50:
                continue
            
            aspect_ratio = w / h
            if aspect_ratio < 1.0 or aspect_ratio > 15.0:  # Text is wider than tall
                continue
            
            confidence = min(0.5, len(region) / 500.0)
            
            elements.append(DetectedElement(
                element_id=f"text_mser_{uuid.uuid4().hex[:8]}",
                element_type="text",
                bbox=(x, y, x + w, y + h),
                center=(x + w//2, y + h//2),
                size=(w, h),
                confidence=confidence,
                detection_method="mser_text_detection",
                visual_features={"region_size": len(region), "aspect_ratio": aspect_ratio}
            ))
        
        return elements


class AdvancedUIDetector:
    """Advanced UI element detector using multiple specialized detectors"""
    
    def __init__(self):
        self.button_detector = ButtonDetector()
        self.form_detector = FormDetector()
        self.link_detector = LinkDetector()
        self.text_detector = TextDetector()
        
        # Initialize modern UI patterns detector
        if MODERN_PATTERNS_AVAILABLE:
            self.modern_patterns = ModernUIPatterns()
            self.use_modern_patterns = True
        else:
            self.modern_patterns = None
            self.use_modern_patterns = False
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
    
    def detect_elements(self, image: np.ndarray) -> List[DetectedElement]:
        """
        Detect UI elements using multi-stage detection pipeline
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of detected UI elements with confidence scores
        """
        all_candidates = []
        
        try:
            # Stage 1: Run all specialized detectors
            self.logger.info("Running button detection...")
            all_candidates.extend(self.button_detector.detect(image))
            
            self.logger.info("Running form detection...")
            all_candidates.extend(self.form_detector.detect(image))
            
            self.logger.info("Running link detection...")
            all_candidates.extend(self.link_detector.detect(image))
            
            self.logger.info("Running text detection...")
            all_candidates.extend(self.text_detector.detect(image))
            
            self.logger.info(f"Total candidates found: {len(all_candidates)}")
            
            # Stage 2: Non-maximum suppression to remove duplicates
            filtered_elements = self._non_maximum_suppression(all_candidates)
            
            # Stage 3: Refine bounding boxes
            refined_elements = self._refine_bounding_boxes(image, filtered_elements)
            
            # Stage 4: Apply modern UI pattern recognition
            if self.use_modern_patterns and self.modern_patterns:
                self.logger.info("Applying modern UI pattern recognition...")
                pattern_enhanced_elements = self.modern_patterns.detect_modern_patterns(image, refined_elements)
                
                # Remove duplicates after pattern detection
                pattern_enhanced_elements = self._non_maximum_suppression(pattern_enhanced_elements, 0.5)
                
                self.logger.info(f"Elements after modern pattern recognition: {len(pattern_enhanced_elements)}")
                refined_elements = pattern_enhanced_elements
            
            # Stage 5: Calculate final confidence scores
            final_elements = self._calculate_final_confidence(refined_elements)
            
            self.logger.info(f"Final elements after processing: {len(final_elements)}")
            
            return final_elements
            
        except Exception as e:
            self.logger.error(f"Element detection failed: {e}")
            return []
    
    def _non_maximum_suppression(self, elements: List[DetectedElement], 
                                overlap_threshold: float = 0.5) -> List[DetectedElement]:
        """Remove overlapping detections, keeping highest confidence ones"""
        if not elements:
            return []
        
        # Sort by confidence (descending)
        sorted_elements = sorted(elements, key=lambda x: x.confidence, reverse=True)
        
        keep = []
        while sorted_elements:
            # Keep the highest confidence element
            current = sorted_elements.pop(0)
            keep.append(current)
            
            # Remove overlapping elements
            remaining = []
            for element in sorted_elements:
                overlap = self._calculate_overlap(current.bbox, element.bbox)
                if overlap < overlap_threshold:
                    remaining.append(element)
            
            sorted_elements = remaining
        
        return keep
    
    def _calculate_overlap(self, bbox1: Tuple[int, int, int, int], 
                          bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate IoU (Intersection over Union) between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def _refine_bounding_boxes(self, image: np.ndarray, 
                              elements: List[DetectedElement]) -> List[DetectedElement]:
        """Refine bounding box coordinates for better accuracy"""
        refined_elements = []
        
        for element in elements:
            try:
                refined_bbox = self._calculate_precise_bbox(image, element.bbox)
                
                # Update element with refined bbox
                x1, y1, x2, y2 = refined_bbox
                refined_element = DetectedElement(
                    element_id=element.element_id,
                    element_type=element.element_type,
                    bbox=refined_bbox,
                    center=(x1 + (x2-x1)//2, y1 + (y2-y1)//2),
                    size=(x2-x1, y2-y1),
                    confidence=element.confidence,
                    detection_method=element.detection_method,
                    visual_features=element.visual_features
                )
                
                refined_elements.append(refined_element)
                
            except Exception as e:
                self.logger.debug(f"Failed to refine bbox for element {element.element_id}: {e}")
                refined_elements.append(element)  # Keep original if refinement fails
        
        return refined_elements
    
    def _calculate_precise_bbox(self, image: np.ndarray, 
                               bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """Calculate precise bounding box using gradient analysis"""
        x1, y1, x2, y2 = bbox
        
        # Extract region of interest
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return bbox
        
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Calculate gradients
        grad_x = cv2.Sobel(gray_roi, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_roi, cv2.CV_64F, 0, 1, ksize=3)
        
        # Find edges based on gradient magnitude
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Find tight bounding box around high gradient areas
        threshold = np.percentile(gradient_magnitude, 75)  # Top 25% of gradients
        edge_mask = gradient_magnitude > threshold
        
        # Find non-zero pixels
        coords = np.column_stack(np.where(edge_mask))
        
        if len(coords) > 0:
            # Calculate tight bounding box
            min_row, min_col = coords.min(axis=0)
            max_row, max_col = coords.max(axis=0)
            
            # Add small padding and convert back to image coordinates
            padding = 2
            refined_x1 = max(0, x1 + min_col - padding)
            refined_y1 = max(0, y1 + min_row - padding)
            refined_x2 = min(image.shape[1], x1 + max_col + padding)
            refined_y2 = min(image.shape[0], y1 + max_row + padding)
            
            return (refined_x1, refined_y1, refined_x2, refined_y2)
        
        return bbox  # Return original if refinement fails
    
    def _calculate_final_confidence(self, elements: List[DetectedElement]) -> List[DetectedElement]:
        """Calculate final confidence scores based on multiple factors"""
        final_elements = []
        
        for element in elements:
            # Base confidence from detection method
            base_confidence = element.confidence
            
            # Size factor (medium-sized elements are more likely to be interactive)
            area = element.size[0] * element.size[1]
            if 500 <= area <= 10000:  # Optimal size range
                size_factor = 1.0
            elif area < 500:
                size_factor = 0.7  # Too small
            else:
                size_factor = 0.8  # Too large
            
            # Aspect ratio factor
            aspect_ratio = element.size[0] / element.size[1] if element.size[1] > 0 else 1.0
            if element.element_type == "button":
                # Buttons work well with aspect ratios between 1.5 and 4.0
                if 1.5 <= aspect_ratio <= 4.0:
                    aspect_factor = 1.0
                else:
                    aspect_factor = 0.8
            else:
                aspect_factor = 1.0
            
            # Detection method factor
            method_confidence = {
                "template_matching": 0.9,
                "color_based": 0.8,
                "morphological": 0.7,
                "border_detection": 0.8,
                "texture_analysis": 0.6,
                "underline_detection": 0.7,
                "blue_text_detection": 0.6,
                "mser_text_detection": 0.5
            }.get(element.detection_method, 0.5)
            
            # Calculate final confidence
            final_confidence = base_confidence * size_factor * aspect_factor * method_confidence
            final_confidence = min(0.95, final_confidence)  # Cap at 95%
            
            # Update element with final confidence
            final_element = DetectedElement(
                element_id=element.element_id,
                element_type=element.element_type,
                bbox=element.bbox,
                center=element.center,
                size=element.size,
                confidence=final_confidence,
                detection_method=element.detection_method,
                visual_features=element.visual_features
            )
            
            final_elements.append(final_element)
        
        # Sort by confidence (descending)
        final_elements.sort(key=lambda x: x.confidence, reverse=True)
        
        return final_elements