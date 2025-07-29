import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
import json
import uuid
import logging

# Import the new advanced UI detector
try:
    from advanced_ui_detector import AdvancedUIDetector, DetectedElement
    ADVANCED_DETECTOR_AVAILABLE = True
except ImportError:
    ADVANCED_DETECTOR_AVAILABLE = False
    logging.warning("Advanced UI detector not available. Falling back to basic detection.")

# Optional imports with error handling
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logging.warning("easyocr not available. OCR functionality will be limited.")

try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available. Color analysis will be limited.")


@dataclass
class UIElement:
    """Represents a detected UI element with all its features"""
    element_id: str
    element_type: str
    text: str
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]
    size: Tuple[int, int]  # width, height
    prominence: float
    visibility: bool
    color_features: Dict[str, Any]
    position_features: Dict[str, Any]


class ScreenshotProcessor:
    """
    Processes PNG screenshots to extract UI elements and their features
    """
    
    def __init__(self, use_advanced_detector: bool = True):
        # Initialize advanced UI detector if available
        self.use_advanced_detector = use_advanced_detector and ADVANCED_DETECTOR_AVAILABLE
        if self.use_advanced_detector:
            try:
                self.advanced_detector = AdvancedUIDetector()
                logging.info("Advanced UI detector initialized successfully")
            except Exception as e:
                logging.error(f"Failed to initialize advanced detector: {e}")
                self.use_advanced_detector = False
                self.advanced_detector = None
        else:
            self.advanced_detector = None
        
        # Initialize OCR reader if available
        if EASYOCR_AVAILABLE:
            try:
                self.ocr_reader = easyocr.Reader(['en'])
            except Exception as e:
                logging.error(f"Failed to initialize OCR reader: {e}")
                self.ocr_reader = None
        else:
            self.ocr_reader = None
            
        self.ui_element_types = {
            'button': ['button', 'btn', 'submit', 'click', 'continue', 'next', 'back'],
            'link': ['link', 'href', 'more', 'learn', 'read', 'view'],
            'form': ['input', 'text', 'email', 'password', 'search'],
            'menu': ['menu', 'nav', 'dropdown', 'options'],
            'image': ['img', 'photo', 'picture', 'banner']
        }
    
    def process_screenshot(self, screenshot_path: str) -> Dict[str, Any]:
        """
        Main processing pipeline for screenshot analysis
        
        Args:
            screenshot_path: Path to PNG screenshot
            
        Returns:
            Dictionary containing detected UI elements and metadata
        """
        # Load and preprocess image
        image = cv2.imread(screenshot_path)
        if image is None:
            raise ValueError(f"Could not load image from {screenshot_path}")
        
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Extract UI elements
        elements = self._extract_ui_elements(image)
        
        # Extract text and enhance elements
        elements = self._extract_text_features(image, elements)
        
        # Calculate visual features
        elements = self._calculate_visual_features(image, elements)
        
        # Calculate position features
        elements = self._calculate_position_features(elements, width, height)
        
        # Calculate prominence scores
        elements = self._calculate_prominence(elements, width, height)
        
        # Filter out elements with empty text and low prominence
        filtered_elements = []
        for elem in elements:
            # Keep element if it has meaningful text OR high visual prominence
            if (elem.text and elem.text.strip()) or elem.prominence > 0.4:
                filtered_elements.append(elem)
        
        logging.info(f"Filtered {len(elements)} -> {len(filtered_elements)} elements (removed empty text)")
        
        return {
            'screen_dimensions': [width, height],
            'total_elements': len(filtered_elements),
            'elements': [self._element_to_dict(elem) for elem in filtered_elements],
            'processing_metadata': {
                'image_path': screenshot_path,
                'processing_success': True,
                'detection_method': 'advanced' if self.use_advanced_detector else 'basic',
                'advanced_detector_available': ADVANCED_DETECTOR_AVAILABLE,
                'ocr_available': EASYOCR_AVAILABLE,
                'original_elements': len(elements),
                'filtered_elements': len(filtered_elements)
            }
        }
    
    def _extract_ui_elements(self, image: np.ndarray) -> List[UIElement]:
        """Extract UI elements using computer vision techniques"""
        if self.use_advanced_detector and self.advanced_detector:
            # Use advanced multi-technique detection
            return self._extract_ui_elements_advanced(image)
        else:
            # Fall back to basic detection
            return self._extract_ui_elements_basic(image)
    
    def _extract_ui_elements_advanced(self, image: np.ndarray) -> List[UIElement]:
        """Extract UI elements using the advanced detector"""
        elements = []
        
        try:
            # Use the advanced detector
            detected_elements = self.advanced_detector.detect_elements(image)
            
            # Convert DetectedElement objects to UIElement objects
            for det_elem in detected_elements:
                ui_element = UIElement(
                    element_id=det_elem.element_id,
                    element_type=det_elem.element_type,
                    text="",  # Will be filled later by OCR
                    bbox=det_elem.bbox,
                    center=det_elem.center,
                    size=det_elem.size,
                    prominence=det_elem.confidence,  # Use detection confidence as initial prominence
                    visibility=True,
                    color_features={},  # Will be calculated later
                    position_features={}  # Will be calculated later
                )
                elements.append(ui_element)
            
            logging.info(f"Advanced detector found {len(elements)} elements")
            return elements
            
        except Exception as e:
            logging.error(f"Advanced detection failed: {e}")
            # Fall back to basic detection
            return self._extract_ui_elements_basic(image)
    
    def _extract_ui_elements_basic(self, image: np.ndarray) -> List[UIElement]:
        """Extract UI elements using basic computer vision techniques (fallback)"""
        elements = []
        
        # Convert to grayscale for better edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect rectangular elements (buttons, forms, etc.)
        elements.extend(self._detect_rectangular_elements(gray))
        
        # Detect text-based elements
        elements.extend(self._detect_text_elements(image))
        
        # Remove duplicate elements
        elements = self._remove_duplicates(elements)
        
        logging.info(f"Basic detector found {len(elements)} elements")
        return elements
    
    def _detect_rectangular_elements(self, gray_image: np.ndarray) -> List[UIElement]:
        """Detect rectangular UI elements like buttons and forms"""
        elements = []
        
        # Apply edge detection
        edges = cv2.Canny(gray_image, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Filter by area
            area = cv2.contourArea(contour)
            if area < 100 or area > 50000:  # Skip very small or very large elements
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by aspect ratio (typical for buttons)
            aspect_ratio = w / h
            if aspect_ratio < 0.3 or aspect_ratio > 10:
                continue
            
            # Create UI element
            element = UIElement(
                element_id=f"rect_{uuid.uuid4().hex[:8]}",
                element_type="button",  # Default, will be refined later
                text="",
                bbox=(x, y, x + w, y + h),
                center=(x + w // 2, y + h // 2),
                size=(w, h),
                prominence=0.0,
                visibility=True,
                color_features={},
                position_features={}
            )
            elements.append(element)
        
        return elements
    
    def _detect_text_elements(self, image: np.ndarray) -> List[UIElement]:
        """Detect text-based UI elements using OCR"""
        elements = []
        
        # Skip OCR if not available
        if not self.ocr_reader:
            logging.warning("OCR not available, skipping text detection")
            return elements
        
        try:
            # Run OCR
            ocr_results = self.ocr_reader.readtext(image)
            
            for (bbox, text, confidence) in ocr_results:
                if confidence < 0.5:  # Skip low confidence text
                    continue
                
                # Convert bbox to standard format
                x1, y1 = int(min([point[0] for point in bbox])), int(min([point[1] for point in bbox]))
                x2, y2 = int(max([point[0] for point in bbox])), int(max([point[1] for point in bbox]))
                
                # Determine element type based on text content
                element_type = self._classify_element_type(text)
                
                element = UIElement(
                    element_id=f"text_{uuid.uuid4().hex[:8]}",
                    element_type=element_type,
                    text=text,
                    bbox=(x1, y1, x2, y2),
                    center=((x1 + x2) // 2, (y1 + y2) // 2),
                    size=(x2 - x1, y2 - y1),
                    prominence=0.0,
                    visibility=True,
                    color_features={},
                    position_features={}
                )
                elements.append(element)
                
        except Exception as e:
            logging.error(f"OCR detection failed: {e}")
        
        return elements
    
    def _classify_element_type(self, text: str) -> str:
        """Classify UI element type based on text content"""
        text_lower = text.lower()
        
        for element_type, keywords in self.ui_element_types.items():
            if any(keyword in text_lower for keyword in keywords):
                return element_type
        
        return "text"
    
    def _extract_text_features(self, image: np.ndarray, elements: List[UIElement]) -> List[UIElement]:
        """Enhanced text extraction with better OCR integration"""
        if not self.ocr_reader:
            return elements
        
        # First, run full-image OCR to get all text with proper grouping
        try:
            ocr_results = self.ocr_reader.readtext(image, detail=1)
            all_text_elements = self._process_ocr_results(ocr_results)
            logging.info(f"OCR found {len(all_text_elements)} text elements")
        except Exception as e:
            logging.error(f"OCR failed: {e}")
            return elements
        
        enhanced_elements = []
        
        # Match visual elements with OCR text
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
                else:
                    # Keep element without text if it has high visual prominence
                    if element.prominence > 0.5:
                        enhanced_elements.append(element)
        
        # Add pure text elements that weren't matched to visual elements
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
                    prominence=min(0.8, text_elem['confidence']),
                    visibility=True,
                    color_features={},
                    position_features={}
                )
                enhanced_elements.append(ui_element)
        
        logging.info(f"Enhanced {len(elements)} visual -> {len(enhanced_elements)} total elements")
        return enhanced_elements
    
    def _process_ocr_results(self, ocr_results: List) -> List[Dict]:
        """Process OCR results with text grouping and merging"""
        if not ocr_results:
            return []
        
        # Convert to workable format and filter low confidence
        text_items = []
        for (bbox, text, confidence) in ocr_results:
            if confidence < 0.3 or not text.strip():
                continue
            
            x1, y1 = int(min([p[0] for p in bbox])), int(min([p[1] for p in bbox]))
            x2, y2 = int(max([p[0] for p in bbox])), int(max([p[1] for p in bbox]))
            
            text_items.append({
                'text': text.strip(),
                'bbox': (x1, y1, x2, y2),
                'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                'confidence': confidence,
                'grouped': False
            })
        
        # Group nearby text items (for multi-word elements like "Sign In")
        grouped_text = self._group_nearby_text(text_items)
        
        return grouped_text
    
    def _group_nearby_text(self, text_items: List[Dict]) -> List[Dict]:
        """Group nearby text items that likely belong together"""
        if not text_items:
            return []
        
        groups = []
        threshold_distance = 40  # pixels
        
        for item in text_items:
            if item['grouped']:
                continue
            
            # Start new group with current item
            current_group = [item]
            item['grouped'] = True
            
            # Find nearby items to group
            for other_item in text_items:
                if other_item['grouped']:
                    continue
                
                # Calculate distance
                dx = abs(item['center'][0] - other_item['center'][0])
                dy = abs(item['center'][1] - other_item['center'][1])
                distance = np.sqrt(dx*dx + dy*dy)
                
                # Group if close and on similar horizontal line
                if distance < threshold_distance and dy < 15:
                    current_group.append(other_item)
                    other_item['grouped'] = True
            
            # Merge the group into a single text element
            merged = self._merge_text_group(current_group)
            if merged:
                groups.append(merged)
        
        return groups
    
    def _merge_text_group(self, group: List[Dict]) -> Dict:
        """Merge a group of text items into a single element"""
        if not group:
            return None
        
        # Sort by horizontal position for proper word order
        group.sort(key=lambda x: x['center'][0])
        
        # Merge text with spaces
        merged_text = ' '.join([item['text'] for item in group])
        
        # Calculate combined bounding box
        x1 = min([item['bbox'][0] for item in group])
        y1 = min([item['bbox'][1] for item in group])
        x2 = max([item['bbox'][2] for item in group])
        y2 = max([item['bbox'][3] for item in group])
        
        # Average confidence
        avg_confidence = sum([item['confidence'] for item in group]) / len(group)
        
        return {
            'text': merged_text,
            'bbox': (x1, y1, x2, y2),
            'center': ((x1 + x2) // 2, (y1 + y2) // 2),
            'confidence': avg_confidence
        }
    
    def _find_matching_text(self, element: UIElement, text_elements: List[Dict]) -> str:
        """Find text that overlaps with a visual element"""
        elem_x1, elem_y1, elem_x2, elem_y2 = element.bbox
        
        best_match = None
        best_overlap = 0
        
        for text_elem in text_elements:
            text_x1, text_y1, text_x2, text_y2 = text_elem['bbox']
            
            # Calculate overlap area
            overlap_x1 = max(elem_x1, text_x1)
            overlap_y1 = max(elem_y1, text_y1) 
            overlap_x2 = min(elem_x2, text_x2)
            overlap_y2 = min(elem_y2, text_y2)
            
            if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
                overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                text_area = (text_x2 - text_x1) * (text_y2 - text_y1)
                
                if text_area > 0:
                    overlap_ratio = overlap_area / text_area
                    
                    # Prefer higher confidence and better overlap
                    score = overlap_ratio * text_elem['confidence']
                    
                    if score > best_overlap and overlap_ratio > 0.2:
                        best_overlap = score
                        best_match = text_elem['text']
        
        return best_match
    
    def _is_text_already_used(self, text_elem: Dict, enhanced_elements: List[UIElement]) -> bool:
        """Check if text is already used in enhanced elements"""
        text_to_check = text_elem['text'].lower().strip()
        
        for element in enhanced_elements:
            if element.text and text_to_check in element.text.lower():
                return True
            
            # Also check for substantial bbox overlap
            elem_x1, elem_y1, elem_x2, elem_y2 = element.bbox
            text_x1, text_y1, text_x2, text_y2 = text_elem['bbox']
            
            overlap_x1 = max(elem_x1, text_x1)
            overlap_y1 = max(elem_y1, text_y1)
            overlap_x2 = min(elem_x2, text_x2)
            overlap_y2 = min(elem_y2, text_y2)
            
            if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
                overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                text_area = (text_x2 - text_x1) * (text_y2 - text_y1)
                
                if text_area > 0 and overlap_area / text_area > 0.5:
                    return True
        
        return False
    
    def _calculate_visual_features(self, image: np.ndarray, elements: List[UIElement]) -> List[UIElement]:
        """Calculate visual features like color and contrast"""
        enhanced_elements = []
        
        for element in elements:
            x1, y1, x2, y2 = element.bbox
            
            # Ensure valid ROI bounds
            if (x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or 
                x2 >= image.shape[1] or y2 >= image.shape[0]):
                enhanced_elements.append(element)
                continue
                
            roi = image[y1:y2, x1:x2]
            
            color_features = {}
            if roi.size > 0:
                try:
                    # Calculate dominant color
                    dominant_color = self._get_dominant_color(roi)
                    
                    # Calculate contrast
                    contrast = self._calculate_contrast(roi)
                    
                    color_features = {
                        'dominant_color': dominant_color,
                        'contrast': contrast,
                        'brightness': np.mean(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
                    }
                except Exception as e:
                    logging.debug(f"Failed to calculate visual features for element {element.element_id}: {e}")
                    color_features = {}
            
            # Create new element with visual features
            enhanced_element = UIElement(
                element_id=element.element_id,
                element_type=element.element_type,
                text=element.text,
                bbox=element.bbox,
                center=element.center,
                size=element.size,
                prominence=element.prominence,
                visibility=element.visibility,
                color_features=color_features,
                position_features=element.position_features
            )
            enhanced_elements.append(enhanced_element)
        
        return enhanced_elements
    
    def _get_dominant_color(self, roi: np.ndarray) -> str:
        """Get dominant color of a region"""
        try:
            if SKLEARN_AVAILABLE:
                # Reshape the image to be a list of pixels
                pixels = roi.reshape(-1, 3)
                
                # Use k-means to find dominant color
                kmeans = KMeans(n_clusters=1, random_state=42, n_init=10)
                kmeans.fit(pixels)
                
                # Get the dominant color
                dominant_color = kmeans.cluster_centers_[0]
                
                # Convert to hex
                return f"#{int(dominant_color[2]):02x}{int(dominant_color[1]):02x}{int(dominant_color[0]):02x}"
            else:
                # Fallback: use mean color
                mean_color = np.mean(roi, axis=(0, 1))
                return f"#{int(mean_color[2]):02x}{int(mean_color[1]):02x}{int(mean_color[0]):02x}"
        except Exception as e:
            logging.debug(f"Failed to calculate dominant color: {e}")
            return "#808080"  # Default gray
    
    def _calculate_contrast(self, roi: np.ndarray) -> float:
        """Calculate contrast of a region"""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return np.std(gray) / 255.0
    
    def _calculate_position_features(self, elements: List[UIElement], width: int, height: int) -> List[UIElement]:
        """Calculate position-based features"""
        enhanced_elements = []
        
        for element in elements:
            center_x, center_y = element.center
            
            # Relative position (0-1)
            rel_x = center_x / width if width > 0 else 0.5
            rel_y = center_y / height if height > 0 else 0.5
            
            # Quadrant
            quadrant = self._get_quadrant(rel_x, rel_y)
            
            # Distance from center
            center_distance = np.sqrt((rel_x - 0.5)**2 + (rel_y - 0.5)**2)
            
            position_features = {
                'relative_x': rel_x,
                'relative_y': rel_y,
                'quadrant': quadrant,
                'center_distance': center_distance,
                'is_top_half': rel_y < 0.5,
                'is_left_half': rel_x < 0.5
            }
            
            # Create new element with position features
            enhanced_element = UIElement(
                element_id=element.element_id,
                element_type=element.element_type,
                text=element.text,
                bbox=element.bbox,
                center=element.center,
                size=element.size,
                prominence=element.prominence,
                visibility=element.visibility,
                color_features=element.color_features,
                position_features=position_features
            )
            enhanced_elements.append(enhanced_element)
        
        return enhanced_elements
    
    def _get_quadrant(self, rel_x: float, rel_y: float) -> str:
        """Determine which quadrant the element is in"""
        if rel_x < 0.5 and rel_y < 0.5:
            return "top_left"
        elif rel_x >= 0.5 and rel_y < 0.5:
            return "top_right"
        elif rel_x < 0.5 and rel_y >= 0.5:
            return "bottom_left"
        else:
            return "bottom_right"
    
    def _calculate_prominence(self, elements: List[UIElement], width: int, height: int) -> List[UIElement]:
        """Calculate prominence score for each element"""
        enhanced_elements = []
        
        for element in elements:
            # Size factor
            element_area = element.size[0] * element.size[1]
            total_area = width * height if width > 0 and height > 0 else 1
            size_factor = element_area / total_area
            
            # Position factor (center elements are more prominent)
            position_factor = 1.0 - element.position_features.get('center_distance', 0.5)
            
            # Contrast factor
            contrast_factor = element.color_features.get('contrast', 0.5)
            
            # Text factor (elements with action words are more prominent)
            text_factor = self._calculate_text_prominence(element.text)
            
            # Combine factors
            prominence = (
                size_factor * 0.3 +
                position_factor * 0.3 +
                contrast_factor * 0.2 +
                text_factor * 0.2
            )
            
            calculated_prominence = min(1.0, prominence)
            
            # Create new element with calculated prominence
            enhanced_element = UIElement(
                element_id=element.element_id,
                element_type=element.element_type,
                text=element.text,
                bbox=element.bbox,
                center=element.center,
                size=element.size,
                prominence=calculated_prominence,
                visibility=element.visibility,
                color_features=element.color_features,
                position_features=element.position_features
            )
            enhanced_elements.append(enhanced_element)
        
        return enhanced_elements
    
    def _calculate_text_prominence(self, text: str) -> float:
        """Calculate how prominent text suggests an element is"""
        if not text:
            return 0.0
        
        action_words = ['click', 'buy', 'purchase', 'continue', 'next', 'submit', 'login', 'signup']
        urgency_words = ['now', 'today', 'limited', 'sale', 'offer']
        
        text_lower = text.lower()
        
        score = 0.0
        for word in action_words:
            if word in text_lower:
                score += 0.3
        
        for word in urgency_words:
            if word in text_lower:
                score += 0.2
        
        return min(1.0, score)
    
    def _remove_duplicates(self, elements: List[UIElement]) -> List[UIElement]:
        """Remove duplicate elements based on overlap"""
        unique_elements = []
        
        for element in elements:
            is_duplicate = False
            for existing in unique_elements:
                if self._calculate_overlap(element.bbox, existing.bbox) > 0.7:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_elements.append(element)
        
        return unique_elements
    
    def _calculate_overlap(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate overlap ratio between two bounding boxes"""
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
    
    def _element_to_dict(self, element: UIElement) -> Dict[str, Any]:
        """Convert UIElement to dictionary for JSON serialization"""
        return {
            'id': element.element_id,
            'type': element.element_type,
            'text': element.text,
            'bbox': element.bbox,
            'center': element.center,
            'size': element.size,
            'prominence': element.prominence,
            'visibility': element.visibility,
            'color_features': element.color_features,
            'position_features': element.position_features
        }