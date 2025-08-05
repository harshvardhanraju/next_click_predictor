import cv2
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
import logging
from sklearn.cluster import DBSCAN
import uuid

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
    logging.warning("scikit-learn not available. Some clustering features will be limited.")


@dataclass
class DetectedElement:
    """Simplified detected UI element with essential features"""
    element_id: str
    element_type: str
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]
    size: Tuple[int, int]
    confidence: float
    text: str
    visual_features: Dict[str, Any]


class ImprovedUIDetector:
    """
    Simplified UI detector focused on accuracy and reliability
    Uses only two proven methods: contour-based detection + OCR
    """
    
    def __init__(self):
        # Initialize OCR reader if available
        if EASYOCR_AVAILABLE:
            try:
                self.ocr_reader = easyocr.Reader(['en'], gpu=False)
                logging.info("EasyOCR initialized successfully")
            except Exception as e:
                logging.error(f"Failed to initialize OCR reader: {e}")
                self.ocr_reader = None
        else:
            self.ocr_reader = None
        
        # Element classification keywords
        self.element_keywords = {
            'button': {
                'high_confidence': ['submit', 'continue', 'next', 'buy', 'purchase', 'add to cart', 
                                  'sign in', 'log in', 'register', 'save', 'confirm', 'apply'],
                'medium_confidence': ['click', 'go', 'start', 'begin', 'proceed', 'finish', 'done']
            },
            'link': {
                'high_confidence': ['learn more', 'read more', 'view details', 'see all', 'more info'],
                'medium_confidence': ['here', 'link', 'visit', 'goto', 'open']
            },
            'form': {
                'high_confidence': ['email', 'password', 'username', 'search', 'enter', 'type'],
                'medium_confidence': ['input', 'field', 'box']
            }
        }
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
    
    def detect_elements(self, image: np.ndarray) -> List[DetectedElement]:
        """
        Main detection pipeline - simplified and focused
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of detected UI elements with confidence scores
        """
        try:
            # Step 1: Detect visual elements using improved contour analysis
            visual_elements = self._detect_visual_elements(image)
            self.logger.info(f"Detected {len(visual_elements)} visual elements")
            
            # Step 2: Detect text elements using improved OCR
            text_elements = self._detect_text_elements(image)
            self.logger.info(f"Detected {len(text_elements)} text elements")
            
            # Step 3: Merge visual and text elements intelligently
            merged_elements = self._merge_elements(visual_elements, text_elements)
            self.logger.info(f"Merged into {len(merged_elements)} final elements")
            
            # Step 4: Filter and rank by confidence
            filtered_elements = self._filter_and_rank(merged_elements)
            
            return filtered_elements
            
        except Exception as e:
            self.logger.error(f"Element detection failed: {e}")
            return []
    
    def _detect_visual_elements(self, image: np.ndarray) -> List[DetectedElement]:
        """Detect visual UI elements using improved contour analysis"""
        elements = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply multiple preprocessing techniques
        preprocessed_images = self._preprocess_for_detection(gray)
        
        for method_name, processed_img in preprocessed_images.items():
            method_elements = self._extract_contours(processed_img, method_name)
            elements.extend(method_elements)
        
        # Remove duplicates based on overlap
        unique_elements = self._remove_duplicate_elements(elements)
        
        return unique_elements
    
    def _preprocess_for_detection(self, gray: np.ndarray) -> Dict[str, np.ndarray]:
        """Apply different preprocessing techniques for robust detection"""
        results = {}
        
        # Method 1: Adaptive threshold for buttons and forms
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        results['adaptive'] = adaptive_thresh
        
        # Method 2: Edge detection for outlined elements
        edges = cv2.Canny(gray, 30, 100)
        
        # Dilate edges to connect broken lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)
        results['edges'] = edges_dilated
        
        # Method 3: Morphological operations for solid elements
        kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        morph = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel_rect)
        _, morph_thresh = cv2.threshold(morph, 50, 255, cv2.THRESH_BINARY)
        results['morphological'] = morph_thresh
        
        return results
    
    def _extract_contours(self, processed_img: np.ndarray, method: str) -> List[DetectedElement]:
        """Extract UI elements from processed image using contour analysis"""
        elements = []
        
        # Find contours
        contours, _ = cv2.findContours(processed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            element = self._analyze_contour(contour, method)
            if element:
                elements.append(element)
        
        return elements
    
    def _analyze_contour(self, contour, method: str) -> Optional[DetectedElement]:
        """Analyze a contour to determine if it's a valid UI element"""
        # Calculate basic properties
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter by area - more permissive range
        if area < 50 or area > 100000:
            return None
        
        # Calculate aspect ratio
        aspect_ratio = w / h if h > 0 else 0
        
        # Filter unrealistic aspect ratios
        if aspect_ratio < 0.1 or aspect_ratio > 20:
            return None
        
        # Calculate shape properties
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return None
            
        # Circularity (4π*area/perimeter²)
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Convexity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        convexity = area / hull_area if hull_area > 0 else 0
        
        # Determine element type and confidence based on shape
        element_type, confidence = self._classify_by_shape(
            area, aspect_ratio, circularity, convexity, method
        )
        
        if confidence < 0.3:  # Minimum confidence threshold
            return None
        
        # Create element
        return DetectedElement(
            element_id=f"visual_{method}_{uuid.uuid4().hex[:8]}",
            element_type=element_type,
            bbox=(x, y, x + w, y + h),
            center=(x + w // 2, y + h // 2),
            size=(w, h),
            confidence=confidence,
            text="",  # Will be filled by text detection
            visual_features={
                'area': area,
                'aspect_ratio': aspect_ratio,
                'circularity': circularity,
                'convexity': convexity,
                'detection_method': method
            }
        )
    
    def _classify_by_shape(self, area: float, aspect_ratio: float, 
                          circularity: float, convexity: float, method: str) -> Tuple[str, float]:
        """Classify element type based on geometric properties"""
        
        base_confidence = 0.5
        element_type = "unknown"
        
        # Button classification
        if (1.5 <= aspect_ratio <= 6.0 and  # Button-like aspect ratio
            0.3 <= circularity <= 1.0 and    # Reasonably round/rectangular
            convexity > 0.7 and              # Convex shape
            500 <= area <= 15000):           # Reasonable button size
            
            element_type = "button"
            base_confidence = 0.7
            
            # Higher confidence for more button-like shapes
            if 2.0 <= aspect_ratio <= 4.0:
                base_confidence += 0.1
            if circularity > 0.5:
                base_confidence += 0.1
                
        # Form field classification
        elif (aspect_ratio > 3.0 and        # Wide and short
              area > 800 and                # Large enough for text input
              convexity > 0.8):             # Very rectangular
            
            element_type = "form"
            base_confidence = 0.6
            
            # Higher confidence for very wide fields
            if aspect_ratio > 5.0:
                base_confidence += 0.1
                
        # Link/text classification (smaller, various shapes)
        elif (area < 3000 and               # Smaller elements
              aspect_ratio > 1.0):          # At least slightly wide
            
            element_type = "text"
            base_confidence = 0.4
        
        # Adjust confidence based on detection method
        method_multipliers = {
            'adaptive': 1.0,
            'edges': 0.9,
            'morphological': 0.8
        }
        
        final_confidence = base_confidence * method_multipliers.get(method, 0.7)
        return element_type, min(0.95, final_confidence)
    
    def _detect_text_elements(self, image: np.ndarray) -> List[DetectedElement]:
        """Detect text elements using improved OCR integration"""
        if not self.ocr_reader:
            self.logger.warning("OCR not available, skipping text detection")
            return []
        
        try:
            # Run OCR with detailed output
            ocr_results = self.ocr_reader.readtext(image, detail=1)
            
            # Convert OCR results to text elements
            text_elements = []
            for (bbox_points, text, confidence) in ocr_results:
                if confidence < 0.4 or not text.strip():  # Filter low confidence
                    continue
                
                # Convert bbox points to rectangle
                x_coords = [point[0] for point in bbox_points]
                y_coords = [point[1] for point in bbox_points]
                x1, y1 = int(min(x_coords)), int(min(y_coords))
                x2, y2 = int(max(x_coords)), int(max(y_coords))
                
                # Classify text element type
                element_type, text_confidence = self._classify_by_text(text)
                
                # Combine OCR confidence with text classification confidence
                final_confidence = confidence * text_confidence
                
                element = DetectedElement(
                    element_id=f"text_{uuid.uuid4().hex[:8]}",
                    element_type=element_type,
                    bbox=(x1, y1, x2, y2),
                    center=((x1 + x2) // 2, (y1 + y2) // 2),
                    size=(x2 - x1, y2 - y1),
                    confidence=final_confidence,
                    text=text.strip(),
                    visual_features={
                        'ocr_confidence': confidence,
                        'text_length': len(text.strip()),
                        'word_count': len(text.strip().split())
                    }
                )
                
                text_elements.append(element)
            
            # Group nearby text elements that belong together
            grouped_elements = self._group_nearby_text(text_elements)
            
            return grouped_elements
            
        except Exception as e:
            self.logger.error(f"Text detection failed: {e}")
            return []
    
    def _classify_by_text(self, text: str) -> Tuple[str, float]:
        """Classify element type based on text content"""
        text_lower = text.lower().strip()
        
        # Check each element type
        for element_type, keywords in self.element_keywords.items():
            # Check high confidence keywords
            for keyword in keywords.get('high_confidence', []):
                if keyword in text_lower:
                    return element_type, 0.9
            
            # Check medium confidence keywords
            for keyword in keywords.get('medium_confidence', []):
                if keyword in text_lower:
                    return element_type, 0.7
        
        # Default classification based on text characteristics
        if len(text_lower) < 3:
            return "text", 0.3
        elif any(char.isdigit() for char in text_lower):
            return "text", 0.4
        else:
            return "text", 0.5
    
    def _group_nearby_text(self, text_elements: List[DetectedElement]) -> List[DetectedElement]:
        """Group nearby text elements that likely belong together"""
        if not text_elements:
            return []
        
        # Use DBSCAN clustering to group nearby elements
        if not SKLEARN_AVAILABLE:
            return text_elements
        
        try:
            # Extract centers for clustering
            centers = np.array([elem.center for elem in text_elements])
            
            # Use DBSCAN with adaptive epsilon based on image size
            avg_size = np.mean([max(elem.size) for elem in text_elements])
            eps = max(30, avg_size * 0.5)  # Adaptive distance threshold
            
            clustering = DBSCAN(eps=eps, min_samples=1).fit(centers)
            labels = clustering.labels_
            
            # Group elements by cluster
            clusters = {}
            for i, label in enumerate(labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(text_elements[i])
            
            # Merge elements in each cluster
            grouped_elements = []
            for cluster_elements in clusters.values():
                if len(cluster_elements) == 1:
                    grouped_elements.append(cluster_elements[0])
                else:
                    merged_element = self._merge_text_cluster(cluster_elements)
                    if merged_element:
                        grouped_elements.append(merged_element)
            
            return grouped_elements
            
        except Exception as e:
            self.logger.debug(f"Text grouping failed: {e}")
            return text_elements
    
    def _merge_text_cluster(self, cluster_elements: List[DetectedElement]) -> Optional[DetectedElement]:
        """Merge a cluster of text elements into a single element"""
        if not cluster_elements:
            return None
        
        # Sort by horizontal position for correct word order
        cluster_elements.sort(key=lambda x: x.center[0])
        
        # Merge text with spaces
        merged_text = ' '.join([elem.text for elem in cluster_elements if elem.text])
        
        # Calculate combined bounding box
        x1 = min([elem.bbox[0] for elem in cluster_elements])
        y1 = min([elem.bbox[1] for elem in cluster_elements])
        x2 = max([elem.bbox[2] for elem in cluster_elements])
        y2 = max([elem.bbox[3] for elem in cluster_elements])
        
        # Use highest confidence and most specific type
        best_element = max(cluster_elements, key=lambda x: x.confidence)
        avg_confidence = np.mean([elem.confidence for elem in cluster_elements])
        
        return DetectedElement(
            element_id=f"merged_{uuid.uuid4().hex[:8]}",
            element_type=best_element.element_type,
            bbox=(x1, y1, x2, y2),
            center=((x1 + x2) // 2, (y1 + y2) // 2),
            size=(x2 - x1, y2 - y1),
            confidence=avg_confidence,
            text=merged_text,
            visual_features={
                'merged_from': len(cluster_elements),
                'original_confidences': [elem.confidence for elem in cluster_elements]
            }
        )
    
    def _merge_elements(self, visual_elements: List[DetectedElement], 
                       text_elements: List[DetectedElement]) -> List[DetectedElement]:
        """Intelligently merge visual and text elements"""
        merged_elements = []
        used_text_elements = set()
        
        # For each visual element, try to find matching text
        for visual_elem in visual_elements:
            best_text_match = None
            best_overlap = 0
            best_text_idx = -1
            
            for i, text_elem in enumerate(text_elements):
                if i in used_text_elements:
                    continue
                
                overlap = self._calculate_overlap_ratio(visual_elem.bbox, text_elem.bbox)
                
                if overlap > best_overlap and overlap > 0.1:  # Minimum overlap threshold
                    best_overlap = overlap
                    best_text_match = text_elem
                    best_text_idx = i
            
            if best_text_match and best_overlap > 0.3:  # Good overlap
                # Merge visual and text element
                merged_element = self._create_merged_element(visual_elem, best_text_match, best_overlap)
                merged_elements.append(merged_element)
                used_text_elements.add(best_text_idx)
            else:
                # Keep visual element without text if it has reasonable confidence
                if visual_elem.confidence > 0.5:
                    merged_elements.append(visual_elem)
        
        # Add unmatched text elements that have high confidence or clear interactive text
        for i, text_elem in enumerate(text_elements):
            if i not in used_text_elements:
                if (text_elem.confidence > 0.6 or 
                    text_elem.element_type in ['button', 'link', 'form']):
                    merged_elements.append(text_elem)
        
        return merged_elements
    
    def _create_merged_element(self, visual_elem: DetectedElement, 
                              text_elem: DetectedElement, overlap: float) -> DetectedElement:
        """Create a merged element from visual and text components"""
        
        # Use the more specific element type
        if text_elem.element_type in ['button', 'link', 'form']:
            element_type = text_elem.element_type
        else:
            element_type = visual_elem.element_type
        
        # Use visual element's bbox if it's larger, otherwise use text bbox
        if (visual_elem.size[0] * visual_elem.size[1] > 
            text_elem.size[0] * text_elem.size[1] * 1.5):
            bbox = visual_elem.bbox
            center = visual_elem.center
            size = visual_elem.size
        else:
            bbox = text_elem.bbox
            center = text_elem.center
            size = text_elem.size
        
        # Combine confidences weighted by overlap
        combined_confidence = (visual_elem.confidence * 0.6 + 
                             text_elem.confidence * 0.4) * (1 + overlap * 0.2)
        
        # Merge visual features
        merged_features = visual_elem.visual_features.copy()
        merged_features.update({
            'text_confidence': text_elem.confidence,
            'overlap_ratio': overlap,
            'merged_from': 'visual_text'
        })
        
        return DetectedElement(
            element_id=f"merged_{uuid.uuid4().hex[:8]}",
            element_type=element_type,
            bbox=bbox,
            center=center,
            size=size,
            confidence=min(0.95, combined_confidence),
            text=text_elem.text,
            visual_features=merged_features
        )
    
    def _calculate_overlap_ratio(self, bbox1: Tuple[int, int, int, int], 
                                bbox2: Tuple[int, int, int, int]) -> float:
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
        
        # Calculate areas
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Use smaller area as denominator for overlap ratio
        smaller_area = min(area1, area2)
        
        return intersection_area / smaller_area if smaller_area > 0 else 0.0
    
    def _remove_duplicate_elements(self, elements: List[DetectedElement]) -> List[DetectedElement]:
        """Remove duplicate elements based on overlap and confidence"""
        if not elements:
            return []
        
        # Sort by confidence (descending)
        sorted_elements = sorted(elements, key=lambda x: x.confidence, reverse=True)
        
        unique_elements = []
        
        for element in sorted_elements:
            is_duplicate = False
            
            for existing in unique_elements:
                overlap = self._calculate_overlap_ratio(element.bbox, existing.bbox)
                
                # Consider as duplicate if high overlap
                if overlap > 0.7:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_elements.append(element)
        
        return unique_elements
    
    def _filter_and_rank(self, elements: List[DetectedElement]) -> List[DetectedElement]:
        """Filter low-quality elements and rank by confidence"""
        
        # Filter elements with minimum quality
        filtered_elements = []
        
        for element in elements:
            # Skip elements that are too small or too large
            area = element.size[0] * element.size[1]
            if area < 25 or area > 500000:
                continue
            
            # Skip elements with very low confidence unless they have good text
            if element.confidence < 0.3 and not (element.text and len(element.text) > 2):
                continue
            
            # Skip elements with suspicious aspect ratios
            aspect_ratio = element.size[0] / element.size[1] if element.size[1] > 0 else 0
            if aspect_ratio < 0.05 or aspect_ratio > 50:
                continue
            
            filtered_elements.append(element)
        
        # Sort by confidence (descending)
        filtered_elements.sort(key=lambda x: x.confidence, reverse=True)
        
        # Limit to top 50 elements to prevent overload
        return filtered_elements[:50]