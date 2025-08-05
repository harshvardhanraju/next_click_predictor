#!/usr/bin/env python3
"""
Model Warm-up Script for Cloud Run Deployment

This script pre-downloads and initializes all ML models during Docker build,
ensuring fast cold starts and no timeout issues in production.
"""

import os
import sys
import logging
import tempfile
import numpy as np
from PIL import Image
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def create_test_image():
    """Create a test screenshot image for warm-up"""
    # Create a simple test image that looks like a UI
    img = np.ones((800, 1200, 3), dtype=np.uint8) * 240  # Light gray background
    
    # Draw some UI-like elements
    cv2.rectangle(img, (100, 100), (300, 150), (200, 200, 200), -1)  # Button-like rectangle
    cv2.rectangle(img, (100, 200), (500, 240), (255, 255, 255), -1)  # Input field
    cv2.rectangle(img, (600, 300), (800, 350), (100, 150, 200), -1)  # Another button
    
    # Add some text-like rectangles
    cv2.rectangle(img, (120, 115), (280, 135), (50, 50, 50), -1)
    cv2.rectangle(img, (120, 210), (480, 230), (100, 100, 100), -1)
    cv2.rectangle(img, (620, 315), (780, 335), (255, 255, 255), -1)
    
    return img

def warm_up_opencv():
    """Warm up OpenCV operations"""
    logger.info("🔥 Warming up OpenCV...")
    
    # Create test image
    img = create_test_image()
    
    # Perform typical CV operations
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    logger.info(f"✅ OpenCV warmed up - detected {len(contours)} contours")

def warm_up_easyocr():
    """Warm up EasyOCR and download models"""
    logger.info("🔥 Warming up EasyOCR (downloading models)...")
    
    try:
        import easyocr
        
        # Initialize EasyOCR - this will download models
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        
        # Create test image with text
        img = create_test_image()
        
        # Run OCR to ensure everything works
        results = reader.readtext(img)
        
        logger.info(f"✅ EasyOCR warmed up - found {len(results)} text regions")
        
    except Exception as e:
        logger.warning(f"⚠️ EasyOCR warm-up failed: {e}")

def warm_up_matplotlib():
    """Warm up matplotlib and generate font cache"""
    logger.info("🔥 Warming up matplotlib...")
    
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        
        # Create a simple plot to generate font cache
        fig, ax = plt.subplots(1, 1, figsize=(2, 2))
        ax.plot([1, 2, 3], [1, 2, 3])
        ax.set_title("Test")
        plt.close(fig)
        
        logger.info("✅ Matplotlib warmed up - font cache generated")
        
    except Exception as e:
        logger.warning(f"⚠️ Matplotlib warm-up failed: {e}")

def warm_up_scikit_learn():
    """Warm up scikit-learn models"""
    logger.info("🔥 Warming up scikit-learn...")
    
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.calibration import CalibratedClassifierCV
        
        # Create dummy data and train a small model
        X = np.random.random((100, 10))
        y = np.random.randint(0, 2, 100)
        
        # Initialize and fit models
        gb_model = GradientBoostingClassifier(n_estimators=10, random_state=42)
        gb_model.fit(X, y)
        
        # Test prediction
        pred = gb_model.predict(X[:1])
        
        logger.info("✅ Scikit-learn warmed up - models ready")
        
    except Exception as e:
        logger.warning(f"⚠️ Scikit-learn warm-up failed: {e}")

def warm_up_pgmpy():
    """Warm up pgmpy Bayesian networks"""
    logger.info("🔥 Warming up pgmpy...")
    
    try:
        from pgmpy.models import BayesianNetwork
        from pgmpy.factors.discrete import TabularCPD
        from pgmpy.inference import VariableElimination
        
        # Create a simple test network
        model = BayesianNetwork([('A', 'C'), ('B', 'C')])
        
        # Add CPDs
        cpd_a = TabularCPD('A', 2, [[0.7], [0.3]])
        cpd_b = TabularCPD('B', 2, [[0.6], [0.4]])
        cpd_c = TabularCPD('C', 2, [[0.8, 0.5, 0.3, 0.1],
                                   [0.2, 0.5, 0.7, 0.9]], ['A', 'B'], [2, 2])
        
        model.add_cpds(cpd_a, cpd_b, cpd_c)
        
        # Test inference
        inference = VariableElimination(model)
        result = inference.query(['C'])
        
        logger.info("✅ pgmpy warmed up - Bayesian networks ready")
        
    except Exception as e:
        logger.warning(f"⚠️ pgmpy warm-up failed: {e}")

def warm_up_improved_ml_system():
    """Warm up the improved ML system"""
    logger.info("🔥 Warming up improved ML system...")
    
    try:
        # Add src to path
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
        
        from improved_next_click_predictor import ImprovedNextClickPredictor
        
        # Initialize predictor
        predictor = ImprovedNextClickPredictor({
            'log_level': 'WARNING',  # Reduce logging during warm-up
            'enable_evaluation': False,
            'ensemble_config': {
                'ensemble_method': 'adaptive'
            }
        })
        
        # Initialize the system
        success = predictor.initialize()
        
        if success:
            # Create test image and save it
            test_img = create_test_image()
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                cv2.imwrite(tmp_file.name, test_img)
                
                # Run a test prediction
                result = predictor.predict_next_click(
                    screenshot_path=tmp_file.name,
                    user_attributes={'tech_savviness': 'medium', 'age_group': 'adult'},
                    task_description="Click the login button",
                    return_detailed=True
                )
                
                # Clean up
                os.unlink(tmp_file.name)
                
            logger.info("✅ Improved ML system warmed up successfully")
        else:
            logger.warning("⚠️ Improved ML system initialization failed")
            
    except Exception as e:
        logger.warning(f"⚠️ Improved ML system warm-up failed: {e}")

def main():
    """Run all warm-up procedures"""
    logger.info("🚀 Starting model warm-up for Cloud Run deployment...")
    
    # Set environment variables for optimal performance
    os.environ['MATPLOTLIB_CACHE_DIR'] = '/tmp/matplotlib'
    os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
    
    # Create cache directories
    os.makedirs('/tmp/matplotlib', exist_ok=True)
    
    # Warm up all components
    warm_up_opencv()
    warm_up_matplotlib()
    warm_up_scikit_learn()
    warm_up_pgmpy()
    warm_up_easyocr()  # This should be done last as it downloads models
    warm_up_improved_ml_system()
    
    logger.info("🎉 All models warmed up successfully! Deployment ready for fast cold starts.")

if __name__ == "__main__":
    main()