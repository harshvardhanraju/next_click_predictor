#!/usr/bin/env python3
"""
Example usage of the Next-Click Prediction System

This script demonstrates how to use the system with various scenarios
and provides examples of the expected inputs and outputs.
"""

import os
import sys
import tempfile
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from next_click_predictor import NextClickPredictor


class ExampleScenarios:
    """Generate example scenarios for testing the prediction system"""
    
    def __init__(self):
        self.predictor = NextClickPredictor()
        self.temp_files = []
    
    def cleanup(self):
        """Clean up temporary files"""
        for file_path in self.temp_files:
            if os.path.exists(file_path):
                os.remove(file_path)
    
    def create_ecommerce_checkout_page(self) -> str:
        """Create a mock e-commerce checkout page"""
        # Create image
        image = np.ones((800, 1200, 3), dtype=np.uint8) * 255
        
        # Header
        cv2.rectangle(image, (0, 0), (1200, 80), (51, 51, 51), -1)
        cv2.putText(image, 'ShopNow.com', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        # Product summary
        cv2.rectangle(image, (50, 120), (750, 200), (245, 245, 245), 2)
        cv2.putText(image, 'Order Summary', (60, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(image, 'Wireless Headphones - $99.99', (60, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Payment section
        cv2.rectangle(image, (50, 220), (750, 400), (245, 245, 245), 2)
        cv2.putText(image, 'Payment Information', (60, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Form fields
        cv2.rectangle(image, (60, 270), (400, 300), (255, 255, 255), -1)
        cv2.rectangle(image, (60, 270), (400, 300), (200, 200, 200), 2)
        cv2.putText(image, 'Card Number', (65, 288), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        cv2.rectangle(image, (60, 320), (200, 350), (255, 255, 255), -1)
        cv2.rectangle(image, (60, 320), (200, 350), (200, 200, 200), 2)
        cv2.putText(image, 'Expiry Date', (65, 338), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        cv2.rectangle(image, (220, 320), (320, 350), (255, 255, 255), -1)
        cv2.rectangle(image, (220, 320), (320, 350), (200, 200, 200), 2)
        cv2.putText(image, 'CVV', (225, 338), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        # Main action buttons
        cv2.rectangle(image, (300, 450), (550, 500), (40, 167, 69), -1)  # Green checkout button
        cv2.putText(image, 'Complete Purchase', (320, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.rectangle(image, (50, 450), (280, 500), (200, 200, 200), -1)  # Gray continue shopping
        cv2.putText(image, 'Continue Shopping', (60, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Help link
        cv2.putText(image, 'Need help?', (50, 550), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        
        # Security badges
        cv2.rectangle(image, (600, 520), (750, 570), (245, 245, 245), -1)
        cv2.putText(image, 'Secure Payment', (610, 550), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            cv2.imwrite(tmp.name, image)
            self.temp_files.append(tmp.name)
            return tmp.name
    
    def create_social_media_feed(self) -> str:
        """Create a mock social media feed"""
        image = np.ones((800, 600, 3), dtype=np.uint8) * 255
        
        # Header
        cv2.rectangle(image, (0, 0), (600, 60), (59, 89, 152), -1)
        cv2.putText(image, 'SocialHub', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Navigation
        cv2.putText(image, 'Home', (400, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(image, 'Profile', (460, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(image, 'Messages', (520, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Post 1
        cv2.rectangle(image, (20, 80), (580, 200), (245, 245, 245), -1)
        cv2.rectangle(image, (20, 80), (580, 200), (200, 200, 200), 2)
        cv2.putText(image, 'John Doe', (30, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(image, 'Just finished an amazing hike!', (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Like/Comment buttons
        cv2.rectangle(image, (30, 160), (80, 185), (66, 103, 178), -1)
        cv2.putText(image, 'Like', (40, 178), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.rectangle(image, (90, 160), (160, 185), (200, 200, 200), -1)
        cv2.putText(image, 'Comment', (95, 178), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        cv2.rectangle(image, (170, 160), (220, 185), (200, 200, 200), -1)
        cv2.putText(image, 'Share', (180, 178), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # Post 2
        cv2.rectangle(image, (20, 220), (580, 340), (245, 245, 245), -1)
        cv2.rectangle(image, (20, 220), (580, 340), (200, 200, 200), 2)
        cv2.putText(image, 'Jane Smith', (30, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(image, 'Check out this new recipe!', (30, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Action buttons
        cv2.rectangle(image, (30, 300), (80, 325), (66, 103, 178), -1)
        cv2.putText(image, 'Like', (40, 318), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.rectangle(image, (90, 300), (160, 325), (200, 200, 200), -1)
        cv2.putText(image, 'Comment', (95, 318), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            cv2.imwrite(tmp.name, image)
            self.temp_files.append(tmp.name)
            return tmp.name
    
    def create_dashboard_interface(self) -> str:
        """Create a mock dashboard interface"""
        image = np.ones((800, 1000, 3), dtype=np.uint8) * 255
        
        # Header
        cv2.rectangle(image, (0, 0), (1000, 80), (52, 73, 94), -1)
        cv2.putText(image, 'Analytics Dashboard', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        # Menu items
        cv2.putText(image, 'Overview', (700, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(image, 'Reports', (700, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(image, 'Settings', (780, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(image, 'Logout', (780, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Sidebar
        cv2.rectangle(image, (0, 80), (200, 800), (236, 240, 241), -1)
        cv2.putText(image, 'Navigation', (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(image, 'Dashboard', (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(image, 'Users', (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(image, 'Products', (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(image, 'Orders', (20, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Main content area
        cv2.rectangle(image, (220, 100), (980, 300), (245, 245, 245), -1)
        cv2.rectangle(image, (220, 100), (980, 300), (200, 200, 200), 2)
        cv2.putText(image, 'Revenue Chart', (240, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Action buttons
        cv2.rectangle(image, (700, 320), (850, 360), (46, 204, 113), -1)
        cv2.putText(image, 'Generate Report', (710, 345), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.rectangle(image, (220, 320), (350, 360), (155, 89, 182), -1)
        cv2.putText(image, 'Export Data', (240, 345), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.rectangle(image, (370, 320), (480, 360), (230, 126, 34), -1)
        cv2.putText(image, 'Refresh', (390, 345), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            cv2.imwrite(tmp.name, image)
            self.temp_files.append(tmp.name)
            return tmp.name
    
    def run_scenario(self, scenario_name: str, image_path: str, user_attrs: dict, task: str):
        """Run a prediction scenario"""
        print(f"\n{'='*60}")
        print(f"SCENARIO: {scenario_name}")
        print(f"{'='*60}")
        
        print(f"Task: {task}")
        print(f"User Profile: {user_attrs}")
        print(f"Screenshot: {image_path}")
        
        try:
            # Run prediction
            result = self.predictor.predict_next_click(image_path, user_attrs, task)
            
            # Display results
            print(f"\n🎯 TOP PREDICTION:")
            print(f"   Element: {result.top_prediction['element_text']}")
            print(f"   Type: {result.top_prediction['element_type']}")
            print(f"   Probability: {result.top_prediction['click_probability']:.1%}")
            print(f"   Confidence: {result.confidence_score:.1%}")
            
            print(f"\n📊 ALL PREDICTIONS:")
            for i, pred in enumerate(result.all_predictions[:5]):
                print(f"   {i+1}. {pred['element_text']} ({pred['element_type']}) - {pred['click_probability']:.1%}")
            
            print(f"\n💡 EXPLANATION:")
            explanation = result.explanation.get('main_explanation', 'No explanation available')
            print(f"   {explanation}")
            
            print(f"\n⚡ PERFORMANCE:")
            print(f"   Processing Time: {result.processing_time:.2f}s")
            print(f"   UI Elements Found: {len(result.ui_elements)}")
            
            return result
            
        except Exception as e:
            print(f"\n❌ ERROR: {str(e)}")
            return None
    
    def run_all_scenarios(self):
        """Run all example scenarios"""
        print("🚀 Next-Click Prediction System - Example Scenarios")
        print("This demo shows how the system predicts user clicks across different interfaces.")
        
        # Scenario 1: E-commerce Checkout
        ecommerce_image = self.create_ecommerce_checkout_page()
        self.run_scenario(
            "E-commerce Checkout",
            ecommerce_image,
            {
                "age_group": "25-34",
                "tech_savviness": "high",
                "mood": "focused",
                "device_type": "desktop"
            },
            "Complete purchase. What would you click next?"
        )
        
        # Scenario 2: Social Media Browsing
        social_image = self.create_social_media_feed()
        self.run_scenario(
            "Social Media Feed",
            social_image,
            {
                "age_group": "18-24",
                "tech_savviness": "medium",
                "mood": "excited",
                "device_type": "mobile"
            },
            "Browse social media posts. What would you click next?"
        )
        
        # Scenario 3: Dashboard Analytics
        dashboard_image = self.create_dashboard_interface()
        self.run_scenario(
            "Business Dashboard",
            dashboard_image,
            {
                "age_group": "35-44",
                "tech_savviness": "expert",
                "mood": "neutral",
                "device_type": "desktop"
            },
            "Generate monthly report. What would you click next?"
        )
        
        # Scenario 4: Different user profiles on same interface
        print(f"\n{'='*60}")
        print("COMPARATIVE ANALYSIS: Different Users, Same Interface")
        print(f"{'='*60}")
        
        user_profiles = [
            {"age_group": "18-24", "tech_savviness": "low", "mood": "frustrated", "device_type": "mobile"},
            {"age_group": "25-34", "tech_savviness": "high", "mood": "focused", "device_type": "desktop"},
            {"age_group": "55-64", "tech_savviness": "medium", "mood": "neutral", "device_type": "tablet"}
        ]
        
        for i, profile in enumerate(user_profiles):
            print(f"\n--- User Profile {i+1} ---")
            result = self.run_scenario(
                f"E-commerce Checkout (User {i+1})",
                ecommerce_image,
                profile,
                "Complete purchase. What would you click next?"
            )


def demonstrate_api_usage():
    """Demonstrate different ways to use the API"""
    print(f"\n{'='*60}")
    print("API USAGE EXAMPLES")
    print(f"{'='*60}")
    
    # Create simple test image
    test_image = np.ones((400, 600, 3), dtype=np.uint8) * 255
    cv2.rectangle(test_image, (200, 150), (400, 200), (52, 152, 219), -1)
    cv2.putText(test_image, 'Click Me!', (220, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        cv2.imwrite(tmp.name, test_image)
        test_image_path = tmp.name
    
    try:
        predictor = NextClickPredictor()
        
        # 1. Basic prediction
        print("\n1. Basic Prediction:")
        result = predictor.predict_next_click(
            test_image_path,
            {"tech_savviness": "medium"},
            "Click the button"
        )
        print(f"   Result: {result.top_prediction['element_text']} - {result.top_prediction['click_probability']:.1%}")
        
        # 2. Screenshot analysis only
        print("\n2. Screenshot Analysis Only:")
        analysis = predictor.analyze_screenshot_only(test_image_path)
        print(f"   Found {analysis['total_elements']} elements")
        print(f"   Screen size: {analysis['screen_dimensions']}")
        
        # 3. System statistics
        print("\n3. System Statistics:")
        stats = predictor.get_system_stats()
        if 'error' not in stats:
            print(f"   Total predictions: {stats['total_predictions']}")
            print(f"   Average processing time: {stats['avg_processing_time']:.2f}s")
        
        # 4. Save/Load results
        print("\n4. Save/Load Results:")
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            predictor.save_prediction_result(result, tmp.name)
            loaded_result = predictor.load_prediction_result(tmp.name)
            print(f"   Saved and loaded result: {loaded_result.top_prediction['element_text']}")
            os.remove(tmp.name)
        
    finally:
        if os.path.exists(test_image_path):
            os.remove(test_image_path)


def main():
    """Main function to run all examples"""
    print("🎯 Next-Click Prediction System - Interactive Examples")
    print("=" * 80)
    
    examples = ExampleScenarios()
    
    try:
        # Run all scenarios
        examples.run_all_scenarios()
        
        # Demonstrate API usage
        demonstrate_api_usage()
        
        print(f"\n{'='*60}")
        print("🎉 All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error running examples: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up
        examples.cleanup()


if __name__ == "__main__":
    main()