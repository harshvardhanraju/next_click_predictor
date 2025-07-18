# Next-Click Prediction System

A system that predicts what a user would click next given their attributes, a screenshot, and a task description.

## Features

- **Screenshot Processing**: Extract UI elements from PNG images using computer vision
- **Bayesian Networks**: Dynamic probabilistic modeling of user decision processes
- **Explainable AI**: Human-readable explanations for predictions
- **Modular Design**: Clean separation between CV, ML, and inference components

## System Overview

```
PNG Screenshot → UI Element Detection → Feature Integration → Bayesian Network → Click Prediction + Explanation
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from next_click_predictor import NextClickPredictor

predictor = NextClickPredictor()

# Make prediction
result = predictor.predict_next_click(
    screenshot_path="screenshot.png",
    user_attributes={"age": "25-34", "tech_savviness": "high", "mood": "focused"},
    task="Complete purchase. What would you click next?"
)

print(f"Predicted click: {result['top_prediction']['element_id']}")
print(f"Probability: {result['top_prediction']['probability']:.2f}")
print(f"Explanation: {result['explanation']}")
```

## Architecture

- `screenshot_processor.py` - UI element detection and feature extraction
- `bayesian_network.py` - Dynamic network construction and inference
- `feature_integrator.py` - Combines user, UI, and task features
- `explanation_generator.py` - Creates human-readable explanations
- `next_click_predictor.py` - Main API and orchestration