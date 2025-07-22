export interface UserAttributes {
  age_group: string;
  tech_savviness: string;
  mood: string;
  device_type: string;
  browsing_speed: string;
}

export interface PredictionRequest {
  user_attributes: UserAttributes;
  task_description: string;
  return_detailed?: boolean;
}

export interface UIElement {
  x: number;
  y: number;
  width: number;
  height: number;
  element_type: string;
  text: string;
  confidence: number;
  prominence: number;
}

export interface Prediction {
  element_id: number;
  click_probability: number;
  element: UIElement;
  reasoning: string[];
}

export interface Explanation {
  main_explanation: string;
  key_factors: Array<{
    factor: string;
    weight: number;
    description: string;
  }>;
  reasoning_chain: string[];
  confidence_analysis: string;
}

export interface PredictionResult {
  top_prediction: Prediction;
  all_predictions: Prediction[];
  explanation: Explanation;
  ui_elements: UIElement[];
  processing_time: number;
  confidence_score: number;
  metadata: Record<string, any>;
}