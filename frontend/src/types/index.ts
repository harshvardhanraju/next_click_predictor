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
  id?: string;
  type?: string;
  element_type?: string;
  text?: string;
  bbox?: number[]; // [x1, y1, x2, y2]
  x?: number;
  y?: number;
  width?: number;
  height?: number;
  center?: number[];
  size?: number[];
  prominence?: number;
  visibility?: boolean;
  confidence?: number;
  color_features?: any;
  position_features?: any;
}

export interface Prediction {
  element_id?: string | number;
  element_type?: string;
  element_text?: string;
  click_probability: number;
  confidence?: number;
  prominence?: number;
  rank?: number;
  element?: UIElement; // For backward compatibility
  reasoning?: string[];
}

export interface Explanation {
  main_explanation: string;
  key_factors?: Array<{
    name?: string;
    factor?: string;
    weight?: number;
    influence?: number;
    importance?: number;
    description: string;
    evidence?: string;
    type?: string;
  }>;
  factor_explanations?: string[];
  reasoning_chain?: string[];
  confidence_analysis?: string;
  confidence_explanation?: string;
  alternative_explanations?: string[];
  prediction_summary?: any;
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