'use client';

import { useState } from 'react';
import { PredictionResult } from '@/types';

interface ResultsDisplayProps {
  result: PredictionResult;
  imagePreview: string | null;
  onReset: () => void;
}

export default function ResultsDisplay({ result, imagePreview, onReset }: ResultsDisplayProps) {
  const [hoveredElement, setHoveredElement] = useState<number | null>(null);
  const [showDetails, setShowDetails] = useState(false);

  const formatProbability = (prob: number) => {
    return `${(prob * 100).toFixed(1)}%`;
  };

  const formatProcessingTime = (time: number) => {
    return `${time.toFixed(2)}s`;
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="bg-white rounded-lg shadow-lg p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-2xl font-semibold text-gray-800">
            Prediction Results
          </h2>
          <div className="flex items-center space-x-4">
            <div className="text-right">
              <p className="text-sm text-gray-500">Confidence Score</p>
              <p className="text-2xl font-bold text-blue-600">
                {formatProbability(result.confidence_score)}
              </p>
            </div>
            <button
              onClick={onReset}
              className="bg-gray-100 hover:bg-gray-200 text-gray-700 px-4 py-2 rounded-md font-medium transition-colors"
            >
              Try Another
            </button>
          </div>
        </div>

        <div className="grid grid-cols-3 gap-4 text-center">
          <div className="bg-blue-50 rounded-lg p-4">
            <p className="text-sm text-gray-600">Processing Time</p>
            <p className="text-lg font-semibold text-blue-600">
              {formatProcessingTime(result.processing_time)}
            </p>
          </div>
          <div className="bg-green-50 rounded-lg p-4">
            <p className="text-sm text-gray-600">UI Elements Found</p>
            <p className="text-lg font-semibold text-green-600">
              {result.ui_elements.length}
            </p>
          </div>
          <div className="bg-purple-50 rounded-lg p-4">
            <p className="text-sm text-gray-600">Top Prediction</p>
            <p className="text-lg font-semibold text-purple-600">
              {formatProbability(result.top_prediction.click_probability)}
            </p>
          </div>
        </div>
      </div>

      <div className="grid lg:grid-cols-2 gap-8">
        {/* Interactive Image */}
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">
            Interactive Prediction Map
          </h3>
          <div className="relative inline-block">
            {imagePreview && (
              <img 
                src={imagePreview} 
                alt="Analysis results" 
                className="max-w-full rounded-lg border"
              />
            )}
            
            {/* Overlay for Top Prediction Only */}
            {result.ui_elements && result.ui_elements.length > 0 && (() => {
              // Only show overlay for the first element (top prediction)
              const element = result.ui_elements[0];
              
              // Handle both API formats - use bbox array if available, fallback to x,y,width,height
              const bbox = element.bbox || [element.x, element.y, element.x + element.width, element.y + element.height];
              const x = bbox[0];
              const y = bbox[1];
              const width = bbox[2] - bbox[0];
              const height = bbox[3] - bbox[1];
              
              // Skip if no valid coordinates
              if ((!x && x !== 0) || (!y && y !== 0)) return null;
              
              return (
                <div
                  key={element.id || 'top-prediction'}
                  className="absolute border-3 border-red-500 bg-red-500 bg-opacity-20 transition-all animate-pulse"
                  style={{
                    left: `${(x / 800) * 100}%`,  // Assuming 800px width, adjust based on actual image
                    top: `${(y / 600) * 100}%`,   // Assuming 600px height, adjust based on actual image  
                    width: `${(width / 800) * 100}%`,
                    height: `${(height / 600) * 100}%`,
                  }}
                  title={`Most Likely Click: ${element.type || element.element_type} - ${result.top_prediction.click_probability ? (result.top_prediction.click_probability * 100).toFixed(1) + '%' : 'N/A'} confidence`}
                >
                  <div className="absolute -top-8 left-0 bg-red-600 text-white text-sm px-3 py-1 rounded-md whitespace-nowrap font-semibold shadow-lg">
                    🎯 Most Likely Click: {result.top_prediction.click_probability ? (result.top_prediction.click_probability * 100).toFixed(1) + '%' : 'N/A'}
                  </div>
                </div>
              );
            })()}
          </div>
          
          <div className="mt-4 flex items-center justify-center">
            <div className="flex items-center space-x-2 bg-red-50 px-4 py-2 rounded-lg">
              <div className="w-4 h-4 border-2 border-red-500 bg-red-500 bg-opacity-20 animate-pulse"></div>
              <span className="text-red-700 font-medium">🎯 Most Likely Next Click</span>
            </div>
          </div>
        </div>

        {/* Predictions List */}
        <div className="space-y-6">
          {/* Top Prediction */}
          <div className="bg-gradient-to-r from-red-50 to-pink-50 rounded-lg shadow-lg p-6 border-l-4 border-red-500">
            <h3 className="text-lg font-semibold text-gray-800 mb-3">
              🎯 Most Likely Next Click
            </h3>
            
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="font-medium text-gray-700">
                  {result.top_prediction.element_type?.toUpperCase() || 'ELEMENT'}
                </span>
                <span className="text-2xl font-bold text-red-600">
                  {formatProbability(result.top_prediction.click_probability)}
                </span>
              </div>
              
              {result.top_prediction.element_text && (
                <div className="bg-white bg-opacity-70 rounded p-3">
                  <p className="text-sm text-gray-600">Text Content:</p>
                  <p className="font-medium">"{result.top_prediction.element_text}"</p>
                </div>
              )}
              
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <p className="text-gray-600">Element ID</p>
                  <p className="font-medium">
                    {result.top_prediction.element_id || 'N/A'}
                  </p>
                </div>
                <div>
                  <p className="text-gray-600">Confidence</p>
                  <p className="font-medium">
                    {formatProbability(result.top_prediction.confidence || result.top_prediction.click_probability)}
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Explanation */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-lg font-semibold text-gray-800 mb-4">
              🧠 AI Explanation
            </h3>
            
            <div className="space-y-4">
              <div className="bg-blue-50 rounded-lg p-4">
                <p className="text-gray-800 leading-relaxed">
                  {result.explanation.main_explanation}
                </p>
              </div>

              <button
                onClick={() => setShowDetails(!showDetails)}
                className="text-blue-600 hover:text-blue-700 font-medium text-sm flex items-center space-x-1"
              >
                <span>{showDetails ? 'Hide' : 'Show'} detailed analysis</span>
                <svg 
                  className={`w-4 h-4 transition-transform ${showDetails ? 'rotate-180' : ''}`}
                  fill="none" stroke="currentColor" viewBox="0 0 24 24"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </button>

              {showDetails && (
                <div className="space-y-4">
                  {/* Key Factors */}
                  {result.explanation.key_factors && result.explanation.key_factors.length > 0 && (
                    <div>
                      <h4 className="font-semibold text-gray-700 mb-2">Key Factors:</h4>
                      <div className="space-y-2">
                        {result.explanation.key_factors.map((factor, index) => (
                          <div key={index} className="bg-gray-50 rounded p-3">
                            <div className="flex justify-between items-center mb-1">
                              <span className="font-medium text-gray-800">
                                {factor.name || factor.factor || 'Factor'}
                              </span>
                              <span className="text-sm font-semibold text-blue-600">
                                {factor.weight ? (factor.weight * 100).toFixed(0) + '%' : 
                                 factor.importance ? (factor.importance * 100).toFixed(0) + '%' : 
                                 factor.influence ? Math.abs(factor.influence * 100).toFixed(0) + '%' : 'N/A'}
                              </span>
                            </div>
                            <p className="text-sm text-gray-600">{factor.description}</p>
                            {factor.evidence && (
                              <p className="text-xs text-gray-500 mt-1">Evidence: {factor.evidence}</p>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Reasoning Chain */}
                  {result.explanation.reasoning_chain && result.explanation.reasoning_chain.length > 0 && (
                    <div>
                      <h4 className="font-semibold text-gray-700 mb-2">Reasoning Process:</h4>
                      <ol className="list-decimal list-inside space-y-1 text-sm text-gray-600">
                        {result.explanation.reasoning_chain.map((step, index) => (
                          <li key={index}>{step}</li>
                        ))}
                      </ol>
                    </div>
                  )}

                  {/* Confidence Analysis */}
                  {(result.explanation.confidence_analysis || result.explanation.confidence_explanation) && (
                    <div className="bg-yellow-50 rounded p-3">
                      <h4 className="font-semibold text-gray-700 mb-1">Confidence Analysis:</h4>
                      <p className="text-sm text-gray-600">
                        {result.explanation.confidence_analysis || result.explanation.confidence_explanation}
                      </p>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Other Predictions */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-lg font-semibold text-gray-800 mb-4">
              📊 All Predictions
            </h3>
            
            <div className="space-y-3">
              {result.all_predictions && result.all_predictions.slice(0, 5).map((prediction, index) => (
                <div 
                  key={prediction.element_id || index}
                  className={`
                    flex items-center justify-between p-3 rounded-lg border-2 transition-colors
                    ${index === 0 ? 'bg-red-50 border-red-300' : 'border-gray-200'}
                  `}
                >
                  <div className="flex items-center space-x-3">
                    <span className={`
                      w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold
                      ${index === 0 ? 'bg-red-500 text-white' : 'bg-gray-200 text-gray-700'}
                    `}>
                      {index + 1}
                    </span>
                    <div>
                      <p className="font-medium text-gray-800">
                        {prediction.element_type?.toUpperCase() || 'ELEMENT'}
                      </p>
                      {prediction.element_text && (
                        <p className="text-sm text-gray-500 truncate max-w-xs">
                          "{prediction.element_text}"
                        </p>
                      )}
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="font-bold text-lg">
                      {formatProbability(prediction.click_probability)}
                    </p>
                    <p className="text-xs text-gray-500">
                      confidence: {((prediction.confidence || prediction.click_probability) * 100).toFixed(0)}%
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}