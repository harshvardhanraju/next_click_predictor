'use client';

import { useState } from 'react';
import ImageUpload from '@/components/ImageUpload';
import UserProfileForm from '@/components/UserProfileForm';
import ResultsDisplay from '@/components/ResultsDisplay';
import { UserAttributes, PredictionResult } from '@/types';

export default function Home() {
  const [step, setStep] = useState<'upload' | 'profile' | 'results'>('upload');
  const [uploadedImage, setUploadedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [userAttributes, setUserAttributes] = useState<UserAttributes | null>(null);
  const [taskDescription, setTaskDescription] = useState<string>('');
  const [predictionResult, setPredictionResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleImageUpload = (file: File, preview: string) => {
    setUploadedImage(file);
    setImagePreview(preview);
    setStep('profile');
  };

  const handleProfileSubmit = async (profile: UserAttributes, task: string) => {
    setUserAttributes(profile);
    setTaskDescription(task);
    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', uploadedImage!);
      formData.append('user_attributes', JSON.stringify(profile));
      formData.append('task_description', task);

      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      console.log('🔗 API URL being used:', apiUrl);
      console.log('📤 Making request to:', `${apiUrl}/predict`);
      
      // Set timeout for ML processing (2 minutes)
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 120000);
      
      const response = await fetch(`${apiUrl}/predict`, {
        method: 'POST',
        body: formData,
        signal: controller.signal,
      });
      
      clearTimeout(timeoutId);

      console.log('📊 Response status:', response.status);
      console.log('📊 Response headers:', Object.fromEntries(response.headers.entries()));
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error('❌ Response error:', errorText);
        throw new Error(`HTTP error! status: ${response.status} - ${errorText}`);
      }

      const result = await response.json();
      console.log('✅ API Response received:', result);
      setPredictionResult(result);
      setStep('results');
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') {
        setError('Request timed out. The AI processing is taking longer than expected. Please try with a simpler image or check your connection.');
      } else {
        setError(err instanceof Error ? err.message : 'An error occurred');
      }
    } finally {
      setLoading(false);
    }
  };

  const resetApp = () => {
    setStep('upload');
    setUploadedImage(null);
    setImagePreview(null);
    setUserAttributes(null);
    setTaskDescription('');
    setPredictionResult(null);
    setError(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-8">
        <header className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            Next Click Predictor
          </h1>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            AI-powered prediction of where users will click next using computer vision and Bayesian networks
          </p>
          <div className="flex justify-center mt-4 space-x-2">
            <span className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm">
              Computer Vision
            </span>
            <span className="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm">
              Bayesian Networks
            </span>
            <span className="px-3 py-1 bg-purple-100 text-purple-800 rounded-full text-sm">
              Explainable AI
            </span>
          </div>
        </header>

        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-6">
            <p><strong>Error:</strong> {error}</p>
            <button 
              onClick={resetApp}
              className="mt-2 text-sm underline hover:no-underline"
            >
              Try again
            </button>
          </div>
        )}

        {loading && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-white rounded-lg p-8 max-w-sm mx-4 text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
              <p className="text-gray-700 font-medium">Analyzing screenshot...</p>
              <p className="text-gray-500 text-sm mt-2">
                Advanced AI processing in progress...<br/>
                Expected time: 5-15 seconds for simple images
              </p>
              <div className="mt-4 text-xs text-gray-400">
                ✓ UI Detection → ✓ Feature Analysis → ✓ Bayesian Inference
              </div>
            </div>
          </div>
        )}

        <div className="max-w-4xl mx-auto">
          {step === 'upload' && (
            <ImageUpload onImageUpload={handleImageUpload} />
          )}

          {step === 'profile' && (
            <UserProfileForm 
              onSubmit={handleProfileSubmit}
              onBack={() => setStep('upload')}
              imagePreview={imagePreview}
            />
          )}

          {step === 'results' && predictionResult && (
            <ResultsDisplay 
              result={predictionResult}
              imagePreview={imagePreview}
              onReset={resetApp}
            />
          )}
        </div>

        <footer className="mt-16 text-center text-gray-500">
          <p className="mb-2">
            Open source project • Built with Next.js, FastAPI, and Railway
          </p>
          <div className="flex justify-center space-x-4 text-sm">
            <a 
              href="https://github.com/harshvardhanraju/next_click_predictor" 
              target="_blank" 
              rel="noopener noreferrer"
              className="hover:text-blue-600 transition-colors"
            >
              View on GitHub
            </a>
            <span>•</span>
            <a 
              href="https://github.com/harshvardhanraju/next_click_predictor/blob/master/SYSTEM_ARCHITECTURE.md" 
              target="_blank" 
              rel="noopener noreferrer"
              className="hover:text-blue-600 transition-colors"
            >
              Documentation
            </a>
          </div>
        </footer>
      </div>
    </div>
  );
}