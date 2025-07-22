'use client';

import { useState } from 'react';
import { UserAttributes } from '@/types';

interface UserProfileFormProps {
  onSubmit: (profile: UserAttributes, taskDescription: string) => void;
  onBack: () => void;
  imagePreview: string | null;
}

export default function UserProfileForm({ onSubmit, onBack, imagePreview }: UserProfileFormProps) {
  const [profile, setProfile] = useState<UserAttributes>({
    age_group: '25-34',
    tech_savviness: 'medium',
    mood: 'neutral',
    device_type: 'desktop',
    browsing_speed: 'medium',
  });
  
  const [taskDescription, setTaskDescription] = useState('');
  
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (taskDescription.trim()) {
      onSubmit(profile, taskDescription.trim());
    }
  };

  const updateProfile = (key: keyof UserAttributes, value: string) => {
    setProfile(prev => ({ ...prev, [key]: value }));
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-8">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-semibold text-gray-800">
          Step 2: User Profile & Task
        </h2>
        <button
          onClick={onBack}
          className="text-blue-600 hover:text-blue-700 font-medium"
        >
          ← Back to upload
        </button>
      </div>

      <div className="grid md:grid-cols-2 gap-8">
        {/* Image Preview */}
        <div>
          <h3 className="font-semibold text-gray-700 mb-3">Uploaded Screenshot</h3>
          {imagePreview && (
            <img 
              src={imagePreview} 
              alt="Uploaded screenshot" 
              className="w-full rounded-lg border shadow-sm"
            />
          )}
        </div>

        {/* Form */}
        <div>
          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Age Group */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Age Group
              </label>
              <select
                value={profile.age_group}
                onChange={(e) => updateProfile('age_group', e.target.value)}
                className="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="18-24">18-24</option>
                <option value="25-34">25-34</option>
                <option value="35-44">35-44</option>
                <option value="45-54">45-54</option>
                <option value="55+">55+</option>
              </select>
            </div>

            {/* Tech Savviness */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Tech Savviness
              </label>
              <select
                value={profile.tech_savviness}
                onChange={(e) => updateProfile('tech_savviness', e.target.value)}
                className="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="low">Beginner</option>
                <option value="medium">Intermediate</option>
                <option value="high">Advanced</option>
              </select>
            </div>

            {/* Device Type */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Device Type
              </label>
              <select
                value={profile.device_type}
                onChange={(e) => updateProfile('device_type', e.target.value)}
                className="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="mobile">Mobile</option>
                <option value="tablet">Tablet</option>
                <option value="desktop">Desktop</option>
              </select>
            </div>

            {/* Browsing Speed */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Browsing Pace
              </label>
              <select
                value={profile.browsing_speed}
                onChange={(e) => updateProfile('browsing_speed', e.target.value)}
                className="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="slow">Careful & deliberate</option>
                <option value="medium">Moderate pace</option>
                <option value="fast">Quick & efficient</option>
              </select>
            </div>

            {/* Task Description */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Task Description *
              </label>
              <textarea
                value={taskDescription}
                onChange={(e) => setTaskDescription(e.target.value)}
                placeholder="Describe what the user is trying to accomplish on this page. For example: 'Complete checkout process and purchase items in cart' or 'Find information about product pricing and features'"
                rows={4}
                required
                className="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
              />
              <p className="text-xs text-gray-500 mt-1">
                Be specific about the user's goal for better predictions
              </p>
            </div>

            <button
              type="submit"
              disabled={!taskDescription.trim()}
              className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed text-white font-medium py-3 px-4 rounded-md transition-colors"
            >
              Predict Next Click →
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}