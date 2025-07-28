#!/bin/bash

# Quick Google Cloud Setup Script for Next Click Predictor
# This script automates the initial setup process

set -e  # Exit on any error

echo "🚀 Google Cloud Run Setup for Next Click Predictor"
echo "=================================================="

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI not found!"
    echo "Please install Google Cloud SDK first:"
    echo "   https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found!"
    echo "Please install Docker first:"
    echo "   https://docs.docker.com/get-docker/"
    exit 1
fi

echo "✅ Prerequisites check passed"

# Prompt for project ID
read -p "Enter your Google Cloud Project ID (or press Enter to create new): " PROJECT_ID

if [ -z "$PROJECT_ID" ]; then
    echo "📋 Creating new Google Cloud project..."
    
    # Generate unique project ID
    TIMESTAMP=$(date +%s)
    PROJECT_ID="next-click-predictor-$TIMESTAMP"
    
    echo "Creating project: $PROJECT_ID"
    gcloud projects create $PROJECT_ID --name="Next Click Predictor"
    
    echo "⚠️  Please enable billing for this project:"
    echo "   https://console.cloud.google.com/billing/linkedaccount?project=$PROJECT_ID"
    read -p "Press Enter after enabling billing..."
fi

echo "🔧 Configuring gcloud..."

# Set project
gcloud config set project $PROJECT_ID
echo "Project set to: $PROJECT_ID"

# Set default region
gcloud config set run/region us-central1
echo "Region set to: us-central1"

# Login if needed
echo "🔐 Checking authentication..."
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q "@"; then
    echo "Please authenticate with Google Cloud:"
    gcloud auth login
fi

echo "✅ Authentication verified"

echo "🔌 Enabling required APIs..."
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com  
gcloud services enable cloudbuild.googleapis.com

echo "✅ APIs enabled"

echo "🐳 Configuring Docker authentication..."
gcloud auth configure-docker --quiet

echo "✅ Docker configured"

# Export environment variables
export GOOGLE_CLOUD_PROJECT=$PROJECT_ID
export GOOGLE_CLOUD_REGION="us-central1"

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "Environment variables set:"
echo "   GOOGLE_CLOUD_PROJECT=$GOOGLE_CLOUD_PROJECT"
echo "   GOOGLE_CLOUD_REGION=$GOOGLE_CLOUD_REGION"
echo ""
echo "Next steps:"
echo "   1. Run: ./deploy-cloudrun.sh"
echo "   2. Update your Vercel frontend with the Cloud Run URL"
echo "   3. Test with: python3 test-cloudrun.py"
echo ""
echo "Project Console: https://console.cloud.google.com/run?project=$PROJECT_ID"