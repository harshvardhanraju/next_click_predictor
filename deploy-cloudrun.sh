#!/bin/bash

# Deploy Next Click Predictor to Google Cloud Run
# Optimized for free tier with minimal resource usage

set -e  # Exit on any error

# Configuration
PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-your-project-id}"
SERVICE_NAME="next-click-predictor"
REGION="${GOOGLE_CLOUD_REGION:-us-central1}"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "🚀 Deploying Next Click Predictor to Google Cloud Run"
echo "   Project: ${PROJECT_ID}"
echo "   Service: ${SERVICE_NAME}"
echo "   Region: ${REGION}"
echo "   Image: ${IMAGE_NAME}"

# Check if gcloud is installed and authenticated
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI not found. Please install Google Cloud SDK."
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker."
    exit 1
fi

# Set project
echo "📋 Setting Google Cloud project..."
gcloud config set project ${PROJECT_ID}

# Enable required APIs
echo "🔧 Enabling required Google Cloud APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Build and push Docker image
echo "🏗️ Building Docker image for Cloud Run..."
docker build -f Dockerfile.cloudrun -t ${IMAGE_NAME}:latest .

echo "📤 Pushing image to Google Container Registry..."
docker push ${IMAGE_NAME}:latest

# Deploy to Cloud Run
echo "🚢 Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
  --image ${IMAGE_NAME}:latest \
  --platform managed \
  --region ${REGION} \
  --allow-unauthenticated \
  --memory 512Mi \
  --cpu 1 \
  --max-instances 10 \
  --min-instances 0 \
  --concurrency 80 \
  --timeout 300 \
  --port 8080 \
  --set-env-vars PORT=8080,WORKERS=1

# Get service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --platform managed --region ${REGION} --format 'value(status.url)')

echo ""
echo "✅ Deployment completed successfully!"
echo "🌐 Service URL: ${SERVICE_URL}"
echo "📊 Health Check: ${SERVICE_URL}/health"
echo "📚 API Docs: ${SERVICE_URL}/docs"
echo ""
echo "🔧 To update Vercel frontend, set environment variable:"
echo "   NEXT_PUBLIC_API_URL=${SERVICE_URL}"
echo ""
echo "📈 Monitor your deployment:"
echo "   gcloud run services describe ${SERVICE_NAME} --region=${REGION}"
echo ""

# Test the deployment
echo "🧪 Testing deployment..."
curl -s "${SERVICE_URL}/health" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    if data.get('status') == 'healthy':
        print('✅ Health check passed!')
    else:
        print('⚠️ Health check returned unexpected status')
except:
    print('❌ Health check failed')
"

echo "🎉 Next Click Predictor is now running on Google Cloud Run!"