#!/bin/bash

# Redeploy Cloud Run with full ML dependencies
# This fixes the "Advanced UI detector not available" warning

set -e

# Configuration
PROJECT_ID="gen-lang-client-0569475118"
SERVICE_NAME="next-click-predictor"
REGION="asia-south1"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "🚀 Redeploying Cloud Run with full ML dependencies..."
echo "📦 Project: ${PROJECT_ID}"
echo "🌏 Region: ${REGION}"
echo "🏷️  Image: ${IMAGE_NAME}"

# Build and push the image
echo "📦 Building Docker image..."
docker build -f Dockerfile.cloudrun -t ${IMAGE_NAME} .

echo "⬆️  Pushing image to Container Registry..."
docker push ${IMAGE_NAME}

# Deploy to Cloud Run
echo "🚀 Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
  --image ${IMAGE_NAME} \
  --platform managed \
  --region ${REGION} \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --concurrency 80 \
  --max-instances 10

# Get service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --region=${REGION} --format="value(status.url)")

echo "✅ Deployment completed!"
echo "🌐 Service URL: ${SERVICE_URL}"
echo "🏥 Health check: ${SERVICE_URL}/health"
echo "📚 API docs: ${SERVICE_URL}/docs"

# Test the deployment
echo "🧪 Testing deployment..."
curl -s "${SERVICE_URL}/health" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(f'✅ Health check passed: {data.get(\"status\", \"unknown\")}')
    print(f'🧠 ML status: {data.get(\"ml_status\", \"unknown\")}')
except:
    print('❌ Health check failed')
    sys.exit(1)
"

echo "🎉 Redeployment with full ML dependencies completed successfully!"
echo "⚠️  Note: The 'Advanced UI detector not available' warning should now be resolved."