#!/bin/bash

# Deploy Simple Version to Google Cloud Run
# This script deploys a minimal fallback version that works reliably

set -e

echo "🚀 DEPLOYING SIMPLE VERSION TO GOOGLE CLOUD RUN"
echo "================================================"

# Configuration
PROJECT_ID="next-click-predictor-439203"
SERVICE_NAME="next-click-predictor"
REGION="asia-south1"
DOCKERFILE="Dockerfile.simple"

echo "📋 Configuration:"
echo "   Project: $PROJECT_ID"
echo "   Service: $SERVICE_NAME"  
echo "   Region: $REGION"
echo "   Dockerfile: $DOCKERFILE"
echo ""

echo "🔧 Setting up gcloud configuration..."
gcloud config set project $PROJECT_ID

echo "🏗️ Building and deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --source . \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --memory 512Mi \
    --cpu 1 \
    --timeout 300 \
    --max-instances 10 \
    --dockerfile $DOCKERFILE

echo ""
echo "✅ DEPLOYMENT COMPLETED!"
echo "================================================"

# Get service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)")
echo "🌐 Service URL: $SERVICE_URL"

echo ""
echo "🧪 Testing deployment..."
echo "Basic health check:"
curl -s "$SERVICE_URL/" | head -3

echo ""
echo "📊 Service status:"
curl -s "$SERVICE_URL/health" | head -3

echo ""
echo "✅ Simple version deployed successfully!"
echo "   This fallback mode provides stable service while ML issues are resolved"