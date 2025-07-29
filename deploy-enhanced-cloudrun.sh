#!/bin/bash

# Deploy Enhanced ML System to Google Cloud Run
# Make sure you're authenticated: gcloud auth login

PROJECT_ID="next-click-predictor-439203"  # Replace with your project ID
SERVICE_NAME="next-click-predictor"
REGION="asia-south1"

echo "🚀 Deploying Enhanced ML System to Google Cloud Run"
echo "=================================================="

# Check if gcloud is configured
if ! command -v gcloud &> /dev/null; then
    echo "❌ Google Cloud CLI not found. Please install it first."
    echo "Install: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Set project
echo "📋 Setting project: $PROJECT_ID"
gcloud config set project $PROJECT_ID

# Deploy the enhanced app
echo "🔄 Deploying enhanced ML system..."
gcloud run deploy $SERVICE_NAME \
    --source . \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --timeout 300 \
    --max-instances 10 \
    --env-vars-file .env.cloudrun \
    --port 8080

echo "✅ Deployment completed!"
echo "🌐 Your service should be available at:"
echo "https://$SERVICE_NAME-[hash].$REGION.run.app"

# Test the deployment
echo "🧪 Testing deployment..."
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)")
echo "Service URL: $SERVICE_URL"

if [ ! -z "$SERVICE_URL" ]; then
    echo "Testing enhanced endpoints..."
    curl -s "$SERVICE_URL/" | python3 -m json.tool
    echo ""
    echo "ML Status:"
    curl -s "$SERVICE_URL/metrics" | python3 -m json.tool
fi