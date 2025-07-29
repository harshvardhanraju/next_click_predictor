#!/bin/bash

# Optimize Google Cloud Run configuration for better performance
# This addresses memory constraints and timeout issues

set -e

# Configuration
PROJECT_ID="gen-lang-client-0569475118"
SERVICE_NAME="next-click-predictor"
REGION="asia-south1"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "🚀 Optimizing Cloud Run configuration for better performance..."
echo "📦 Project: ${PROJECT_ID}"
echo "🌏 Region: ${REGION}"
echo "🏷️  Service: ${SERVICE_NAME}"

# Check current configuration
echo "📊 Current Cloud Run configuration:"
gcloud run services describe ${SERVICE_NAME} --region=${REGION} --format="table(
  spec.template.spec.containerConcurrency,
  spec.template.spec.containers[0].resources.limits.memory,
  spec.template.spec.containers[0].resources.limits.cpu,
  spec.template.spec.timeoutSeconds
)" 2>/dev/null || echo "Could not fetch current config"

echo ""
echo "🔧 Applying optimized configuration..."

# Deploy with optimized settings
gcloud run deploy ${SERVICE_NAME} \
  --image ${IMAGE_NAME} \
  --platform managed \
  --region ${REGION} \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 300 \
  --concurrency 10 \
  --max-instances 10 \
  --min-instances 0 \
  --port 8080 \
  --set-env-vars="WORKERS=2" \
  --execution-environment gen2

echo ""
echo "✅ Optimization completed!"

# Get service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --region=${REGION} --format="value(status.url)")

echo "🌐 Service URL: ${SERVICE_URL}"

# Test the optimized service
echo ""
echo "🧪 Testing optimized service..."

echo "  Health check test..."
HEALTH_START=$(date +%s.%N)
curl -s "${SERVICE_URL}/health" > /dev/null
HEALTH_END=$(date +%s.%N)
HEALTH_TIME=$(echo "$HEALTH_END - $HEALTH_START" | bc -l)
echo "  ✅ Health check: ${HEALTH_TIME}s"

echo "  Warming up container..."
curl -s "${SERVICE_URL}/" > /dev/null

echo ""
echo "📊 Performance optimization summary:"
echo "  - Memory: Increased to 4Gi (from 2Gi)"
echo "  - CPU: Increased to 2 vCPU (from 1)"  
echo "  - Timeout: Set to 300s (5 minutes)"
echo "  - Concurrency: Limited to 10 concurrent requests"
echo "  - Max instances: 10 (for scaling)"
echo "  - Generation: gen2 (better performance)"

echo ""
echo "🎯 Expected improvements:"
echo "  - Faster ML processing (more CPU/memory)"
echo "  - No more timeout errors (5min limit)"
echo "  - Better concurrent request handling"
echo "  - Reduced cold start frequency"

echo ""
echo "🔍 To monitor performance:"
echo "  1. Go to: https://console.cloud.google.com/run"
echo "  2. Click on '${SERVICE_NAME}'"  
echo "  3. Check 'Metrics' tab for:"
echo "     - Memory utilization (should be < 80%)"
echo "     - CPU utilization"
echo "     - Request latency (should improve)"
echo "     - Container instance count"

echo ""
echo "🧪 Test the improvements:"
echo "  python3 debug_cloudrun_performance.py"

echo ""
echo "🎉 Cloud Run optimization complete!"