# 🔧 Manual Cloud Run Optimization Commands

Since gcloud isn't configured in this environment, please run these commands manually:

## Option 1: Google Cloud Console (Easiest)
1. Go to: https://console.cloud.google.com/run
2. Select project: `gen-lang-client-0569475118`
3. Click on `next-click-predictor` service
4. Click "Edit & Deploy New Revision"
5. Update these settings:
   - **Memory**: 4GiB (increase from 2GiB)
   - **CPU**: 2 (increase from 1)
   - **Request timeout**: 300 seconds (5 minutes)
   - **Maximum concurrent requests**: 10
   - **Maximum instances**: 10

## Option 2: Command Line (if you have gcloud configured)
```bash
gcloud run deploy next-click-predictor \
  --image gcr.io/gen-lang-client-0569475118/next-click-predictor \
  --region asia-south1 \
  --memory 4Gi \
  --cpu 2 \
  --timeout 300 \
  --concurrency 10 \
  --max-instances 10 \
  --allow-unauthenticated
```

## Expected Benefits
- ✅ Faster ML processing (more CPU/memory)
- ✅ No timeout errors (5-minute limit)
- ✅ Better concurrent request handling
- ✅ Improved response times under load

## Current vs Optimized Performance
- **Current**: 4-5 seconds for predictions
- **Optimized**: Expected 2-3 seconds for predictions
- **Memory**: Better headroom for complex ML inference