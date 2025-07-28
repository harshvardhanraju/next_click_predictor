# Next Click Predictor - Google Cloud Run Deployment 🚀

A lightweight, production-ready implementation optimized for Google Cloud Run's free tier.

## 🏗️ Architecture

**Backend**: FastAPI + Google Cloud Run (lightweight Docker container)  
**Frontend**: Next.js + Vercel  
**Integration**: CORS-enabled API with optimized cold start performance

## 🚀 Quick Deployment

### Prerequisites

1. **Google Cloud SDK**: [Install gcloud CLI](https://cloud.google.com/sdk/docs/install)
2. **Docker**: [Install Docker](https://docs.docker.com/get-docker/)
3. **Google Cloud Project**: Create a project in [Google Cloud Console](https://console.cloud.google.com)

### Step 1: Deploy Backend to Cloud Run

```bash
# Set your Google Cloud project ID
export GOOGLE_CLOUD_PROJECT="your-project-id"

# Deploy to Cloud Run (automated script)
./deploy-cloudrun.sh
```

### Step 2: Update Frontend for Cloud Run

```bash
# Update Vercel environment variable
# Copy the Cloud Run URL from deployment output
echo "NEXT_PUBLIC_API_URL=https://next-click-predictor-xyz-uc.a.run.app" > frontend/.env.production

# Deploy frontend to Vercel
cd frontend
npx vercel --prod
```

### Step 3: Test Integration

```bash
# Test the full stack
CLOUDRUN_URL="your-cloud-run-url" python3 test-cloudrun.py
```

## 📊 Cloud Run Optimizations

### Free Tier Friendly
- **Memory**: 512Mi (within free tier limit)
- **CPU**: 1 vCPU (optimized allocation)
- **Cold Start**: <2 seconds (lightweight dependencies)
- **Image Size**: ~150MB (multi-stage Docker build)

### Performance Features
- **Concurrency**: 80 requests per instance
- **Auto-scaling**: 0-10 instances
- **Timeout**: 300 seconds
- **Health Checks**: Built-in liveness/readiness probes

## 🔧 Local Development

### Backend Development
```bash
# Install dependencies
pip install -r requirements.cloudrun.txt

# Start backend locally
python3 cloudrun_app.py
# Server runs on http://localhost:8080
```

### Frontend Development
```bash
cd frontend

# Install dependencies
npm install

# Start frontend (connects to local backend)
npm run dev
# Frontend runs on http://localhost:3000
```

### Full Stack Testing
```bash
# Test both backend and frontend
python3 test-cloudrun.py
```

## 📁 Project Structure

```
├── cloudrun_app.py              # Cloud Run optimized FastAPI backend
├── Dockerfile.cloudrun          # Multi-stage Docker build
├── requirements.cloudrun.txt    # Minimal dependencies
├── cloudrun.yaml               # Cloud Run service configuration
├── deploy-cloudrun.sh          # Automated deployment script
├── test-cloudrun.py           # Integration testing
└── frontend/
    ├── .env.cloudrun          # Cloud Run URL configuration
    ├── .env.local             # Local development
    └── src/                   # Next.js application
```

## 🧪 API Endpoints

### Base URL
- **Local**: `http://localhost:8080`
- **Cloud Run**: `https://next-click-predictor-[hash]-uc.a.run.app`

### Endpoints
- `GET /` - Service information
- `GET /health` - Health check (for Cloud Run monitoring)
- `POST /predict` - Click prediction API
- `GET /docs` - Interactive API documentation

### Example API Usage

```bash
# Health check
curl https://your-service-url/health

# Predict next click
curl -X POST "https://your-service-url/predict" \
  -F "file=@screenshot.png" \
  -F "user_attributes={\"age_group\":\"25-34\",\"tech_savviness\":\"medium\",\"device_type\":\"desktop\"}" \
  -F "task_description=Complete checkout process"
```

## 🔒 Security Features

- **Non-root container**: Runs as user `cloudrun` (UID 65532)
- **Minimal attack surface**: Only essential dependencies
- **CORS protection**: Configured for Vercel domains
- **Input validation**: File type and size limits
- **Error handling**: Secure error responses

## 💰 Cost Optimization

### Google Cloud Run Free Tier
- **2 million requests** per month
- **400,000 GB-seconds** compute time
- **200,000 vCPU-seconds** per month
- **1GB outbound traffic** per month

### Estimated Usage (Free Tier)
- **Typical request**: ~100ms processing time
- **Memory usage**: ~128MB average
- **Monthly capacity**: ~500,000 predictions (well within free tier)

## 🔍 Monitoring & Debugging

### Cloud Run Monitoring
```bash
# View service details
gcloud run services describe next-click-predictor --region=us-central1

# View logs
gcloud logs read "resource.type=cloud_run_revision AND resource.labels.service_name=next-click-predictor" --limit=50

# Monitor metrics
gcloud monitoring metrics list --filter="resource.type=cloud_run_revision"
```

### Local Debugging
```bash
# View backend logs
python3 cloudrun_app.py

# Test specific endpoints
python3 test-cloudrun.py

# Frontend debugging
cd frontend && npm run dev
```

## 🚨 Troubleshooting

### Common Issues

1. **Cold Start Timeout**
   ```bash
   # Increase timeout in cloudrun.yaml
   timeoutSeconds: 300
   ```

2. **Memory Exceeded**
   ```bash
   # Monitor memory usage
   gcloud run services update next-click-predictor --memory=1Gi
   ```

3. **CORS Errors**
   ```bash
   # Update CORS origins in cloudrun_app.py
   allow_origins=["https://your-frontend-domain.vercel.app"]
   ```

4. **Frontend API Connection**
   ```bash
   # Verify environment variable
   echo $NEXT_PUBLIC_API_URL
   
   # Test API endpoint
   curl https://your-cloud-run-url/health
   ```

## 🔄 Updates & Maintenance

### Update Backend
```bash
# Make changes to cloudrun_app.py
# Redeploy
./deploy-cloudrun.sh
```

### Update Frontend
```bash
cd frontend
# Update API URL if needed
echo "NEXT_PUBLIC_API_URL=new-url" > .env.production
# Redeploy to Vercel
npx vercel --prod
```

## 📈 Scaling Beyond Free Tier

When you exceed free tier limits:

1. **Enable billing** in Google Cloud Console
2. **Increase resources** in `cloudrun.yaml`:
   ```yaml
   resources:
     limits:
       memory: "1Gi"
       cpu: "2000m"
   ```
3. **Add autoscaling**:
   ```yaml
   autoscaling.knative.dev/maxScale: "100"
   ```

## 🤝 Contributing

1. Fork the repository
2. Make changes to `cloudrun_app.py` or frontend
3. Test locally with `python3 test-cloudrun.py`
4. Submit pull request

## 📄 License

MIT License - see LICENSE file for details.

---

**🎯 Ready for production!** This implementation provides a robust, scalable, and cost-effective solution for click prediction using Google Cloud Run's free tier.