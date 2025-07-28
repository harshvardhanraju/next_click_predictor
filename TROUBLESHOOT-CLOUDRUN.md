# Google Cloud Run Troubleshooting Guide 🔧

Common issues and solutions when deploying Next Click Predictor to Google Cloud Run.

## 🚨 Common Deployment Issues

### Issue 1: "gcloud: command not found"

**Problem**: Google Cloud SDK not installed or not in PATH

**Solution**:
```bash
# Check if gcloud is installed
which gcloud

# If not found, install Google Cloud SDK:
# macOS with Homebrew:
brew install google-cloud-sdk

# Linux:
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Windows: Download installer from https://cloud.google.com/sdk/docs/install
```

### Issue 2: "The user does not have access to project"

**Problem**: Authentication or permission issues

**Solution**:
```bash
# Re-authenticate
gcloud auth login

# Set correct project
gcloud config set project YOUR-PROJECT-ID

# Check current configuration
gcloud config list

# If needed, add yourself as project editor
gcloud projects add-iam-policy-binding YOUR-PROJECT-ID \
  --member="user:your-email@gmail.com" \
  --role="roles/editor"
```

### Issue 3: "API [run.googleapis.com] not enabled"

**Problem**: Required APIs not enabled

**Solution**:
```bash
# Enable all required APIs
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable cloudbuild.googleapis.com

# Verify APIs are enabled
gcloud services list --enabled | grep -E "(run|container|build)"
```

### Issue 4: "Docker push permission denied"

**Problem**: Docker not authenticated with Google Cloud Registry

**Solution**:
```bash
# Configure Docker authentication
gcloud auth configure-docker

# If still failing, try explicit authentication
gcloud auth print-access-token | docker login -u oauth2accesstoken --password-stdin https://gcr.io

# Test Docker authentication
docker pull gcr.io/google-containers/busybox
```

### Issue 5: "Build failed: Dockerfile not found"

**Problem**: Wrong directory or missing Dockerfile

**Solution**:
```bash
# Check you're in the correct directory
pwd
ls -la | grep -E "(cloudrun_app.py|Dockerfile.cloudrun)"

# If files missing, ensure you're in the project root
cd /path/to/next_click_predictor

# Verify all required files exist
ls -la cloudrun_app.py Dockerfile.cloudrun requirements.cloudrun.txt
```

## 🔥 Runtime Issues

### Issue 6: Service returns 502/503 errors

**Problem**: Application failing to start or respond

**Solution**:
```bash
# Check service logs
gcloud logs read "resource.type=cloud_run_revision AND resource.labels.service_name=next-click-predictor" --limit=50

# Common fixes:
# 1. Increase memory allocation
gcloud run services update next-click-predictor --memory=1Gi --region=us-central1

# 2. Increase timeout
gcloud run services update next-click-predictor --timeout=600 --region=us-central1

# 3. Check if app is binding to correct port
# Ensure cloudrun_app.py uses PORT environment variable
```

### Issue 7: "Container failed to start"

**Problem**: Application not starting properly

**Solution**:
```bash
# Test Docker image locally first
docker build -f Dockerfile.cloudrun -t test-image .
docker run -p 8080:8080 -e PORT=8080 test-image

# Check application logs
gcloud logs tail "resource.type=cloud_run_revision"

# Common issues:
# - Missing dependencies in requirements.cloudrun.txt
# - Python path issues
# - Port binding problems
```

### Issue 8: CORS errors from frontend

**Problem**: Frontend can't connect to backend due to CORS

**Solution**:
```bash
# Update CORS origins in cloudrun_app.py
# Add your Vercel domain to allow_origins list:

allow_origins=[
    "https://your-frontend.vercel.app",
    "https://*.vercel.app",
    "*"  # Remove in production
]

# Redeploy after changes
./deploy-cloudrun.sh
```

## 💰 Free Tier Issues

### Issue 9: "Quota exceeded" errors

**Problem**: Exceeding free tier limits

**Solution**:
```bash
# Check current usage
gcloud logging read "resource.type=cloud_run_revision" --limit=1 --format="table(timestamp,resource.labels.service_name)"

# Monitor usage in console
echo "Visit: https://console.cloud.google.com/billing/reports"

# Optimize for free tier:
# 1. Reduce max instances
gcloud run services update next-click-predictor --max-instances=3

# 2. Set minimum instances to 0 (default)
gcloud run services update next-click-predictor --min-instances=0

# 3. Reduce memory if possible
gcloud run services update next-click-predictor --memory=512Mi
```

### Issue 10: Billing account required

**Problem**: Need to enable billing even for free tier

**Solution**:
1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Navigate to **Billing**
3. Create billing account (credit card required but won't be charged for free tier usage)
4. Link billing account to your project

## 🐛 Development Issues

### Issue 11: Local testing fails

**Problem**: Can't test locally before deployment

**Solution**:
```bash
# Test Python app directly
python3 cloudrun_app.py

# Test with uvicorn
pip install -r requirements.cloudrun.txt
uvicorn cloudrun_app:app --host 0.0.0.0 --port 8080

# Test Docker image locally
docker build -f Dockerfile.cloudrun -t local-test .
docker run -p 8080:8080 -e PORT=8080 local-test

# Test endpoints
curl http://localhost:8080/health
```

### Issue 12: "Module not found" errors

**Problem**: Python dependencies not installed correctly

**Solution**:
```bash
# Check requirements file
cat requirements.cloudrun.txt

# Test locally first
pip install -r requirements.cloudrun.txt
python3 -c "import fastapi, uvicorn; print('Dependencies OK')"

# If using virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate  # Windows
pip install -r requirements.cloudrun.txt
```

## 🔍 Diagnostic Commands

### Check Service Status
```bash
# Get service details
gcloud run services describe next-click-predictor --region=us-central1

# List all Cloud Run services
gcloud run services list

# Check service URL
gcloud run services describe next-click-predictor --region=us-central1 --format='value(status.url)'
```

### View Logs
```bash
# Recent logs
gcloud logs read "resource.type=cloud_run_revision" --limit=100

# Follow logs in real-time
gcloud logs tail "resource.type=cloud_run_revision"

# Filter by service
gcloud logs read "resource.type=cloud_run_revision AND resource.labels.service_name=next-click-predictor" --limit=50
```

### Test Connectivity
```bash
# Test health endpoint
SERVICE_URL=$(gcloud run services describe next-click-predictor --region=us-central1 --format='value(status.url)')
curl "$SERVICE_URL/health"

# Test with verbose output
curl -v "$SERVICE_URL/"

# Test from different locations
curl -H "User-Agent: Test" "$SERVICE_URL/health"
```

## 🆘 Getting Help

### Google Cloud Support
- **Documentation**: [cloud.google.com/run/docs](https://cloud.google.com/run/docs)
- **Community**: [Stack Overflow](https://stackoverflow.com/questions/tagged/google-cloud-run)
- **Discord**: [Google Cloud Community](https://discord.gg/google-cloud)

### Project Specific Help
```bash
# Run diagnostic script
python3 test-cloudrun.py

# Check project configuration
gcloud config list
gcloud projects describe $GOOGLE_CLOUD_PROJECT

# Verify permissions
gcloud projects get-iam-policy $GOOGLE_CLOUD_PROJECT
```

### Emergency Reset
If everything fails, you can start fresh:

```bash
# Delete existing service
gcloud run services delete next-click-predictor --region=us-central1

# Clean up Docker images
docker system prune -a

# Re-authenticate
gcloud auth login
gcloud auth configure-docker

# Start over with deployment
./deploy-cloudrun.sh
```

## 📞 Contact Support

If you're still having issues:

1. **Check the logs first**: `gcloud logs read "resource.type=cloud_run_revision" --limit=50`
2. **Try local testing**: `python3 test-cloudrun.py`
3. **Verify configuration**: `gcloud config list`
4. **Search existing issues**: [GitHub Issues](https://github.com/harshvardhanraju/next_click_predictor/issues)

Most issues are configuration-related and can be resolved by following the setup guide carefully.