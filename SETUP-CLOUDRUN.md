# Google Cloud Run Setup Guide 🚀

Complete step-by-step guide to deploy Next Click Predictor to Google Cloud Run's free tier.

## 📋 Prerequisites

- Google account
- Command line access (Terminal/PowerShell)
- Git repository cloned locally
- Basic familiarity with command line

## Step 1: Create Google Cloud Project 🏗️

### 1.1 Go to Google Cloud Console
1. Visit [Google Cloud Console](https://console.cloud.google.com)
2. Sign in with your Google account
3. Accept terms of service if prompted

### 1.2 Create New Project
1. Click **"Select a project"** dropdown at the top
2. Click **"New Project"**
3. Enter project details:
   - **Project name**: `next-click-predictor` (or your preferred name)
   - **Organization**: Leave as default (usually None)
   - **Location**: Leave as default
4. Click **"Create"**
5. Wait for project creation (30-60 seconds)
6. **Note your Project ID** (e.g., `next-click-predictor-123456`)

### 1.3 Enable Billing (Free Tier Available)
1. Go to **Billing** in the sidebar
2. Click **"Link a billing account"**
3. Create new billing account or use existing
4. **Note**: Free tier includes $300 credit + always-free Cloud Run usage

## Step 2: Install Google Cloud SDK 🛠️

### For Windows:
```powershell
# Download and run installer
# Visit: https://cloud.google.com/sdk/docs/install-sdk
# Download GoogleCloudSDKInstaller.exe and run it
```

### For macOS:
```bash
# Using Homebrew (recommended)
brew install google-cloud-sdk

# OR download installer
curl https://sdk.cloud.google.com | bash
exec -l $SHELL  # Restart shell
```

### For Linux/Ubuntu:
```bash
# Add Google Cloud SDK repository
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list

# Import Google Cloud public key
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -

# Update and install
sudo apt-get update && sudo apt-get install google-cloud-cli
```

### Verify Installation:
```bash
gcloud --version
# Should show: Google Cloud SDK 400.0.0+ (or similar)
```

## Step 3: Authentication & Configuration 🔐

### 3.1 Login to Google Cloud
```bash
# Initialize gcloud and login
gcloud init

# Follow the prompts:
# 1. Choose "Log in with a new account"
# 2. Browser will open - sign in with your Google account
# 3. Select your project (next-click-predictor-123456)
# 4. Choose default region: us-central1 (recommended for free tier)
```

### 3.2 Set Project Configuration
```bash
# Set your project ID (replace with your actual project ID)
export PROJECT_ID="next-click-predictor-123456"
gcloud config set project $PROJECT_ID

# Set default region (free tier eligible)
gcloud config set run/region us-central1

# Verify configuration
gcloud config list
```

### 3.3 Enable Required APIs
```bash
# Enable Cloud Run API
gcloud services enable run.googleapis.com

# Enable Container Registry API
gcloud services enable containerregistry.googleapis.com

# Enable Cloud Build API (for easier Docker builds)
gcloud services enable cloudbuild.googleapis.com

# Verify APIs are enabled
gcloud services list --enabled
```

## Step 4: Configure Docker Authentication 🐳

### 4.1 Install Docker (if not already installed)

**Windows/Mac**: Download from [docker.com](https://www.docker.com/products/docker-desktop)

**Linux**:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install docker.io
sudo usermod -aG docker $USER
# Log out and back in for group changes to take effect
```

### 4.2 Configure Docker for Google Cloud
```bash
# Configure Docker to use gcloud as credential helper
gcloud auth configure-docker

# Test Docker authentication
docker --version
gcloud auth list
```

## Step 5: Deploy to Cloud Run 🚢

### 5.1 Navigate to Project Directory
```bash
# Go to your project directory
cd /path/to/next_click_predictor
# or wherever you cloned the repository

# Verify you have the Cloud Run files
ls -la cloudrun_app.py Dockerfile.cloudrun requirements.cloudrun.txt
```

### 5.2 Set Environment Variables
```bash
# Set your project ID (replace with actual)
export GOOGLE_CLOUD_PROJECT="next-click-predictor-123456"
export GOOGLE_CLOUD_REGION="us-central1"

# Verify environment variables
echo "Project: $GOOGLE_CLOUD_PROJECT"
echo "Region: $GOOGLE_CLOUD_REGION"
```

### 5.3 Option A: Automated Deployment (Recommended)
```bash
# Make deployment script executable
chmod +x deploy-cloudrun.sh

# Run automated deployment
./deploy-cloudrun.sh
```

### 5.4 Option B: Manual Deployment
```bash
# Build and push Docker image
SERVICE_NAME="next-click-predictor"
IMAGE_NAME="gcr.io/$GOOGLE_CLOUD_PROJECT/$SERVICE_NAME"

# Build Docker image
docker build -f Dockerfile.cloudrun -t $IMAGE_NAME:latest .

# Push to Google Container Registry
docker push $IMAGE_NAME:latest

# Deploy to Cloud Run
gcloud run deploy $SERVICE_NAME \
  --image $IMAGE_NAME:latest \
  --platform managed \
  --region $GOOGLE_CLOUD_REGION \
  --allow-unauthenticated \
  --memory 512Mi \
  --cpu 1 \
  --max-instances 10 \
  --min-instances 0 \
  --concurrency 80 \
  --timeout 300 \
  --port 8080
```

## Step 6: Get Your Service URL 🌐

After deployment completes, you'll see output like:
```
Service URL: https://next-click-predictor-abcd1234-uc.a.run.app
```

### Manually get service URL:
```bash
# Get service URL
gcloud run services describe next-click-predictor \
  --region=$GOOGLE_CLOUD_REGION \
  --format='value(status.url)'
```

## Step 7: Test Your Deployment 🧪

### 7.1 Test Health Endpoint
```bash
# Replace with your actual service URL
SERVICE_URL="https://next-click-predictor-abcd1234-uc.a.run.app"

# Test health check
curl "$SERVICE_URL/health"

# Should return:
# {"status":"healthy","platform":"Google Cloud Run",...}
```

### 7.2 Test with Automated Script
```bash
# Set your Cloud Run URL
export CLOUDRUN_URL="https://next-click-predictor-abcd1234-uc.a.run.app"

# Run comprehensive tests
python3 test-cloudrun.py
```

### 7.3 Test API Documentation
Visit your service URL + `/docs`:
```
https://next-click-predictor-abcd1234-uc.a.run.app/docs
```

## Step 8: Update Vercel Frontend 🎨

### 8.1 Update Environment Variable
```bash
# Go to frontend directory
cd frontend

# Create production environment file
echo "NEXT_PUBLIC_API_URL=https://next-click-predictor-abcd1234-uc.a.run.app" > .env.production

# Update local development (optional)
echo "NEXT_PUBLIC_API_URL=https://next-click-predictor-abcd1234-uc.a.run.app" > .env.local
```

### 8.2 Deploy Frontend to Vercel
```bash
# If you haven't installed Vercel CLI
npm install -g vercel

# Deploy to Vercel
vercel --prod

# Or update environment variable in Vercel dashboard:
# 1. Go to vercel.com/dashboard
# 2. Select your project
# 3. Go to Settings > Environment Variables
# 4. Add: NEXT_PUBLIC_API_URL = your-cloud-run-url
# 5. Redeploy
```

## Step 9: Monitor and Manage 📊

### 9.1 View Logs
```bash
# View recent logs
gcloud logs read "resource.type=cloud_run_revision" --limit=50

# Follow logs in real-time
gcloud logs tail "resource.type=cloud_run_revision"
```

### 9.2 Monitor Usage (Free Tier)
```bash
# Check service status
gcloud run services describe next-click-predictor --region=$GOOGLE_CLOUD_REGION

# View metrics in console
echo "Visit: https://console.cloud.google.com/run"
```

### 9.3 Update Service
```bash
# To update your service, simply run the deployment again
./deploy-cloudrun.sh

# Or manually rebuild and deploy
docker build -f Dockerfile.cloudrun -t $IMAGE_NAME:latest .
docker push $IMAGE_NAME:latest
gcloud run deploy next-click-predictor --image $IMAGE_NAME:latest
```

## 🚨 Troubleshooting

### Common Issues:

#### 1. Authentication Errors
```bash
# Re-authenticate
gcloud auth login
gcloud auth application-default login
```

#### 2. Permission Denied
```bash
# Check project permissions
gcloud projects get-iam-policy $GOOGLE_CLOUD_PROJECT

# Add yourself as editor (if needed)
gcloud projects add-iam-policy-binding $GOOGLE_CLOUD_PROJECT \
  --member="user:your-email@gmail.com" \
  --role="roles/editor"
```

#### 3. Docker Push Fails
```bash
# Reconfigure Docker authentication
gcloud auth configure-docker --quiet
```

#### 4. Service Won't Start
```bash
# Check logs for errors
gcloud logs read "resource.type=cloud_run_revision AND resource.labels.service_name=next-click-predictor" --limit=50
```

#### 5. 502/503 Errors
```bash
# Check service health
curl "https://your-service-url/health"

# Increase memory if needed
gcloud run services update next-click-predictor --memory=1Gi
```

## 💰 Cost Management

### Free Tier Limits:
- **2 million requests** per month
- **400,000 GB-seconds** compute time
- **200,000 vCPU-seconds** per month
- **1GB egress** per month

### Monitor Usage:
1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Navigate to **Billing > Reports**
3. Filter by **Cloud Run** service

### Set Budget Alerts:
```bash
# Create budget alert (optional)
gcloud billing budgets create \
  --billing-account=YOUR-BILLING-ACCOUNT-ID \
  --display-name="Cloud Run Budget" \
  --budget-amount=10USD
```

## 🎉 Success!

Your Next Click Predictor is now running on Google Cloud Run!

### Final URLs:
- **Backend API**: `https://next-click-predictor-abcd1234-uc.a.run.app`
- **API Docs**: `https://next-click-predictor-abcd1234-uc.a.run.app/docs`
- **Health Check**: `https://next-click-predictor-abcd1234-uc.a.run.app/health`

### Next Steps:
1. Update your Vercel frontend with the Cloud Run URL
2. Test the full stack integration
3. Monitor usage in Google Cloud Console
4. Scale up when you exceed free tier limits

## 📞 Support

- **Google Cloud Run Docs**: [cloud.google.com/run/docs](https://cloud.google.com/run/docs)
- **Pricing Calculator**: [cloud.google.com/products/calculator](https://cloud.google.com/products/calculator)
- **Community Support**: [stackoverflow.com/questions/tagged/google-cloud-run](https://stackoverflow.com/questions/tagged/google-cloud-run)