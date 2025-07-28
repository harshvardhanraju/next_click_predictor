# Deploy to Google Cloud Run via Web UI 🌐

Easy deployment guide using Google Cloud Console - no SDK installation required!

## 🚀 Web UI Deployment Steps

### Step 1: Access Cloud Run Console
You already have the perfect link! Go to:
```
https://console.cloud.google.com/run/deploy/asia-south1/next-click-predictor?project=gen-lang-client-0569475118&hl=en&inv=1&invt=Ab380A
```

### Step 2: Choose Deployment Source

You'll see options for deployment source. Choose **"Deploy one revision from a source"**

#### Option A: Deploy from GitHub (Recommended)
1. Click **"Continuously deploy from a repository"**
2. Click **"Set up with Cloud Build"**
3. Choose **"GitHub"** as source
4. Authenticate with GitHub if prompted
5. Select repository: `harshvardhan.raju/next_click_predictor`
6. Select branch: `main` or `master`
7. Build configuration:
   - **Build Type**: `Dockerfile`
   - **Dockerfile location**: `/Dockerfile.cloudrun`

#### Option B: Deploy from Uploaded Container
If GitHub connection fails, you can build locally and upload:

1. Choose **"Deploy one revision from an existing container image"**
2. We'll build the container in the next section

### Step 3: Container Configuration

#### If using GitHub (Option A):
- The system will automatically build using `Dockerfile.cloudrun`
- Skip to Step 4

#### If uploading container (Option B):
1. Open Google Cloud Shell (click the terminal icon in the top bar)
2. Clone your repository:
   ```bash
   git clone https://github.com/harshvardhanraju/next_click_predictor.git
   cd next_click_predictor
   ```
3. Build and push container:
   ```bash
   # Build the image
   docker build -f Dockerfile.cloudrun -t gcr.io/gen-lang-client-0569475118/next-click-predictor:latest .
   
   # Push to Container Registry
   docker push gcr.io/gen-lang-client-0569475118/next-click-predictor:latest
   ```
4. Back in the UI, enter container URL:
   ```
   gcr.io/gen-lang-client-0569475118/next-click-predictor:latest
   ```

### Step 4: Service Configuration

Fill in these settings:

#### Basic Settings:
- **Service name**: `next-click-predictor`
- **Region**: `asia-south1` (already selected)
- **Platform**: `Cloud Run (fully managed)` ✅

#### Advanced Settings (click "Show Advanced Settings"):

**Container Tab:**
- **Container port**: `8080`
- **Memory**: `512 MiB` (free tier friendly)
- **CPU**: `1 vCPU`
- **Request timeout**: `300 seconds`

**Variables & Secrets Tab:**
Add these environment variables:
- `PORT` = `8080`
- `WORKERS` = `1`

**Connections Tab:**
- **Concurrency**: `80`
- **Maximum instances**: `10`
- **Minimum instances**: `0` (for free tier)

**Security Tab:**
- **Authentication**: `Allow unauthenticated invocations` ✅ (for public API)

### Step 5: Deploy!

1. Review all settings
2. Click **"Deploy"** button
3. Wait for deployment (usually 2-5 minutes)
4. You'll see a green checkmark when complete

### Step 6: Get Your Service URL

After deployment completes:
1. You'll see your service URL like:
   ```
   https://next-click-predictor-[random-hash]-an.a.run.app
   ```
2. **Copy this URL** - you'll need it for the frontend!

### Step 7: Test Your Deployment

1. Click on the service URL
2. You should see:
   ```json
   {
     "service": "Next Click Predictor",
     "version": "2.0.0",
     "platform": "Google Cloud Run",
     "status": "healthy"
   }
   ```

3. Test the health endpoint by adding `/health` to your URL:
   ```
   https://your-service-url/health
   ```

4. Test the API documentation at:
   ```
   https://your-service-url/docs
   ```

## 🎯 Update Your Frontend

### Step 8: Configure Vercel Frontend

1. **Update environment variable**:
   - Go to your project files
   - Edit `frontend/.env.production`:
   ```bash
   NEXT_PUBLIC_API_URL=https://next-click-predictor-[your-hash]-an.a.run.app
   ```

2. **Deploy to Vercel**:
   - If using Vercel Dashboard:
     - Go to [vercel.com/dashboard](https://vercel.com/dashboard)
     - Select your project
     - Go to **Settings > Environment Variables**
     - Add: `NEXT_PUBLIC_API_URL` = `your-cloud-run-url`
     - Redeploy from **Deployments** tab

   - If using Vercel CLI:
     ```bash
     cd frontend
     vercel --prod
     ```

## 🔧 Troubleshooting Web UI Deployment

### Issue: Build Fails
**Solution**: Check the build logs in the Cloud Console
- Look for missing files or dependency issues
- Ensure `Dockerfile.cloudrun` exists in repository root

### Issue: Container Won't Start
**Solution**: 
1. Check logs in Cloud Run service page
2. Verify `PORT=8080` environment variable is set
3. Ensure container listens on `0.0.0.0:$PORT`

### Issue: 502/503 Errors
**Solutions**:
1. Increase memory to `1 GiB`
2. Check application logs for startup errors
3. Verify health check endpoint `/health` works

### Issue: CORS Errors
**Solution**: Frontend can't connect due to CORS
- The app is already configured for Vercel domains
- If using custom domain, update CORS settings in `cloudrun_app.py`

## 📊 Monitor Your Service

### View Logs:
1. Go to your Cloud Run service page
2. Click **"Logs"** tab
3. Monitor requests and errors in real-time

### Check Metrics:
1. Click **"Metrics"** tab
2. Monitor:
   - Request count
   - Response time
   - Memory usage
   - Error rate

### Manage Revisions:
1. Click **"Revisions"** tab
2. See deployment history
3. Rollback if needed

## 💰 Free Tier Usage

Your service is configured for Google Cloud's free tier:
- **2 million requests/month**
- **400,000 GB-seconds compute**
- **200,000 vCPU-seconds/month**

Monitor usage at: [console.cloud.google.com/billing](https://console.cloud.google.com/billing)

## ✅ Success Checklist

- [ ] Service deploys without errors
- [ ] Root endpoint returns service info
- [ ] Health check (`/health`) returns `{"status":"healthy"}`
- [ ] API docs accessible at `/docs`
- [ ] Frontend updated with Cloud Run URL
- [ ] End-to-end test works (upload image → get prediction)

## 🎉 You're Done!

Your Next Click Predictor is now running on Google Cloud Run! 

**Service URL**: `https://next-click-predictor-[hash]-an.a.run.app`
**API Docs**: `https://next-click-predictor-[hash]-an.a.run.app/docs`
**Cost**: Free tier (up to 2M requests/month)