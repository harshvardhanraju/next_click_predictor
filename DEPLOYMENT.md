# Deployment Guide

This guide explains how to deploy the Next Click Predictor app using Vercel (frontend) and Railway (backend).

## Architecture

- **Frontend**: Next.js app hosted on Vercel
- **Backend**: FastAPI service hosted on Railway
- **Database**: None required (stateless application)

## Prerequisites

1. GitHub account
2. Vercel account (free)
3. Railway account (free tier available)

## Backend Deployment (Railway)

### 1. Push Code to GitHub
Ensure your code is pushed to your GitHub repository.

### 2. Deploy to Railway

1. Go to [Railway.app](https://railway.app) and sign in
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Choose your `next_click_predictor` repository
5. Railway will automatically detect the Dockerfile and deploy

### 3. Configure Environment Variables (if needed)
- No additional environment variables required for basic deployment
- The service will be available at: `https://your-app-name.railway.app`

### 4. Verify Deployment
- Check the health endpoint: `https://your-app-name.railway.app/health`
- View API documentation: `https://your-app-name.railway.app/docs`

## Frontend Deployment (Vercel)

### 1. Install Dependencies
```bash
cd frontend
npm install
```

### 2. Configure Environment Variables
1. Copy `.env.example` to `.env.local`:
```bash
cp .env.example .env.local
```

2. Update the API URL in `.env.local`:
```
NEXT_PUBLIC_API_URL=https://your-railway-app-name.railway.app
```

### 3. Deploy to Vercel

**Option A: Vercel CLI**
```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
cd frontend
vercel --prod
```

**Option B: Vercel Dashboard**
1. Go to [vercel.com](https://vercel.com) and sign in
2. Click "New Project"
3. Import your GitHub repository
4. Set the root directory to `frontend`
5. Add environment variable:
   - `NEXT_PUBLIC_API_URL` = `https://your-railway-app-name.railway.app`
6. Deploy

### 4. Verify Deployment
- Your frontend will be available at: `https://your-project-name.vercel.app`

## Testing the Deployment

1. **Backend Health Check**:
   ```bash
   curl https://your-railway-app-name.railway.app/health
   ```

2. **Frontend Access**:
   Open `https://your-project-name.vercel.app` in your browser

3. **End-to-End Test**:
   - Upload a screenshot through the frontend
   - Fill in user profile and task description
   - Verify predictions are returned

## Local Development

### Backend
```bash
# Install dependencies
pip install -r requirements.txt

# Run server
uvicorn src.web_service:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

Access the app at `http://localhost:3000`

## Troubleshooting

### Common Issues

1. **CORS Errors**:
   - Ensure the backend CORS middleware is properly configured
   - Check that the frontend is using the correct API URL

2. **Cold Start Delays**:
   - Railway free tier has cold starts after inactivity
   - First request may take 30-60 seconds

3. **Memory Issues**:
   - Computer vision libraries require significant memory
   - Railway free tier provides 512MB-1GB RAM
   - Consider optimizing image processing if issues occur

4. **Build Failures**:
   - Check Railway build logs for dependency installation errors
   - Ensure Docker image size is reasonable (<1GB recommended)

### Monitoring

- **Railway Logs**: Available in Railway dashboard
- **Vercel Logs**: Available in Vercel dashboard
- **Backend Health**: `/health` endpoint
- **Backend Stats**: `/stats` endpoint

## Cost Estimation

### Railway (Backend)
- **Free Tier**: $5 credit monthly
- **Usage**: ~$3-5/month for light usage
- **Paid Plans**: Start at $5/month

### Vercel (Frontend)
- **Free Tier**: 100GB bandwidth, unlimited static sites
- **Usage**: Likely to stay within free tier
- **Paid Plans**: Start at $20/month for teams

**Total Monthly Cost**: $0-5 for personal use

## Scaling Considerations

### Performance Optimization
1. **Image Compression**: Compress uploaded images before processing
2. **Caching**: Cache prediction results for identical inputs  
3. **Async Processing**: Use background tasks for heavy processing
4. **CDN**: Use Vercel's CDN for static assets

### Horizontal Scaling
1. **Multiple Railway Services**: Deploy multiple backend instances
2. **Load Balancer**: Use Railway's built-in load balancing
3. **Database**: Add Redis for caching if needed

## Security Notes

1. **CORS**: Update CORS origins to specific domains in production
2. **Rate Limiting**: Add rate limiting to prevent abuse
3. **Input Validation**: Validate file sizes and types
4. **API Keys**: No API keys required for basic functionality

## Support

- **Issues**: Report at GitHub repository
- **Documentation**: See `SYSTEM_ARCHITECTURE.md`
- **API Documentation**: Available at `/docs` endpoint