# Fix Vercel Environment Variable Issue 🔧

The `process is not defined` error is normal - you can't access `process.env` in browser console. Here's how to properly check and fix your Vercel deployment:

## 🔍 **Proper Way to Check Environment Variable**

### Method 1: Check Network Tab
1. Open your Vercel frontend URL
2. Open Developer Tools (F12)
3. Go to **Network** tab
4. Try to upload an image (trigger the API call)
5. Look for the request URL - it should show your Cloud Run URL

### Method 2: Check Source Code
1. In Developer Tools, go to **Sources** tab
2. Find your main JavaScript file
3. Search for "NEXT_PUBLIC_API_URL"
4. See what value it shows

## 🛠️ **Fix Steps:**

### Step 1: Verify Vercel Environment Variable

**Go to Vercel Dashboard:**
1. Visit [vercel.com/dashboard](https://vercel.com/dashboard)
2. Select your project
3. Go to **Settings → Environment Variables**
4. Make sure you have:
   ```
   Key: NEXT_PUBLIC_API_URL
   Value: https://next-click-predictor-157954281090.asia-south1.run.app
   ```

### Step 2: Force Rebuild

**Important:** Just redeploying might not be enough. You need to **rebuild**:

1. Go to **Settings → Functions**
2. Scroll down to **Environment Variables**
3. Make sure `NEXT_PUBLIC_API_URL` is there
4. Go to **Deployments**
5. Click **"..."** on latest deployment
6. Click **"Redeploy"**
7. **Check "Use existing Build Cache"** should be **OFF**

### Step 3: Alternative - Deploy from CLI

```bash
cd frontend
# Make sure your .env.production file has the correct URL
echo "NEXT_PUBLIC_API_URL=https://next-click-predictor-157954281090.asia-south1.run.app" > .env.production

# Deploy
vercel --prod
```

## 🧪 **Test After Fix:**

### Check 1: Look at Network Requests
1. Open your Vercel URL
2. Try to make a prediction
3. In Network tab, you should see:
   ```
   POST https://next-click-predictor-157954281090.asia-south1.run.app/predict
   ```
   
   Instead of:
   ```
   POST undefined/predict  ← This means env var is missing
   ```

### Check 2: Look at Console Errors
After fix, you should NOT see:
- "Failed to fetch"
- "Cannot read property of undefined"
- "Network error"

## 🚨 **Common Issue: Build Cache**

Vercel sometimes caches the old build. Make sure to:

1. **Turn OFF build cache** when redeploying
2. **Wait 2-3 minutes** for full deployment
3. **Hard refresh** your browser (Ctrl+F5 or Cmd+Shift+R)

## 📱 **Mobile Test:**

After fixing:
1. **Clear mobile browser cache**
2. **Try incognito/private mode**
3. **Test your Vercel URL again**

## 🎯 **Expected Result:**

After proper deployment with environment variable:
- ✅ Upload image should work
- ✅ No "Failed to fetch" errors
- ✅ API calls go to `next-click-predictor-157954281090.asia-south1.run.app`
- ✅ Mobile and desktop both work

## 🆘 **If Still Not Working:**

Try this **nuclear option**:

1. **Delete your Vercel project completely**
2. **Import from GitHub again**
3. **Set environment variable during import**
4. **Deploy fresh**

This guarantees no cached issues.