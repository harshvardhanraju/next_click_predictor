// Paste this code into your mobile browser console to check environment variable
// Go to your Vercel URL, open developer tools, paste this in Console tab:

console.log('=== VERCEL ENVIRONMENT CHECK ===');
console.log('Current URL:', window.location.href);
console.log('API URL should be:', 'https://next-click-predictor-157954281090.asia-south1.run.app');

// Check if the API URL is embedded in the page
try {
    // Look for the API URL in the page source
    const pageSource = document.documentElement.outerHTML;
    const hasCloudRunURL = pageSource.includes('next-click-predictor-157954281090.asia-south1.run.app');
    
    console.log('Cloud Run URL found in page source:', hasCloudRunURL);
    
    if (!hasCloudRunURL) {
        console.error('❌ ENVIRONMENT VARIABLE NOT SET');
        console.log('The API URL is not embedded in your Vercel build');
        console.log('You need to:');
        console.log('1. Set NEXT_PUBLIC_API_URL in Vercel dashboard');
        console.log('2. Redeploy WITHOUT build cache');
    } else {
        console.log('✅ Environment variable appears to be set correctly');
    }
    
    // Try to find where the API calls are made
    const scripts = document.querySelectorAll('script');
    let foundAPICall = false;
    
    scripts.forEach(script => {
        if (script.innerHTML && script.innerHTML.includes('predict')) {
            console.log('Found API call code in script tag');
            foundAPICall = true;
        }
    });
    
    if (!foundAPICall) {
        console.log('No API call code found in current page');
    }
    
} catch (error) {
    console.error('Error checking page source:', error);
}

// Instructions for next steps
console.log('\n=== NEXT STEPS ===');
console.log('1. If you see "❌ ENVIRONMENT VARIABLE NOT SET" above:');
console.log('   - Go to Vercel dashboard');
console.log('   - Add NEXT_PUBLIC_API_URL environment variable');
console.log('   - Redeploy without build cache');
console.log('2. Try uploading an image and check Network tab');
console.log('3. Look for the API request URL in Network tab');