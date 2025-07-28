#!/bin/bash

# Pre-deployment verification script for Google Cloud Run
# Checks all prerequisites before attempting deployment

set -e

echo "🔍 Pre-Deployment Verification for Google Cloud Run"
echo "================================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
PASSED=0
FAILED=0

check_pass() {
    echo -e "${GREEN}✅ $1${NC}"
    ((PASSED++))
}

check_fail() {
    echo -e "${RED}❌ $1${NC}"
    ((FAILED++))
}

check_warn() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

echo ""
echo "1. Checking System Prerequisites..."

# Check if gcloud is installed
if command -v gcloud &> /dev/null; then
    GCLOUD_VERSION=$(gcloud --version | head -1)
    check_pass "Google Cloud SDK installed: $GCLOUD_VERSION"
else
    check_fail "Google Cloud SDK not installed"
    echo "   Install from: https://cloud.google.com/sdk/docs/install"
fi

# Check if Docker is installed
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version)
    check_pass "Docker installed: $DOCKER_VERSION"
else
    check_fail "Docker not installed"
    echo "   Install from: https://docs.docker.com/get-docker/"
fi

# Check if Python is installed
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    check_pass "Python 3 installed: $PYTHON_VERSION"
else
    check_fail "Python 3 not installed"
fi

echo ""
echo "2. Checking Google Cloud Configuration..."

# Check if authenticated
if gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q "@"; then
    ACTIVE_ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format="value(account)")
    check_pass "Authenticated as: $ACTIVE_ACCOUNT"
else
    check_fail "Not authenticated with Google Cloud"
    echo "   Run: gcloud auth login"
fi

# Check project configuration
if gcloud config get-value project &> /dev/null; then
    PROJECT_ID=$(gcloud config get-value project)
    check_pass "Project configured: $PROJECT_ID"
    
    # Check if project exists
    if gcloud projects describe $PROJECT_ID &> /dev/null; then
        check_pass "Project exists and accessible"
    else
        check_fail "Project not accessible or doesn't exist"
    fi
else
    check_fail "No project configured"
    echo "   Run: gcloud config set project YOUR-PROJECT-ID"
fi

# Check region configuration
if gcloud config get-value run/region &> /dev/null; then
    REGION=$(gcloud config get-value run/region)
    check_pass "Region configured: $REGION"
else
    check_warn "No default region set"
    echo "   Recommended: gcloud config set run/region us-central1"
fi

echo ""
echo "3. Checking Required APIs..."

if command -v gcloud &> /dev/null && gcloud config get-value project &> /dev/null; then
    PROJECT_ID=$(gcloud config get-value project)
    
    # Check Cloud Run API
    if gcloud services list --enabled --filter="name:run.googleapis.com" --format="value(name)" | grep -q "run.googleapis.com"; then
        check_pass "Cloud Run API enabled"
    else
        check_fail "Cloud Run API not enabled"
        echo "   Run: gcloud services enable run.googleapis.com"
    fi
    
    # Check Container Registry API
    if gcloud services list --enabled --filter="name:containerregistry.googleapis.com" --format="value(name)" | grep -q "containerregistry.googleapis.com"; then
        check_pass "Container Registry API enabled"
    else
        check_fail "Container Registry API not enabled"
        echo "   Run: gcloud services enable containerregistry.googleapis.com"
    fi
    
    # Check Cloud Build API
    if gcloud services list --enabled --filter="name:cloudbuild.googleapis.com" --format="value(name)" | grep -q "cloudbuild.googleapis.com"; then
        check_pass "Cloud Build API enabled"
    else
        check_warn "Cloud Build API not enabled (optional but recommended)"
        echo "   Run: gcloud services enable cloudbuild.googleapis.com"
    fi
fi

echo ""
echo "4. Checking Project Files..."

# Check required files exist
if [ -f "cloudrun_app.py" ]; then
    check_pass "cloudrun_app.py found"
else
    check_fail "cloudrun_app.py not found"
fi

if [ -f "Dockerfile.cloudrun" ]; then
    check_pass "Dockerfile.cloudrun found"
else
    check_fail "Dockerfile.cloudrun not found"
fi

if [ -f "requirements.cloudrun.txt" ]; then
    check_pass "requirements.cloudrun.txt found"
else
    check_fail "requirements.cloudrun.txt not found"
fi

if [ -f "deploy-cloudrun.sh" ]; then
    if [ -x "deploy-cloudrun.sh" ]; then
        check_pass "deploy-cloudrun.sh found and executable"
    else
        check_warn "deploy-cloudrun.sh found but not executable"
        echo "   Run: chmod +x deploy-cloudrun.sh"
    fi
else
    check_fail "deploy-cloudrun.sh not found"
fi

echo ""
echo "5. Testing Docker Configuration..."

# Test Docker daemon
if docker info &> /dev/null; then
    check_pass "Docker daemon running"
else
    check_fail "Docker daemon not running"
    echo "   Start Docker Desktop or run: sudo systemctl start docker"
fi

# Test Docker authentication
if docker pull gcr.io/google-containers/busybox &> /dev/null; then
    check_pass "Docker authenticated with Google Cloud Registry"
    # Clean up test image
    docker rmi gcr.io/google-containers/busybox &> /dev/null || true
else
    check_fail "Docker not authenticated with Google Cloud Registry"
    echo "   Run: gcloud auth configure-docker"
fi

echo ""
echo "6. Testing Local Application..."

# Test Python dependencies
if [ -f "requirements.cloudrun.txt" ]; then
    if python3 -c "import fastapi, uvicorn" &> /dev/null; then
        check_pass "Required Python packages available"
    else
        check_warn "Python packages not installed locally"
        echo "   Run: pip install -r requirements.cloudrun.txt"
    fi
fi

# Test application import
if python3 -c "import cloudrun_app" &> /dev/null; then
    check_pass "Application imports successfully"
else
    check_fail "Application import failed"
    echo "   Check cloudrun_app.py for syntax errors"
fi

echo ""
echo "📊 Verification Summary"
echo "====================="
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"

if [ $FAILED -eq 0 ]; then
    echo ""
    echo -e "${GREEN}🎉 All checks passed! Ready for deployment.${NC}"
    echo ""
    echo "Next steps:"
    echo "   1. Run: ./deploy-cloudrun.sh"
    echo "   2. Test deployment: python3 test-cloudrun.py"
    echo "   3. Update Vercel frontend with Cloud Run URL"
    echo ""
    exit 0
else
    echo ""
    echo -e "${RED}❌ $FAILED checks failed. Please fix the issues above before deploying.${NC}"
    echo ""
    echo "Common solutions:"
    echo "   • Install missing tools (gcloud, Docker, Python)"
    echo "   • Authenticate: gcloud auth login"
    echo "   • Enable APIs: gcloud services enable run.googleapis.com"
    echo "   • Configure Docker: gcloud auth configure-docker"
    echo ""
    exit 1
fi