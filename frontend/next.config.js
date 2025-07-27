/** @type {import('next').NextConfig} */
const nextConfig = {
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
  },
  experimental: {
    // Enable path mapping for better import resolution
    typedRoutes: false
  }
}

module.exports = nextConfig