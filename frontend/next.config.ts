import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Enable standalone output for optimized Docker deployment
  output: "standalone",

  // Experimental features
  // Type assertion needed: proxyClientMaxBodySize is valid in Next.js 15 but types lag behind
  experimental: {
    // Increase proxy body size limit for file uploads (default is 10MB)
    // This allows larger files to be uploaded through the /api/* rewrite proxy to FastAPI
    proxyClientMaxBodySize: '100mb',
  } as NextConfig['experimental'],

  // API Rewrites: Proxy /api/* requests to FastAPI backend
  // This simplifies reverse proxy configuration - users only need to proxy to port 8502
  // Next.js handles internal routing to the API backend on port 5055
  async rewrites() {
    // INTERNAL_API_URL: Where Next.js server-side should proxy API requests
    // Default: http://localhost:5055 (single-container deployment)
    // Override for multi-container: INTERNAL_API_URL=http://api-service:5055
    const internalApiUrl = process.env.INTERNAL_API_URL || 'http://localhost:5055'

    console.log(`[Next.js Rewrites] Proxying /api/* to ${internalApiUrl}/api/*`)

    // INTERNAL_API_URL points at /on/<adapter root>; strip trailing /on for routes
    // that live above the adapter on the main FastAPI app (admin panel).
    const apiRoot = internalApiUrl.replace(/\/on$/, '')

    return [
      {
        source: '/api/:path*',
        destination: `${internalApiUrl}/api/:path*`,
      },
      // /admin (reasoning inspector) is mounted on the MAIN FastAPI app, not
      // under /on. Proxy so users can hit it from the Next.js origin.
      {
        source: '/admin',
        destination: `${apiRoot}/admin`,
      },
      {
        source: '/admin/:path*',
        destination: `${apiRoot}/admin/:path*`,
      },
    ]
  },
};

export default nextConfig;
