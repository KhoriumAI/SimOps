/**
 * Centralized configuration for the web frontend.
 * This handles environment-specific API endpoints.
 */

const ALB_DNS = 'webdev-alb-1882895883.us-west-1.elb.amazonaws.com';
const PUBLIC_IP = '54.183.252.115';

// For production, we want to ensure we're hitting the correct domain even if accessed via S3 URL
const getApiBase = () => {
    if (import.meta.env.VITE_API_URL) return import.meta.env.VITE_API_URL;

    const hostname = window.location.hostname;
    // If we're on the S3 website URL, force the API to the custom domain
    if (hostname.includes('s3-website') || hostname.includes('amazonaws.com')) {
        return 'https://development.khorium.ai/api';
    }

    // Default to relative path which works best through CloudFront
    return '/api';
};

const getWsUrl = () => {
    if (import.meta.env.VITE_WS_URL) return import.meta.env.VITE_WS_URL;

    const hostname = window.location.hostname;
    if (hostname === 'localhost' || hostname === '127.0.0.1') {
        return 'http://localhost:5000';
    }
    if (hostname.includes('s3-website') || hostname.includes('amazonaws.com')) {
        return 'https://development.khorium.ai';
    }
    return window.location.origin;
};

export const API_BASE = getApiBase();
export const WS_URL = getWsUrl();

console.log('[Config] API_BASE:', API_BASE);
console.log('[Config] WS_URL:', WS_URL);
console.log('[Config] Hostname:', window.location.hostname);

export default {
    API_BASE,
    WS_URL,
    ALB_DNS,
    PUBLIC_IP
};
