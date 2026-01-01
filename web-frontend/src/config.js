/**
 * Centralized configuration for the web frontend.
 * This handles environment-specific API endpoints.
 */

const ALB_DNS = 'webdev-alb-1882895883.us-west-1.elb.amazonaws.com';

export const API_BASE = import.meta.env.VITE_API_URL ||
    ((window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
        ? '/api'
        : (window.location.hostname.includes('s3-website')
            ? `http://${ALB_DNS}/api`
            : `http://${window.location.hostname}:5000/api`));

export const WS_URL = import.meta.env.VITE_WS_URL ||
    ((window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
        ? window.location.origin
        : (window.location.hostname.includes('s3-website')
            ? `http://${ALB_DNS}`
            : `http://${window.location.hostname}:5000`));

export default {
    API_BASE,
    WS_URL,
    ALB_DNS
};
