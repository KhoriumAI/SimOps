/**
 * Centralized configuration for the web frontend.
 * This handles environment-specific API endpoints.
 */

const ALB_DNS = 'webdev-alb-1882895883.us-west-1.elb.amazonaws.com';
const PUBLIC_IP = '54.183.252.115';

export const API_BASE = import.meta.env.VITE_API_URL ||
    ((window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
        ? '/api'
        : `http://${ALB_DNS}/api`);

export const WS_URL = import.meta.env.VITE_WS_URL ||
    ((window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
        ? 'http://localhost:5000'
        : `http://${ALB_DNS}`);

console.log('[Config] API_BASE:', API_BASE);
console.log('[Config] WS_URL:', WS_URL);
console.log('[Config] Hostname:', window.location.hostname);

export default {
    API_BASE,
    WS_URL,
    ALB_DNS,
    PUBLIC_IP
};
