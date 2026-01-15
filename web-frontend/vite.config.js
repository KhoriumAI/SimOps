import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

/**
 * Vite Configuration
 * 
 * DEVELOPMENT:
 *   - API requests to /api/* are proxied to localhost:5000
 *   - No environment variables needed
 *   - Run: npm run dev
 * 
 * PRODUCTION BUILD:
 *   - Set VITE_API_URL environment variable before building
 *   - Example: VITE_API_URL=https://api.your-domain.com npm run build
 *   - Or create .env.production file with: VITE_API_URL=https://api.your-domain.com
 */
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    // Development proxy - forwards /api requests to Flask backend
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:5000',
        changeOrigin: true,
        secure: false,
        configure: (proxy, options) => {
          proxy.on('proxyReq', (proxyReq, req, res) => {
            console.log('Proxying:', req.method, req.url)
          })
        }
      }
    }
  },
  // Production build settings
  build: {
    outDir: 'dist',
    sourcemap: false,
    // Chunk splitting for better caching
    rollupOptions: {
      output: {
        manualChunks: {
          'react-vendor': ['react', 'react-dom'],
          'three-vendor': ['three', '@react-three/fiber', '@react-three/drei']
        }
      }
    }
  }
})
