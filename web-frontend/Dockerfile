# SimOps Web Frontend Dockerfile
# ================================
# Multi-stage build: Node.js build -> Nginx serve

# =================================================================
# STAGE 1: Build the React/Vite application
# =================================================================
FROM node:20-alpine AS builder

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci

# Copy source code
COPY . .

# Build the production bundle
# API URL is configured at runtime via nginx config
RUN npm run build

# =================================================================
# STAGE 2: Serve with Nginx
# =================================================================
FROM nginx:alpine AS production

LABEL maintainer="SimOps Team"
LABEL description="SimOps Web Frontend"
LABEL version="1.0"

# Remove default nginx config
RUN rm /etc/nginx/conf.d/default.conf

# Copy custom nginx config
COPY nginx.conf /etc/nginx/conf.d/simops.conf

# Copy built assets from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Expose port 80
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
