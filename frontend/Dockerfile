# React Frontend
FROM node:18-alpine as build

WORKDIR /app

# Copy package files first for better caching
COPY frontend/package*.json ./
RUN npm ci --only=production

# Copy source code and build
COPY frontend/ .
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy custom nginx config
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf

# Copy built app
COPY --from=build /app/build /usr/share/nginx/html

# Create non-root user
RUN addgroup -g 1000 appuser && adduser -u 1000 -G appuser -s /bin/sh -D appuser
RUN chown -R appuser:appuser /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]