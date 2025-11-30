# syntax=docker/dockerfile:1

# ============================================
# Stage 1: Build the application
# ============================================
FROM rust:1.83-bookworm AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy manifests first for better layer caching
COPY Cargo.toml Cargo.lock ./

# Create a dummy main.rs to build dependencies
RUN mkdir -p src && \
    echo "fn main() {}" > src/main.rs && \
    cargo build --release && \
    rm -rf src

# Copy actual source code
COPY src ./src

# Build the actual application
RUN touch src/main.rs && cargo build --release

# ============================================
# Stage 2: Runtime image
# ============================================
FROM debian:bookworm-slim AS runtime

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -r -s /bin/false -u 1001 appuser

# Copy binary from builder
COPY --from=builder /app/target/release/embedding_service /usr/local/bin/embedding_service

# Create cache directory for models
RUN mkdir -p /app/.fastembed_cache && chown -R appuser:appuser /app

# Set environment variables
ENV HOST=0.0.0.0
ENV PORT=8001
ENV RUST_LOG=info
ENV FASTEMBED_CACHE_PATH=/app/.fastembed_cache

# Switch to non-root user
USER appuser

# Expose the service port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/api/v1/health || exit 1

# Run the service
CMD ["embedding_service"]
