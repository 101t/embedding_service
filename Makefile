.PHONY: help build run test clean docker-build docker-run docker-stop lint fmt check release install dev

# Default target
help:
	@echo "FastEmbed Service - Available Commands"
	@echo "======================================="
	@echo ""
	@echo "Development:"
	@echo "  make dev          - Run in development mode with hot reload info"
	@echo "  make build        - Build debug version"
	@echo "  make run          - Run the service locally"
	@echo "  make test         - Run tests"
	@echo "  make lint         - Run clippy linter"
	@echo "  make fmt          - Format code"
	@echo "  make check        - Run all checks (fmt, lint, test)"
	@echo ""
	@echo "Production:"
	@echo "  make release      - Build optimized release version"
	@echo "  make install      - Install to system (requires sudo)"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-run   - Run with Docker Compose"
	@echo "  make docker-stop  - Stop Docker containers"
	@echo "  make docker-logs  - View Docker logs"
	@echo "  make docker-clean - Remove Docker images and volumes"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean        - Remove build artifacts"
	@echo "  make clean-cache  - Remove model cache"

# ============================================
# Development
# ============================================

dev:
	RUST_LOG=debug cargo run

build:
	cargo build

run:
	cargo run --release

test:
	cargo test

lint:
	cargo clippy -- -D warnings

fmt:
	cargo fmt

fmt-check:
	cargo fmt -- --check

check: fmt-check lint test
	@echo "All checks passed!"

# ============================================
# Production
# ============================================

release:
	cargo build --release
	@echo "Binary available at: target/release/embedding_service"

install: release
	sudo cp target/release/embedding_service /usr/local/bin/
	@echo "Installed to /usr/local/bin/embedding_service"

# ============================================
# Docker
# ============================================

docker-build:
	docker build -t embedding_service:latest .

docker-run:
	docker compose up -d

docker-stop:
	docker compose down

docker-logs:
	docker compose logs -f

docker-clean:
	docker compose down -v --rmi local

docker-shell:
	docker compose exec embedding_service /bin/sh

# ============================================
# Maintenance
# ============================================

clean:
	cargo clean

clean-cache:
	rm -rf .fastembed_cache

clean-all: clean clean-cache

# ============================================
# CI/CD Helpers
# ============================================

ci: check
	@echo "CI checks completed successfully"

# Version info
version:
	@cargo pkgid | cut -d# -f2
