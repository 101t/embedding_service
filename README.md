# FastEmbed Service

[![Rust](https://img.shields.io/badge/rust-1.75+-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

A fast, local text embedding service built with Rust using [FastEmbed](https://github.com/Anush008/fastembed-rs) and ONNX Runtime. Perfect for semantic search, recommendation systems, and RAG applications.

## ✨ Features

- **Fast & Local** - No external API calls, runs entirely on your machine
- **ONNX Runtime** - Optimized inference with ONNX for best performance
- **Multiple Models** - Support for BGE, MiniLM, and other popular embedding models
- **REST API** - Simple HTTP API with OpenAPI/Swagger documentation
- **Batch Processing** - Efficient batch embedding for multiple texts
- **Docker Ready** - Easy deployment with Docker and Docker Compose
- **Zero Config** - Works out of the box with sensible defaults

## 🚀 Quick Start

### Using Cargo

```bash
# Clone the repository
git clone https://github.com/101t/embedding_service.git
cd embedding_service

# Run with default settings
cargo run --release

# Or with custom model
EMBEDDING_MODEL="BAAI/bge-base-en-v1.5" cargo run --release
```

### Using Docker

```bash
# Build and run with Docker Compose
docker compose up -d

# Or build manually
docker build -t embedding_service .
docker run -p 8001:8001 embedding_service
```

### Using Make

```bash
make run          # Run locally
make docker-run   # Run with Docker
make help         # See all commands
```

## 📖 API Usage

### Embed Single Text

```bash
curl -X POST http://localhost:8001/api/v1/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!"}'
```

```json
{
  "embedding": [0.123, -0.456, ...],
  "dimension": 384
}
```

### Batch Embed

```bash
curl -X POST http://localhost:8001/api/v1/embed/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["First text", "Second text"]}'
```

```json
{
  "embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]],
  "dimension": 384,
  "count": 2
}
```

### Health Check

```bash
curl http://localhost:8001/api/v1/health
```

### Model Info

```bash
curl http://localhost:8001/api/v1/model
```

## 📚 API Documentation

Interactive Swagger UI available at: **http://localhost:8001/docs/**

## 🤖 Available Models

| Model | Dimension | Description |
|-------|-----------|-------------|
| `BAAI/bge-small-en-v1.5` | 384 | Small, fast model **(default)** |
| `BAAI/bge-base-en-v1.5` | 768 | Balanced performance |
| `BAAI/bge-large-en-v1.5` | 1024 | Best quality |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | Popular lightweight model |
| `sentence-transformers/all-MiniLM-L12-v2` | 384 | Slightly larger MiniLM |

## ⚙️ Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Host to bind to |
| `PORT` | `8001` | Port to listen on |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model to use |
| `RUST_LOG` | `info` | Log level (debug, info, warn, error) |

Copy `.env.example` to `.env` and customize as needed:

```bash
cp .env.example .env
```

## 🛠️ Development

```bash
# Install Rust (if not installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Build
make build

# Run tests
make test

# Run linter
make lint

# Format code
make fmt

# Run all checks
make check
```

## 🐳 Docker

### Docker Compose (Recommended)

```bash
# Start service
docker compose up -d

# View logs
docker compose logs -f

# Stop service
docker compose down
```

### Environment Variables with Docker

```bash
# Using docker-compose with custom model
EMBEDDING_MODEL="BAAI/bge-base-en-v1.5" docker compose up -d
```

## 📊 Performance

The service uses ONNX Runtime for optimized inference. Performance varies by model:

| Model | Latency (single) | Throughput (batch of 32) |
|-------|------------------|-------------------------|
| BGE-small | ~5ms | ~50ms |
| BGE-base | ~10ms | ~100ms |
| BGE-large | ~20ms | ~200ms |

*Benchmarks on Intel i7-12700K, actual performance may vary*

## 🔗 Integration Example

### Python

```python
import requests

def get_embedding(text: str) -> list[float]:
    response = requests.post(
        "http://localhost:8001/api/v1/embed",
        json={"text": text}
    )
    return response.json()["embedding"]

embedding = get_embedding("Hello, world!")
```

### JavaScript/TypeScript

```typescript
async function getEmbedding(text: string): Promise<number[]> {
  const response = await fetch("http://localhost:8001/api/v1/embed", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  const data = await response.json();
  return data.embedding;
}
```

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Run checks (`make check`)
4. Commit your changes (`git commit -m 'feat: add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [FastEmbed-rs](https://github.com/Anush008/fastembed-rs) - Rust bindings for FastEmbed
- [ONNX Runtime](https://onnxruntime.ai/) - High-performance inference engine
- [Actix-web](https://actix.rs/) - Powerful web framework for Rust
