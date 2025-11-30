use actix_cors::Cors;
use actix_web::{get, post, web, App, HttpResponse, HttpServer};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use utoipa::{OpenApi, ToSchema};
use utoipa_swagger_ui::SwaggerUi;

/// Request body for single text embedding
#[derive(Debug, Deserialize, ToSchema)]
pub struct EmbedRequest {
    /// Text to embed
    pub text: String,
}

/// Request body for batch text embedding
#[derive(Debug, Deserialize, ToSchema)]
pub struct BatchEmbedRequest {
    /// List of texts to embed
    pub texts: Vec<String>,
}

/// Response containing embedding vector
#[derive(Debug, Serialize, ToSchema)]
pub struct EmbedResponse {
    /// The embedding vector
    pub embedding: Vec<f32>,
    /// Dimension of the embedding
    pub dimension: usize,
}

/// Response containing multiple embedding vectors
#[derive(Debug, Serialize, ToSchema)]
pub struct BatchEmbedResponse {
    /// List of embedding vectors
    pub embeddings: Vec<Vec<f32>>,
    /// Dimension of each embedding
    pub dimension: usize,
    /// Number of embeddings generated
    pub count: usize,
}

/// Health check response
#[derive(Debug, Serialize, ToSchema)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub model: String,
}

/// Error response
#[derive(Debug, Serialize, ToSchema)]
pub struct ErrorResponse {
    pub error: String,
}

/// Model information response
#[derive(Debug, Serialize, ToSchema)]
pub struct ModelInfoResponse {
    pub model_name: String,
    pub dimension: usize,
    pub max_tokens: usize,
}

pub struct AppState {
    pub embedder: Arc<RwLock<TextEmbedding>>,
    pub model_name: String,
    pub dimension: usize,
}

#[derive(OpenApi)]
#[openapi(
    info(
        title = "FastEmbed Service API",
        description = "Local text embedding service using FastEmbed",
        version = "0.1.0",
        contact(name = "API Support"),
        license(name = "MIT")
    ),
    paths(
        health_check,
        embed_text,
        batch_embed_texts,
        get_model_info,
    ),
    components(schemas(
        EmbedRequest,
        BatchEmbedRequest,
        EmbedResponse,
        BatchEmbedResponse,
        HealthResponse,
        ErrorResponse,
        ModelInfoResponse,
    )),
    tags(
        (name = "Health", description = "Health check endpoints"),
        (name = "Embedding", description = "Text embedding endpoints"),
        (name = "Info", description = "Model information endpoints")
    )
)]
struct ApiDoc;

/// Health check endpoint
#[utoipa::path(
    get,
    path = "/api/v1/health",
    tag = "Health",
    responses(
        (status = 200, description = "Service is healthy", body = HealthResponse)
    )
)]
#[get("/health")]
async fn health_check(app_state: web::Data<AppState>) -> HttpResponse {
    HttpResponse::Ok().json(HealthResponse {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        model: app_state.model_name.clone(),
    })
}

/// Embed a single text
#[utoipa::path(
    post,
    path = "/api/v1/embed",
    tag = "Embedding",
    request_body = EmbedRequest,
    responses(
        (status = 200, description = "Text embedded successfully", body = EmbedResponse),
        (status = 500, description = "Internal server error", body = ErrorResponse)
    )
)]
#[post("/embed")]
async fn embed_text(
    req: web::Json<EmbedRequest>,
    app_state: web::Data<AppState>,
) -> HttpResponse {
    let embedder = app_state.embedder.read().await;
    
    match embedder.embed(vec![req.text.clone()], None) {
        Ok(embeddings) => {
            if let Some(embedding) = embeddings.into_iter().next() {
                let dimension = embedding.len();
                HttpResponse::Ok().json(EmbedResponse {
                    embedding,
                    dimension,
                })
            } else {
                HttpResponse::InternalServerError().json(ErrorResponse {
                    error: "No embedding generated".to_string(),
                })
            }
        }
        Err(e) => HttpResponse::InternalServerError().json(ErrorResponse {
            error: format!("Failed to generate embedding: {}", e),
        }),
    }
}

/// Embed multiple texts in batch
#[utoipa::path(
    post,
    path = "/api/v1/embed/batch",
    tag = "Embedding",
    request_body = BatchEmbedRequest,
    responses(
        (status = 200, description = "Texts embedded successfully", body = BatchEmbedResponse),
        (status = 400, description = "Bad request"),
        (status = 500, description = "Internal server error", body = ErrorResponse)
    )
)]
#[post("/embed/batch")]
async fn batch_embed_texts(
    req: web::Json<BatchEmbedRequest>,
    app_state: web::Data<AppState>,
) -> HttpResponse {
    if req.texts.is_empty() {
        return HttpResponse::BadRequest().json(ErrorResponse {
            error: "No texts provided".to_string(),
        });
    }
    
    let embedder = app_state.embedder.read().await;
    
    match embedder.embed(req.texts.clone(), None) {
        Ok(embeddings) => {
            let count = embeddings.len();
            let dimension = embeddings.first().map(|e| e.len()).unwrap_or(0);
            HttpResponse::Ok().json(BatchEmbedResponse {
                embeddings,
                dimension,
                count,
            })
        }
        Err(e) => HttpResponse::InternalServerError().json(ErrorResponse {
            error: format!("Failed to generate embeddings: {}", e),
        }),
    }
}

/// Get model information
#[utoipa::path(
    get,
    path = "/api/v1/model",
    tag = "Info",
    responses(
        (status = 200, description = "Model information", body = ModelInfoResponse)
    )
)]
#[get("/model")]
async fn get_model_info(app_state: web::Data<AppState>) -> HttpResponse {
    HttpResponse::Ok().json(ModelInfoResponse {
        model_name: app_state.model_name.clone(),
        dimension: app_state.dimension,
        max_tokens: 512, // Default for most models
    })
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::init();
    
    // Get model from environment or use default
    let model_name = std::env::var("EMBEDDING_MODEL")
        .unwrap_or_else(|_| "BAAI/bge-small-en-v1.5".to_string());
    
    log::info!("Loading embedding model: {}", model_name);
    
    // Initialize FastEmbed model
    let model = match model_name.as_str() {
        "BAAI/bge-small-en-v1.5" => EmbeddingModel::BGESmallENV15,
        "BAAI/bge-base-en-v1.5" => EmbeddingModel::BGEBaseENV15,
        "BAAI/bge-large-en-v1.5" => EmbeddingModel::BGELargeENV15,
        "sentence-transformers/all-MiniLM-L6-v2" => EmbeddingModel::AllMiniLML6V2,
        "sentence-transformers/all-MiniLM-L12-v2" => EmbeddingModel::AllMiniLML12V2,
        _ => {
            log::warn!("Unknown model '{}', using default BGESmallENV15", model_name);
            EmbeddingModel::BGESmallENV15
        }
    };
    
    let embedder = TextEmbedding::try_new(InitOptions::new(model.clone()))
        .expect("Failed to initialize embedding model");
    
    // Get embedding dimension by running a test embedding
    let test_embedding = embedder.embed(vec!["test"], None)
        .expect("Failed to generate test embedding");
    let dimension = test_embedding.first().map(|e| e.len()).unwrap_or(384);
    
    log::info!("Model loaded successfully. Dimension: {}", dimension);
    
    let app_state = AppState {
        embedder: Arc::new(RwLock::new(embedder)),
        model_name: format!("{:?}", model),
        dimension,
    };
    
    let openapi = ApiDoc::openapi();
    
    let host = std::env::var("HOST").unwrap_or_else(|_| "0.0.0.0".to_string());
    let port = std::env::var("PORT").unwrap_or_else(|_| "8001".to_string());
    let bind_addr = format!("{}:{}", host, port);
    
    log::info!("Starting embedding server at http://{}", bind_addr);
    log::info!("Swagger UI available at http://{}/docs/", bind_addr);
    
    HttpServer::new(move || {
        let cors = Cors::default()
            .allow_any_origin()
            .allow_any_method()
            .allow_any_header()
            .max_age(3600);
            
        App::new()
            .app_data(web::Data::new(app_state.clone()))
            .wrap(cors)
            .service(
                web::scope("/api/v1")
                    .service(health_check)
                    .service(embed_text)
                    .service(batch_embed_texts)
                    .service(get_model_info)
            )
            .service(
                SwaggerUi::new("/docs/{_:.*}")
                    .url("/api-docs/openapi.json", openapi.clone())
            )
    })
    .bind(&bind_addr)?
    .run()
    .await
}

impl Clone for AppState {
    fn clone(&self) -> Self {
        Self {
            embedder: Arc::clone(&self.embedder),
            model_name: self.model_name.clone(),
            dimension: self.dimension,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embed_request_deserialization() {
        let json = r#"{"text": "Hello world"}"#;
        let request: EmbedRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.text, "Hello world");
    }

    #[test]
    fn test_batch_embed_request_deserialization() {
        let json = r#"{"texts": ["Hello", "World"]}"#;
        let request: BatchEmbedRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.texts.len(), 2);
        assert_eq!(request.texts[0], "Hello");
        assert_eq!(request.texts[1], "World");
    }

    #[test]
    fn test_embed_response_serialization() {
        let response = EmbedResponse {
            embedding: vec![0.1, 0.2, 0.3],
            dimension: 3,
        };
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("embedding"));
        assert!(json.contains("dimension"));
    }

    #[test]
    fn test_batch_embed_response_serialization() {
        let response = BatchEmbedResponse {
            embeddings: vec![vec![0.1, 0.2], vec![0.3, 0.4]],
            dimension: 2,
            count: 2,
        };
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("embeddings"));
        assert!(json.contains("count"));
    }

    #[test]
    fn test_health_response_serialization() {
        let response = HealthResponse {
            status: "healthy".to_string(),
            version: "0.1.0".to_string(),
            model: "BGESmallENV15".to_string(),
        };
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("healthy"));
        assert!(json.contains("BGESmallENV15"));
    }

    #[test]
    fn test_model_info_response_serialization() {
        let response = ModelInfoResponse {
            model_name: "BGESmallENV15".to_string(),
            dimension: 384,
            max_tokens: 512,
        };
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("384"));
        assert!(json.contains("512"));
    }
}
