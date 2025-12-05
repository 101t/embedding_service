use actix_cors::Cors;
use actix_web::{get, post, web, App, HttpResponse, HttpServer};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};
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

/// Available model entry
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct AvailableModel {
    /// Model identifier to use with EMBEDDING_MODEL env var
    pub id: &'static str,
    /// Embedding dimension
    pub dimension: usize,
    /// Model description
    pub description: &'static str,
}

/// Response containing list of available models
#[derive(Debug, Serialize, ToSchema)]
pub struct ModelsListResponse {
    /// List of available models
    pub models: Vec<AvailableModel>,
    /// Total number of models
    pub count: usize,
}

/// Model registry entry with all information
struct ModelEntry {
    id: &'static str,
    model: EmbeddingModel,
    dimension: usize,
    description: &'static str,
}

/// Single source of truth for all model configurations
const MODEL_REGISTRY: &[ModelEntry] = &[
    // BGE English models
    ModelEntry { id: "BAAI/bge-small-en-v1.5", model: EmbeddingModel::BGESmallENV15, dimension: 384, description: "Small, fast English model (default)" },
    ModelEntry { id: "BAAI/bge-base-en-v1.5", model: EmbeddingModel::BGEBaseENV15, dimension: 768, description: "Base English model, balanced performance" },
    ModelEntry { id: "BAAI/bge-large-en-v1.5", model: EmbeddingModel::BGELargeENV15, dimension: 1024, description: "Large English model, best quality" },
    // BGE Chinese models
    ModelEntry { id: "Xenova/bge-small-zh-v1.5", model: EmbeddingModel::BGESmallZHV15, dimension: 512, description: "Small Chinese model" },
    ModelEntry { id: "Xenova/bge-large-zh-v1.5", model: EmbeddingModel::BGELargeZHV15, dimension: 1024, description: "Large Chinese model" },
    // MiniLM models
    ModelEntry { id: "sentence-transformers/all-MiniLM-L6-v2", model: EmbeddingModel::AllMiniLML6V2, dimension: 384, description: "Fast, lightweight English model" },
    ModelEntry { id: "sentence-transformers/all-MiniLM-L12-v2", model: EmbeddingModel::AllMiniLML12V2, dimension: 384, description: "Slightly larger MiniLM model" },
    // MPNet models
    ModelEntry { id: "sentence-transformers/all-mpnet-base-v2", model: EmbeddingModel::AllMpnetBaseV2, dimension: 768, description: "MPNet base model" },
    // Paraphrase multilingual models
    ModelEntry { id: "Xenova/paraphrase-multilingual-MiniLM-L12-v2", model: EmbeddingModel::ParaphraseMLMiniLML12V2, dimension: 384, description: "Multilingual paraphrase model" },
    ModelEntry { id: "Xenova/paraphrase-multilingual-mpnet-base-v2", model: EmbeddingModel::ParaphraseMLMpnetBaseV2, dimension: 768, description: "Multilingual MPNet model" },
    // Nomic models
    ModelEntry { id: "nomic-ai/nomic-embed-text-v1", model: EmbeddingModel::NomicEmbedTextV1, dimension: 768, description: "Nomic v1, 8192 context length" },
    ModelEntry { id: "nomic-ai/nomic-embed-text-v1.5", model: EmbeddingModel::NomicEmbedTextV15, dimension: 768, description: "Nomic v1.5, 8192 context length" },
    // Multilingual E5 models
    ModelEntry { id: "intfloat/multilingual-e5-small", model: EmbeddingModel::MultilingualE5Small, dimension: 384, description: "E5 small multilingual" },
    ModelEntry { id: "intfloat/multilingual-e5-base", model: EmbeddingModel::MultilingualE5Base, dimension: 768, description: "E5 base multilingual" },
    ModelEntry { id: "intfloat/multilingual-e5-large", model: EmbeddingModel::MultilingualE5Large, dimension: 1024, description: "E5 large multilingual" },
    // MxBai models
    ModelEntry { id: "mixedbread-ai/mxbai-embed-large-v1", model: EmbeddingModel::MxbaiEmbedLargeV1, dimension: 1024, description: "MxBai large English model" },
    // GTE models (Alibaba)
    ModelEntry { id: "Alibaba-NLP/gte-base-en-v1.5", model: EmbeddingModel::GTEBaseENV15, dimension: 768, description: "GTE base English model" },
    ModelEntry { id: "Alibaba-NLP/gte-large-en-v1.5", model: EmbeddingModel::GTELargeENV15, dimension: 1024, description: "GTE large English model" },
    // Other models
    ModelEntry { id: "lightonai/modernbert-embed-large", model: EmbeddingModel::ModernBertEmbedLarge, dimension: 1024, description: "ModernBERT large model" },
    ModelEntry { id: "Qdrant/clip-ViT-B-32-text", model: EmbeddingModel::ClipVitB32, dimension: 512, description: "CLIP text encoder ViT-B/32" },
    ModelEntry { id: "jinaai/jina-embeddings-v2-base-code", model: EmbeddingModel::JinaEmbeddingsV2BaseCode, dimension: 768, description: "Jina code embeddings" },
    ModelEntry { id: "onnx-community/embeddinggemma-300m-ONNX", model: EmbeddingModel::EmbeddingGemma300M, dimension: 768, description: "Google EmbeddingGemma 300M" },
];

/// Lazy-initialized HashMap for O(1) model lookup
static MODEL_MAP: Lazy<HashMap<&'static str, &'static ModelEntry>> = Lazy::new(|| {
    MODEL_REGISTRY.iter().map(|e| (e.id, e)).collect()
});

/// Get EmbeddingModel by ID, returns default if not found
fn get_model_by_id(id: &str) -> EmbeddingModel {
    MODEL_MAP
        .get(id)
        .map(|e| e.model.clone())
        .unwrap_or_else(|| {
            log::warn!("Unknown model '{}', using default BGESmallENV15", id);
            EmbeddingModel::BGESmallENV15
        })
}

/// Get available models for API response
fn get_available_models() -> Vec<AvailableModel> {
    MODEL_REGISTRY
        .iter()
        .map(|e| AvailableModel {
            id: e.id,
            dimension: e.dimension,
            description: e.description,
        })
        .collect()
}

#[derive(Clone)]
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
        list_available_models,
    ),
    components(schemas(
        EmbedRequest,
        BatchEmbedRequest,
        EmbedResponse,
        BatchEmbedResponse,
        HealthResponse,
        ErrorResponse,
        ModelInfoResponse,
        AvailableModel,
        ModelsListResponse,
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
    let mut embedder = app_state.embedder.write().await;
    
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
    
    let mut embedder = app_state.embedder.write().await;
    
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

/// List all available embedding models
#[utoipa::path(
    get,
    path = "/api/v1/models",
    tag = "Info",
    responses(
        (status = 200, description = "List of available models", body = ModelsListResponse)
    )
)]
#[get("/models")]
async fn list_available_models() -> HttpResponse {
    let models = get_available_models();
    let count = models.len();
    HttpResponse::Ok().json(ModelsListResponse { models, count })
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::init();
    
    // Get model from environment or use default
    let model_name = std::env::var("EMBEDDING_MODEL")
        .unwrap_or_else(|_| "BAAI/bge-small-en-v1.5".to_string());
    
    log::info!("Loading embedding model: {}", model_name);
    
    // Initialize FastEmbed model using registry lookup
    let model = get_model_by_id(&model_name);
    
    let mut embedder = TextEmbedding::try_new(InitOptions::new(model.clone()))
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
                    .service(list_available_models)
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

    #[test]
    fn test_model_registry() {
        assert!(MODEL_REGISTRY.len() >= 22);
        assert!(MODEL_MAP.contains_key("BAAI/bge-small-en-v1.5"));
    }

    #[test]
    fn test_get_model_by_id() {
        let model = get_model_by_id("BAAI/bge-small-en-v1.5");
        assert!(matches!(model, EmbeddingModel::BGESmallENV15));
        
        // Unknown model should return default
        let default = get_model_by_id("unknown/model");
        assert!(matches!(default, EmbeddingModel::BGESmallENV15));
    }

    #[test]
    fn test_models_list_response_serialization() {
        let models = get_available_models();
        let response = ModelsListResponse {
            count: models.len(),
            models,
        };
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("models"));
        assert!(json.contains("count"));
        assert!(json.contains("BAAI/bge-small-en-v1.5"));
    }
}
