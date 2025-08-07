"""
AI Foundation - Real-time chat application with document processing.

This is the main entry point for the FastAPI application.
The application has been refactored into modular components for better maintainability.
"""

import logging
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

# Import our modular components
from core.config import (
    AppConfig, 
    create_fastapi_app, 
    setup_middleware, 
    setup_static_files_and_templates,
    create_file_validator,
    validate_environment,
    setup_logging
)
from api.endpoints import (
    read_root, 
    get_sessions, 
    get_session_history, 
    delete_session, 
    stream_chat,
    get_available_models,
    set_templates,
    set_file_validator,
    set_document_processor_factory,
    initialize_enhanced_session_management,
    get_session_documents,
    get_document_details,
    share_document,
    remove_document,
    get_cleanup_statistics,
    run_manual_cleanup,
    chat_sessions
)
from api.vector_endpoints import (
    semantic_search,
    find_similar_documents,
    generate_embeddings,
    generate_embeddings_stream,
    get_embedding_status,
    regenerate_embeddings,
    get_vector_database_stats,
    get_vector_database_health,
    get_search_analytics,
    clear_search_cache,
    initialize_vector_services
)
from document_processing.processors import DocumentProcessorFactory
from core.embedding_service import EmbeddingService
from core.vector_store import VectorStore
from core.semantic_search import SemanticSearchService


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    # Validate environment before starting
    validate_environment()
    
    # Initialize configuration
    config = AppConfig()
    
    # Setup logging
    setup_logging()
    
    # Create FastAPI app
    app = create_fastapi_app()
    
    # Setup middleware
    setup_middleware(app, config)
    
    # Setup static files and templates
    templates = setup_static_files_and_templates(app)
    
    # Create file validator
    file_validator = create_file_validator(config)
    
    # Create document processor factory with config
    document_processor_factory = DocumentProcessorFactory(app_config=config)
    
    # Initialize enhanced session management
    document_persistence_config = config.get_document_persistence_config()
    initialize_enhanced_session_management(document_persistence_config)
    
    # Initialize vector services if enabled
    embedding_service = None
    vector_store = None
    semantic_search_service = None
    
    if config.enable_vector_search:
        # Initialize vector services
        vector_config = config.get_vector_search_config()
        vector_db_config = config.get_vector_database_config()
        search_config = config.get_semantic_search_config()
        
        # Create embedding service
        embedding_service = EmbeddingService(
            model_name=vector_config["embedding_model_name"],
            embedding_dimension=vector_config["embedding_dimension"],
            cache_size=vector_config["embedding_cache_size"],
            batch_size=vector_config["embedding_batch_size"],
            enable_gpu=vector_config["enable_gpu_acceleration"]
        )
        
        # Create vector store
        vector_store = VectorStore(
            persistence_path=vector_db_config["vector_db_path"],
            collection_prefix=vector_db_config["collection_prefix"],
            enable_backup=vector_db_config["enable_vector_db_backup"],
            backup_interval_hours=vector_db_config["backup_interval_hours"],
            max_backup_files=vector_db_config["max_backup_files"],
            expected_dimension=vector_db_config["expected_dimension"]
        )
        
        # Create semantic search service
        semantic_search_service = SemanticSearchService(
            embedding_service=embedding_service,
            vector_store=vector_store,
            cache_size=search_config["search_cache_size"],
            cache_ttl_minutes=search_config["search_cache_ttl_minutes"],
            enable_query_expansion=search_config["enable_query_expansion"],
            enable_deduplication=search_config["enable_result_deduplication"],
            similarity_threshold=search_config["similarity_threshold_for_dedup"]
        )
        
        # Schedule async initialization of vector services
        async def init_vector_services():
            await embedding_service.initialize()
            await vector_store.initialize()
            await semantic_search_service.initialize()
            
            # Initialize vector services for API endpoints
            from api.endpoints import session_document_manager
            if session_document_manager:
                initialize_vector_services(
                    embedding_service,
                    vector_store,
                    semantic_search_service,
                    session_document_manager
                )
            
            logging.info("Vector services async initialization completed")
        
        # Add startup event to initialize services
        @app.on_event("startup")
        async def startup_event():
            await init_vector_services()
        
        logging.info("Vector search services initialized successfully")
    else:
        logging.info("Vector search disabled in configuration")
    
    # Inject dependencies into endpoints
    set_templates(templates)
    set_file_validator(file_validator)
    set_document_processor_factory(document_processor_factory)
    
    # Register existing routes
    app.get("/", response_class=HTMLResponse)(read_root)
    app.get("/sessions")(get_sessions)
    app.get("/sessions/{session_id}")(get_session_history)
    app.delete("/sessions/{session_id}")(delete_session)
    app.post("/stream")(stream_chat)
    app.get("/models")(get_available_models)
    
    # Register new document management routes
    app.get("/sessions/{session_id}/documents")(get_session_documents)
    app.get("/documents/{document_id}")(get_document_details)
    app.post("/documents/{document_id}/share")(share_document)
    app.delete("/documents/{document_id}")(remove_document)
    
    # Register administrative routes
    app.get("/admin/cleanup/statistics")(get_cleanup_statistics)
    app.post("/admin/cleanup/run")(run_manual_cleanup)
    
    # Register vector search routes if enabled
    if config.enable_vector_search:
        app.post("/search/semantic")(semantic_search)
        app.post("/search/similar")(find_similar_documents)
        app.post("/embeddings/generate")(generate_embeddings)
        app.get("/embeddings/{document_id}/stream")(generate_embeddings_stream)
        app.get("/embeddings/{document_id}/status")(get_embedding_status)
        app.post("/embeddings/{document_id}/regenerate")(regenerate_embeddings)
        app.get("/admin/vector/stats")(get_vector_database_stats)
        app.get("/admin/vector/health")(get_vector_database_health)
        app.get("/admin/search/analytics")(get_search_analytics)
        app.post("/admin/search/cache/clear")(clear_search_cache)
        
        logging.info("Vector search endpoints registered successfully")
    
    logging.info("FastAPI application created and configured successfully")
    logging.info(f"Configuration: max_file_size={config.max_file_size}, security_scanning={config.enable_security_scanning}, vector_search={config.enable_vector_search}")
    
    return app


# Create the application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )