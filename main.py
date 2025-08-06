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
from document_processing.processors import DocumentProcessorFactory


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
    
    logging.info("FastAPI application created and configured successfully")
    logging.info(f"Configuration: max_file_size={config.max_file_size}, security_scanning={config.enable_security_scanning}")
    
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