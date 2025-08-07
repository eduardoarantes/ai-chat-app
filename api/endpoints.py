"""FastAPI route handlers and API endpoints."""

import base64
import hashlib
import json
import logging
import os
import subprocess
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from langchain.schema import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

from core.access_controller import DocumentAccessController
from core.cleanup_manager import SessionCleanupManager
from core.document_store import DocumentStore
from core.session_document_manager import SessionDocumentManager
from document_processing.processors import DocumentProcessorFactory
from error_handling.handlers import ErrorHandler, SSEErrorStreamer
from models.errors import (
    ContentValidationError,
    ErrorSeverity,
    FileSizeError,
    LLMError,
    MimeTypeError,
    NetworkError,
    SecurityError,
)
from models.session import DocumentAccessLevel, DocumentMetadata, DocumentProcessingStatus, SessionDocument
from models.validation import ValidationStatus

# Enhanced session management system
document_store = DocumentStore()
session_document_manager = None
access_controller = None
cleanup_manager = None

# Legacy compatibility: maintain the old chat_sessions dict structure for backward compatibility
chat_sessions: Dict[str, Dict] = {}

# Document processor factory will be initialized with config in main.py
document_processor_factory = None

# Templates will be initialized in main.py
templates: Optional[Jinja2Templates] = None

# File validator will be injected from main.py
file_validator = None


def initialize_enhanced_session_management(config: dict) -> None:
    """
    Initialize enhanced session management system with configuration.

    Args:
        config: Application configuration dictionary
    """
    global session_document_manager, access_controller, cleanup_manager

    # Initialize components
    session_document_manager = SessionDocumentManager(document_store, config)
    access_controller = DocumentAccessController(config)
    cleanup_manager = SessionCleanupManager(
        session_manager=session_document_manager,
        document_store=document_store,
        access_controller=access_controller,
        config=config,
    )

    logging.info("Enhanced session management system initialized")


def _ensure_enhanced_session_exists(session_id: str, title: str = "New Chat") -> None:
    """
    Ensure both legacy and enhanced session structures exist for backward compatibility.

    Args:
        session_id: Session ID
        title: Session title
    """
    # Create enhanced session if it doesn't exist
    if session_document_manager:
        session_document_manager.get_or_create_session(session_id, title)

    # Maintain legacy session structure for backward compatibility
    if session_id not in chat_sessions:
        chat_sessions[session_id] = {"id": session_id, "title": title, "messages": []}


def _sync_session_data_to_legacy(session_id: str) -> None:
    """
    Sync enhanced session data back to legacy format for backward compatibility.

    Args:
        session_id: Session ID to sync
    """
    if not session_document_manager:
        return

    enhanced_session = session_document_manager.get_session(session_id)
    if enhanced_session and session_id in chat_sessions:
        # Sync basic data - messages are handled separately
        chat_sessions[session_id]["title"] = enhanced_session.title
        chat_sessions[session_id]["messages"] = enhanced_session.messages


def get_session_id_from_request(request: Request) -> Optional[str]:
    """
    Safely extract session ID from request.

    Priority order:
    1. Session storage (if SessionMiddleware is configured)
    2. Query parameters
    3. None (will trigger session creation in endpoints)

    Args:
        request: FastAPI Request object

    Returns:
        Session ID string or None
    """
    try:
        # First try session storage
        if hasattr(request, "session") and request.session:
            session_id = request.session.get("session_id")
            if session_id:
                return session_id
    except Exception as e:
        logging.warning(f"Session access error: {e}")

    # Fallback to query parameters
    session_id = request.query_params.get("session_id")
    if session_id:
        return session_id

    return None


def set_session_id_in_request(request: Request, session_id: str) -> None:
    """
    Safely store session ID in request session.

    Args:
        request: FastAPI Request object
        session_id: Session ID to store
    """
    try:
        if hasattr(request, "session"):
            request.session["session_id"] = session_id
    except Exception as e:
        logging.warning(f"Session storage error: {e}")


def serialize_message(message):
    """Serialize LangChain messages for JSON response."""
    if isinstance(message, HumanMessage):
        if isinstance(message.content, list):
            serialized_content = []
            for item in message.content:
                if isinstance(item, dict) and "type" in item:
                    if item["type"] == "text":
                        serialized_content.append({"type": "text", "text": item["text"]})
                    elif item["type"] == "image_url":
                        # Log image serialization but don't include the full base64 string
                        logging.info("Serializing image for history...")
                        serialized_content.append({"type": "image_url", "image_url": item["image_url"]})
                else:
                    serialized_content.append(item)
            return {"type": "user", "content": serialized_content}
        else:
            return {"type": "user", "content": message.content}
    if isinstance(message, AIMessage):
        return {"type": "ai", "content": message.content}
    logging.warning(f"Attempted to serialize unknown message type: {type(message)}")
    return message


# Supported models configuration (will be dynamically updated)
SUPPORTED_MODELS = {
    # Google Gemini models
    "gemini-1.5-flash": {
        "name": "Gemini 1.5 Flash",
        "description": "Fast and efficient - good for most tasks",
        "provider": "google",
    },
    "gemini-1.5-pro": {"name": "Gemini 1.5 Pro", "description": "Most capable - best for complex tasks", "provider": "google"},
    "gemini-1.0-pro": {"name": "Gemini 1.0 Pro", "description": "Legacy model - reliable and stable", "provider": "google"},
}

DEFAULT_MODEL = "gemini-1.5-flash"


def get_ollama_models():
    """Get available Ollama models dynamically."""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")[1:]  # Skip header
            models = []
            for line in lines:
                if line.strip():
                    model_name = line.split()[0]  # Get first column (NAME)
                    models.append(model_name)
            return models
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
        logging.warning(f"Failed to get Ollama models: {e}")
    return []


def refresh_supported_models():
    """Refresh the supported models list with available Ollama models."""
    global SUPPORTED_MODELS

    # Start with base models
    updated_models = {
        # Google Gemini models
        "gemini-1.5-flash": {
            "name": "Gemini 1.5 Flash",
            "description": "Fast and efficient - good for most tasks",
            "provider": "google",
        },
        "gemini-1.5-pro": {
            "name": "Gemini 1.5 Pro",
            "description": "Most capable - best for complex tasks",
            "provider": "google",
        },
        "gemini-1.0-pro": {
            "name": "Gemini 1.0 Pro",
            "description": "Legacy model - reliable and stable",
            "provider": "google",
        },
    }

    # Add available Ollama models
    ollama_models = get_ollama_models()
    for model_name in ollama_models:
        updated_models[model_name] = {
            "name": f"{model_name.replace(':', ' ').title()} (Local)",
            "description": "Local Ollama model - private and fast",
            "provider": "ollama",
        }

    SUPPORTED_MODELS = updated_models
    logging.info(f"Refreshed models: {len(ollama_models)} Ollama models found")


def validate_model(model: str) -> str:
    """Validate and return a supported model name."""
    if model in SUPPORTED_MODELS:
        return model
    logging.warning(f"Invalid model '{model}' requested, falling back to default: {DEFAULT_MODEL}")
    return DEFAULT_MODEL


def create_llm(model: str):
    """Create an LLM instance based on the model provider."""
    validated_model = validate_model(model)
    model_config = SUPPORTED_MODELS[validated_model]
    provider = model_config["provider"]

    if provider == "google":
        return ChatGoogleGenerativeAI(model=validated_model, google_api_key=os.getenv("GOOGLE_API_KEY"), temperature=0)
    elif provider == "ollama":
        return ChatOllama(model=validated_model, base_url="http://localhost:11434", temperature=0)
    else:
        logging.error(f"Unsupported provider: {provider}")
        raise ValueError(f"Unsupported model provider: {provider}")


async def generate_title(prompt, response, model: str = DEFAULT_MODEL):
    """Generate a title for the chat session based on the conversation."""
    # Validate and use the specified model
    validated_model = validate_model(model)

    title_prompt = f'''Based on the following exchange, create a short title (4 words max):

User: "{prompt}"
AI: "{response}"'''
    logging.info(f"Generating title using model: {validated_model}...")
    try:
        llm = create_llm(validated_model)
        result = await llm.ainvoke([HumanMessage(content=title_prompt)])
        title = result.content.strip('"\n ')
        logging.info(f"Generated title: '{title}'")
        return title
    except Exception as e:
        logging.error(f"Error generating title: {e}")
        return "New Chat"


async def read_root(request: Request):
    """Serve the main chat interface with session initialization."""
    logging.info("Serving root HTML.")

    # Initialize session if needed
    session_id = get_session_id_from_request(request)
    if not session_id:
        session_id = str(uuid.uuid4())
        set_session_id_in_request(request, session_id)

    # Ensure both legacy and enhanced sessions exist for backward compatibility
    _ensure_enhanced_session_exists(session_id, "New Chat")

    return templates.TemplateResponse(request, "index.html")


async def get_sessions(request: Request):
    """Get all chat sessions for the current user/session context."""
    logging.info(f"Fetching all sessions. Count: {len(chat_sessions)}")
    current_session_id = get_session_id_from_request(request)

    # Return all sessions (in future could filter by user context)
    return JSONResponse(content=[{"id": sid, "title": sdata["title"]} for sid, sdata in chat_sessions.items()])


async def get_available_models():
    """Get list of available LLM models."""
    logging.info("Fetching available models")

    # Refresh models to include current Ollama models
    try:
        refresh_supported_models()
    except Exception as e:
        logging.warning(f"Failed to refresh Ollama models: {e}")

    return JSONResponse(content={"models": SUPPORTED_MODELS, "default": DEFAULT_MODEL})


def get_session_history(session_id: str):
    """Get chat history for a specific session."""
    logging.info(f"Fetching history for session_id: {session_id}")
    session_data = chat_sessions.get(session_id)
    if not session_data:
        logging.warning(f"Session not found for session_id: {session_id}")
        return JSONResponse(content=[], status_code=404)
    serialized_history = [serialize_message(msg) for msg in session_data["messages"]]
    return JSONResponse(content=serialized_history)


def delete_session(session_id: str):
    """Delete a specific chat session."""
    logging.info(f"Deleting session_id: {session_id}")

    # Delete from enhanced session management if available
    enhanced_deleted = False
    if session_document_manager:
        enhanced_deleted = session_document_manager.delete_session(session_id)
        if enhanced_deleted:
            logging.info(f"Successfully deleted enhanced session: {session_id}")

    # Delete from legacy sessions for backward compatibility
    legacy_deleted = False
    if session_id in chat_sessions:
        del chat_sessions[session_id]
        legacy_deleted = True
        logging.info(f"Successfully deleted legacy session: {session_id}")

    if enhanced_deleted or legacy_deleted:
        return JSONResponse(content={"status": "success"}, status_code=200)
    else:
        logging.warning(f"Session not found for deletion: {session_id}")
        return JSONResponse(content={"status": "error", "message": "Session not found"}, status_code=404)


async def stream_chat(
    request: Request,
    session_id: str = Form(...),
    prompt: str = Form(...),
    file: UploadFile = File(None),
    model: str = Form(DEFAULT_MODEL),
):
    """Stream chat responses with file processing support."""
    logging.info(f"Received stream request for session_id: {session_id}")

    # Ensure session is registered in request context
    set_session_id_in_request(request, session_id)

    user_message_content = []
    if prompt:
        logging.info(f"Prompt received (length: {len(prompt)})")
        user_message_content.append({"type": "text", "text": prompt})

    if file:
        logging.info(f"File received: {file.filename}, Content-Type: {file.content_type}")

        # Validate file using our comprehensive validation framework
        validation_result = await file_validator.validate_file(file)

        # Check validation status and use enhanced error streaming
        error_handler = ErrorHandler()
        sse_streamer = SSEErrorStreamer(error_handler)

        if validation_result.status == ValidationStatus.ERROR:
            logging.error(f"File validation error: {validation_result.validation_errors}")
            # Create a validation error from the result
            error = ContentValidationError("; ".join(validation_result.validation_errors))
            return StreamingResponse(sse_streamer.stream_error(error, session_id, request), media_type="text/event-stream")

        elif validation_result.status == ValidationStatus.INVALID:
            logging.warning(f"File validation failed: {validation_result.validation_errors}")
            # Create appropriate validation error based on the specific failure
            if any("size" in error.lower() for error in validation_result.validation_errors):
                error = FileSizeError("; ".join(validation_result.validation_errors))
            elif any("mime" in error.lower() or "type" in error.lower() for error in validation_result.validation_errors):
                error = MimeTypeError("; ".join(validation_result.validation_errors))
            else:
                error = ContentValidationError("; ".join(validation_result.validation_errors))

            return StreamingResponse(sse_streamer.stream_error(error, session_id, request), media_type="text/event-stream")

        elif validation_result.status == ValidationStatus.SUSPICIOUS:
            logging.warning(f"File flagged as suspicious (score: {validation_result.security_score:.2f}): {file.filename}")
            error = SecurityError(
                f"File flagged as potentially suspicious (security score: {validation_result.security_score:.2f})"
            )
            return StreamingResponse(sse_streamer.stream_error(error, session_id, request), media_type="text/event-stream")

        # File passed validation - proceed with processing
        logging.info(
            f"File validation passed: {file.filename}, hash: {validation_result.file_hash}, "
            f"mime_type: {validation_result.mime_type}, security_score: {validation_result.security_score:.2f}"
        )

        # Read file content (file was already read during validation, but we need to reset)
        await file.seek(0)
        file_content = await file.read()
        mime_type = validation_result.mime_type

        # Create document metadata for persistence (if enabled)
        document_metadata = None
        if session_document_manager and session_document_manager._enable_document_persistence:
            document_metadata = DocumentMetadata(
                id=str(uuid.uuid4()),
                hash=validation_result.file_hash,
                filename=file.filename or "untitled",
                mime_type=mime_type,
                size=validation_result.file_size,
                uploaded_at=datetime.utcnow(),
                last_accessed=datetime.utcnow(),
                processing_status=DocumentProcessingStatus.PROCESSING,
            )

        # Enhanced Document Processing with LangChain Loaders
        # Try to process the document using appropriate LangChain loaders first
        processed_document = None
        try:
            processed_document = await document_processor_factory.process_document(
                file_content=file_content, filename=file.filename, mime_type=mime_type, session_id=session_id
            )
        except Exception as e:
            logging.warning(f"Document processing failed for {file.filename}: {e}, falling back to base64")

        if processed_document is not None and processed_document.content and processed_document.content.strip():
            # Successfully processed with LangChain - use enhanced content
            logging.info(f"Successfully processed {file.filename} using {processed_document.processing_method}")

            # Update document metadata with processing results
            if document_metadata:
                document_metadata.processing_status = DocumentProcessingStatus.COMPLETED
                document_metadata.metadata.update(processed_document.metadata)

                # Persist document to session
                try:
                    success, error = session_document_manager.add_document_to_session(
                        session_id, document_metadata, processed_document
                    )
                    if success:
                        logging.info(f"Document {file.filename} persisted to session {session_id}")
                    else:
                        logging.warning(f"Failed to persist document {file.filename}: {error}")
                except Exception as e:
                    logging.error(f"Error persisting document {file.filename}: {e}")

            # For PDFs and other structured documents, provide rich content to LLM
            enhanced_content = f"""Document: {processed_document.original_filename}
Type: {mime_type}
Processing Method: {processed_document.processing_method}
Pages: {processed_document.structure.get('total_pages', 'N/A')}

Document Metadata:
- Title: {processed_document.metadata.get('title', 'N/A')}
- Author: {processed_document.metadata.get('author', 'N/A')}
- Creation Date: {processed_document.metadata.get('creation_date', 'N/A')}
- Page Count: {processed_document.metadata.get('page_count', 'N/A')}

Document Content:
{processed_document.content}

Document Structure Information:
{json.dumps(processed_document.structure, indent=2)}"""

            # Send enhanced content as text for better LLM understanding
            user_message_content.append({"type": "text", "text": enhanced_content})

            # Only include base64 for visual analysis if it's an actual image format
            # PDFs are handled through text content extraction, not as images
            if mime_type.startswith("image/"):
                base64_encoded_file = base64.b64encode(file_content).decode("utf-8")
                user_message_content.append(
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_encoded_file}"}}
                )

            logging.info(
                f"Enhanced processing completed for {file.filename}: "
                f"{processed_document.structure.get('content_length', 0)} chars, "
                f"{processed_document.structure.get('total_pages', 0)} pages"
            )
        else:
            # Fallback to original base64 processing
            logging.info(f"Using fallback base64 processing for {file.filename}")

            # Mark document as failed if we have metadata
            if document_metadata:
                document_metadata.processing_status = DocumentProcessingStatus.FAILED
                document_metadata.processing_error = "Failed to process with LangChain loaders, using base64 fallback"

                # Still try to persist the document metadata (without chunks)
                try:
                    success, error = session_document_manager.add_document_to_session(session_id, document_metadata, None)
                    if success:
                        logging.info(f"Document metadata for {file.filename} persisted to session {session_id}")
                    else:
                        logging.warning(f"Failed to persist document metadata {file.filename}: {error}")
                except Exception as e:
                    logging.error(f"Error persisting document metadata {file.filename}: {e}")

            base64_encoded_file = base64.b64encode(file_content).decode("utf-8")

            # Google Gemini's LangChain integration is strict about MIME types for image_url format
            # Only actual image files can be sent as image_url - PDFs must be processed as text
            if mime_type.startswith("image/"):
                # Only images are directly supported by Gemini's image_url format
                user_message_content.append(
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_encoded_file}"}}
                )
            elif mime_type == "application/pdf":
                # For PDFs in fallback mode, attempt basic text extraction using PyPDF
                try:
                    import io

                    from pypdf import PdfReader

                    pdf_reader = PdfReader(io.BytesIO(file_content))
                    pdf_text = ""
                    for page_num, page in enumerate(pdf_reader.pages):
                        pdf_text += f"\n--- Page {page_num + 1} ---\n"
                        pdf_text += page.extract_text()

                    if pdf_text.strip():
                        user_message_content.append(
                            {"type": "text", "text": f"PDF Document: {file.filename}\nExtracted Content:\n{pdf_text}"}
                        )
                    else:
                        user_message_content.append(
                            {
                                "type": "text",
                                "text": f"PDF file '{file.filename}' uploaded ({validation_result.file_size} bytes) but no text content could be extracted. The PDF may be image-based or encrypted.",
                            }
                        )
                except Exception as pdf_error:
                    logging.warning(f"Basic PDF text extraction failed for {file.filename}: {pdf_error}")
                    user_message_content.append(
                        {
                            "type": "text",
                            "text": f"PDF file '{file.filename}' uploaded: {validation_result.file_size} bytes. Advanced PDF processing failed, and basic extraction also failed. Please try a different file format.",
                        }
                    )
            else:
                # For other file types (text, etc.), include them as text content with context
                try:
                    # Try to decode as text if it's a text-based file
                    if mime_type.startswith("text/") or mime_type in ["application/json", "application/xml"]:
                        file_text = file_content.decode("utf-8", errors="replace")
                        user_message_content.append(
                            {"type": "text", "text": f"File '{file.filename}' content ({mime_type}):\n```\n{file_text}\n```"}
                        )
                    else:
                        # For binary files, provide metadata only
                        user_message_content.append(
                            {
                                "type": "text",
                                "text": f"File '{file.filename}' uploaded: {mime_type}, {validation_result.file_size} bytes. Content cannot be displayed as text.",
                            }
                        )
                except Exception as e:
                    logging.warning(f"Could not process file content as text: {e}")
                    user_message_content.append(
                        {
                            "type": "text",
                            "text": f"File '{file.filename}' uploaded: {mime_type}, {validation_result.file_size} bytes. Content could not be processed.",
                        }
                    )

        logging.info(
            f"Completed file processing for: {file.filename}, "
            f"mime_type: {mime_type}, size: {validation_result.file_size} bytes"
        )

    if not user_message_content:
        logging.warning("Stream request with no prompt or file.")

        async def no_content_stream():
            yield "data: An error occurred: Please provide a prompt or a file.\n\n"

        return StreamingResponse(no_content_stream(), media_type="text/event-stream")

    async def event_stream(initial_user_message_content):
        # Ensure enhanced session exists
        _ensure_enhanced_session_exists(session_id, "New Chat")

        # Use legacy session for message history (backward compatibility)
        history = chat_sessions[session_id]["messages"]
        history.append(HumanMessage(content=initial_user_message_content))
        logging.info(f"Appended user message to history for session {session_id}.")

        try:
            # Validate and use the specified model
            validated_model = validate_model(model)
            logging.info(f"Using model: {validated_model} for session {session_id}")

            llm = create_llm(validated_model)

            full_response = ""
            logging.info("Starting LLM stream...")
            async for chunk in llm.astream(history):
                content = chunk.content
                if content:
                    full_response += content
                    yield f"data: {content}\n\n"

            logging.info("LLM stream finished.")
            history.append(AIMessage(content=full_response))

            # Sync messages to enhanced session
            if session_document_manager:
                enhanced_session = session_document_manager.get_session(session_id)
                if enhanced_session:
                    enhanced_session.messages = history
                    enhanced_session.update_activity()

            if len(history) == 2:
                new_title = await generate_title(prompt, full_response, validated_model)
                chat_sessions[session_id]["title"] = new_title

                # Update enhanced session title too
                if session_document_manager:
                    enhanced_session = session_document_manager.get_session(session_id)
                    if enhanced_session:
                        enhanced_session.title = new_title

                yield f"event: title\ndata: {new_title}\n\n"  # Send title update event

            yield "data: [DONE]\n\n"
            logging.info(f"Stream completed successfully for session {session_id}.")

        except Exception as e:
            logging.error(f"LLM streaming error: {e}")

            # Use enhanced error streaming for LLM errors
            error_handler = ErrorHandler()
            sse_streamer = SSEErrorStreamer(error_handler)

            # Create appropriate LLM error based on the exception
            if "rate limit" in str(e).lower() or "429" in str(e):
                llm_error = LLMError(
                    message=str(e),
                    error_code="LLM_RATE_LIMIT",
                    severity=ErrorSeverity.HIGH,
                    user_message="The AI service is currently busy. Please wait a moment and try again.",
                    suggested_actions=[
                        "Wait a moment and try again",
                        "Try with a shorter message",
                        "Check your internet connection",
                    ],
                )
            elif "timeout" in str(e).lower() or "connection" in str(e).lower():
                llm_error = NetworkError(
                    message=str(e),
                    error_code="LLM_CONNECTION_ERROR",
                    severity=ErrorSeverity.HIGH,
                    user_message="Connection to AI service failed. Please try again.",
                    suggested_actions=[
                        "Check your internet connection",
                        "Try again in a moment",
                        "Refresh the page if needed",
                    ],
                )
            else:
                llm_error = LLMError(
                    message=str(e),
                    error_code="LLM_GENERAL_ERROR",
                    severity=ErrorSeverity.HIGH,
                    user_message="The AI service encountered an error. Please try again.",
                    suggested_actions=[
                        "Try sending your message again",
                        "Try with a different prompt",
                        "Contact support if the problem persists",
                    ],
                )

            # Stream the error using our enhanced error streamer
            async for error_chunk in sse_streamer.stream_error(llm_error, session_id):
                yield error_chunk

    return StreamingResponse(event_stream(user_message_content), media_type="text/event-stream")


def set_templates(template_instance: Jinja2Templates):
    """Set the templates instance for use in endpoints."""
    global templates
    templates = template_instance


def set_file_validator(validator_instance):
    """Set the file validator instance for use in endpoints."""
    global file_validator
    file_validator = validator_instance


def set_document_processor_factory(factory_instance):
    """Set the document processor factory instance for use in endpoints."""
    global document_processor_factory
    document_processor_factory = factory_instance


def get_session_document_manager():
    """Get the session document manager instance."""
    return session_document_manager


def get_document_store():
    """Get the document store instance."""
    return document_store


def get_access_controller():
    """Get the access controller instance."""
    return access_controller


def get_cleanup_manager():
    """Get the cleanup manager instance."""
    return cleanup_manager


# New API endpoints for document management


async def get_session_documents(session_id: str, include_chunks: bool = False):
    """
    Get all documents for a specific session.

    Args:
        session_id: Session ID
        include_chunks: Whether to include chunked document information

    Returns:
        JSONResponse with session documents information
    """
    if not session_document_manager:
        return JSONResponse(content={"error": "Document persistence not enabled"}, status_code=501)

    session = session_document_manager.get_session(session_id)
    if not session:
        return JSONResponse(content={"error": "Session not found"}, status_code=404)

    try:
        documents = session_document_manager.get_session_documents(session_id, include_chunks)
        total_size = sum(doc["size"] for doc in documents)

        response_data = {
            "session_id": session_id,
            "documents": documents,
            "total_count": len(documents),
            "total_size_bytes": total_size,
            "session_info": {
                "title": session.title,
                "created_at": session.created_at.isoformat(),
                "last_activity": session.last_activity.isoformat(),
            },
        }

        return JSONResponse(content=response_data)

    except Exception as e:
        logging.error(f"Error getting session documents for {session_id}: {e}")
        return JSONResponse(content={"error": "Internal server error"}, status_code=500)


async def get_document_details(document_id: str, session_id: Optional[str] = None):
    """
    Get detailed information about a specific document.

    Args:
        document_id: Document ID
        session_id: Optional session ID for access control

    Returns:
        JSONResponse with document details
    """
    if not session_document_manager or not document_store:
        return JSONResponse(content={"error": "Document persistence not enabled"}, status_code=501)

    try:
        # Get document from store
        document_metadata = document_store.get_document_by_id(document_id)
        if not document_metadata:
            return JSONResponse(content={"error": "Document not found"}, status_code=404)

        # Get sessions that have access to this document
        doc_hash = document_metadata.hash
        sessions_with_access = document_store.get_sessions_for_document(doc_hash)

        # If session_id provided, verify access
        if session_id and session_id not in sessions_with_access:
            return JSONResponse(content={"error": "Access denied"}, status_code=403)

        # Get detailed document information from a session that has access
        session_document = None
        for sid in sessions_with_access:
            session_document = session_document_manager.get_document_from_session(sid, doc_hash)
            if session_document:
                break

        if not session_document:
            return JSONResponse(content={"error": "Document not found in any session"}, status_code=404)

        response_data = {
            "id": document_metadata.id,
            "hash": document_metadata.hash,
            "filename": document_metadata.filename,
            "mime_type": document_metadata.mime_type,
            "size": document_metadata.size,
            "uploaded_at": document_metadata.uploaded_at.isoformat(),
            "last_accessed": document_metadata.last_accessed.isoformat(),
            "processing_status": document_metadata.processing_status.value,
            "processing_error": document_metadata.processing_error,
            "metadata": document_metadata.metadata,
            "access_level": session_document.access_level.value,
            "owned": session_document.owned,
            "shared_from": session_document.shared_from,
            "shared_to": list(session_document.shared_to),
            "chunks": {
                "available": session_document.chunks is not None,
                "chunk_count": session_document.chunks.chunk_count if session_document.chunks else 0,
                "total_content_length": session_document.chunks.total_content_length if session_document.chunks else 0,
            },
            "sessions_with_access": list(sessions_with_access),
        }

        return JSONResponse(content=response_data)

    except Exception as e:
        logging.error(f"Error getting document details for {document_id}: {e}")
        return JSONResponse(content={"error": "Internal server error"}, status_code=500)


async def share_document(
    document_id: str,
    source_session_id: str = Form(...),
    target_session_id: str = Form(...),
    access_level: str = Form("shared"),
):
    """
    Share a document between sessions.

    Args:
        document_id: Document ID to share
        source_session_id: Source session ID (form data)
        target_session_id: Target session ID (form data)
        access_level: Access level for shared document (form data)

    Returns:
        JSONResponse with sharing result
    """
    if not session_document_manager or not access_controller:
        return JSONResponse(content={"error": "Document sharing not enabled"}, status_code=501)

    try:
        # Get document metadata
        document_metadata = document_store.get_document_by_id(document_id)
        if not document_metadata:
            return JSONResponse(content={"error": "Document not found"}, status_code=404)

        doc_hash = document_metadata.hash

        # Get document from source session
        source_document = session_document_manager.get_document_from_session(source_session_id, doc_hash)
        if not source_document:
            return JSONResponse(content={"error": "Document not found in source session"}, status_code=404)

        # Check if sharing is allowed
        can_share, reason = access_controller.can_share_document(source_session_id, target_session_id, source_document)
        if not can_share:
            return JSONResponse(content={"error": f"Sharing not allowed: {reason}"}, status_code=403)

        # Attempt to share the document
        success, error = session_document_manager.share_document_between_sessions(
            source_session_id, target_session_id, doc_hash
        )

        if success:
            return JSONResponse(
                content={
                    "success": True,
                    "message": "Document shared successfully",
                    "auto_approved": True,  # Direct sharing is auto-approved
                }
            )
        else:
            return JSONResponse(content={"error": f"Sharing failed: {error}"}, status_code=400)

    except Exception as e:
        logging.error(f"Error sharing document {document_id}: {e}")
        return JSONResponse(content={"error": "Internal server error"}, status_code=500)


async def remove_document(document_id: str, session_id: str = Query(...)):
    """
    Remove a document from a session.

    Args:
        document_id: Document ID to remove
        session_id: Session ID to remove document from (query parameter)

    Returns:
        JSONResponse with removal result
    """
    if not session_document_manager or not document_store:
        return JSONResponse(content={"error": "Document persistence not enabled"}, status_code=501)

    try:
        # Get document metadata
        document_metadata = document_store.get_document_by_id(document_id)
        if not document_metadata:
            return JSONResponse(content={"error": "Document not found"}, status_code=404)

        doc_hash = document_metadata.hash

        # Check if document exists in session
        session_document = session_document_manager.get_document_from_session(session_id, doc_hash)
        if not session_document:
            return JSONResponse(content={"error": "Document not found in session"}, status_code=404)

        # Remove document from session
        removed = session_document_manager.remove_document_from_session(session_id, doc_hash)

        # Check if document was completely deleted (no more references)
        remaining_sessions = document_store.get_sessions_for_document(doc_hash)
        completely_deleted = len(remaining_sessions) == 0

        if removed:
            return JSONResponse(
                content={
                    "success": True,
                    "message": "Document removed from session successfully",
                    "document_removed_from_session": True,
                    "document_completely_deleted": completely_deleted,
                }
            )
        else:
            return JSONResponse(content={"error": "Failed to remove document from session"}, status_code=500)

    except Exception as e:
        logging.error(f"Error removing document {document_id} from session {session_id}: {e}")
        return JSONResponse(content={"error": "Internal server error"}, status_code=500)


async def get_cleanup_statistics():
    """
    Get system cleanup statistics and recommendations.

    Returns:
        JSONResponse with cleanup statistics
    """
    if not cleanup_manager:
        return JSONResponse(content={"error": "Cleanup manager not available"}, status_code=501)

    try:
        stats = cleanup_manager.get_cleanup_statistics()
        return JSONResponse(content=stats)

    except Exception as e:
        logging.error(f"Error getting cleanup statistics: {e}")
        return JSONResponse(content={"error": "Internal server error"}, status_code=500)


async def run_manual_cleanup():
    """
    Manually trigger system cleanup.

    Returns:
        JSONResponse with cleanup results
    """
    if not cleanup_manager:
        return JSONResponse(content={"error": "Cleanup manager not available"}, status_code=501)

    try:
        results = cleanup_manager.run_full_cleanup()
        return JSONResponse(content=results)

    except Exception as e:
        logging.error(f"Error running manual cleanup: {e}")
        return JSONResponse(content={"error": "Internal server error"}, status_code=500)
