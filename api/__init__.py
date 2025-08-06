"""API endpoints and route handlers."""

from .endpoints import (
    read_root,
    get_sessions,
    get_session_history,
    delete_session,
    stream_chat,
    set_templates,
    set_file_validator,
    set_document_processor_factory,
    chat_sessions
)

__all__ = [
    'read_root',
    'get_sessions', 
    'get_session_history',
    'delete_session',
    'stream_chat',
    'set_templates',
    'set_file_validator',
    'set_document_processor_factory',
    'chat_sessions'
]