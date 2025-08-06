"""
Mock data generators for AI Foundation chat application tests.

This module provides functions to generate realistic mock data for testing
various scenarios including sessions, messages, API responses, and file uploads.
"""

import uuid
import random
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta


class MockDataGenerator:
    """Generator class for creating mock test data."""
    
    @staticmethod
    def generate_session_id() -> str:
        """
        Generate a random session ID.
        
        Returns:
            str: Random UUID session ID
        """
        return str(uuid.uuid4())
    
    @staticmethod
    def generate_session_title(length: Optional[int] = None) -> str:
        """
        Generate a random session title.
        
        Args:
            length: Optional specific length for the title
            
        Returns:
            str: Random session title
        """
        titles = [
            "Chat about Python Programming",
            "Help with JavaScript Functions",
            "Machine Learning Discussion",
            "Web Development Questions",
            "Data Analysis Help",
            "API Integration Support",
            "Database Design Discussion", 
            "React Component Help",
            "FastAPI Development",
            "Testing Strategies",
            "Code Review Session",
            "Bug Troubleshooting",
            "Architecture Planning",
            "Performance Optimization",
            "Security Best Practices"
        ]
        
        title = random.choice(titles)
        if length:
            if len(title) > length:
                title = title[:length-3] + "..."
            elif len(title) < length:
                title = title + " " + str(random.randint(1, 999))
        
        return title
    
    @staticmethod
    def generate_user_message(content: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a mock user message.
        
        Args:
            content: Optional specific content for the message
            
        Returns:
            Dict[str, Any]: Mock user message
        """
        if content is None:
            user_prompts = [
                "Hello, can you help me?",
                "I'm having trouble with my code",
                "Can you explain how this works?",
                "What's the best way to approach this?",
                "I'm getting an error when I run this",
                "How do I implement this feature?",
                "Can you review my code?",
                "What are the best practices for this?",
                "I need help debugging this issue",
                "Can you walk me through this process?"
            ]
            content = random.choice(user_prompts)
        
        return {
            "type": "user",
            "content": content
        }
    
    @staticmethod
    def generate_ai_message(content: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a mock AI message.
        
        Args:
            content: Optional specific content for the message
            
        Returns:
            Dict[str, Any]: Mock AI message
        """
        if content is None:
            ai_responses = [
                "I'd be happy to help you with that!",
                "Let me walk you through this step by step.",
                "That's a great question. Here's how you can approach it:",
                "I can see the issue in your code. Here's the fix:",
                "This is a common pattern in programming. Let me explain:",
                "You're on the right track! Here are some suggestions:",
                "Here's a better way to implement that:",
                "That error usually means this. Try this solution:",
                "Let me help you debug this problem:",
                "I can definitely assist you with this task."
            ]
            content = random.choice(ai_responses)
        
        return {
            "type": "ai",
            "content": content
        }
    
    @staticmethod
    def generate_conversation(
        num_exchanges: int = 3,
        include_multimodal: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Generate a mock conversation with alternating user/AI messages.
        
        Args:
            num_exchanges: Number of user-AI exchanges to generate
            include_multimodal: Whether to include multimodal messages
            
        Returns:
            List[Dict[str, Any]]: List of conversation messages
        """
        messages = []
        
        for i in range(num_exchanges):
            # Add user message
            if include_multimodal and i == 0:
                # First message could be multimodal
                messages.append({
                    "type": "user",
                    "content": [
                        {"type": "text", "text": "Can you help me understand this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
                            }
                        }
                    ]
                })
            else:
                messages.append(MockDataGenerator.generate_user_message())
            
            # Add AI response
            messages.append(MockDataGenerator.generate_ai_message())
        
        return messages
    
    @staticmethod
    def generate_session(
        session_id: Optional[str] = None,
        title: Optional[str] = None,
        num_messages: int = 6
    ) -> Dict[str, Any]:
        """
        Generate a complete mock session.
        
        Args:
            session_id: Optional specific session ID
            title: Optional specific title
            num_messages: Number of messages to generate
            
        Returns:
            Dict[str, Any]: Complete mock session
        """
        if session_id is None:
            session_id = MockDataGenerator.generate_session_id()
        
        if title is None:
            title = MockDataGenerator.generate_session_title()
        
        # Generate alternating messages
        messages = []
        for i in range(num_messages):
            if i % 2 == 0:
                messages.append(MockDataGenerator.generate_user_message())
            else:
                messages.append(MockDataGenerator.generate_ai_message())
        
        return {
            "id": session_id,
            "title": title,
            "messages": messages
        }
    
    @staticmethod
    def generate_multiple_sessions(count: int = 5) -> List[Dict[str, Any]]:
        """
        Generate multiple mock sessions.
        
        Args:
            count: Number of sessions to generate
            
        Returns:
            List[Dict[str, Any]]: List of mock sessions
        """
        sessions = []
        for _ in range(count):
            num_messages = random.randint(2, 12)  # Vary message count
            session = MockDataGenerator.generate_session(num_messages=num_messages)
            sessions.append(session)
        
        return sessions
    
    @staticmethod
    def generate_streaming_response_chunks() -> List[str]:
        """
        Generate realistic chunks for streaming response testing.
        
        Returns:
            List[str]: List of response chunks
        """
        chunks = [
            "I can",
            " help you",
            " with that",
            " question.",
            " Let me",
            " break it",
            " down",
            " step by",
            " step:",
            "\n\n1.",
            " First,",
            " you need",
            " to understand",
            " the basic",
            " concept.",
            "\n2.",
            " Then,",
            " implement",
            " the solution",
            " carefully.",
            "\n3.",
            " Finally,",
            " test your",
            " implementation",
            " thoroughly."
        ]
        return chunks
    
    @staticmethod
    def generate_title_generation_prompts() -> List[str]:
        """
        Generate prompts that would trigger title generation.
        
        Returns:
            List[str]: List of title generation prompts
        """
        return [
            "Based on our conversation about Python programming, generate a concise title.",
            "Create a title that summarizes our discussion about web development.",
            "Generate a short title for our chat about machine learning.",
            "Summarize our conversation in a brief title.",
            "Create a descriptive title for this discussion."
        ]
    
    @staticmethod
    def generate_file_upload_scenarios() -> List[Dict[str, Any]]:
        """
        Generate various file upload test scenarios.
        
        Returns:
            List[Dict[str, Any]]: List of file upload scenarios
        """
        return [
            {
                "filename": "test.txt",
                "content": b"This is a test text file.",
                "content_type": "text/plain",
                "prompt": "Please analyze this text file."
            },
            {
                "filename": "image.png",
                "content": b"fake_png_data",
                "content_type": "image/png",
                "prompt": "What do you see in this image?"
            },
            {
                "filename": "document.pdf",
                "content": b"fake_pdf_data",
                "content_type": "application/pdf",
                "prompt": "Summarize this PDF document."
            },
            {
                "filename": "data.json",
                "content": b'{"key": "value", "numbers": [1, 2, 3]}',
                "content_type": "application/json",
                "prompt": "Process this JSON data."
            }
        ]
    
    @staticmethod
    def generate_error_scenarios() -> List[Dict[str, Any]]:
        """
        Generate error scenarios for testing error handling.
        
        Returns:
            List[Dict[str, Any]]: List of error scenarios
        """
        return [
            {
                "scenario": "missing_session_id",
                "data": {"prompt": "Hello"},
                "expected_error": "Session ID is required"
            },
            {
                "scenario": "empty_prompt",
                "data": {"session_id": "test-123", "prompt": ""},
                "expected_error": "Prompt cannot be empty"
            },
            {
                "scenario": "invalid_session_id",
                "data": {"session_id": "invalid-session", "prompt": "Hello"},
                "expected_error": "Session not found"
            },
            {
                "scenario": "large_file",
                "data": {
                    "session_id": "test-123",
                    "prompt": "Process this file",
                    "file_size": 50 * 1024 * 1024  # 50MB
                },
                "expected_error": "File too large"
            }
        ]