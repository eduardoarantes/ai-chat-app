# SSE ChatGPT with LangChain and FastAPI

This project demonstrates a real-time chat application using Server-Sent Events (SSE), FastAPI for the backend, and LangChain for integrating with Large Language Models (LLMs) like Google Gemini. It supports multi-session conversations, automatic chat titling, and multimodal input (text and images).

## Features

*   **Real-time Streaming:** Get responses from the LLM token by token using Server-Sent Events.
*   **Multi-session Chat:** Create, manage, and switch between different chat conversations.
*   **Automatic Chat Titling:** New conversations are automatically given a short, descriptive title after the first exchange.
*   **Multimodal Input:** Send both text prompts and image files to the LLM.
*   **Syntax Highlighting:** Code blocks in AI responses are automatically highlighted for readability.
*   **Responsive UI:** A clean, chat-like interface with user and AI message bubbles.

## Setup and Running the Application

Follow these steps to get the application up and running on your local machine.

### Prerequisites

*   Python 3.8+
*   `pip` (Python package installer)

### 1. Navigate to the Project Directory

First, ensure you are in the `sse-chatgpt` directory:

```bash
cd sse-chatgpt
```

### 2. Create and Activate a Python Virtual Environment

It's highly recommended to use a virtual environment to manage project dependencies.

```bash
python3 -m venv .venv
```

Activate the virtual environment:

*   **On macOS/Linux:**
    ```bash
    source .venv/bin/activate
    ```
*   **On Windows (Command Prompt):**
    ```bash
    .venv\Scripts\activate.bat
    ```
*   **On Windows (PowerShell):**
    ```powershell
    .venv\Scripts\Activate.ps1
    ```

### 3. Install Dependencies

With your virtual environment activated, install the required Python packages:

```bash
pip install -r requirements.txt
```

### 4. Set Up Your API Key

The application uses a Google Gemini API key. You need to obtain one from the Google AI Studio (https://aistudio.google.com/app/apikey).

Once you have your API key, open the `.env` file in the `sse-chatgpt` directory and replace `"your-api-key"` with your actual key:

```
GOOGLE_API_KEY="YOUR_ACTUAL_GOOGLE_API_KEY_HERE"
```

**Important:** Do not share your API key publicly or commit it to version control.

### 5. Run the Application

Start the FastAPI application using Uvicorn:

```bash
uvicorn main:app --reload --app-dir .
```

*   The `--reload` flag will automatically restart the server when you make changes to the code.
*   The `--app-dir .` flag tells Uvicorn to look for `main:app` in the current directory.

If you encounter an `Address already in use` error, it means another process is using port 8000. You can either stop that process or run the application on a different port:

```bash
uvicorn main:app --reload --app-dir . --port 8001
```

### 6. Access the Application

Open your web browser and navigate to:

```
http://127.0.0.1:8000
```
(Or `http://127.0.0.1:8001` if you used a different port)

You should now see the chat interface. You can start a new conversation, switch between sessions, and send both text and image inputs to the LLM.
