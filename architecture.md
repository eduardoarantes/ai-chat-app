```mermaid
graph TD
    subgraph Client (Browser)
        A[User Input & File Selection] --> B(JavaScript Frontend);
        B -- POST /stream (Prompt, File, Session ID) --> C;
        B -- GET /sessions --> D;
        B -- GET /sessions/{session_id} --> E;
        C -- SSE Stream (AI Response) --> B;
        D -- JSON (Session List) --> B;
        E -- JSON (Chat History) --> B;
    end

    subgraph Server (FastAPI Backend)
        C(Stream Endpoint) --> F{Process Request};
        D(Sessions Endpoint) --> G{Retrieve Sessions};
        E(History Endpoint) --> H{Retrieve History};

        F -- Update History --> I[In-memory Chat Sessions];
        F -- Call LLM --> J[LangChain Integration];
        G -- Read Sessions --> I;
        H -- Read History --> I;
    end

    subgraph External Service
        J --> K[Google Gemini LLM];
    end

    K -- LLM Response --> J;
```