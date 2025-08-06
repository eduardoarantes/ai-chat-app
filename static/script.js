let activeSessionId = null;
let chatContainer;
let promptInput;
let sessionList;
let fileInput;
let newChatBtn;
let attachFileBtn;
let sendBtn;
let filePreviewContainer;
let fileNameSpan;
let removeFileBtn;
let modelSelect;

// Available models (will be loaded from backend)
let availableModels = {};
let defaultModel = 'gemini-1.5-flash';

// Store last message context for retry functionality
let lastMessageContext = {
    prompt: null,
    file: null,
    model: null
};

// Model preference management
const MODEL_STORAGE_KEY = 'preferred-llm-model';

// Get user's preferred model from localStorage
function getPreferredModel() {
    return localStorage.getItem(MODEL_STORAGE_KEY) || defaultModel;
}

// Save user's preferred model to localStorage
function savePreferredModel(model) {
    localStorage.setItem(MODEL_STORAGE_KEY, model);
    log('info', `Saved preferred model: ${model}`);
}

// Load available models from backend
async function loadAvailableModels() {
    try {
        const response = await fetch('/models');
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const modelsData = await response.json();
        availableModels = modelsData.models;
        defaultModel = modelsData.default;
        
        // Update the model select dropdown
        updateModelSelectOptions();
        
        // Set user's preferred model
        const preferredModel = getPreferredModel();
        if (preferredModel && availableModels[preferredModel]) {
            modelSelect.value = preferredModel;
        }
        
        log('info', `Loaded ${Object.keys(availableModels).length} available models`);
    } catch (error) {
        log('error', 'Failed to load available models:', error);
        // Use fallback models if backend request fails
        availableModels = {
            'gemini-1.5-flash': { name: 'Gemini 1.5 Flash', description: 'Fast and efficient' },
            'gemini-1.5-pro': { name: 'Gemini 1.5 Pro', description: 'Most capable' },
            'gemini-1.0-pro': { name: 'Gemini 1.0 Pro', description: 'Legacy model' }
        };
        updateModelSelectOptions();
    }
}

// Update model select dropdown options
function updateModelSelectOptions() {
    if (!modelSelect) return;
    
    modelSelect.innerHTML = '';
    
    Object.entries(availableModels).forEach(([modelId, modelInfo]) => {
        const option = document.createElement('option');
        option.value = modelId;
        option.textContent = modelInfo.name;
        option.title = modelInfo.description;
        modelSelect.appendChild(option);
    });
}

// --- Logging Setup ---
const log = (level, message, ...args) => {
    console[level](`[${level.toUpperCase()}] ${new Date().toISOString()} - ${message}`, ...args);
};

// Configure marked.js to use highlight.js
marked.setOptions({
    highlight: function(code, lang) {
        const language = hljs.getLanguage(lang) ? lang : 'plaintext';
        return hljs.highlight(code, { language }).value;
    }
});

// --- Core Functions ---
async function loadSessions() {
    log('info', 'Loading sessions...');
    try {
        const response = await fetch(`/sessions?t=${new Date().getTime()}`); // Cache-busting
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const sessions = await response.json();
        sessionList.innerHTML = "";
        sessions.forEach(session => {
            const sessionDiv = document.createElement("div");
            sessionDiv.className = "session-item";
            sessionDiv.dataset.sessionId = session.id;

            const titleSpan = document.createElement("span");
            titleSpan.textContent = session.title;
            titleSpan.style.flexGrow = "1";
            titleSpan.style.overflow = "hidden";
            titleSpan.style.textOverflow = "ellipsis";
            titleSpan.addEventListener("click", () => loadSessionHistory(session.id));

            const deleteBtn = document.createElement("button");
            deleteBtn.innerHTML = "&times;";
            deleteBtn.className = "delete-session-btn";
            deleteBtn.title = "Delete session";
            deleteBtn.addEventListener("click", (e) => {
                e.stopPropagation(); // Prevent session from loading when clicking delete
                deleteSession(session.id, sessionDiv);
            });

            sessionDiv.appendChild(titleSpan);
            sessionDiv.appendChild(deleteBtn);
            sessionList.appendChild(sessionDiv);
        });
        log('info', `Loaded ${sessions.length} sessions.`);
        updateActiveSessionUI();
    } catch (error) {
        log('error', 'Failed to load sessions:', error);
        if (window.errorHandler) {
            window.errorHandler.handleNetworkError(error, {
                retryHandler: () => loadSessions(),
                userMessage: 'Failed to load your chat sessions. Please check your connection.',
                suggestedActions: [
                    'Check your internet connection',
                    'Refresh the page',
                    'Try again in a moment'
                ]
            });
        }
    }
}

async function loadSessionHistory(sessionId) {
    log('info', `Loading history for session: ${sessionId}`);
    activeSessionId = sessionId;
    chatContainer.innerHTML = "";
    try {
        const response = await fetch(`/sessions/${sessionId}`);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const history = await response.json();
        history.forEach(message => {
            const messageDiv = document.createElement("div");
            messageDiv.className = `message ${message.type === 'user' ? 'user-message' : 'ai-message'}`;
            
            if (message.type === 'user' && Array.isArray(message.content)) {
                message.content.forEach(item => {
                    if (item.type === 'text') {
                        messageDiv.innerHTML += marked.parse(item.text);
                    } else if (item.type === 'image_url') {
                        const img = document.createElement('img');
                        img.src = item.image_url.url;
                        img.style.maxWidth = '100%';
                        img.style.height = 'auto';
                        messageDiv.appendChild(img);
                        log('info', 'Displayed image from history.');
                    }
                });
            } else {
                messageDiv.innerHTML = marked.parse(message.content);
            }
            chatContainer.appendChild(messageDiv);
        });
        hljs.highlightAll();
        chatContainer.scrollTop = chatContainer.scrollHeight;
        log('info', `Successfully loaded history for session: ${sessionId}`);
        updateActiveSessionUI();
    } catch (error) {
        log('error', `Failed to load session history for ${sessionId}:`, error);
        if (window.errorHandler) {
            window.errorHandler.handleNetworkError(error, {
                retryHandler: () => loadSessionHistory(sessionId),
                userMessage: 'Failed to load chat history. Please try again.',
                suggestedActions: [
                    'Check your connection',
                    'Try selecting the session again',
                    'Refresh the page if needed'
                ]
            });
        }
    }
}

async function deleteSession(sessionId, sessionDiv) {
    log('info', `Deleting session: ${sessionId}`);
    try {
        const response = await fetch(`/sessions/${sessionId}`, {
            method: 'DELETE',
        });
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

        log('info', `Successfully deleted session: ${sessionId}`);
        sessionDiv.remove(); // Remove from UI

        // If the deleted session was the active one, create a new one
        if (activeSessionId === sessionId) {
            createNewSession();
        }
    } catch (error) {
        log('error', `Failed to delete session ${sessionId}:`, error);
        if (window.errorHandler) {
            window.errorHandler.handleNetworkError(error, {
                retryHandler: () => deleteSession(sessionId, sessionDiv),
                userMessage: 'Failed to delete the chat session. Please try again.',
                suggestedActions: [
                    'Try deleting the session again',
                    'Refresh the page to see current sessions',
                    'Check your connection'
                ]
            });
        }
    }
}

function createNewSession() {
    // Don't create a new placeholder if one already exists and is active
    const existingActive = document.querySelector('.session-item.active');
    if (existingActive && existingActive.dataset.isNew === 'true') {
        log('info', 'A new chat session is already the active session.');
        return;
    }

    const newId = crypto.randomUUID();
    log('info', `Creating new session with ID: ${newId}`);
    activeSessionId = newId;
    chatContainer.innerHTML = "";
    clearFileInput();

    // Create a new session element and add it to the top of the list
    const sessionDiv = document.createElement("div");
    sessionDiv.className = "session-item";
    sessionDiv.dataset.sessionId = newId;
    sessionDiv.dataset.isNew = 'true'; // Mark as a new, unsaved session

    const titleSpan = document.createElement("span");
    titleSpan.textContent = "New Chat"; // Placeholder title
    titleSpan.style.flexGrow = "1";
    titleSpan.style.overflow = "hidden";
    titleSpan.style.textOverflow = "ellipsis";
    titleSpan.addEventListener("click", () => {
        // If user clicks on a new chat placeholder, do nothing as it's already active
        if (activeSessionId === newId) return;
        loadSessionHistory(newId);
    });

    const deleteBtn = document.createElement("button");
    deleteBtn.innerHTML = "&times;";
    deleteBtn.className = "delete-session-btn";
    deleteBtn.title = "Delete session";
    deleteBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        // This session doesn't exist on the server yet, so just remove from UI
        log('warn', `Removing new, unsaved session placeholder: ${newId}`);
        sessionDiv.remove();
        // If we deleted the active session, load the top one or create a new one
        if (activeSessionId === newId) {
            if (sessionList.children.length > 0) {
                sessionList.children[0].click();
            } else {
                createNewSession();
            }
        }
    });

    sessionDiv.appendChild(titleSpan);
    sessionDiv.appendChild(deleteBtn);
    sessionList.prepend(sessionDiv); // Add to the top of the list

    updateActiveSessionUI();
}

async function startStream() {
    const prompt = promptInput.value;
    const file = fileInput.files[0];

    if (!prompt && !file) {
        log('warn', 'startStream called with no prompt or file.');
        return;
    }
    if (!activeSessionId) {
        log('error', 'startStream called with no active session ID.');
        return;
    }

    // Get selected model
    const selectedModel = modelSelect ? modelSelect.value : defaultModel;
    
    // Store message context for retry functionality
    lastMessageContext = {
        prompt: prompt,
        file: file,
        model: selectedModel
    };

    log('info', `Starting stream for session: ${activeSessionId}`);

    const userMessageDiv = document.createElement("div");
    userMessageDiv.className = "message user-message";
    if (prompt) {
        userMessageDiv.innerHTML += marked.parse(prompt);
    }
    if (file) {
        if (file.type.startsWith('image/')) {
            const img = document.createElement('img');
            img.src = URL.createObjectURL(file);
            img.style.maxWidth = '100%';
            img.style.height = 'auto';
            userMessageDiv.appendChild(img);
            log('info', 'Displaying user-uploaded image.');
        } 
    }
    chatContainer.appendChild(userMessageDiv);

    promptInput.value = "";
    clearFileInput();

    const aiMessageDiv = document.createElement("div");
    aiMessageDiv.className = "message ai-message";
    chatContainer.appendChild(aiMessageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;

    const formData = new FormData();
    formData.append("session_id", activeSessionId);
    formData.append("prompt", prompt);
    formData.append("model", selectedModel);
    if (file) {
        formData.append("file", file);
        log('info', `Appending file to FormData: ${file.name}`);
    }
    
    log('info', `Using model: ${selectedModel} for session: ${activeSessionId}`);

    try {
        log('info', 'Sending POST request to /stream');
        
        // Start file upload progress if file is present
        if (file && window.enhancedUIManager) {
            window.enhancedUIManager.startFileUpload();
        }
        
        const response = await fetch("/stream", {
            method: "POST",
            body: formData,
        });

        if (!response.ok) throw new Error(`Stream request failed with status ${response.status}`);

        // Transition to processing stage
        if (window.enhancedUIManager) {
            if (file) {
                // If we had a file, move to validation
                window.enhancedUIManager.startValidation();
            } else {
                // No file, go straight to processing
                window.enhancedUIManager.startProcessing();
            }
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let fullResponse = "";

        log('info', 'Starting to read from stream...');
        while (true) {
            const { value, done } = await reader.read();
            if (done) {
                log('info', 'Stream reading complete.');
                break;
            }

            buffer += decoder.decode(value, { stream: true });
            const eventMessages = buffer.split('\n\n');
            buffer = eventMessages.pop(); // Keep incomplete message in buffer

            for (const eventMessage of eventMessages) {
                if (eventMessage.startsWith("event: title")) {
                    const title = eventMessage.split("data: ")[1];
                    log('info', `Received title update: ${title}`);
                    const activeSessionDiv = document.querySelector(`.session-item[data-session-id='${activeSessionId}']`);
                    if (activeSessionDiv) {
                        activeSessionDiv.querySelector('span').textContent = title;
                        delete activeSessionDiv.dataset.isNew;
                    }
                    continue; // Move to the next message
                }
                
                // Handle enhanced UI events
                if (eventMessage.startsWith("event: upload_progress")) {
                    const data = eventMessage.split("data: ")[1];
                    try {
                        const progressData = JSON.parse(data);
                        if (window.enhancedUIManager && progressData.percentage !== undefined) {
                            window.enhancedUIManager.updateUploadProgress(progressData.percentage);
                        }
                    } catch (e) {
                        log('warn', 'Failed to parse upload progress data:', data);
                    }
                    continue;
                }
                
                if (eventMessage.startsWith("event: validation_progress")) {
                    const data = eventMessage.split("data: ")[1];
                    try {
                        const progressData = JSON.parse(data);
                        if (window.enhancedUIManager && progressData.progress !== undefined) {
                            const percentage = progressData.progress * 100;
                            const stage = progressData.stage || null;
                            window.enhancedUIManager.updateValidationProgress(percentage, stage);
                        }
                    } catch (e) {
                        log('warn', 'Failed to parse validation progress data:', data);
                    }
                    continue;
                }
                
                if (eventMessage.startsWith("event: validation_complete")) {
                    const data = eventMessage.split("data: ")[1];
                    try {
                        const validationData = JSON.parse(data);
                        if (window.enhancedUIManager) {
                            window.enhancedUIManager.completeValidation(validationData);
                            
                            // Update file preview with validation results
                            if (file && validationData) {
                                window.enhancedUIManager.filePreviewManager.updateValidationData(validationData);
                            }
                            
                            // Transition to processing if validation passed
                            if (validationData.status === 'valid') {
                                setTimeout(() => {
                                    window.enhancedUIManager.startProcessing();
                                }, 500);
                            }
                        }
                    } catch (e) {
                        log('warn', 'Failed to parse validation complete data:', data);
                    }
                    continue;
                }
                
                if (eventMessage.startsWith("event: processing_progress")) {
                    const data = eventMessage.split("data: ")[1];
                    try {
                        const progressData = JSON.parse(data);
                        if (window.enhancedUIManager && progressData.progress !== undefined) {
                            const percentage = progressData.progress * 100;
                            window.enhancedUIManager.updateProcessingProgress(percentage);
                        }
                    } catch (e) {
                        log('warn', 'Failed to parse processing progress data:', data);
                    }
                    continue;
                }
                
                if (eventMessage.startsWith("event: validation_error") || eventMessage.startsWith("event: error")) {
                    const data = eventMessage.split("data: ")[1];
                    try {
                        const errorData = JSON.parse(data);
                        
                        // Show error as chat message
                        showErrorInChat(errorData);
                        
                        // Trigger enhanced UI manager for cleanup but don't show additional error messages
                        if (window.enhancedUIManager) {
                            if (errorData.category === 'validation') {
                                // Just reset validation UI state without showing error in chat
                                window.enhancedUIManager.progressBarManager.showError();
                                window.enhancedUIManager.processingIndicatorManager.showValidationError(errorData);
                                window.enhancedUIManager.isProcessing = false;
                                window.enhancedUIManager.currentStage = 'error';
                            } else {
                                // Just reset processing UI state without showing error in chat
                                window.enhancedUIManager.progressBarManager.showError();
                                window.enhancedUIManager.processingIndicatorManager.showValidationError(errorData);
                                window.enhancedUIManager.isProcessing = false;
                                window.enhancedUIManager.currentStage = 'error';
                            }
                        }
                    } catch (e) {
                        log('warn', 'Failed to parse error event data:', data, e);
                        // Show basic error message if parsing fails
                        showErrorInChat({
                            error_code: 'PARSE_ERROR',
                            user_message: 'Error processing server response',
                            category: 'system',
                            severity: 'medium',
                            recoverable: true,
                            suggested_actions: [
                                'Try your request again',
                                'Refresh the page if problems persist'
                            ]
                        });
                    }
                    continue;
                }

                if (eventMessage.startsWith("data: ")) {
                    const data = eventMessage.substring(6);
                    if (data === "[DONE]") {
                        log('info', 'Received [DONE] signal.');
                        
                        // Complete processing with enhanced UI
                        if (window.enhancedUIManager) {
                            window.enhancedUIManager.completeProcessing();
                        }
                        
                        reader.cancel();
                        // No need to reload all sessions anymore, just ensure the state is clean
                        const activeSessionDiv = document.querySelector(`.session-item[data-session-id='${activeSessionId}']`);
                        if (activeSessionDiv && activeSessionDiv.dataset.isNew === 'true') {
                             log('warn', '[DONE] received but title was not updated. Forcing a reload.');
                             await loadSessions();
                             updateActiveSessionUI();
                        }
                        return;
                    }
                    if (data.startsWith("An error occurred:") || data.startsWith("{")) {
                        // Handle both legacy string errors and new structured error data
                        if (data.startsWith("{")) {
                            try {
                                const errorData = JSON.parse(data);
                                if (errorData.error_code || errorData.user_message) {
                                    log('error', `Structured server error:`, errorData);
                                    
                                    // Show error in chat instead of existing AI message
                                    aiMessageDiv.remove(); // Remove the empty AI message div
                                    showErrorInChat(errorData);
                                    reader.cancel();
                                    return;
                                }
                            } catch (e) {
                                // Not valid JSON, fall through to legacy handling
                            }
                        }
                        
                        // Legacy error handling
                        log('error', `Server-side error: ${data}`);
                        aiMessageDiv.style.color = 'red';
                        aiMessageDiv.textContent = data;
                        if (window.errorHandler) {
                            window.errorHandler.handleGenericError(data, {
                                category: 'llm',
                                retryHandler: () => startStream(),
                                userMessage: 'The AI service encountered an error. Please try again.',
                                suggestedActions: [
                                    'Try sending your message again',
                                    'Try with a shorter message',
                                    'Wait a moment and retry'
                                ]
                            });
                        }
                        reader.cancel();
                        return;
                    } else {
                        fullResponse += data;
                        aiMessageDiv.innerHTML = marked.parse(fullResponse);
                        setTimeout(() => {
                            aiMessageDiv.querySelectorAll('pre code').forEach(block => {
                                hljs.highlightElement(block);
                            });
                        }, 0); // Small delay to allow DOM to update
                        chatContainer.scrollTop = chatContainer.scrollHeight;
                    }
                }
            }
        }
    } catch (error) {
        log('error', 'Fetch or stream processing error:', error);
        
        // Show network error in chat
        aiMessageDiv.remove(); // Remove the empty AI message div
        showErrorInChat({
            error_code: 'NETWORK_ERROR',
            user_message: 'Connection lost while streaming. Please check your connection and try again.',
            category: 'network',
            severity: 'high',
            recoverable: true,
            suggested_actions: [
                'Check your internet connection',
                'Try sending your message again',
                'Refresh the page if problems persist'
            ]
        });
        
        // Also reset enhanced UI
        if (window.enhancedUIManager) {
            window.enhancedUIManager.handleProcessingError({
                error_code: 'NETWORK_ERROR',
                user_message: 'Connection lost while streaming. Please try again.',
                category: 'network',
                severity: 'high',
                recoverable: true
            });
        }
    }
}

function updateActiveSessionUI() {
    const allSessions = document.querySelectorAll(".session-item");
    allSessions.forEach(s => {
        if (s.dataset.sessionId === activeSessionId) {
            s.classList.add("active");
        } else {
            s.classList.remove("active");
        }
    });
    log('info', `UI updated for active session: ${activeSessionId}`);
}

function handleFileSelection() {
    if (fileInput.files.length > 0) {
        const file = fileInput.files[0];
        log('info', `File selected: ${file.name}, type: ${file.type}`);
        
        // Use enhanced UI manager for file display
        if (window.enhancedUIManager) {
            window.enhancedUIManager.handleFileSelection(file);
        } else {
            // Fallback to basic display
            fileNameSpan.textContent = file.name;
            filePreviewContainer.style.display = 'flex';
        }
    } else {
        log('info', 'File selection cleared.');
        
        // Use enhanced UI manager for file removal
        if (window.enhancedUIManager) {
            window.enhancedUIManager.handleFileRemoval();
        } else {
            // Fallback to basic hiding
            filePreviewContainer.style.display = 'none';
        }
    }
}

function clearFileInput() {
    fileInput.value = "";
    handleFileSelection();
}

// Retry last message function
async function retryLastMessage() {
    if (!lastMessageContext.prompt && !lastMessageContext.file) {
        log('warn', 'No previous message context available for retry');
        return;
    }

    log('info', 'Retrying last message:', {
        prompt: lastMessageContext.prompt || '(no text)',
        hasFile: !!lastMessageContext.file
    });

    // Restore the previous message context to input fields
    if (lastMessageContext.prompt) {
        promptInput.value = lastMessageContext.prompt;
    }
    
    if (lastMessageContext.file) {
        // Create a new FileList with the stored file
        const dt = new DataTransfer();
        dt.items.add(lastMessageContext.file);
        fileInput.files = dt.files;
        handleFileSelection(); // Update UI to show the file
    }
    
    if (lastMessageContext.model && modelSelect) {
        modelSelect.value = lastMessageContext.model;
    }

    // Start the stream with the restored context
    await startStream();
}

// Global file removal handler for enhanced UI components
window.handleFileRemoval = clearFileInput;

// Make error chat function globally available
window.showErrorInChat = showErrorInChat;

// Make retry function globally available
window.retryLastMessage = retryLastMessage;

// Global test function to demo chat error messages
window.testChatError = () => {
    showErrorInChat({
        error_code: 'TEST_ERROR',
        category: 'validation',
        user_message: 'This is a test error message displayed as a chat message instead of a popup.',
        suggested_actions: [
            'This error appears inline in the chat',
            'It looks like a regular AI message but with error styling',
            'You can include helpful suggestions here'
        ],
        recoverable: true
    });
};

// Function to display errors as chat messages
function showErrorInChat(errorData) {
    const errorMessageDiv = document.createElement("div");
    errorMessageDiv.className = "message ai-message error-message";
    
    // Create error content with structured layout
    const errorContent = document.createElement("div");
    errorContent.className = "error-content";
    
    // Error title/summary
    const errorTitle = document.createElement("div");
    errorTitle.className = "error-title";
    errorTitle.textContent = errorData.user_message || errorData.message || 'An error occurred';
    errorContent.appendChild(errorTitle);
    
    // Error details if available
    if (errorData.error_code || errorData.category) {
        const errorDetails = document.createElement("div");
        errorDetails.className = "error-details";
        errorDetails.innerHTML = `
            ${errorData.error_code ? `<span class="error-code">Code: ${errorData.error_code}</span>` : ''}
            ${errorData.category ? `<span class="error-category">Type: ${errorData.category}</span>` : ''}
        `;
        errorContent.appendChild(errorDetails);
    }
    
    // Suggested actions if provided
    if (errorData.suggested_actions && errorData.suggested_actions.length > 0) {
        const suggestionsTitle = document.createElement("div");
        suggestionsTitle.className = "error-suggestions-title";
        suggestionsTitle.textContent = "What you can do:";
        errorContent.appendChild(suggestionsTitle);
        
        const suggestionsList = document.createElement("ul");
        suggestionsList.className = "error-suggestions";
        errorData.suggested_actions.forEach(action => {
            const item = document.createElement("li");
            item.textContent = action;
            suggestionsList.appendChild(item);
        });
        errorContent.appendChild(suggestionsList);
    }
    
    // Retry button if recoverable
    if (errorData.recoverable) {
        const retryButton = document.createElement("button");
        retryButton.className = "retry-button";
        retryButton.textContent = "Try Again";
        retryButton.onclick = () => {
            // Remove the error message
            errorMessageDiv.remove();
            
            // Retry the last message automatically
            retryLastMessage();
        };
        errorContent.appendChild(retryButton);
    }
    
    errorMessageDiv.appendChild(errorContent);
    chatContainer.appendChild(errorMessageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    
    log('info', 'Error displayed in chat:', errorData.error_code || 'unknown');
}

// --- Initial Load and Event Listeners ---
document.addEventListener("DOMContentLoaded", async () => {
    log('info', 'DOM fully loaded and parsed.');
    chatContainer = document.getElementById("chat-container");
    promptInput = document.getElementById("prompt");
    sessionList = document.getElementById("session-list");
    fileInput = document.getElementById("file-input");
    newChatBtn = document.getElementById("new-chat-btn");
    attachFileBtn = document.getElementById("attach-file-btn");
    sendBtn = document.getElementById("send-btn");
    filePreviewContainer = document.getElementById('file-preview-container');
    fileNameSpan = document.getElementById('file-name');
    removeFileBtn = document.getElementById('remove-file-btn');
    modelSelect = document.getElementById('model-select');

    log('info', 'All UI elements have been captured.');

    // Initialize model selection
    await loadAvailableModels();

    newChatBtn.addEventListener("click", createNewSession);
    attachFileBtn.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelection);
    removeFileBtn.addEventListener('click', clearFileInput);
    sendBtn.addEventListener("click", startStream);
    promptInput.addEventListener("keyup", (event) => {
        if (event.key === "Enter") {
            event.preventDefault();
            startStream();
        }
    });
    
    // Model selection change handler
    if (modelSelect) {
        modelSelect.addEventListener('change', (event) => {
            const selectedModel = event.target.value;
            savePreferredModel(selectedModel);
            log('info', `Model changed to: ${selectedModel}`);
        });
    }

    log('info', 'Event listeners have been attached.');

    await loadSessions(); // Wait for sessions to load first

    // If no sessions were loaded, or no active session is set, create a new one.
    if (sessionList.children.length === 0) {
        createNewSession();
    } else {
        // Optionally, load the first session in the list
        const firstSession = sessionList.children[0];
        if (firstSession) {
            loadSessionHistory(firstSession.dataset.sessionId);
        }
    }
});

