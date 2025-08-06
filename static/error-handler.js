/**
 * Frontend Error Handling System (CARD_003)
 * 
 * Centralized error handling for the frontend with user-friendly error display,
 * toast notifications, error recovery mechanisms, and integration with backend
 * error streaming.
 */

class ErrorSeverity {
    static LOW = 'low';
    static MEDIUM = 'medium';
    static HIGH = 'high';
    static CRITICAL = 'critical';
}

class ErrorCategory {
    static VALIDATION = 'validation';
    static SESSION = 'session';
    static LLM = 'llm';
    static NETWORK = 'network';
    static CONFIGURATION = 'configuration';
    static SYSTEM = 'system';
}

/**
 * Toast notification component for displaying temporary error messages
 */
class ToastNotification {
    constructor() {
        this.container = null;
        this.init();
    }

    init() {
        // Create toast container if it doesn't exist
        this.container = document.getElementById('toast-container');
        if (!this.container) {
            // Ensure document.body exists before trying to append
            if (document.body) {
                this.container = document.createElement('div');
                this.container.id = 'toast-container';
                this.container.className = 'toast-container';
                document.body.appendChild(this.container);
            } else {
                // If body isn't ready, wait for DOM to be ready
                const initToastContainer = () => {
                    if (document.body) {
                        this.container = document.createElement('div');
                        this.container.id = 'toast-container';
                        this.container.className = 'toast-container';
                        document.body.appendChild(this.container);
                    }
                };
                
                if (document.readyState === 'loading') {
                    document.addEventListener('DOMContentLoaded', initToastContainer);
                } else {
                    // DOM is already loaded, try again in next tick
                    setTimeout(initToastContainer, 0);
                }
            }
        }
    }

    show(message, severity = ErrorSeverity.MEDIUM, duration = 5000, actions = []) {
        // Toast notifications disabled - all errors should go to chat
        console.log('Toast notification blocked:', message);
        return null;
    }

    hide(toast) {
        toast.classList.remove('toast-show');
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, 300); // Animation duration
    }

    clear() {
        if (this.container) {
            this.container.innerHTML = '';
        }
    }
}

/**
 * Error panel component for displaying persistent error information
 */
class ErrorPanel {
    constructor() {
        this.panel = null;
        this.init();
    }

    init() {
        this.panel = document.createElement('div');
        this.panel.id = 'error-panel';
        this.panel.className = 'error-panel hidden';
        document.body.appendChild(this.panel);
    }

    show(errorResult, onRetry = null, onDismiss = null) {
        this.panel.innerHTML = '';
        this.panel.className = `error-panel error-${errorResult.severity}`;

        // Error header
        const header = document.createElement('div');
        header.className = 'error-header';
        
        const title = document.createElement('h3');
        title.textContent = this.getErrorTitle(errorResult.category);
        header.appendChild(title);

        const closeBtn = document.createElement('button');
        closeBtn.className = 'error-close';
        closeBtn.innerHTML = '&times;';
        closeBtn.onclick = () => {
            this.hide();
            if (onDismiss) onDismiss();
        };
        header.appendChild(closeBtn);

        this.panel.appendChild(header);

        // Error message
        const message = document.createElement('div');
        message.className = 'error-message';
        message.textContent = errorResult.user_message;
        this.panel.appendChild(message);

        // Suggested actions
        if (errorResult.suggested_actions && errorResult.suggested_actions.length > 0) {
            const actionsTitle = document.createElement('h4');
            actionsTitle.textContent = 'What you can do:';
            this.panel.appendChild(actionsTitle);

            const actionsList = document.createElement('ul');
            actionsList.className = 'error-actions';
            errorResult.suggested_actions.forEach(action => {
                const item = document.createElement('li');
                item.textContent = action;
                actionsList.appendChild(item);
            });
            this.panel.appendChild(actionsList);
        }

        // Action buttons
        const buttonContainer = document.createElement('div');
        buttonContainer.className = 'error-buttons';

        if (errorResult.recoverable && onRetry) {
            const retryBtn = document.createElement('button');
            retryBtn.className = 'error-btn error-btn-retry';
            retryBtn.textContent = errorResult.retry_after 
                ? `Retry in ${errorResult.retry_after}s` 
                : 'Try Again';
            retryBtn.onclick = () => {
                this.hide();
                onRetry();
            };
            buttonContainer.appendChild(retryBtn);

            // Handle retry countdown
            if (errorResult.retry_after) {
                this.startRetryCountdown(retryBtn, errorResult.retry_after);
            }
        }

        const dismissBtn = document.createElement('button');
        dismissBtn.className = 'error-btn error-btn-dismiss';
        dismissBtn.textContent = 'Dismiss';
        dismissBtn.onclick = () => {
            this.hide();
            if (onDismiss) onDismiss();
        };
        buttonContainer.appendChild(dismissBtn);

        this.panel.appendChild(buttonContainer);

        // Show panel
        this.panel.classList.remove('hidden');
    }

    hide() {
        this.panel.classList.add('hidden');
    }

    getErrorTitle(category) {
        const titles = {
            [ErrorCategory.VALIDATION]: 'File Validation Error',
            [ErrorCategory.SESSION]: 'Session Error',
            [ErrorCategory.LLM]: 'AI Service Error',
            [ErrorCategory.NETWORK]: 'Connection Error',
            [ErrorCategory.CONFIGURATION]: 'Service Configuration Error',
            [ErrorCategory.SYSTEM]: 'System Error'
        };
        return titles[category] || 'Error';
    }

    startRetryCountdown(button, seconds) {
        let remaining = seconds;
        button.disabled = true;
        
        const interval = setInterval(() => {
            remaining--;
            button.textContent = `Retry in ${remaining}s`;
            
            if (remaining <= 0) {
                clearInterval(interval);
                button.textContent = 'Try Again';
                button.disabled = false;
            }
        }, 1000);
    }
}

/**
 * Error state manager for persistent error tracking
 */
class ErrorStateManager {
    constructor() {
        this.errors = new Map();
        this.listeners = [];
    }

    addError(errorId, errorResult) {
        this.errors.set(errorId, {
            ...errorResult,
            timestamp: Date.now(),
            dismissed: false
        });
        this.notifyListeners('error_added', errorId, errorResult);
    }

    removeError(errorId) {
        const error = this.errors.get(errorId);
        if (error) {
            this.errors.delete(errorId);
            this.notifyListeners('error_removed', errorId, error);
        }
    }

    dismissError(errorId) {
        const error = this.errors.get(errorId);
        if (error) {
            error.dismissed = true;
            this.notifyListeners('error_dismissed', errorId, error);
        }
    }

    getError(errorId) {
        return this.errors.get(errorId);
    }

    getAllErrors() {
        return Array.from(this.errors.entries()).map(([id, error]) => ({ id, ...error }));
    }

    getActiveErrors() {
        return this.getAllErrors().filter(error => !error.dismissed);
    }

    clearOldErrors(maxAge = 5 * 60 * 1000) { // 5 minutes default
        const now = Date.now();
        for (const [id, error] of this.errors.entries()) {
            if (now - error.timestamp > maxAge) {
                this.removeError(id);
            }
        }
    }

    addEventListener(listener) {
        this.listeners.push(listener);
    }

    removeEventListener(listener) {
        const index = this.listeners.indexOf(listener);
        if (index > -1) {
            this.listeners.splice(index, 1);
        }
    }

    notifyListeners(event, errorId, errorData) {
        this.listeners.forEach(listener => {
            try {
                listener(event, errorId, errorData);
            } catch (e) {
                console.error('Error in error state listener:', e);
            }
        });
    }
}

/**
 * Main error handler that orchestrates error processing and display
 */
class FrontendErrorHandler {
    constructor() {
        this.toast = new ToastNotification();
        this.panel = new ErrorPanel();
        this.state = new ErrorStateManager();
        this.retryHandlers = new Map();
        this.init();
    }

    init() {
        // Set up global error handlers
        window.addEventListener('error', (event) => {
            this.handleJavaScriptError(event.error, event.filename, event.lineno);
        });

        window.addEventListener('unhandledrejection', (event) => {
            this.handleUnhandledPromiseRejection(event.reason);
        });

        // Clean up old errors periodically
        setInterval(() => {
            this.state.clearOldErrors();
        }, 60000); // Every minute
    }

    /**
     * Process a structured error result from the backend
     */
    handleErrorResult(errorResult, context = {}) {
        const errorId = errorResult.context?.error_id || this.generateErrorId();
        
        // Add to state
        this.state.addError(errorId, errorResult);

        // Route to chat message instead of popups/toasts
        if (window.showErrorInChat) {
            window.showErrorInChat(errorResult);
        } else {
            console.log('Error would be displayed in chat:', errorResult);
        }

        // Set up retry mechanism if applicable
        if (errorResult.recoverable && context.retryHandler) {
            this.retryHandlers.set(errorId, context.retryHandler);
        }

        return errorId;
    }

    /**
     * Handle SSE error events from the backend
     */
    handleSSEError(eventData, context = {}) {
        try {
            const errorResult = JSON.parse(eventData);
            return this.handleErrorResult(errorResult, { ...context, source: 'sse' });
        } catch (e) {
            // Fallback for non-JSON error data
            return this.handleGenericError(eventData, { ...context, source: 'sse' });
        }
    }

    /**
     * Handle validation errors with file context
     */
    handleValidationError(error, filename = null, context = {}) {
        const errorResult = {
            error_code: 'VALIDATION_ERROR',
            severity: ErrorSeverity.MEDIUM,
            category: ErrorCategory.VALIDATION,
            user_message: error.message || 'File validation failed',
            suggested_actions: [
                'Check your file format and size',
                'Try a different file',
                'Contact support if the problem persists'
            ],
            recoverable: false,
            context: {
                error_id: this.generateErrorId(),
                timestamp: new Date().toISOString(),
                filename: filename
            }
        };

        return this.handleErrorResult(errorResult, { ...context, showPanel: true });
    }

    /**
     * Handle network errors with retry capability
     */
    handleNetworkError(error, context = {}) {
        const errorResult = {
            error_code: 'NETWORK_ERROR',
            severity: ErrorSeverity.HIGH,
            category: ErrorCategory.NETWORK,
            user_message: 'Network connection failed. Please check your internet connection.',
            suggested_actions: [
                'Check your internet connection',
                'Try again in a few moments',
                'Refresh the page if the problem persists'
            ],
            recoverable: true,
            retry_after: 30,
            context: {
                error_id: this.generateErrorId(),
                timestamp: new Date().toISOString(),
                original_error: error.message
            }
        };

        return this.handleErrorResult(errorResult, context);
    }

    /**
     * Handle generic JavaScript errors
     */
    handleJavaScriptError(error, filename = null, lineno = null) {
        const errorResult = {
            error_code: 'JS_ERROR',
            severity: ErrorSeverity.MEDIUM,
            category: ErrorCategory.SYSTEM,
            user_message: 'An unexpected error occurred. The page may need to be refreshed.',
            suggested_actions: [
                'Refresh the page',
                'Try again',
                'Contact support if the problem continues'
            ],
            recoverable: false,
            context: {
                error_id: this.generateErrorId(),
                timestamp: new Date().toISOString(),
                filename: filename,
                lineno: lineno,
                stack: error.stack
            }
        };

        return this.handleErrorResult(errorResult, { showPanel: false });
    }

    /**
     * Handle unhandled promise rejections
     */
    handleUnhandledPromiseRejection(reason) {
        const errorResult = {
            error_code: 'UNHANDLED_PROMISE',
            severity: ErrorSeverity.HIGH,
            category: ErrorCategory.SYSTEM,
            user_message: 'An unexpected error occurred. Please try again.',
            suggested_actions: [
                'Try your last action again',
                'Refresh the page if needed',
                'Contact support if the problem persists'
            ],
            recoverable: true,
            context: {
                error_id: this.generateErrorId(),
                timestamp: new Date().toISOString(),
                reason: String(reason)
            }
        };

        return this.handleErrorResult(errorResult);
    }

    /**
     * Handle generic errors with fallback
     */
    handleGenericError(error, context = {}) {
        const message = error.message || error || 'An unknown error occurred';
        const errorResult = {
            error_code: 'GENERIC_ERROR',
            severity: context.severity || ErrorSeverity.MEDIUM,
            category: context.category || ErrorCategory.SYSTEM,
            user_message: context.userMessage || message,
            suggested_actions: context.suggestedActions || [
                'Try again',
                'Refresh the page',
                'Contact support if needed'
            ],
            recoverable: context.recoverable !== false,
            context: {
                error_id: this.generateErrorId(),
                timestamp: new Date().toISOString(),
                original_error: message,
                source: context.source || 'generic'
            }
        };

        return this.handleErrorResult(errorResult, context);
    }

    showErrorToast(errorResult, context = {}) {
        // Don't show toast notifications - let chat messages handle errors
        console.log('Toast notification suppressed in favor of chat message');
        return null;
    }

    showErrorPanel(errorResult, context = {}) {
        // Don't show error panels - let chat messages handle errors
        console.log('Error panel suppressed in favor of chat message');
        return null;
    }

    retry(errorId) {
        const retryHandler = this.retryHandlers.get(errorId);
        if (retryHandler) {
            try {
                retryHandler();
                this.state.removeError(errorId);
                this.retryHandlers.delete(errorId);
            } catch (e) {
                console.error('Error during retry:', e);
                this.handleGenericError(e, { severity: ErrorSeverity.HIGH });
            }
        }
    }

    getToastDuration(severity) {
        const durations = {
            [ErrorSeverity.LOW]: 3000,
            [ErrorSeverity.MEDIUM]: 5000,
            [ErrorSeverity.HIGH]: 8000,
            [ErrorSeverity.CRITICAL]: 0 // Don't auto-hide critical errors
        };
        return durations[severity] || 5000;
    }

    generateErrorId() {
        return 'error_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }

    // Public API methods
    clearAll() {
        this.toast.clear();
        this.panel.hide();
        this.state.errors.clear();
        this.retryHandlers.clear();
    }

    getErrorState() {
        return this.state;
    }
}

// Global error handler instance
window.errorHandler = new FrontendErrorHandler();

// Export for testing
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        ErrorSeverity,
        ErrorCategory,
        ToastNotification,
        ErrorPanel,
        ErrorStateManager,
        FrontendErrorHandler
    };
}