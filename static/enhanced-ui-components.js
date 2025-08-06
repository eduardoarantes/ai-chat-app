/**
 * Enhanced UI Components for File Processing Feedback (TASK-003)
 * 
 * This module provides enhanced UI components for file processing:
 * - FilePreviewManager: Enhanced file display with metadata (CARD_001)
 * - ProgressBarManager: Upload and validation progress bars (CARD_003)
 * - ProcessingIndicatorManager: Animated processing indicators (CARD_002)
 * 
 * Integration with TASK-001 (validation data) and TASK-002 (error handling)
 */

/**
 * Utility function to format file sizes
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

/**
 * Utility function to get file type icon class
 */
function getFileTypeIcon(mimeType) {
    if (!mimeType) return 'generic-icon';
    
    if (mimeType.startsWith('image/')) return 'image-icon';
    if (mimeType === 'application/pdf') return 'pdf-icon';
    if (mimeType.startsWith('text/') || mimeType === 'application/json') return 'text-icon';
    if (mimeType === 'application/json') return 'json-icon';
    
    return 'generic-icon';
}

/**
 * Utility function to get security score classification
 */
function getSecurityClassification(score) {
    if (score >= 0.8) return 'security-high';
    if (score >= 0.6) return 'security-medium';
    if (score >= 0.3) return 'security-low';
    return 'security-critical';
}

/**
 * Enhanced File Preview Manager (CARD_001)
 * 
 * Manages enhanced file preview display with metadata, security indicators,
 * and integration with TASK-001 validation results.
 */
class FilePreviewManager {
    constructor() {
        this.previewContainer = document.getElementById('file-preview-container');
        this.metadataPanel = document.getElementById('file-metadata-panel');
        this.metadataToggleBtn = document.getElementById('metadata-toggle-btn');
        this.currentFile = null;
        this.validationData = null;
        this.isMetadataExpanded = false;
        
        this.initializeEventListeners();
    }
    
    initializeEventListeners() {
        // Metadata toggle functionality
        if (this.metadataToggleBtn) {
            this.metadataToggleBtn.addEventListener('click', () => {
                this.toggleMetadataPanel();
            });
        }
        
        // Remove file button
        const removeBtn = document.getElementById('remove-file-btn');
        if (removeBtn) {
            removeBtn.addEventListener('click', () => {
                this.hidePreview();
                // Trigger file removal in main script
                if (window.handleFileRemoval) {
                    window.handleFileRemoval();
                }
            });
        }
    }
    
    /**
     * Display file with optional validation result from TASK-001
     */
    displayFile(file, validationResult = null) {
        this.currentFile = file;
        this.validationData = validationResult;
        
        this.renderFilePreview();
        this.renderFileDetails();
        
        if (validationResult) {
            this.renderSecurityIndicators();
            this.renderMetadataPanel();
            this.metadataToggleBtn.style.display = 'block';
        }
        
        this.showPreview();
    }
    
    /**
     * Render basic file preview elements
     */
    renderFilePreview() {
        const fileIcon = document.getElementById('file-type-icon');
        const fileName = document.getElementById('file-name');
        
        if (fileIcon && this.currentFile) {
            const iconClass = getFileTypeIcon(this.currentFile.type);
            fileIcon.className = `file-type-icon ${iconClass}`;
        }
        
        if (fileName && this.currentFile) {
            fileName.textContent = this.currentFile.name;
            fileName.title = this.currentFile.name; // Tooltip for long names
        }
    }
    
    /**
     * Render file details (size, type)
     */
    renderFileDetails() {
        const fileSize = document.getElementById('file-size');
        const fileType = document.getElementById('file-type');
        
        if (fileSize && this.currentFile) {
            fileSize.textContent = formatFileSize(this.currentFile.size);
        }
        
        if (fileType && this.currentFile) {
            const typeDisplay = this.currentFile.type || 'Unknown';
            fileType.textContent = typeDisplay;
        }
    }
    
    /**
     * Render security indicators from TASK-001 validation
     */
    renderSecurityIndicators() {
        if (!this.validationData || typeof this.validationData.security_score === 'undefined') {
            return;
        }
        
        const securityIndicator = document.getElementById('security-indicator');
        const securityFill = document.getElementById('security-score-fill');
        const securityText = document.getElementById('security-score-text');
        
        if (securityIndicator && securityFill && securityText) {
            const score = this.validationData.security_score;
            const percentage = Math.round(score * 100);
            const classification = getSecurityClassification(score);
            
            // Update security bar
            securityFill.style.width = `${percentage}%`;
            securityFill.className = `security-score-fill ${classification}`;
            
            // Update security text
            securityText.textContent = `${percentage}%`;
            
            // Show security indicator
            securityIndicator.style.display = 'flex';
        }
    }
    
    /**
     * Render expandable metadata panel
     */
    renderMetadataPanel() {
        if (!this.validationData) return;
        
        const fileHash = document.getElementById('file-hash');
        const fileMime = document.getElementById('file-mime');
        const detailedSecurityScore = document.getElementById('detailed-security-score');
        const validationResults = document.getElementById('validation-results');
        const validationDetails = document.getElementById('validation-details');
        
        // Update basic metadata
        if (fileHash && this.validationData.file_hash) {
            fileHash.textContent = this.validationData.file_hash.substring(0, 16) + '...';
            fileHash.title = this.validationData.file_hash; // Full hash in tooltip
        }
        
        if (fileMime && this.validationData.mime_type) {
            fileMime.textContent = this.validationData.mime_type;
        }
        
        if (detailedSecurityScore && typeof this.validationData.security_score !== 'undefined') {
            const score = Math.round(this.validationData.security_score * 100);
            const classification = getSecurityClassification(this.validationData.security_score);
            detailedSecurityScore.textContent = `${score}% (${classification.replace('security-', '')})`;
            detailedSecurityScore.className = `metadata-value security-score ${classification}`;
        }
        
        // Update validation results
        if (validationResults && validationDetails && this.validationData.validation_errors) {
            validationDetails.innerHTML = '';
            
            if (this.validationData.validation_errors.length === 0) {
                // Show success state
                const successDiv = document.createElement('div');
                successDiv.className = 'validation-success';
                successDiv.textContent = 'File validation passed successfully';
                validationDetails.appendChild(successDiv);
            } else {
                // Show errors
                this.validationData.validation_errors.forEach(error => {
                    const errorDiv = document.createElement('div');
                    errorDiv.className = 'validation-error';
                    errorDiv.textContent = error;
                    validationDetails.appendChild(errorDiv);
                });
            }
            
            validationResults.style.display = 'block';
        }
    }
    
    /**
     * Toggle metadata panel visibility
     */
    toggleMetadataPanel() {
        if (!this.metadataPanel || !this.metadataToggleBtn) return;
        
        this.isMetadataExpanded = !this.isMetadataExpanded;
        
        if (this.isMetadataExpanded) {
            this.metadataPanel.style.display = 'block';
            this.metadataToggleBtn.classList.add('expanded');
        } else {
            this.metadataPanel.style.display = 'none';
            this.metadataToggleBtn.classList.remove('expanded');
        }
    }
    
    /**
     * Show the file preview container
     */
    showPreview() {
        if (this.previewContainer) {
            this.previewContainer.style.display = 'block';
        }
    }
    
    /**
     * Hide the file preview container
     */
    hidePreview() {
        if (this.previewContainer) {
            this.previewContainer.style.display = 'none';
        }
        
        // Reset metadata panel state
        if (this.metadataPanel) {
            this.metadataPanel.style.display = 'none';
        }
        
        if (this.metadataToggleBtn) {
            this.metadataToggleBtn.style.display = 'none';
            this.metadataToggleBtn.classList.remove('expanded');
        }
        
        this.isMetadataExpanded = false;
        this.currentFile = null;
        this.validationData = null;
    }
    
    /**
     * Update file with new validation data
     */
    updateValidationData(validationResult) {
        this.validationData = validationResult;
        this.renderSecurityIndicators();
        this.renderMetadataPanel();
        
        if (validationResult && this.metadataToggleBtn) {
            this.metadataToggleBtn.style.display = 'block';
        }
    }
}

/**
 * Progress Bar Manager (CARD_003)
 * 
 * Manages upload, validation, and processing progress bars with
 * real-time updates via SSE integration.
 */
class ProgressBarManager {
    constructor() {
        this.progressContainer = document.getElementById('progress-container');
        this.uploadProgressWrapper = document.getElementById('upload-progress');
        this.validationProgressWrapper = document.getElementById('validation-progress');
        this.processingProgressWrapper = document.getElementById('processing-progress');
        
        this.uploadProgressFill = document.getElementById('upload-progress-fill');
        this.validationProgressFill = document.getElementById('validation-progress-fill');
        this.processingProgressFill = document.getElementById('processing-progress-fill');
        
        this.uploadPercentage = document.getElementById('upload-percentage');
        this.validationPercentage = document.getElementById('validation-percentage');
        this.processingPercentage = document.getElementById('processing-percentage');
        
        this.validationStageText = document.getElementById('validation-stage-text');
        
        this.currentStage = 'idle';
        this.stageProgress = {
            upload: 0,
            validation: 0,
            processing: 0
        };
    }
    
    /**
     * Start upload progress display
     */
    startUpload() {
        this.currentStage = 'upload';
        this.showProgressContainer();
        this.showUploadProgress();
        this.updateProgress('upload', 0, 'Starting upload...');
    }
    
    /**
     * Transition to validation stage
     */
    transitionToValidation() {
        this.currentStage = 'validation';
        this.hideUploadProgress();
        this.showValidationProgress();
        this.updateProgress('validation', 0, 'Starting validation...');
    }
    
    /**
     * Transition to processing stage
     */
    transitionToProcessing() {
        this.currentStage = 'processing';
        this.hideValidationProgress();
        this.showProcessingProgress();
        this.updateProgress('processing', 0, 'Starting processing...');
    }
    
    /**
     * Update progress for specific stage
     */
    updateProgress(stage, percentage, message = null) {
        this.stageProgress[stage] = Math.max(0, Math.min(100, percentage));
        
        const progressFill = this[`${stage}ProgressFill`];
        const percentageElement = this[`${stage}Percentage`];
        
        if (progressFill) {
            progressFill.style.width = `${this.stageProgress[stage]}%`;
        }
        
        if (percentageElement) {
            percentageElement.textContent = `${Math.round(this.stageProgress[stage])}%`;
        }
        
        // Update stage-specific messages
        if (message && stage === 'validation' && this.validationStageText) {
            this.validationStageText.textContent = message;
        }
    }
    
    /**
     * Show completion state
     */
    showCompletion() {
        this.currentStage = 'complete';
        this.hideAllProgressBars();
        
        // Could show completion animation here
        setTimeout(() => {
            this.hideProgressContainer();
        }, 1500);
    }
    
    /**
     * Show error state
     */
    showError(errorMessage = 'Processing failed') {
        this.currentStage = 'error';
        // Error handling is managed by the error handling system (TASK-002)
        // This just hides the progress bars
        this.hideAllProgressBars();
        
        setTimeout(() => {
            this.hideProgressContainer();
        }, 500);
    }
    
    /**
     * Reset all progress
     */
    resetProgress() {
        this.currentStage = 'idle';
        this.stageProgress = { upload: 0, validation: 0, processing: 0 };
        this.hideAllProgressBars();
        this.hideProgressContainer();
    }
    
    // UI Helper Methods
    showProgressContainer() {
        if (this.progressContainer) {
            this.progressContainer.style.display = 'block';
        }
    }
    
    hideProgressContainer() {
        if (this.progressContainer) {
            this.progressContainer.style.display = 'none';
        }
    }
    
    showUploadProgress() {
        if (this.uploadProgressWrapper) {
            this.uploadProgressWrapper.style.display = 'block';
        }
    }
    
    hideUploadProgress() {
        if (this.uploadProgressWrapper) {
            this.uploadProgressWrapper.style.display = 'none';
        }
    }
    
    showValidationProgress() {
        if (this.validationProgressWrapper) {
            this.validationProgressWrapper.style.display = 'block';
        }
    }
    
    hideValidationProgress() {
        if (this.validationProgressWrapper) {
            this.validationProgressWrapper.style.display = 'none';
        }
    }
    
    showProcessingProgress() {
        if (this.processingProgressWrapper) {
            this.processingProgressWrapper.style.display = 'block';
        }
    }
    
    hideProcessingProgress() {
        if (this.processingProgressWrapper) {
            this.processingProgressWrapper.style.display = 'none';
        }
    }
    
    hideAllProgressBars() {
        this.hideUploadProgress();
        this.hideValidationProgress();
        this.hideProcessingProgress();
    }
}

/**
 * Processing Indicator Manager (CARD_002)
 * 
 * Manages animated processing indicators, spinners, and status messages
 * with integration to SSE real-time updates.
 */
class ProcessingIndicatorManager {
    constructor() {
        this.indicatorsContainer = document.getElementById('processing-indicators');
        this.processingSpinner = document.getElementById('processing-spinner');
        this.processingMessage = document.getElementById('processing-message');
        this.processingStatus = document.getElementById('processing-status');
        
        this.currentState = 'idle';
        this.isVisible = false;
    }
    
    /**
     * Show processing state with message and status
     */
    showProcessingState(state, message = null, statusText = null) {
        this.currentState = state;
        this.showIndicators();
        
        if (message && this.processingMessage) {
            this.processingMessage.textContent = message;
        }
        
        if (statusText && this.processingStatus) {
            this.processingStatus.textContent = statusText;
            this.processingStatus.className = `processing-status ${state}`;
        }
        
        // Show appropriate spinner state
        if (this.processingSpinner) {
            this.processingSpinner.style.display = state === 'complete' ? 'none' : 'block';
        }
    }
    
    /**
     * Update processing message without changing state
     */
    updateProcessingMessage(message) {
        if (this.processingMessage) {
            this.processingMessage.textContent = message;
        }
    }
    
    /**
     * Show validation success indicators
     */
    showValidationSuccess(validationResult) {
        this.showProcessingState('complete', 'Validation completed successfully', 'Valid');
        
        setTimeout(() => {
            this.hideIndicators();
        }, 2000);
    }
    
    /**
     * Show validation error indicators
     */
    showValidationError(errorResult) {
        const message = errorResult.user_message || 'Validation failed';
        this.showProcessingState('error', message, 'Error');
        
        // Error indicators stay visible longer to allow user action
        setTimeout(() => {
            this.hideIndicators();
        }, 5000);
    }
    
    /**
     * Show LLM processing indicators
     */
    showLLMProcessing(message = 'Processing with AI...') {
        this.showProcessingState('processing', message, 'Processing');
    }
    
    /**
     * Show completion indicators
     */
    showCompletion(message = 'Processing completed') {
        this.showProcessingState('complete', message, 'Complete');
        
        setTimeout(() => {
            this.hideIndicators();
        }, 2000);
    }
    
    /**
     * Show processing indicators container
     */
    showIndicators() {
        if (this.indicatorsContainer && !this.isVisible) {
            this.indicatorsContainer.style.display = 'flex';
            this.isVisible = true;
        }
    }
    
    /**
     * Hide processing indicators container
     */
    hideIndicators() {
        if (this.indicatorsContainer && this.isVisible) {
            this.indicatorsContainer.style.display = 'none';
            this.isVisible = false;
            this.currentState = 'idle';
        }
    }
    
    /**
     * Reset indicators to initial state
     */
    resetIndicators() {
        this.hideIndicators();
        this.currentState = 'idle';
        
        if (this.processingMessage) {
            this.processingMessage.textContent = '';
        }
        
        if (this.processingStatus) {
            this.processingStatus.textContent = '';
            this.processingStatus.className = 'processing-status';
        }
    }
}

/**
 * Enhanced UI Manager
 * 
 * Coordinates all enhanced UI components and provides a unified interface
 * for integration with the main application.
 */
class EnhancedUIManager {
    constructor() {
        this.filePreviewManager = new FilePreviewManager();
        this.progressBarManager = new ProgressBarManager();
        this.processingIndicatorManager = new ProcessingIndicatorManager();
        
        // Integration flags
        this.isProcessing = false;
        this.currentStage = 'idle';
    }
    
    /**
     * Handle file selection with enhanced preview
     */
    handleFileSelection(file, validationResult = null) {
        this.filePreviewManager.displayFile(file, validationResult);
        this.resetProcessingState();
    }
    
    /**
     * Handle file removal
     */
    handleFileRemoval() {
        this.filePreviewManager.hidePreview();
        this.resetProcessingState();
    }
    
    /**
     * Start file upload process
     */
    startFileUpload() {
        this.isProcessing = true;
        this.currentStage = 'upload';
        this.progressBarManager.startUpload();
        this.processingIndicatorManager.showProcessingState('uploading', 'Uploading file...', 'Uploading');
    }
    
    /**
     * Update upload progress
     */
    updateUploadProgress(percentage) {
        if (this.currentStage === 'upload') {
            this.progressBarManager.updateProgress('upload', percentage);
        }
    }
    
    /**
     * Transition to validation stage
     */
    startValidation() {
        this.currentStage = 'validation';
        this.progressBarManager.transitionToValidation();
        this.processingIndicatorManager.showProcessingState('validating', 'Validating file...', 'Validating');
    }
    
    /**
     * Update validation progress
     */
    updateValidationProgress(percentage, stage = null) {
        if (this.currentStage === 'validation') {
            const message = stage ? `Validating: ${stage}` : 'Validating file...';
            this.progressBarManager.updateProgress('validation', percentage, message);
        }
    }
    
    /**
     * Handle validation completion
     */
    completeValidation(validationResult) {
        if (validationResult.status === 'valid') {
            this.progressBarManager.updateProgress('validation', 100, 'Validation complete');
            this.processingIndicatorManager.showValidationSuccess(validationResult);
            
            // Update file preview with validation data
            this.filePreviewManager.updateValidationData(validationResult);
        } else {
            this.handleValidationError(validationResult);
        }
    }
    
    /**
     * Handle validation errors
     */
    handleValidationError(errorResult) {
        this.progressBarManager.showError();
        this.processingIndicatorManager.showValidationError(errorResult);
        this.isProcessing = false;
        this.currentStage = 'error';
        // Note: Chat error display is handled by the SSE event handler in script.js
    }
    
    /**
     * Start processing stage
     */
    startProcessing() {
        this.currentStage = 'processing';
        this.progressBarManager.transitionToProcessing();
        this.processingIndicatorManager.showLLMProcessing();
    }
    
    /**
     * Update processing progress
     */
    updateProcessingProgress(percentage) {
        if (this.currentStage === 'processing') {
            this.progressBarManager.updateProgress('processing', percentage);
        }
    }
    
    /**
     * Complete processing
     */
    completeProcessing() {
        this.progressBarManager.showCompletion();
        this.processingIndicatorManager.showCompletion();
        this.isProcessing = false;
        this.currentStage = 'complete';
    }
    
    /**
     * Handle processing errors
     */
    handleProcessingError(errorResult) {
        this.progressBarManager.showError();
        this.processingIndicatorManager.showValidationError(errorResult);
        this.isProcessing = false;
        this.currentStage = 'error';
        // Note: Chat error display is handled by the SSE event handler in script.js
    }
    
    /**
     * Reset all processing state
     */
    resetProcessingState() {
        this.progressBarManager.resetProgress();
        this.processingIndicatorManager.resetIndicators();
        this.isProcessing = false;
        this.currentStage = 'idle';
    }
    
    /**
     * Check if currently processing
     */
    isCurrentlyProcessing() {
        return this.isProcessing;
    }
    
    /**
     * Get current processing stage
     */
    getCurrentStage() {
        return this.currentStage;
    }
}

// Global instance for integration with main script
window.enhancedUIManager = new EnhancedUIManager();

// Export for module usage if needed
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        FilePreviewManager,
        ProgressBarManager,
        ProcessingIndicatorManager,
        EnhancedUIManager,
        formatFileSize,
        getFileTypeIcon,
        getSecurityClassification
    };
}