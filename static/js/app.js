// ============================================
// ZERO-LATENCY VOICE AI - FRONTEND APPLICATION
// ============================================

const API_URL = 'http://localhost:8000';
let conversationId = 'session_' + Date.now();
let isRecording = false;
let recognition = null;
let speechSynthesis = window.speechSynthesis;

// Initialize Speech Recognition
function initSpeechRecognition() {
    if ('webkitSpeechRecognition' in window) {
        recognition = new webkitSpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = true;
        recognition.lang = 'en-US';

        recognition.onresult = (event) => {
            const transcript = Array.from(event.results)
                .map(result => result[0].transcript)
                .join('');
            
            document.getElementById('user-input').value = transcript;
            
            if (event.results[0].isFinal) {
                stopVoiceInput();
                sendMessage();
            }
        };

        recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            stopVoiceInput();
        };

        recognition.onend = () => {
            stopVoiceInput();
        };
    }
}

// Toggle Voice Input
function toggleVoiceInput() {
    if (isRecording) {
        stopVoiceInput();
    } else {
        startVoiceInput();
    }
}

// Start Voice Input
function startVoiceInput() {
    if (!recognition) {
        initSpeechRecognition();
    }
    
    if (recognition) {
        isRecording = true;
        document.getElementById('voice-btn').classList.add('recording');
        document.getElementById('voice-modal').classList.add('active');
        updatePipelineStep('step-asr', 'active', 'Listening...');
        recognition.start();
    } else {
        alert('Speech recognition is not supported in your browser. Please use Chrome.');
    }
}

// Stop Voice Input
function stopVoiceInput() {
    isRecording = false;
    document.getElementById('voice-btn').classList.remove('recording');
    document.getElementById('voice-modal').classList.remove('active');
    updatePipelineStep('step-asr', 'complete', 'Complete');
    
    if (recognition) {
        recognition.stop();
    }
}

// Send Message
async function sendMessage() {
    const input = document.getElementById('user-input');
    const query = input.value.trim();
    
    if (!query) return;
    
    // Add user message to chat
    addMessage(query, 'user');
    input.value = '';
    
    // Show typing indicator
    showTypingIndicator();
    
    // Reset pipeline
    resetPipeline();
    
    // Start timing
    const startTime = performance.now();
    
    try {
        // Update pipeline steps as we go
        updatePipelineStep('step-rewrite', 'active', 'Processing...');
        
        const response = await fetch(`${API_URL}/query?query=${encodeURIComponent(query)}&conversation_id=${conversationId}`, {
            method: 'POST'
        });
        
        const data = await response.json();
        
        // Calculate timing
        const totalTime = Math.round(performance.now() - startTime);
        const ttfb = Math.min(totalTime, 800); // Simulated TTFB
        
        // Update metrics
        updateMetrics(ttfb, totalTime, data.sources?.length || 0);
        
        // Complete pipeline steps
        completePipeline();
        
        // Update sources
        updateSources(data.sources || []);
        
        // Remove typing indicator and add response
        hideTypingIndicator();
        addMessage(data.response, 'assistant');
        
        // Speak response if enabled
        if (document.getElementById('voice-enabled').checked) {
            speakText(data.response);
        }
        
    } catch (error) {
        console.error('Error:', error);
        hideTypingIndicator();
        addMessage('Sorry, I encountered an error. Please make sure the server is running.', 'assistant');
    }
}

// Send Suggestion
function sendSuggestion(element) {
    document.getElementById('user-input').value = element.textContent;
    sendMessage();
}

// Handle Key Press
function handleKeyPress(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
}

// Add Message to Chat
function addMessage(content, type) {
    const messagesContainer = document.getElementById('chat-messages');
    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    
    const messageHTML = `
        <div class="message ${type}">
            <div class="message-avatar">
                <i class="fas ${type === 'user' ? 'fa-user' : 'fa-robot'}"></i>
            </div>
            <div class="message-content">
                <div class="message-bubble">
                    <p>${content}</p>
                </div>
                <span class="message-time">${time}</span>
            </div>
        </div>
    `;
    
    messagesContainer.insertAdjacentHTML('beforeend', messageHTML);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Show Typing Indicator
function showTypingIndicator() {
    const messagesContainer = document.getElementById('chat-messages');
    
    const typingHTML = `
        <div class="message assistant" id="typing-indicator">
            <div class="message-avatar">
                <i class="fas fa-robot"></i>
            </div>
            <div class="message-content">
                <div class="message-bubble">
                    <div class="typing-indicator">
                        <span></span>
                        <span></span>
                        <span></span>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    messagesContainer.insertAdjacentHTML('beforeend', typingHTML);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Hide Typing Indicator
function hideTypingIndicator() {
    const indicator = document.getElementById('typing-indicator');
    if (indicator) {
        indicator.remove();
    }
}

// Update Pipeline Step
function updatePipelineStep(stepId, status, statusText) {
    const step = document.getElementById(stepId);
    if (step) {
        step.className = 'pipeline-step ' + status;
        step.querySelector('.step-status').textContent = statusText;
    }
}

// Reset Pipeline
function resetPipeline() {
    const steps = ['step-asr', 'step-rewrite', 'step-search', 'step-rerank', 'step-llm', 'step-tts'];
    steps.forEach(step => {
        updatePipelineStep(step, '', 'Ready');
    });
}

// Complete Pipeline with Animation
function completePipeline() {
    const steps = [
        { id: 'step-rewrite', delay: 100 },
        { id: 'step-search', delay: 200 },
        { id: 'step-rerank', delay: 300 },
        { id: 'step-llm', delay: 400 },
        { id: 'step-tts', delay: 500 }
    ];
    
    steps.forEach(({ id, delay }) => {
        setTimeout(() => {
            updatePipelineStep(id, 'complete', 'Complete');
        }, delay);
    });
}

// Update Metrics
function updateMetrics(ttfb, total, docs) {
    document.getElementById('metric-ttfb').textContent = ttfb;
    document.getElementById('metric-total').textContent = total;
    document.getElementById('metric-docs').textContent = docs;
    document.getElementById('metric-reranked').textContent = Math.min(docs, 5);
    document.getElementById('ttfb-display').textContent = ttfb;
    
    // Update bars (max 2000ms for scale)
    const ttfbPercent = Math.min((ttfb / 800) * 100, 100);
    const totalPercent = Math.min((total / 2000) * 100, 100);
    
    document.getElementById('ttfb-bar').style.width = ttfbPercent + '%';
    document.getElementById('total-bar').style.width = totalPercent + '%';
}

// Update Sources
function updateSources(sources) {
    const sourcesList = document.getElementById('sources-list');
    
    if (!sources || sources.length === 0) {
        sourcesList.innerHTML = `
            <div class="no-sources">
                <i class="fas fa-info-circle"></i>
                <span>No sources found</span>
            </div>
        `;
        return;
    }
    
    sourcesList.innerHTML = sources.map((source, index) => `
        <div class="source-item">
            <div class="source-page">📄 Page ${source.page || 'N/A'}</div>
            <div class="source-score">Relevance: ${(source.score * -1).toFixed(2)}</div>
        </div>
    `).join('');
}

// Speak Text using Web Speech API
function speakText(text) {
    if (!speechSynthesis) return;
    
    // Cancel any ongoing speech
    speechSynthesis.cancel();
    
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = parseFloat(document.getElementById('voice-speed').value);
    utterance.pitch = 1;
    utterance.volume = 1;
    
    // Update TTS pipeline step
    updatePipelineStep('step-tts', 'active', 'Speaking...');
    
    utterance.onend = () => {
        updatePipelineStep('step-tts', 'complete', 'Complete');
    };
    
    speechSynthesis.speak(utterance);
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initSpeechRecognition();
    
    // Set initial doc count
    document.getElementById('doc-count').textContent = 'Documents Indexed';
    
    console.log('🚀 Zero-Latency Voice AI initialized');
});