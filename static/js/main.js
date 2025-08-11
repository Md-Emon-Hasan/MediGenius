// static/js/main.js (Premium Interactions)
document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const chatContainer = document.getElementById('chat-container');
    const chatMessages = document.getElementById('chat-messages');
    const textInput = document.getElementById('text-input');
    const sendBtn = document.getElementById('send-btn');
    const welcomeScreen = document.getElementById('welcome-screen');
    const typingIndicator = document.getElementById('typing-indicator');
    const toast = document.getElementById('toast');
    const themeToggle = document.getElementById('theme-toggle');
    const quickBtns = document.querySelectorAll('.quick-btn');
    const voiceBtn = document.getElementById('voice-btn');
    
    // Initialize variables
    let isDarkMode = false;
    
    // Initialize animations
    initAnimations();
    
    // Event Listeners
    sendBtn.addEventListener('click', sendMessage);
    textInput.addEventListener('keydown', handleInputKeydown);
    themeToggle.addEventListener('click', toggleTheme);
    voiceBtn.addEventListener('click', toggleVoiceRecognition);
    quickBtns.forEach(btn => {
        btn.addEventListener('click', handleQuickQuestion);
    });
    
    // Initialize functions
    function initAnimations() {
        // Animated gradient text
        const gradientText = document.querySelector('.animate-charcter');
        anime({
            targets: gradientText,
            backgroundPosition: ['0% center', '200% center'],
            duration: 10000,
            easing: 'linear',
            loop: true
        });
        
        // Floating elements animation
        anime({
            targets: '.floating-circle',
            translateX: function() {
                return anime.random(-50, 50);
            },
            translateY: function() {
                return anime.random(-50, 50);
            },
            duration: function() {
                return anime.random(15000, 30000);
            },
            easing: 'easeInOutQuad',
            direction: 'alternate',
            loop: true
        });
    }
    
    function sendMessage() {
        const message = textInput.textContent.trim();
        if (!message) return;
        
        // Add user message
        addMessage(message, true);
        textInput.textContent = '';
        
        // Hide welcome screen if visible
        if (!welcomeScreen.classList.contains('hidden')) {
            welcomeScreen.classList.add('hidden');
            anime({
                targets: welcomeScreen,
                opacity: 0,
                duration: 300,
                easing: 'easeOutQuad'
            });
        }
        
        // Show typing indicator
        typingIndicator.classList.add('active');
        
        // Send to server
        fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message })
        })
        .then(response => response.json())
        .then(data => {
            // Add bot response
            addMessage(data.response, false, data.timestamp, data.source);
            
            // Show toast notification
            showToast('Message received');
        })
        .catch(error => {
            console.error('Error:', error);
            addMessage("Sorry, I encountered an error. Please try again.", false);
            showToast('Error occurred', true);
        })
        .finally(() => {
            typingIndicator.classList.remove('active');
        });
    }
    
    function addMessage(content, isUser, timestamp = null) {  // Removed source parameter
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
        
        const avatarDiv = document.createElement('div');
        avatarDiv.className = 'message-avatar';
        avatarDiv.innerHTML = `<i class="fas ${isUser ? 'fa-user' : 'fa-robot'}"></i>`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.innerHTML = `<p>${content}</p>`;
        
        if (timestamp) {
            const timeSpan = document.createElement('span');
            timeSpan.className = 'message-time';
            timeSpan.textContent = timestamp;
            contentDiv.appendChild(timeSpan);
        }
        
        messageDiv.appendChild(avatarDiv);
        messageDiv.appendChild(contentDiv);
        chatMessages.appendChild(messageDiv);
        
        anime({
            targets: messageDiv,
            opacity: [0, 1],
            translateY: [20, 0],
            duration: 300,
            easing: 'easeOutQuad'
        });
        
        chatContainer.scrollTo({
            top: chatContainer.scrollHeight,
            behavior: 'smooth'
        });
    }
    
    function handleInputKeydown(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    }
    
    function toggleTheme() {
        isDarkMode = !isDarkMode;
        document.documentElement.setAttribute('data-theme', isDarkMode ? 'dark' : 'light');
        
        // Update icon
        themeToggle.innerHTML = `<i class="fas ${isDarkMode ? 'fa-sun' : 'fa-moon'}"></i>`;
        
        // Animate toggle
        anime({
            targets: themeToggle,
            rotate: 180,
            duration: 500,
            easing: 'easeOutBack'
        });
    }
    
    function toggleVoiceRecognition() {
        // This would be implemented with the Web Speech API
        showToast('Voice recognition coming soon!');
    }
    
    function handleQuickQuestion(e) {
        const question = e.currentTarget.dataset.question;
        textInput.textContent = question;
        sendMessage();
    }
    
    function showToast(message, isError = false) {
        toast.innerHTML = `
            <div class="toast-content">
                <i class="fas ${isError ? 'fa-exclamation-circle' : 'fa-check-circle'}"></i>
                <span>${message}</span>
            </div>
        `;
        
        toast.classList.add('show');
        
        // Animate toast
        anime({
            targets: toast,
            opacity: [0, 1],
            translateY: [100, 0],
            duration: 300,
            easing: 'easeOutBack'
        });
        
        // Hide after delay
        setTimeout(() => {
            anime({
                targets: toast,
                opacity: 0,
                translateY: 100,
                duration: 300,
                easing: 'easeInBack',
                complete: () => {
                    toast.classList.remove('show');
                }
            });
        }, 3000);
    }
    
    // Parallax effect for chat background
    chatContainer.addEventListener('mousemove', (e) => {
        const x = e.clientX / window.innerWidth;
        const y = e.clientY / window.innerHeight;
        
        const parallaxBg = document.querySelector('.parallax-bg');
        parallaxBg.style.transform = `translate(-${x * 20}px, -${y * 20}px)`;
    });
    
    // Initial animations
    anime({
        targets: '.sidebar',
        translateX: [-50, 0],
        opacity: [0, 1],
        duration: 800,
        easing: 'easeOutQuad'
    });
    
    anime({
        targets: '.main-content',
        translateX: [50, 0],
        opacity: [0, 1],
        duration: 800,
        easing: 'easeOutQuad',
        delay: 200
    });
});