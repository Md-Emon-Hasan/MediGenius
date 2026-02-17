import React, { useState } from 'react';

const InputArea = ({ onSendMessage, disabled }) => {
    const [message, setMessage] = useState('');

    const handleSend = () => {
        if (message.trim() && !disabled) {
            onSendMessage(message);
            setMessage('');
        }
    };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    return (
        <div className="input-area">
            <div className="input-wrapper">
                <div className="input-container glass-effect">
                    <button className="input-btn" title="Attach file">
                        <i className="fas fa-paperclip"></i>
                    </button>
                    <textarea
                        value={message}
                        onChange={(e) => setMessage(e.target.value)}
                        onKeyPress={handleKeyPress}
                        className="message-input"
                        placeholder="Ask your medical question..."
                        rows="1"
                        disabled={disabled}
                    ></textarea>
                    <button className="input-btn" title="Voice input">
                        <i className="fas fa-microphone"></i>
                    </button>
                    <button
                        className="send-btn"
                        onClick={handleSend}
                        disabled={!message.trim() || disabled}
                        aria-label="Send message"
                    >
                        <i className="fas fa-paper-plane"></i>
                    </button>
                </div>
                <div className="input-info">
                    <i className="fas fa-info-circle"></i>
                    <span>AI can make mistakes. Always consult healthcare professionals for medical advice.</span>
                </div>
            </div>
        </div>
    );
};

export default InputArea;
