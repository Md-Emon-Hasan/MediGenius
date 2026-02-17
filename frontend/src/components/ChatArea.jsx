import React, { useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';

const ChatArea = ({ messages, isTyping, onQuickQuestion }) => {
    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages, isTyping]);

    if (messages.length === 0) {
        return (
            <div className="chat-area" id="chatArea">
                <div className="welcome-screen">
                    <div className="welcome-content">
                        <div className="logo-3d">
                            <i className="fas fa-stethoscope"></i>
                        </div>

                        <h1 className="welcome-title">Welcome to MediGenius</h1>
                        <p className="welcome-subtitle">Your AI-powered medical assistant is ready to help</p>

                        <div className="quick-actions">
                            <h3>Quick Questions:</h3>
                            <div className="quick-buttons">
                                {[
                                    { icon: "fa-thermometer", text: "Fever Symptoms", query: "What are the symptoms of fever?" },
                                    { icon: "fa-head-side-virus", text: "Headache Treatment", query: "How to treat a headache?" },
                                    { icon: "fa-heart-pulse", text: "High Blood Pressure", query: "What causes high blood pressure?" },
                                    { icon: "fa-notes-medical", text: "Diabetes Management", query: "Tell me about diabetes management" },
                                    { icon: "fa-virus-covid", text: "COVID Prevention", query: "COVID-19 prevention tips" },
                                    { icon: "fa-pills", text: "Cold Remedies", query: "Common cold remedies" }
                                ].map((btn, index) => (
                                    <button
                                        key={index}
                                        className="quick-btn glass-effect"
                                        onClick={() => onQuickQuestion(btn.query)}
                                    >
                                        <i className={`fas ${btn.icon}`}></i>
                                        <span>{btn.text}</span>
                                    </button>
                                ))}
                            </div>
                        </div>

                        <div className="features">
                            <div className="feature-card glass-effect">
                                <i className="fas fa-brain"></i>
                                <span>AI-Powered</span>
                            </div>
                            <div className="feature-card glass-effect">
                                <i className="fas fa-database"></i>
                                <span>Medical Database</span>
                            </div>
                            <div className="feature-card glass-effect">
                                <i className="fas fa-shield-alt"></i>
                                <span>Reliable Info</span>
                            </div>
                            <div className="feature-card glass-effect">
                                <i className="fas fa-clock"></i>
                                <span>24/7 Available</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="chat-area" id="chatArea">
            <div className="messages-container">
                {messages.map((msg, index) => (
                    <div key={index} className={`message ${msg.role === 'user' ? 'user-message' : 'bot-message'}`}>
                        <div className="message-wrapper">
                            {msg.role === 'assistant' && (
                                <div className="message-avatar">
                                    <i className="fas fa-robot"></i>
                                </div>
                            )}

                            <div className="message-content">
                                <div className="message-text">
                                    <ReactMarkdown>{msg.content}</ReactMarkdown>
                                </div>
                                <div className="message-time">
                                    {msg.timestamp || new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                </div>

                                {msg.role === 'assistant' && (
                                    <div className="message-footer">
                                        {msg.source && (
                                            <div className="message-source">
                                                <i className="fas fa-book-medical"></i>
                                                <span>{msg.source}</span>
                                            </div>
                                        )}
                                        <div className="message-actions">
                                            <button className="message-action" title="Copy">
                                                <i className="fas fa-copy"></i>
                                            </button>
                                            <button className="message-action" title="Regenerate">
                                                <i className="fas fa-sync-alt"></i>
                                            </button>
                                        </div>
                                    </div>
                                )}
                            </div>

                            {msg.role === 'user' && (
                                <div className="message-avatar">
                                    <i className="fas fa-user"></i>
                                </div>
                            )}
                        </div>
                    </div>
                ))}

                {isTyping && (
                    <div className="typing-indicator active">
                        <div className="typing-bubble glass-effect">
                            <div className="typing-content">
                                <span className="typing-text">MediGenius is thinking</span>
                                <div className="typing-dots">
                                    <span className="dot"></span>
                                    <span className="dot"></span>
                                    <span className="dot"></span>
                                </div>
                            </div>
                        </div>
                    </div>
                )}
                <div ref={messagesEndRef} />
            </div>
        </div>
    );
};

export default ChatArea;
