import React from 'react';

const Sidebar = ({ isOpen, toggleSidebar, onNewChat, chatHistory, currentSessionId, onLoadSession, onDeleteSession }) => {
    return (
        <>
            {/* Toggle Button */}
            <button
                className="sidebar-toggle-btn"
                onClick={toggleSidebar}
            >
                <i className="fas fa-bars"></i>
            </button>

            {/* Sidebar */}
            <aside className={`sidebar glass-effect ${!isOpen ? 'collapsed hidden md:flex md:w-0' : 'w-[280px]'}`} style={{ transform: !isOpen ? 'translateX(-100%)' : 'translateX(0)' }}>
                <div className="sidebar-content">
                    {/* Logo and New Chat */}
                    <div className="sidebar-header">
                        <div className="logo-wrapper">
                            <div className="logo-animated">
                                <div className="logo-pulse absolute w-full h-full rounded-2xl"></div>
                                <i className="fas fa-heartbeat relative z-10 text-white text-2xl"></i>
                            </div>
                            <div className="logo-text">
                                <h1 className="text-xl font-bold text-[var(--text-primary)]">MediGenius</h1>
                                <span className="version">AI Assistant v3.0</span>
                            </div>
                        </div>
                        <button className="new-chat-btn" onClick={onNewChat}>
                            <i className="fas fa-plus"></i>
                            <span>New Chat</span>
                        </button>
                    </div>

                    {/* Chat History */}
                    <div className="chat-history-section">
                        <div className="section-header">
                            <span>Chat History</span>
                            <div className="section-line"></div>
                        </div>
                        <div className="chat-list">
                            {chatHistory.length === 0 ? (
                                <div className="text-center p-5 text-[var(--text-tertiary)] text-sm">
                                    <div className="loading-spinner mx-auto mb-2"></div>
                                    Loading chats...
                                </div>
                            ) : (
                                chatHistory.map((session) => (
                                    <div
                                        key={session.session_id}
                                        className={`chat-item ${currentSessionId === session.session_id ? 'active' : ''}`}
                                        onClick={() => onLoadSession(session.session_id)}
                                    >
                                        <div className="chat-item-content">
                                            <div className="chat-item-title">{session.preview || "New Chat"}</div>
                                            <div className="chat-item-time">{new Date(session.last_active).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</div>
                                        </div>
                                        <button
                                            className="chat-item-delete"
                                            onClick={(e) => {
                                                e.stopPropagation();
                                                onDeleteSession(session.session_id);
                                            }}
                                        >
                                            <i className="fas fa-trash-alt text-xs"></i>
                                        </button>
                                    </div>
                                ))
                            )}
                        </div>
                    </div>

                    {/* Developer Info */}
                    <div className="sidebar-footer">
                        <div className="developer-card glass-effect">
                            <div className="dev-header">
                                <i className="fas fa-code"></i>
                                <span>Developer</span>
                            </div>
                            <div className="dev-info">
                                <p className="font-semibold">Md. Hasan Imon</p>
                                <div className="social-links">
                                    <a href="https://github.com/Md-Emon-Hasan" className="social-link" target="_blank" rel="noreferrer">
                                        <i className="fab fa-github"></i>
                                    </a>
                                    <a href="https://www.linkedin.com/in/md-emon-hasan-695483237/" className="social-link" target="_blank" rel="noreferrer">
                                        <i className="fab fa-linkedin"></i>
                                    </a>
                                    <a href="https://md-emon-hasan.github.io/My-Resume/" className="social-link" target="_blank" rel="noreferrer">
                                        <i className="fas fa-globe"></i>
                                    </a>
                                </div>
                            </div>
                        </div>
                        <button className="theme-btn glass-effect">
                            <i className="fas fa-moon"></i>
                        </button>
                    </div>
                </div>
            </aside>
        </>
    );
};

export default Sidebar;
