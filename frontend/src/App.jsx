import React, { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import ChatArea from './components/ChatArea';
import InputArea from './components/InputArea';

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [messages, setMessages] = useState([]);
  const [isTyping, setIsTyping] = useState(false);
  const [currentSessionId, setCurrentSessionId] = useState(null);
  const [chatHistory, setChatHistory] = useState([]);
  const [toast, setToast] = useState({ show: false, message: '' });

  // Initial Data Load
  useEffect(() => {
    fetchSessions();
    // Check if there is an active session
    fetch('/api/v1/history')
      .then(res => res.json())
      .then(data => {
        if (data.success && data.messages) {
          setMessages(data.messages);
        }
      })
      .catch(err => console.error("Failed to load history", err));
  }, []);

  const fetchSessions = async () => {
    try {
      const res = await fetch('/api/v1/sessions');
      const data = await res.json();
      if (data.success) {
        setChatHistory(data.sessions);
      }
    } catch (error) {
      console.error("Error fetching sessions:", error);
    }
  };

  const showToast = (message) => {
    setToast({ show: true, message });
    setTimeout(() => setToast({ show: false, message: '' }), 3000);
  };

  const handleSendMessage = async (text) => {
    // Optimistic UI update
    const userMsg = { role: 'user', content: text, timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) };
    setMessages(prev => [...prev, userMsg]);
    setIsTyping(true);

    try {
      const res = await fetch('/api/v1/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text })
      });
      const data = await res.json();

      if (data.success) {
        const botMsg = {
          role: 'assistant',
          content: data.response,
          source: data.source,
          timestamp: data.timestamp
        };
        setMessages(prev => [...prev, botMsg]);
        fetchSessions(); // Update history list
      } else {
        showToast("Failed to generate response");
      }
    } catch (error) {
      console.error("Chat error:", error);
      showToast("Error communicating with server");
    } finally {
      setIsTyping(false);
    }
  };

  const handleNewChat = async () => {
    try {
      const res = await fetch('/api/v1/new-chat', { method: 'POST' });
      const data = await res.json();
      if (data.success) {
        setMessages([]);
        setCurrentSessionId(data.session_id);
        fetchSessions();
        showToast("New chat started");
      }
    } catch (error) {
      console.error("Error creating new chat:", error);
    }
  };

  const handleLoadSession = async (sessionId) => {
    try {
      const res = await fetch(`/api/v1/session/${sessionId}`);
      const data = await res.json();
      if (data.success) {
        setMessages(data.messages);
        setCurrentSessionId(sessionId);
        // Mobile: auto close sidebar
        if (window.innerWidth < 768) setSidebarOpen(false);
      }
    } catch (error) {
      console.error("Error loading session:", error);
    }
  };

  const handleDeleteSession = async (sessionId) => {
    if (!window.confirm("Delete this chat?")) return;

    try {
      const res = await fetch(`/api/v1/session/${sessionId}`, { method: 'DELETE' });
      const data = await res.json();
      if (data.success) {
        fetchSessions();
        if (currentSessionId === sessionId) {
          setMessages([]); // Cleared current
          setCurrentSessionId(null);
        }
        showToast("Chat deleted");
      }
    } catch (error) {
      console.error("Error deleting session:", error);
    }
  };

  return (
    <div className="app-container">
      {/* Background Animation */}
      <div className="animated-background">
        <div className="gradient-overlay"></div>
        <div className="floating-circles">
          <div className="circle circle-1"></div>
          <div className="circle circle-2"></div>
          <div className="circle circle-3"></div>
        </div>
      </div>

      <Sidebar
        isOpen={sidebarOpen}
        toggleSidebar={() => setSidebarOpen(!sidebarOpen)}
        onNewChat={handleNewChat}
        chatHistory={chatHistory}
        currentSessionId={currentSessionId}
        onLoadSession={handleLoadSession}
        onDeleteSession={handleDeleteSession}
      />

      <main className={`main-content ${sidebarOpen ? 'sidebar-open' : ''}`} style={{ marginLeft: sidebarOpen && window.innerWidth > 768 ? '280px' : '0' }}>
        {/* Header */}
        <header className="app-header glass-header">
          <div className="header-content">
            <h2 className="gradient-text">Medical AI Assistant</h2>
            <div className="status-indicator">
              <div className="status-ring">
                <span className="ring-pulse"></span>
              </div>
              <span>AI Ready</span>
            </div>
          </div>

          <div className="header-actions">
            <button className="action-btn" title="Clear conversation" onClick={() => setMessages([])}>
              <i className="fas fa-trash"></i>
            </button>
            <button className="action-btn" title="Download chat">
              <i className="fas fa-download"></i>
            </button>
            <button className="action-btn" title="Settings">
              <i className="fas fa-cog"></i>
            </button>
          </div>
        </header>

        {/* Chat Area */}
        <ChatArea
          messages={messages}
          isTyping={isTyping}
          onQuickQuestion={handleSendMessage}
        />

        {/* Input Area */}
        <InputArea
          onSendMessage={handleSendMessage}
          disabled={isTyping}
        />
      </main>

      {/* Toast */}
      <div className={`toast glass-effect ${toast.show ? 'show' : ''}`}>
        <i className="fas fa-check-circle"></i>
        <span>{toast.message}</span>
      </div>
    </div>
  );
}

export default App;
