import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import './AIChat.css';

const AIChat = ({ apiBase, merchants }) => {
  const [messages, setMessages] = useState([
    {
      id: 1,
      type: 'assistant',
      content: 'Hello! I\'m your AI assistant for merchant risk intelligence. I can help you understand risk factors, compliance requirements, and fraud patterns. How can I assist you today?',
      timestamp: new Date()
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [selectedMerchant, setSelectedMerchant] = useState('');
  const [conversationId, setConversationId] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottom, [messages]);

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!inputMessage.trim()) return;

    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: inputMessage,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      const response = await axios.post(`${apiBase}/chat`, {
        message: inputMessage,
        merchant_id: selectedMerchant || null,
        conversation_id: conversationId
      });

      const assistantMessage = {
        id: Date.now() + 1,
        type: 'assistant',
        content: response.data.response,
        sources: response.data.sources || [],
        timestamp: new Date(response.data.timestamp)
      };

      setMessages(prev => [...prev, assistantMessage]);
      setConversationId(response.data.conversation_id);

    } catch (error) {
      console.error('Chat error:', error);
      const errorMessage = {
        id: Date.now() + 1,
        type: 'assistant',
        content: 'I apologize, but I\'m having trouble connecting to the AI service right now. Please try again in a moment.',
        timestamp: new Date(),
        isError: true
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const clearConversation = () => {
    setMessages([
      {
        id: 1,
        type: 'assistant',
        content: 'Conversation cleared. How can I help you with merchant risk intelligence?',
        timestamp: new Date()
      }
    ]);
    setConversationId(null);
  };

  const suggestedQuestions = [
    "What are the main risk factors for high-risk merchants?",
    "Explain PCI DSS compliance requirements",
    "How do I identify potential fraud patterns?",
    "What are the best practices for chargeback prevention?",
    "Tell me about AML regulations for payment processors"
  ];

  return (
    <div className="ai-chat">
      <div className="chat-header">
        <h2>🤖 AI Risk Intelligence Assistant</h2>
        <div className="chat-controls">
          <select
            value={selectedMerchant}
            onChange={(e) => setSelectedMerchant(e.target.value)}
            className="merchant-selector"
          >
            <option value="">Select merchant for context (optional)</option>
            {merchants.map(id => (
              <option key={id} value={id}>{id}</option>
            ))}
          </select>
          <button onClick={clearConversation} className="clear-button">
            Clear Chat
          </button>
        </div>
      </div>

      <div className="chat-container">
        <div className="messages">
          {messages.map(message => (
            <div key={message.id} className={`message ${message.type}`}>
              <div className="message-content">
                <div className="message-text">
                  {message.content}
                </div>
                {message.sources && message.sources.length > 0 && (
                  <div className="message-sources">
                    <details>
                      <summary>Sources ({message.sources.length})</summary>
                      {message.sources.map((source, index) => (
                        <div key={index} className="source-item">
                          <strong>{source.metadata?.title || 'Knowledge Base'}</strong>
                          <p>{source.content}</p>
                        </div>
                      ))}
                    </details>
                  </div>
                )}
                <div className="message-timestamp">
                  {message.timestamp.toLocaleTimeString()}
                </div>
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="message assistant">
              <div className="message-content">
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="suggested-questions">
          <h4>Suggested Questions:</h4>
          <div className="question-buttons">
            {suggestedQuestions.map((question, index) => (
              <button
                key={index}
                className="suggestion-button"
                onClick={() => setInputMessage(question)}
              >
                {question}
              </button>
            ))}
          </div>
        </div>

        <form onSubmit={handleSendMessage} className="chat-input-form">
          <div className="input-container">
            <input
              type="text"
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              placeholder="Ask about merchant risk, compliance, fraud detection..."
              className="message-input"
              disabled={isLoading}
            />
            <button
              type="submit"
              disabled={isLoading || !inputMessage.trim()}
              className="send-button"
            >
              Send
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default AIChat;