import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';
import MerchantLookup from './components/MerchantLookup';
import AIChat from './components/AIChat';
import Dashboard from './components/Dashboard';
import Navigation from './components/Navigation';

const API_BASE = process.env.REACT_APP_API_BASE || 'http://localhost:8000';

function App() {
  const [merchants, setMerchants] = useState([]);
  const [currentView, setCurrentView] = useState('lookup');
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchMerchants();
  }, []);

  const fetchMerchants = async () => {
    try {
      setIsLoading(true);
      const response = await axios.get(`${API_BASE}/merchants/`);
      setMerchants(response.data.merchant_ids);
      setError(null);
    } catch (error) {
      console.error("Error fetching merchants:", error);
      setError("Failed to load merchant data. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const renderCurrentView = () => {
    switch (currentView) {
      case 'lookup':
        return <MerchantLookup merchants={merchants} apiBase={API_BASE} />;
      case 'chat':
        return <AIChat apiBase={API_BASE} merchants={merchants} />;
      case 'dashboard':
        return <Dashboard apiBase={API_BASE} />;
      default:
        return <MerchantLookup merchants={merchants} apiBase={API_BASE} />;
    }
  };

  if (isLoading) {
    return (
      <div className="App">
        <div className="loading-container">
          <div className="spinner"></div>
          <p>Loading Merchant Pulse...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="App">
      <header className="App-header">
        <div className="header-content">
          <div className="header-text">
            <h1>Merchant Pulse Enterprise</h1>
            <p>AI-Powered Merchant Risk Intelligence Platform</p>
          </div>
          <div className="header-stats">
            <div className="stat-card">
              <span className="stat-number">{merchants.length}</span>
              <span className="stat-label">Merchants</span>
            </div>
            <div className="stat-card">
              <span className="stat-number">99.9%</span>
              <span className="stat-label">Uptime</span>
            </div>
            <div className="stat-card">
              <span className="stat-number">AI</span>
              <span className="stat-label">Powered</span>
            </div>
          </div>
        </div>
      </header>

      <Navigation currentView={currentView} onViewChange={setCurrentView} />

      {error && (
        <div className="error-banner">
          <span className="error-text">{error}</span>
          <button onClick={fetchMerchants} className="retry-button">
            Retry
          </button>
        </div>
      )}

      <main className="main-content">
        {renderCurrentView()}
      </main>

      <footer className="App-footer">
        <div className="footer-content">
          <div className="footer-section">
            <h4>Merchant Pulse Enterprise</h4>
            <p>Advanced risk intelligence for the digital payment ecosystem</p>
          </div>
          <div className="footer-section">
            <h4>Features</h4>
            <ul>
              <li>Real-time Risk Assessment</li>
              <li>AI-Powered Explanations</li>
              <li>Conversational Intelligence</li>
              <li>Predictive Analytics</li>
            </ul>
          </div>
          <div className="footer-section">
            <h4>Technology</h4>
            <ul>
              <li>Machine Learning</li>
              <li>Natural Language Processing</li>
              <li>Microservices Architecture</li>
              <li>Cloud-Native Deployment</li>
            </ul>
          </div>
        </div>
        <div className="footer-bottom">
          <p>&copy; 2024 Merchant Pulse Enterprise - Built for 4th Year Graduate Project</p>
        </div>
      </footer>
    </div>
  );
}

export default App;