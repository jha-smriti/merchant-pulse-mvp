import React from 'react';
import './Navigation.css';

const Navigation = ({ currentView, onViewChange }) => {
  const navItems = [
    { id: 'lookup', label: 'Risk Assessment', icon: '🔍' },
    { id: 'chat', label: 'AI Assistant', icon: '🤖' },
    { id: 'dashboard', label: 'Analytics', icon: '📊' }
  ];

  return (
    <nav className="navigation">
      <div className="nav-container">
        {navItems.map(item => (
          <button
            key={item.id}
            className={`nav-item ${currentView === item.id ? 'active' : ''}`}
            onClick={() => onViewChange(item.id)}
          >
            <span className="nav-icon">{item.icon}</span>
            <span className="nav-label">{item.label}</span>
          </button>
        ))}
      </div>
    </nav>
  );
};

export default Navigation;