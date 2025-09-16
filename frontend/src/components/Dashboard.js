import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './Dashboard.css';

const Dashboard = ({ apiBase }) => {
  const [analytics, setAnalytics] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchAnalytics();
  }, []);

  const fetchAnalytics = async () => {
    try {
      setIsLoading(true);
      const response = await axios.get(`${apiBase}/analytics/dashboard`);
      setAnalytics(response.data);
      setError(null);
    } catch (error) {
      console.error('Analytics error:', error);
      setError('Failed to load analytics data');
    } finally {
      setIsLoading(false);
    }
  };

  if (isLoading) {
    return (
      <div className="dashboard">
        <div className="loading">Loading analytics...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="dashboard">
        <div className="error">{error}</div>
        <button onClick={fetchAnalytics}>Retry</button>
      </div>
    );
  }

  const { summary, risk_distribution } = analytics;

  return (
    <div className="dashboard">
      <div className="dashboard-header">
        <h2>📊 Risk Analytics Dashboard</h2>
        <button onClick={fetchAnalytics} className="refresh-button">
          Refresh Data
        </button>
      </div>

      <div className="metrics-grid">
        <div className="metric-card">
          <div className="metric-icon">🏪</div>
          <div className="metric-content">
            <h3>Total Merchants</h3>
            <div className="metric-value">{summary.total_merchants.toLocaleString()}</div>
          </div>
        </div>

        <div className="metric-card high-risk">
          <div className="metric-icon">⚠️</div>
          <div className="metric-content">
            <h3>High Risk Merchants</h3>
            <div className="metric-value">{summary.high_risk_merchants}</div>
            <div className="metric-subtitle">
              {summary.high_risk_percentage.toFixed(1)}% of total
            </div>
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-icon">💳</div>
          <div className="metric-content">
            <h3>Total Transactions (30d)</h3>
            <div className="metric-value">{summary.total_transactions_30d.toLocaleString()}</div>
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-icon">💰</div>
          <div className="metric-content">
            <h3>Total Volume (30d)</h3>
            <div className="metric-value">
              ${(summary.total_volume_30d / 1000000).toFixed(1)}M
            </div>
          </div>
        </div>

        <div className="metric-card chargeback">
          <div className="metric-icon">🔄</div>
          <div className="metric-content">
            <h3>Avg Chargeback Rate</h3>
            <div className="metric-value">
              {(summary.avg_chargeback_rate * 100).toFixed(2)}%
            </div>
          </div>
        </div>
      </div>

      <div className="charts-grid">
        <div className="chart-card">
          <h3>Risk Distribution</h3>
          <div className="risk-chart">
            <div className="risk-bar-chart">
              <div className="risk-bar low-risk">
                <div className="bar-fill" style={{
                  height: `${(risk_distribution.low_risk / summary.total_merchants) * 100}%`
                }}></div>
                <div className="bar-label">
                  <span>Low Risk</span>
                  <span>{risk_distribution.low_risk}</span>
                </div>
              </div>
              <div className="risk-bar medium-risk">
                <div className="bar-fill" style={{
                  height: `${(risk_distribution.medium_risk / summary.total_merchants) * 100}%`
                }}></div>
                <div className="bar-label">
                  <span>Medium Risk</span>
                  <span>{risk_distribution.medium_risk}</span>
                </div>
              </div>
              <div className="risk-bar high-risk">
                <div className="bar-fill" style={{
                  height: `${(risk_distribution.high_risk / summary.total_merchants) * 100}%`
                }}></div>
                <div className="bar-label">
                  <span>High Risk</span>
                  <span>{risk_distribution.high_risk}</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="chart-card">
          <h3>Key Insights</h3>
          <div className="insights-list">
            <div className="insight-item">
              <span className="insight-icon">📈</span>
              <div className="insight-text">
                <strong>Risk Trend:</strong> {
                  summary.high_risk_percentage > 10 
                    ? 'Higher than industry average' 
                    : 'Within normal range'
                }
              </div>
            </div>
            <div className="insight-item">
              <span className="insight-icon">💡</span>
              <div className="insight-text">
                <strong>Recommendation:</strong> {
                  summary.avg_chargeback_rate > 0.02
                    ? 'Focus on chargeback prevention measures'
                    : 'Maintain current risk management practices'
                }
              </div>
            </div>
            <div className="insight-item">
              <span className="insight-icon">🎯</span>
              <div className="insight-text">
                <strong>Action Items:</strong> Review high-risk merchants for additional monitoring
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="features-showcase">
        <h3>Platform Features</h3>
        <div className="features-grid">
          <div className="feature-card">
            <div className="feature-icon">🤖</div>
            <h4>AI-Powered Analysis</h4>
            <p>Advanced machine learning models for accurate risk prediction</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">🔍</div>
            <h4>Real-time Monitoring</h4>
            <p>Continuous monitoring of merchant behavior and transaction patterns</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">📊</div>
            <h4>Predictive Analytics</h4>
            <p>Forecast potential risks before they become problems</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">🛡️</div>
            <h4>Fraud Detection</h4>
            <p>Advanced algorithms to identify suspicious activities</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;