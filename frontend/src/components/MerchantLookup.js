import React, { useState } from 'react';
import axios from 'axios';

const API_BASE = 'http://localhost:8000';

const MerchantLookup = ({ merchants }) => {
  const [selectedMerchant, setSelectedMerchant] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!selectedMerchant) return;

    setLoading(true);
    setError('');
    setResult(null);

    try {
      const response = await axios.post(`${API_BASE}/predict/`, {
        merchant_id: selectedMerchant,
      });
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.error || 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (score) => {
    if (score < 0.3) return 'green';
    if (score < 0.7) return 'orange';
    return 'red';
  };

  return (
    <div>
      <form onSubmit={handleSubmit}>
        <label htmlFor="merchant-select">Select a Merchant: </label>
        <select
          id="merchant-select"
          value={selectedMerchant}
          onChange={(e) => setSelectedMerchant(e.target.value)}
        >
          <option value="">-- Choose a Merchant --</option>
          {merchants.map((id) => (
            <option key={id} value={id}>{id}</option>
          ))}
        </select>
        <button type="submit" disabled={loading || !selectedMerchant}>
          {loading ? 'Analyzing...' : 'Get Risk Score'}
        </button>
      </form>

      {error && <div style={{ color: 'red' }}>Error: {error}</div>}

      {result && (
        <div className="result-container">
          <h2>Results for {result.merchant_id}</h2>
          <div className="risk-score" style={{ color: getRiskColor(result.risk_score) }}>
            Risk Score: <strong>{(result.risk_score * 100).toFixed(0)}%</strong>
          </div>
          <p><strong>Primary Reason:</strong> {result.risk_reason}</p>
          <div className="details">
            <h3>Transaction Details:</h3>
            <ul>
              <li>Avg. Transaction Value: ${result.details.avg_transaction_value}</li>
              <li>Transactions (30d): {result.details.transaction_count_30d}</li>
              <li>Chargebacks (30d): {result.details.chargeback_count_30d}</li>
            </ul>
          </div>
        </div>
      )}
    </div>
  );
};

export default MerchantLookup;