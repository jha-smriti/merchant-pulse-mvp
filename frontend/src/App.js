import React, { useState } from 'react';
import axios from 'axios';
import './App.css';
import MerchantLookup from './components/MerchantLookup';

const API_BASE = 'http://localhost:8000';

function App() {
  const [merchants, setMerchants] = useState([]);

  React.useEffect(() => {
    axios.get(`${API_BASE}/merchants/`)
      .then(response => setMerchants(response.data.merchant_ids))
      .catch(error => console.error("Error fetching merchants:", error));
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        <h1>Merchant Pulse Lite</h1>
        <p>AI-Powered Merchant Risk Intelligence</p>
      </header>
      <main>
        <MerchantLookup merchants={merchants} />
      </main>
    </div>
  );
}

export default App;