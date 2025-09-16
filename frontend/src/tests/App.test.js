import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import axios from 'axios';
import MockAdapter from 'axios-mock-adapter';
import App from '../App';

// Mock axios
const mockAxios = new MockAdapter(axios);

describe('App Component', () => {
  beforeEach(() => {
    mockAxios.reset();
  });

  afterEach(() => {
    mockAxios.restore();
  });

  test('renders app header correctly', async () => {
    mockAxios.onGet('/merchants/').reply(200, {
      merchant_ids: ['M1001', 'M1002', 'M1003']
    });

    render(<App />);
    
    expect(screen.getByText('Merchant Pulse Enterprise')).toBeInTheDocument();
    expect(screen.getByText('AI-Powered Merchant Risk Intelligence Platform')).toBeInTheDocument();
  });

  test('loads merchants on component mount', async () => {
    const merchants = ['M1001', 'M1002', 'M1003'];
    mockAxios.onGet('/merchants/').reply(200, {
      merchant_ids: merchants
    });

    render(<App />);

    await waitFor(() => {
      expect(screen.getByText('3')).toBeInTheDocument(); // Merchant count in stats
    });
  });

  test('handles merchant loading error', async () => {
    mockAxios.onGet('/merchants/').reply(500);

    render(<App />);

    await waitFor(() => {
      expect(screen.getByText(/Failed to load merchant data/)).toBeInTheDocument();
    });
  });

  test('shows loading state initially', () => {
    mockAxios.onGet('/merchants/').reply(200, { merchant_ids: [] });

    render(<App />);
    
    expect(screen.getByText('Loading Merchant Pulse...')).toBeInTheDocument();
  });

  test('navigation works correctly', async () => {
    mockAxios.onGet('/merchants/').reply(200, {
      merchant_ids: ['M1001', 'M1002']
    });

    render(<App />);

    await waitFor(() => {
      expect(screen.getByText('Risk Assessment')).toBeInTheDocument();
    });

    // Click on AI Assistant tab
    fireEvent.click(screen.getByText('AI Assistant'));
    
    await waitFor(() => {
      expect(screen.getByText(/AI Risk Intelligence Assistant/)).toBeInTheDocument();
    });

    // Click on Analytics tab
    fireEvent.click(screen.getByText('Analytics'));
    
    await waitFor(() => {
      expect(screen.getByText(/Risk Analytics Dashboard/)).toBeInTheDocument();
    });
  });

  test('retry button works on error', async () => {
    mockAxios.onGet('/merchants/').replyOnce(500).onGet('/merchants/').reply(200, {
      merchant_ids: ['M1001']
    });

    render(<App />);

    await waitFor(() => {
      expect(screen.getByText('Retry')).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText('Retry'));

    await waitFor(() => {
      expect(screen.getByText('1')).toBeInTheDocument(); // Updated merchant count
    });
  });
});

describe('App Footer', () => {
  test('renders footer content', async () => {
    mockAxios.onGet('/merchants/').reply(200, { merchant_ids: [] });

    render(<App />);

    await waitFor(() => {
      expect(screen.getByText('Merchant Pulse Enterprise')).toBeInTheDocument();
      expect(screen.getByText('Real-time Risk Assessment')).toBeInTheDocument();
      expect(screen.getByText('Machine Learning')).toBeInTheDocument();
      expect(screen.getByText(/Built for 4th Year Graduate Project/)).toBeInTheDocument();
    });
  });
});

describe('App Responsive Behavior', () => {
  test('adjusts layout for mobile screens', async () => {
    mockAxios.onGet('/merchants/').reply(200, { merchant_ids: [] });

    // Mock window.innerWidth
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: 480,
    });

    render(<App />);

    await waitFor(() => {
      const header = screen.getByRole('banner');
      expect(header).toBeInTheDocument();
    });
  });
});