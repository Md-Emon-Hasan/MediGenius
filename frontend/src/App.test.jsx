import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import App from './App';

describe('App Integration', () => {
    // Mock fetch
    const mockFetch = vi.fn();
    global.fetch = mockFetch;

    beforeEach(() => {
        mockFetch.mockClear();
        // Default mocks
        mockFetch.mockResolvedValue({
            json: () => Promise.resolve({ success: true, messages: [], sessions: [] }),
            ok: true
        });
    });

    afterEach(() => {
        vi.restoreAllMocks();
    });

    it('renders app structure', async () => {
        render(<App />);
        expect(screen.getByText('Medical AI Assistant')).toBeInTheDocument();
        expect(screen.getByText('MediGenius')).toBeInTheDocument();
        expect(screen.getByPlaceholderText('Ask your medical question...')).toBeInTheDocument();
    });

    it('loads sessions on mount', async () => {
        const mockSessions = [{ session_id: '1', preview: 'Flu symptoms', last_active: '2023' }];
        mockFetch.mockImplementation((url) => {
            if (url === '/api/v1/sessions') {
                return Promise.resolve({
                    json: () => Promise.resolve({ success: true, sessions: mockSessions }),
                    ok: true
                });
            }
            return Promise.resolve({ json: () => Promise.resolve({ success: true }), ok: true });
        });

        render(<App />);

        await waitFor(() => {
            expect(screen.getByText('Flu symptoms')).toBeInTheDocument();
        });
    });

    it('sends a message and displays response', async () => {
        // Setup mocks
        mockFetch.mockImplementation((url, options) => {
            if (url === '/api/v1/chat') {
                return Promise.resolve({
                    json: () => Promise.resolve({
                        success: true,
                        response: 'I can help with that.',
                        source: 'test-source',
                        timestamp: 'now'
                    }),
                    ok: true
                });
            }
            return Promise.resolve({ json: () => Promise.resolve({ success: true, sessions: [] }), ok: true });
        });

        render(<App />);

        const input = screen.getByPlaceholderText('Ask your medical question...');
        fireEvent.change(input, { target: { value: 'Headache' } });

        const sendBtn = screen.getByRole('button', { name: /Send message/i });
        fireEvent.click(sendBtn);

        expect(screen.getByText('Headache')).toBeInTheDocument(); // Optimistic UI

        await waitFor(() => {
            expect(screen.getByText('I can help with that.')).toBeInTheDocument();
        });
    });

    it('creates new chat', async () => {
        mockFetch.mockImplementation((url) => {
            if (url === '/api/v1/new-chat') {
                return Promise.resolve({
                    json: () => Promise.resolve({ success: true, session_id: 'new-session' }),
                    ok: true
                });
            }
            return Promise.resolve({ json: () => Promise.resolve({ success: true, sessions: [] }), ok: true });
        });

        render(<App />);

        const newChatBtn = screen.getByText('New Chat');
        fireEvent.click(newChatBtn);

        await waitFor(() => {
            // Check if new chat API was called
            expect(mockFetch).toHaveBeenCalledWith('/api/v1/new-chat', expect.any(Object));
        });
    });
});
