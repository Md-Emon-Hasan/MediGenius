import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import Sidebar from './Sidebar';

describe('Sidebar Component', () => {
    const mockSessions = [
        { session_id: '1', preview: 'Session 1', last_active: '2023-01-01' },
        { session_id: '2', preview: 'Session 2', last_active: '2023-01-02' }
    ];

    it('renders correctly when open', () => {
        render(
            <Sidebar
                isOpen={true}
                toggleSidebar={() => { }}
                chatHistory={mockSessions}
                currentSessionId="1"
                onLoadSession={() => { }}
                onNewChat={() => { }}
                onDeleteSession={() => { }}
            />
        );
        expect(screen.getByText('MediGenius')).toBeInTheDocument();
        expect(screen.getByText('Session 1')).toBeInTheDocument();
    });

    it('calls createNewChat on button click', () => {
        const onNewChat = vi.fn();
        render(
            <Sidebar
                isOpen={true}
                toggleSidebar={() => { }}
                chatHistory={[]}
                currentSessionId={null}
                onLoadSession={() => { }}
                onNewChat={onNewChat}
                onDeleteSession={() => { }}
            />
        );

        const newChatBtn = screen.getByText('New Chat');
        fireEvent.click(newChatBtn);
        expect(onNewChat).toHaveBeenCalled();
    });
});
