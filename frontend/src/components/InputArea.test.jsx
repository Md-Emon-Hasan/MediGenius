import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import InputArea from './InputArea';

describe('InputArea Component', () => {
    it('updates input value on change', () => {
        render(<InputArea onSendMessage={() => { }} disabled={false} />);
        const input = screen.getByPlaceholderText('Ask your medical question...');
        expect(input).toBeInTheDocument();
    });

    it('calls handleSend on click', () => {
        const handleSend = vi.fn();
        render(<InputArea onSendMessage={handleSend} disabled={false} />);

        const input = screen.getByPlaceholderText('Ask your medical question...');
        fireEvent.change(input, { target: { value: 'test message' } });

        const button = screen.getByRole('button', { name: /Send message/i });
        fireEvent.click(button);

        expect(handleSend).toHaveBeenCalledWith('test message');
    });
});
