import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import CADDropZone from './CADDropZone';
import '@testing-library/jest-dom';

describe('CADDropZone', () => {
    const mockOnAccepted = jest.fn();
    const mockOnRejected = jest.fn();

    beforeEach(() => {
        jest.clearAllMocks();
    });

    test('renders idle state correctly', () => {
        render(<CADDropZone onFileAccepted={mockOnAccepted} onFileRejected={mockOnRejected} />);
        expect(screen.getByText(/Import Mesh/i)).toBeInTheDocument();
        expect(screen.getByText(/Gmsh MSH Format/i)).toBeInTheDocument();
    });

    test('shows error for invalid file type', async () => {
        render(<CADDropZone onFileAccepted={mockOnAccepted} onFileRejected={mockOnRejected} />);
        const input = screen.getByLabelText(/Import Mesh/i, { selector: 'input' }); // Note: This might need adjustment based on how dropzone labels inputs

        // Simulating drop is complex in RTL, often easier to test FileValidator directly
        // and test UI state transitions via props if they were exposed, but we'll try a basic input change
        const file = new File(['test'], 'test.txt', { type: 'text/plain' });
        Object.defineProperty(input, 'files', { value: [file] });
        fireEvent.change(input);

        await waitFor(() => {
            expect(screen.getByText(/Upload Failed/i)).toBeInTheDocument();
        });
    });
});
