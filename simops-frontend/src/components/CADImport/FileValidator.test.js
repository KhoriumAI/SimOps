import { FileValidator, MAX_SIZE_MB, VALID_EXTENSIONS } from './FileValidator';

describe('FileValidator', () => {
    test('should validate correct MSH file', () => {
        const file = { name: 'test.msh', size: 1024 * 1024 }; // 1MB
        const result = FileValidator.validate(file);
        expect(result.isValid).toBe(true);
    });

    test('should reject invalid extension', () => {
        const file = { name: 'test.pdf', size: 1024 };
        const result = FileValidator.validate(file);
        expect(result.isValid).toBe(false);
        expect(result.error).toContain('Invalid format');
    });

    test('should reject oversized file', () => {
        const file = { name: 'huge.msh', size: (MAX_SIZE_MB + 1) * 1024 * 1024 };
        const result = FileValidator.validate(file);
        expect(result.isValid).toBe(false);
        expect(result.error).toContain('File too large');
    });

    test('should handle missing file', () => {
        const result = FileValidator.validate(null);
        expect(result.isValid).toBe(false);
        expect(result.error).toBe('No file selected');
    });
});
