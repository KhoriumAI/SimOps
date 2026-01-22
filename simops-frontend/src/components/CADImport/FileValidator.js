/**
 * Utility for validating CAD files
 */
export const VALID_EXTENSIONS = ['.msh'];
export const MAX_SIZE_MB = 100;

export class FileValidator {
    /**
     * Validates a file against allowed extensions and size
     * @param {File} file 
     * @returns {{isValid: boolean, error: string|null}}
     */
    static validate(file) {
        if (!file) {
            return { isValid: false, error: 'No file selected' };
        }

        const extension = '.' + file.name.split('.').pop().toLowerCase();
        if (!VALID_EXTENSIONS.includes(extension)) {
            return {
                isValid: false,
                error: `Invalid format: ${extension}. Supported formats: ${VALID_EXTENSIONS.join(', ')}`
            };
        }

        const sizeMB = file.size / (1024 * 1024);
        if (sizeMB > MAX_SIZE_MB) {
            return {
                isValid: false,
                error: `File too large: ${sizeMB.toFixed(1)}MB. Maximum size is ${MAX_SIZE_MB}MB.`
            };
        }

        return { isValid: true, error: null };
    }

    static async verifyMagicBytes(file) {
        // In a production app, we would read the first few KB 
        // to search for 'Gmsh' or '$MeshFormat' markers.
        return true;
    }
}
