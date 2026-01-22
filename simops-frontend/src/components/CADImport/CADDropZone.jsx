import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, FileBox, X, CheckCircle, AlertCircle, Loader2 } from 'lucide-react';
import { FileValidator } from './FileValidator';
import UploadProgress from './UploadProgress';

/**
 * Premium CAD Drop Zone Component
 * Supports Idle, Hover, Uploading, Success, and Error states
 */
export default function CADDropZone({
    onFileAccepted,
    onFileRejected,
    acceptedFormats = ['.msh'],
    maxSizeMB = 100
}) {
    const [file, setFile] = useState(null);
    const [uploadState, setUploadState] = useState('idle'); // idle, uploading, success, error
    const [progress, setProgress] = useState(0);
    const [errorMessage, setErrorMessage] = useState(null);

    const performUpload = async (selectedFile) => {
        setUploadState('uploading');
        setProgress(30); // Show immediate progress feedback

        try {
            // Call the parent's handler which does the real upload
            await onFileAccepted(selectedFile);
            setProgress(100);
            setUploadState('success');
        } catch (error) {
            setUploadState('error');
            setErrorMessage(error.message || 'Upload failed');
            if (onFileRejected) onFileRejected(error.message);
        }
    };


    const onDrop = useCallback((acceptedFiles, rejectedFiles) => {
        setErrorMessage(null);
        setUploadState('idle');

        if (rejectedFiles.length > 0) {
            setUploadState('error');
            setErrorMessage('Invalid file type or size.');
            if (onFileRejected) onFileRejected('Invalid file type or size.');
            return;
        }

        if (acceptedFiles.length > 0) {
            const selectedFile = acceptedFiles[0];
            const validation = FileValidator.validate(selectedFile);

            if (!validation.isValid) {
                setUploadState('error');
                setErrorMessage(validation.error);
                if (onFileRejected) onFileRejected(validation.error);
                return;
            }

            setFile(selectedFile);
            performUpload(selectedFile);
        }
    }, [onFileAccepted, onFileRejected]);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        maxFiles: 1,
        accept: {
            'application/x-gmsh': ['.msh'],
        },
        disabled: uploadState === 'uploading'
    });

    const reset = (e) => {
        if (e) e.stopPropagation();
        setFile(null);
        setUploadState('idle');
        setProgress(0);
        setErrorMessage(null);
    };

    return (
        <div className="w-full max-w-lg mx-auto py-8">
            <div
                {...getRootProps()}
                className={`
          relative overflow-hidden transition-colors duration-200 min-h-[12rem]
          bg-card/50 border border-dashed rounded-lg
          flex flex-col items-center justify-center text-center cursor-pointer group
          ${isDragActive ? 'border-primary bg-primary/5' : ''}
          ${uploadState === 'error' ? 'border-destructive bg-destructive/5' : ''}
          ${uploadState === 'success' ? 'border-green-500 bg-green-500/5' : ''}
          ${uploadState === 'idle' && !isDragActive ? 'border-border hover:border-primary hover:bg-muted/10' : ''}
        `}
            >
                <input {...getInputProps()} />

                <AnimatePresence mode="wait">
                    {uploadState === 'idle' && (
                        <motion.div
                            key="idle"
                            initial={{ opacity: 0, y: 5 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, scale: 0.95 }}
                            className="flex flex-col items-center gap-4 px-8"
                        >
                            <div className={`
                w-12 h-12 rounded flex items-center justify-center transition-transform duration-200
                ${isDragActive ? 'text-primary scale-110' : 'text-muted-foreground group-hover:text-primary'}
              `}>
                                <Upload className="w-8 h-8" />
                            </div>
                            <div className="space-y-1">
                                <h3 className="text-sm font-bold text-foreground uppercase tracking-wider">
                                    {isDragActive ? 'Drop to Import' : 'Import Mesh'}
                                </h3>
                                <p className="text-[10px] font-mono text-muted-foreground max-w-xs">
                                    Gmsh MSH Format
                                </p>
                            </div>
                        </motion.div>
                    )}

                    {uploadState === 'uploading' && (
                        <motion.div
                            key="uploading"
                            initial={{ opacity: 0, scale: 0.9 }}
                            animate={{ opacity: 1, scale: 1 }}
                            className="w-full px-12"
                        >
                            <UploadProgress progress={progress} filename={file?.name} />
                        </motion.div>
                    )}

                    {uploadState === 'success' && (
                        <motion.div
                            key="success"
                            data-testid="success-state"
                            initial={{ opacity: 0, scale: 0.9 }}
                            animate={{ opacity: 1, scale: 1 }}
                            className="flex flex-col items-center gap-4 px-10"
                        >
                            <div className="w-20 h-20 bg-green-500 text-white rounded-full flex items-center justify-center shadow-xl shadow-green-500/20">
                                <CheckCircle className="w-10 h-10" />
                            </div>
                            <div className="space-y-1">
                                <h3 className="text-xl font-bold text-gray-800">{file?.name}</h3>
                                <p className="text-sm font-medium text-green-600 flex items-center justify-center gap-1.5">
                                    Successfully imported & ready to solve
                                </p>
                            </div>
                            <button
                                onClick={reset}
                                className="mt-4 px-6 py-2 rounded-xl bg-gray-100 text-gray-600 text-sm font-semibold hover:bg-gray-200 transition-colors"
                            >
                                Change File
                            </button>
                        </motion.div>
                    )}

                    {uploadState === 'error' && (
                        <motion.div
                            key="error"
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            className="flex flex-col items-center gap-5 px-10"
                        >
                            <div className="w-16 h-16 bg-red-100 text-red-500 rounded-2xl flex items-center justify-center">
                                <AlertCircle className="w-8 h-8" />
                            </div>
                            <div className="space-y-1">
                                <h3 className="text-lg font-bold text-gray-800">Upload Failed</h3>
                                <p className="text-sm text-red-500 font-medium">{errorMessage}</p>
                            </div>
                            <button
                                onClick={reset}
                                className="mt-2 text-sm font-bold text-brand-600 hover:text-brand-700 underline underline-offset-4"
                            >
                                Try again
                            </button>
                        </motion.div>
                    )}
                </AnimatePresence>

                {/* Decorative glass highlight */}
                <div className="absolute -top-24 -right-24 w-48 h-48 bg-brand-400/10 blur-[60px] rounded-full" />
                <div className="absolute -bottom-24 -left-24 w-48 h-48 bg-brand-600/10 blur-[60px] rounded-full" />
            </div>
        </div>
    );
}
