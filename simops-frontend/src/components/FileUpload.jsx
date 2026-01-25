import React, { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { motion, AnimatePresence } from 'framer-motion'
import { Upload, FileType, X, CheckCircle, Loader2, FileBox } from 'lucide-react'

export default function FileUpload({ onFileSelect, selectedFile, isUploading = false, compact = false }) {
    const [error, setError] = useState(null)

    const onDrop = useCallback((acceptedFiles, rejectedFiles) => {
        setError(null)

        if (rejectedFiles.length > 0) {
            setError('Invalid file type. Please upload MSH files.')
            return
        }

        if (acceptedFiles.length > 0) {
            const selectedFile = acceptedFiles[0]
            const ext = selectedFile.name.split('.').pop().toLowerCase()
            if (ext === 'msh') {
                onFileSelect(selectedFile)
            } else {
                setError('Unsupported format. Please upload Gmsh MSH files.')
            }
        }
    }, [onFileSelect])

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        maxFiles: 1,
        accept: {
            'application/x-gmsh': ['.msh']
        },
        disabled: isUploading || !!selectedFile
    })

    const clearFile = (e) => {
        if (e) e.stopPropagation()
        onFileSelect(null)
        setError(null)
    }

    // Compact variation for sidebar (Engineering Dense)
    if (compact) {
        return (
            <div className="w-full">
                {!selectedFile ? (
                    <div
                        {...getRootProps()}
                        className={`
              relative group cursor-pointer
              border border-dashed rounded px-2 py-4 transition-colors
              ${isDragActive
                                ? 'border-primary bg-primary/5'
                                : 'border-border hover:border-primary hover:bg-muted/30'
                            }
            `}
                    >
                        <input {...getInputProps()} />
                        <div className="flex flex-col items-center justify-center text-center gap-1">
                            <Upload className={`w-4 h-4 ${isDragActive ? 'text-primary' : 'text-muted-foreground'}`} />
                            <p className="text-[10px] uppercase tracking-wider font-bold text-muted-foreground">
                                {isDragActive ? 'Drop MSH' : 'Import Mesh'}
                            </p>
                        </div>
                    </div>
                ) : (
                    <div className="bg-card border border-border rounded p-2 relative overflow-hidden group">
                        {isUploading && (
                            <div className="absolute inset-0 bg-background/80 backdrop-blur-[1px] z-10 flex items-center justify-center">
                                <Loader2 className="w-4 h-4 text-primary animate-spin" />
                            </div>
                        )}

                        <div className="flex items-start gap-2">
                            <div className="p-1 bg-primary/10 text-primary rounded">
                                <FileBox className="w-4 h-4" />
                            </div>
                            <div className="flex-1 min-w-0">
                                <p className="text-[10px] font-bold text-foreground truncate">{selectedFile.name}</p>
                                <p className="text-[9px] font-mono text-muted-foreground">{(selectedFile.size / 1024 / 1024).toFixed(2)} MB</p>
                            </div>
                            {!isUploading && (
                                <button
                                    onClick={clearFile}
                                    className="p-0.5 text-muted-foreground hover:text-destructive hover:bg-destructive/10 rounded transition-colors"
                                >
                                    <X className="w-3 h-3" />
                                </button>
                            )}
                        </div>
                        {/* Status Bar */}
                        <div className="mt-1 flex items-center gap-1 text-[9px] font-mono text-green-500">
                            <CheckCircle className="w-3 h-3" />
                            <span>READY TO SOLVE</span>
                        </div>
                    </div>
                )}
            </div>
        )
    }

    // Full screen / Hero variation (Fallback)
    return (
        <div className="w-full max-w-xl mx-auto">
            <AnimatePresence>
                {error && (
                    <motion.div
                        initial={{ opacity: 0, y: -10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -10 }}
                        className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-600 flex items-center gap-2"
                    >
                        <X className="w-4 h-4" />
                        {error}
                    </motion.div>
                )}
            </AnimatePresence>

            <div
                {...getRootProps()}
                className={`
          relative overflow-hidden transition-all duration-500 ease-out
          ${selectedFile ? 'h-32' : 'h-64'}
          bg-white/40 backdrop-blur-md border-2 border-dashed rounded-3xl
          flex flex-col items-center justify-center text-center cursor-pointer group
          ${isDragActive
                        ? 'border-brand-500 bg-brand-50/60 scale-[1.01] shadow-xl shadow-brand-500/10'
                        : selectedFile
                            ? 'border-green-500/50 bg-green-50/30'
                            : 'border-gray-300 hover:border-brand-400 hover:bg-white/60 hover:shadow-lg'
                    }
        `}
            >
                <input {...getInputProps()} />

                <AnimatePresence mode="wait">
                    {!selectedFile ? (
                        <motion.div
                            key="drop-prompt"
                            initial={{ opacity: 0, scale: 0.9 }}
                            animate={{ opacity: 1, scale: 1 }}
                            exit={{ opacity: 0, scale: 0.9 }}
                            className="flex flex-col items-center gap-4 px-6"
                        >
                            <div className={`
                w-16 h-16 rounded-2xl flex items-center justify-center shadow-sm transition-all duration-300
                ${isDragActive ? 'bg-brand-500 text-white rotate-12 scale-110' : 'bg-white text-brand-500 group-hover:shadow-md group-hover:-translate-y-1'}
              `}>
                                <Upload className="w-8 h-8" />
                            </div>
                            <div className="space-y-1">
                                <p className="text-lg font-semibold text-gray-700 font-sans">
                                    {isDragActive ? 'Drop MSH' : 'Drag & Drop MSH File'}
                                </p>
                                <p className="text-sm text-gray-500">
                                    Support for Gmsh MSH Format
                                </p>
                            </div>
                        </motion.div>
                    ) : (
                        <motion.div
                            key="file-preview"
                            initial={{ opacity: 0, scale: 0.9 }}
                            animate={{ opacity: 1, scale: 1 }}
                            className="flex items-center gap-6 px-8 w-full"
                        >
                            <div className="w-16 h-16 bg-green-100 text-green-600 rounded-2xl flex items-center justify-center shadow-inner">
                                <CheckCircle className="w-8 h-8" />
                            </div>
                            <div className="flex-1 text-left min-w-0">
                                <h3 className="text-lg font-bold text-gray-800 truncate">{selectedFile.name}</h3>
                                <p className="text-sm text-gray-500 font-medium">
                                    {(selectedFile.size / 1024 / 1024).toFixed(2)} MB • Ready to Process
                                </p>
                            </div>
                            <div
                                onClick={clearFile}
                                className="w-10 h-10 rounded-full flex items-center justify-center text-gray-400 hover:bg-red-50 hover:text-red-500 transition-colors cursor-pointer"
                                title="Remove file"
                            >
                                <X className="w-5 h-5" />
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>
        </div>
    )
}
