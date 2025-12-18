import { useState, useRef, useCallback } from 'react'
import { Upload, File, X, FolderOpen, CheckCircle, AlertCircle, Loader2 } from 'lucide-react'

/**
 * BatchUpload Component
 * 
 * Multi-file dropzone for batch mesh generation.
 * Supports drag & drop of multiple files or folder selection.
 */
export default function BatchUpload({ 
  onFilesSelected, 
  maxFiles = 10, 
  maxFileSize = 500 * 1024 * 1024,  // 500MB
  disabled = false 
}) {
  const [isDragging, setIsDragging] = useState(false)
  const [selectedFiles, setSelectedFiles] = useState([])
  const [errors, setErrors] = useState([])
  const fileInputRef = useRef(null)
  const folderInputRef = useRef(null)

  const allowedExtensions = ['.step', '.stp', '.stl']

  const validateFile = useCallback((file) => {
    const ext = file.name.toLowerCase().slice(file.name.lastIndexOf('.'))
    
    if (!allowedExtensions.includes(ext)) {
      return { valid: false, error: `Invalid type: ${ext}` }
    }
    
    if (file.size > maxFileSize) {
      return { valid: false, error: `Too large: ${(file.size / (1024 * 1024)).toFixed(1)}MB` }
    }
    
    return { valid: true }
  }, [maxFileSize])

  const processFiles = useCallback((files) => {
    const fileArray = Array.from(files)
    const newFiles = []
    const newErrors = []

    // Check total count
    const totalCount = selectedFiles.length + fileArray.length
    if (totalCount > maxFiles) {
      newErrors.push(`Max ${maxFiles} files allowed. Selected: ${fileArray.length}, Current: ${selectedFiles.length}`)
    }

    for (const file of fileArray) {
      if (selectedFiles.length + newFiles.length >= maxFiles) break

      // Check for duplicates
      const isDuplicate = selectedFiles.some(f => f.name === file.name && f.size === file.size)
      if (isDuplicate) {
        newErrors.push(`Duplicate: ${file.name}`)
        continue
      }

      const validation = validateFile(file)
      if (validation.valid) {
        newFiles.push({
          file,
          name: file.name,
          size: file.size,
          status: 'pending',  // pending, uploading, uploaded, error
          id: `${file.name}-${file.size}-${Date.now()}`
        })
      } else {
        newErrors.push(`${file.name}: ${validation.error}`)
      }
    }

    if (newFiles.length > 0) {
      const updated = [...selectedFiles, ...newFiles]
      setSelectedFiles(updated)
      onFilesSelected?.(updated.map(f => f.file))
    }

    if (newErrors.length > 0) {
      setErrors(prev => [...prev, ...newErrors].slice(-5))  // Keep last 5 errors
    }
  }, [selectedFiles, maxFiles, validateFile, onFilesSelected])

  const handleDragOver = (e) => {
    e.preventDefault()
    if (!disabled) setIsDragging(true)
  }

  const handleDragLeave = (e) => {
    e.preventDefault()
    setIsDragging(false)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragging(false)
    if (disabled) return

    const items = e.dataTransfer.items
    const files = []

    // Handle folder drop
    if (items) {
      for (const item of items) {
        if (item.kind === 'file') {
          const file = item.getAsFile()
          if (file) files.push(file)
        }
      }
    } else {
      files.push(...e.dataTransfer.files)
    }

    processFiles(files)
  }

  const handleFileSelect = (e) => {
    if (e.target.files) {
      processFiles(e.target.files)
    }
    e.target.value = ''  // Reset to allow re-selecting same files
  }

  const removeFile = (id) => {
    const updated = selectedFiles.filter(f => f.id !== id)
    setSelectedFiles(updated)
    onFilesSelected?.(updated.map(f => f.file))
  }

  const clearAll = () => {
    setSelectedFiles([])
    setErrors([])
    onFilesSelected?.([])
  }

  const formatFileSize = (bytes) => {
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  }

  const getStatusIcon = (status) => {
    switch (status) {
      case 'uploaded':
        return <CheckCircle className="w-4 h-4 text-green-500" />
      case 'error':
        return <AlertCircle className="w-4 h-4 text-red-500" />
      case 'uploading':
        return <Loader2 className="w-4 h-4 text-blue-500 animate-spin" />
      default:
        return <File className="w-4 h-4 text-gray-400" />
    }
  }

  return (
    <div className="space-y-3">
      {/* Drop Zone */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`
          border-2 border-dashed rounded-lg p-6 transition-all cursor-pointer
          ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
          ${isDragging
            ? 'border-blue-500 bg-blue-500/10'
            : 'border-gray-300 hover:border-gray-400 bg-gray-50'
          }
        `}
        onClick={() => !disabled && fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".step,.stp,.stl"
          onChange={handleFileSelect}
          className="hidden"
          disabled={disabled}
        />
        <input
          ref={folderInputRef}
          type="file"
          webkitdirectory=""
          directory=""
          multiple
          onChange={handleFileSelect}
          className="hidden"
          disabled={disabled}
        />

        <div className="text-center">
          <Upload className={`w-10 h-10 mx-auto mb-3 ${isDragging ? 'text-blue-500' : 'text-gray-400'}`} />
          <p className="text-sm font-medium text-gray-700">
            Drop CAD files here
          </p>
          <p className="text-xs text-gray-500 mt-1">
            or click to browse
          </p>
          <div className="flex gap-2 justify-center mt-3">
            <button
              onClick={(e) => { e.stopPropagation(); fileInputRef.current?.click() }}
              className="px-3 py-1.5 text-xs bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
              disabled={disabled}
            >
              Select Files
            </button>
            <button
              onClick={(e) => { e.stopPropagation(); folderInputRef.current?.click() }}
              className="px-3 py-1.5 text-xs bg-gray-200 text-gray-700 rounded hover:bg-gray-300 transition-colors flex items-center gap-1"
              disabled={disabled}
            >
              <FolderOpen className="w-3 h-3" />
              Folder
            </button>
          </div>
          <p className="text-xs text-gray-400 mt-3">
            Max {maxFiles} files • STEP, STP, STL • Up to {maxFileSize / (1024 * 1024)}MB each
          </p>
        </div>
      </div>

      {/* Selected Files List */}
      {selectedFiles.length > 0 && (
        <div className="bg-white rounded-lg border border-gray-200">
          <div className="flex items-center justify-between px-3 py-2 border-b border-gray-100 bg-gray-50">
            <span className="text-xs font-medium text-gray-600">
              {selectedFiles.length} file{selectedFiles.length !== 1 ? 's' : ''} selected
            </span>
            <button
              onClick={clearAll}
              className="text-xs text-red-500 hover:text-red-600"
              disabled={disabled}
            >
              Clear All
            </button>
          </div>
          <div className="max-h-48 overflow-y-auto divide-y divide-gray-100">
            {selectedFiles.map((item) => (
              <div
                key={item.id}
                className="flex items-center gap-2 px-3 py-2 hover:bg-gray-50"
              >
                {getStatusIcon(item.status)}
                <div className="flex-1 min-w-0">
                  <p className="text-xs font-medium text-gray-800 truncate">
                    {item.name}
                  </p>
                  <p className="text-[10px] text-gray-500">
                    {formatFileSize(item.size)}
                  </p>
                </div>
                <button
                  onClick={() => removeFile(item.id)}
                  className="p-1 hover:bg-gray-200 rounded text-gray-400 hover:text-gray-600"
                  disabled={disabled || item.status === 'uploading'}
                >
                  <X className="w-3 h-3" />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Errors */}
      {errors.length > 0 && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-2">
          <div className="flex items-center justify-between mb-1">
            <span className="text-xs font-medium text-red-700">Errors</span>
            <button
              onClick={() => setErrors([])}
              className="text-xs text-red-500 hover:text-red-600"
            >
              Clear
            </button>
          </div>
          <ul className="space-y-0.5">
            {errors.map((error, i) => (
              <li key={i} className="text-xs text-red-600 flex items-start gap-1">
                <AlertCircle className="w-3 h-3 mt-0.5 flex-shrink-0" />
                {error}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
