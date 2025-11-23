import { useState, useRef } from 'react'
import { Upload, File } from 'lucide-react'

export default function FileUpload({ onFileUpload }) {
  const [isDragging, setIsDragging] = useState(false)
  const [selectedFile, setSelectedFile] = useState(null)
  const fileInputRef = useRef(null)

  const handleDragOver = (e) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = () => {
    setIsDragging(false)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragging(false)

    const file = e.dataTransfer.files[0]
    if (file && isValidFile(file)) {
      setSelectedFile(file)
      onFileUpload(file)
    } else {
      alert('Please upload a valid CAD file (.step, .stp, or .stl)')
    }
  }

  const handleFileSelect = (e) => {
    const file = e.target.files[0]
    if (file && isValidFile(file)) {
      setSelectedFile(file)
      onFileUpload(file)
    }
  }

  const isValidFile = (file) => {
    const validExtensions = ['.step', '.stp', '.stl']
    const extension = file.name.toLowerCase().match(/\.[^.]+$/)?.[0]
    return validExtensions.includes(extension)
  }

  return (
    <div className="space-y-3">
      <h3 className="font-semibold text-sm uppercase text-gray-400">Load CAD File</h3>

      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className={`border-2 border-dashed rounded-lg p-6 cursor-pointer transition-all ${
          isDragging
            ? 'border-blue-500 bg-blue-500/10'
            : 'border-gray-600 hover:border-gray-500 bg-gray-900/50'
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".step,.stp,.stl"
          onChange={handleFileSelect}
          className="hidden"
        />

        <div className="text-center">
          {selectedFile ? (
            <>
              <File className="w-8 h-8 mx-auto mb-2 text-blue-400" />
              <p className="text-sm font-medium text-blue-400">{selectedFile.name}</p>
              <p className="text-xs text-gray-500 mt-1">
                {(selectedFile.size / 1024).toFixed(1)} KB
              </p>
            </>
          ) : (
            <>
              <Upload className="w-8 h-8 mx-auto mb-2 text-gray-500" />
              <p className="text-sm text-gray-400">Drop CAD file here</p>
              <p className="text-xs text-gray-600 mt-1">or click to browse</p>
              <p className="text-xs text-gray-700 mt-2">STEP, STP, STL</p>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
