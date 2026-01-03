import { useState, useRef } from 'react'
import { Upload, File, FolderOpen } from 'lucide-react'

export default function FileUpload({ onFileUpload, compact = false }) {
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
      alert('Please upload a valid CAD or mesh file (.step, .stp, .stl, .msh, etc.)')
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
    const validExtensions = ['.step', '.stp', '.stl', '.msh', '.iges', '.igs', '.brep', '.x_t', '.x_b', '.prt', '.sldprt', '.obj', '.vtk']
    const extension = file.name.toLowerCase().match(/\.[^.]+$/)?.[0]
    return validExtensions.includes(extension)
  }

  if (compact) {
    return (
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className={`border-2 border-dashed rounded-lg px-3 py-3 cursor-pointer transition-all ${isDragging
          ? 'border-blue-500 bg-blue-50'
          : 'border-gray-300 bg-white hover:border-blue-400 hover:bg-gray-50'
          }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".step,.stp,.stl,.msh,.iges,.igs,.brep,.x_t,.x_b,.prt,.sldprt,.obj,.vtk"
          onChange={handleFileSelect}
          className="hidden"
        />
        <div className="flex items-center gap-2">
          <FolderOpen className="w-5 h-5 text-blue-500 flex-shrink-0" />
          <span className="text-xs text-gray-700 font-medium truncate">
            {selectedFile ? selectedFile.name : 'Open CAD file...'}
          </span>
        </div>
      </div>
    )
  }

  return (
    <div
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={() => fileInputRef.current?.click()}
      className={`border-2 border-dashed rounded-lg p-6 cursor-pointer transition-all bg-white ${isDragging
        ? 'border-blue-500 bg-blue-50'
        : 'border-gray-300 hover:border-gray-400'
        }`}
    >
      <input
        ref={fileInputRef}
        type="file"
        accept=".step,.stp,.stl,.msh,.iges,.igs,.brep,.x_t,.x_b,.prt,.sldprt,.obj,.vtk"
        onChange={handleFileSelect}
        className="hidden"
      />

      <div className="text-center">
        {selectedFile ? (
          <>
            <File className="w-8 h-8 mx-auto mb-2 text-blue-500" />
            <p className="text-sm font-medium text-blue-600 line-clamp-3 break-all">{selectedFile.name}</p>
            <p className="text-xs text-gray-500 mt-1">
              {(selectedFile.size / 1024).toFixed(1)} KB
            </p>
          </>
        ) : (
          <>
            <p className="text-sm text-gray-700 font-medium">Drag and drop file here</p>
            <p className="text-xs text-gray-500 mt-1">Limit 200MB per file • STEP, STP</p>
            <button className="mt-3 px-4 py-2 border border-gray-300 rounded-md text-sm text-gray-700 hover:bg-gray-50 transition-colors">
              Browse files
            </button>
          </>
        )}
      </div>
    </div>
  )
}
