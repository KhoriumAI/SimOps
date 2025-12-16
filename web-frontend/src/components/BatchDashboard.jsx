import { useState, useEffect } from 'react'
import { 
  Play, Pause, Download, Trash2, ChevronDown, ChevronRight,
  CheckCircle, AlertCircle, Clock, Loader2, FileText, Settings
} from 'lucide-react'

/**
 * BatchDashboard Component
 * 
 * Displays batch processing status, file list, job progress,
 * and provides controls for starting/stopping batch operations.
 */
export default function BatchDashboard({ 
  batch,
  onStart,
  onCancel,
  onDownload,
  onDelete,
  isLoading = false
}) {
  const [expandedFiles, setExpandedFiles] = useState({})

  if (!batch) {
    return (
      <div className="bg-white rounded-lg border border-gray-200 p-6 text-center">
        <FileText className="w-12 h-12 mx-auto mb-3 text-gray-300" />
        <p className="text-sm text-gray-500">No batch selected</p>
        <p className="text-xs text-gray-400 mt-1">Upload files to create a batch</p>
      </div>
    )
  }

  const toggleFile = (fileId) => {
    setExpandedFiles(prev => ({
      ...prev,
      [fileId]: !prev[fileId]
    }))
  }

  const getStatusColor = (status) => {
    const colors = {
      pending: 'bg-gray-100 text-gray-600',
      ready: 'bg-blue-100 text-blue-600',
      uploading: 'bg-yellow-100 text-yellow-600',
      processing: 'bg-blue-100 text-blue-600',
      completed: 'bg-green-100 text-green-600',
      failed: 'bg-red-100 text-red-600',
      cancelled: 'bg-gray-100 text-gray-500',
      queued: 'bg-purple-100 text-purple-600'
    }
    return colors[status] || 'bg-gray-100 text-gray-600'
  }

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-green-500" />
      case 'failed':
        return <AlertCircle className="w-4 h-4 text-red-500" />
      case 'processing':
      case 'uploading':
        return <Loader2 className="w-4 h-4 text-blue-500 animate-spin" />
      case 'queued':
        return <Clock className="w-4 h-4 text-purple-500" />
      default:
        return <Clock className="w-4 h-4 text-gray-400" />
    }
  }

  const formatDuration = (seconds) => {
    if (!seconds) return '-'
    if (seconds < 60) return `${seconds.toFixed(1)}s`
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}m ${secs}s`
  }

  const formatFileSize = (bytes) => {
    if (!bytes) return '-'
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  }

  const canStart = batch.status === 'ready' && batch.total_jobs > 0
  const canCancel = batch.status === 'processing'
  const canDownload = batch.completed_jobs > 0
  const canDelete = !['processing', 'uploading'].includes(batch.status)

  return (
    <div className="bg-white rounded-lg border border-gray-200 overflow-hidden relative">
      {/* Loading Overlay */}
      {isLoading && (
        <div className="absolute inset-0 bg-white/80 flex items-center justify-center z-10">
          <div className="flex items-center gap-2 text-blue-600">
            <Loader2 className="w-5 h-5 animate-spin" />
            <span className="text-sm font-medium">Processing...</span>
          </div>
        </div>
      )}
      
      {/* Processing Animation in Header */}
      {batch.status === 'processing' && (
        <div className="absolute top-0 left-0 right-0 h-1 bg-gray-200 overflow-hidden">
          <div className="h-full w-1/3 bg-blue-500 animate-pulse rounded" 
               style={{ animation: 'slide 1.5s infinite ease-in-out' }} />
        </div>
      )}
      
      {/* Header */}
      <div className="px-4 py-3 bg-gray-50 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="font-semibold text-gray-800">
              {batch.name || `Batch ${batch.id.slice(0, 8)}`}
            </h3>
            <div className="flex items-center gap-2 mt-1">
              <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${getStatusColor(batch.status)}`}>
                {batch.status}
              </span>
              {batch.mesh_independence && (
                <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-purple-100 text-purple-600">
                  Independence Study
                </span>
              )}
            </div>
          </div>
          
          {/* Actions */}
          <div className="flex items-center gap-2">
            {canStart && (
              <button
                onClick={onStart}
                disabled={isLoading}
                className="flex items-center gap-1.5 px-3 py-1.5 bg-green-600 text-white text-xs font-medium rounded hover:bg-green-700 transition-colors disabled:opacity-50"
              >
                <Play className="w-3.5 h-3.5" />
                Start
              </button>
            )}
            {canCancel && (
              <button
                onClick={onCancel}
                disabled={isLoading}
                className="flex items-center gap-1.5 px-3 py-1.5 bg-orange-600 text-white text-xs font-medium rounded hover:bg-orange-700 transition-colors disabled:opacity-50"
              >
                <Pause className="w-3.5 h-3.5" />
                Cancel
              </button>
            )}
            {canDownload && (
              <button
                onClick={onDownload}
                disabled={isLoading}
                className="flex items-center gap-1.5 px-3 py-1.5 bg-blue-600 text-white text-xs font-medium rounded hover:bg-blue-700 transition-colors disabled:opacity-50"
              >
                <Download className="w-3.5 h-3.5" />
                Download ZIP
              </button>
            )}
            {canDelete && (
              <button
                onClick={onDelete}
                disabled={isLoading}
                className="flex items-center gap-1.5 px-2 py-1.5 text-red-600 text-xs font-medium rounded hover:bg-red-50 transition-colors disabled:opacity-50"
                title="Delete Batch"
              >
                <Trash2 className="w-3.5 h-3.5" />
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Progress Bar */}
      {batch.total_jobs > 0 && (
        <div className="px-4 py-3 border-b border-gray-100">
          <div className="flex items-center justify-between text-xs text-gray-600 mb-1.5">
            <span>Progress</span>
            <span>
              {batch.completed_jobs} / {batch.total_jobs} jobs
              {batch.failed_jobs > 0 && (
                <span className="text-red-500 ml-1">({batch.failed_jobs} failed)</span>
              )}
            </span>
          </div>
          <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
            <div className="h-full flex">
              <div 
                className="bg-green-500 transition-all duration-300"
                style={{ width: `${(batch.completed_jobs / batch.total_jobs) * 100}%` }}
              />
              <div 
                className="bg-red-500 transition-all duration-300"
                style={{ width: `${(batch.failed_jobs / batch.total_jobs) * 100}%` }}
              />
            </div>
          </div>
          <div className="flex items-center justify-between text-[10px] text-gray-500 mt-1">
            <span>{batch.total_files} files</span>
            <span>{batch.progress?.toFixed(1) || 0}% complete</span>
          </div>
        </div>
      )}

      {/* Settings Summary */}
      <div className="px-4 py-2 border-b border-gray-100 bg-gray-50/50">
        <div className="flex items-center gap-4 text-xs text-gray-600">
          <div className="flex items-center gap-1">
            <Settings className="w-3 h-3" />
            <span>{batch.mesh_strategy}</span>
          </div>
          {batch.mesh_independence && (
            <span className="text-purple-600">Coarse + Medium + Fine</span>
          )}
          <span>Parallel: {batch.parallel_limit}</span>
        </div>
      </div>

      {/* Files List */}
      {batch.files && batch.files.length > 0 && (
        <div className="max-h-80 overflow-y-auto">
          {batch.files.map((file) => (
            <div key={file.id} className="border-b border-gray-100 last:border-0">
              {/* File Header */}
              <div 
                className="flex items-center gap-2 px-4 py-2 hover:bg-gray-50 cursor-pointer"
                onClick={() => toggleFile(file.id)}
              >
                {expandedFiles[file.id] 
                  ? <ChevronDown className="w-4 h-4 text-gray-400" />
                  : <ChevronRight className="w-4 h-4 text-gray-400" />
                }
                {getStatusIcon(file.status)}
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-gray-800 truncate">
                    {file.original_filename}
                  </p>
                  <p className="text-xs text-gray-500">
                    {formatFileSize(file.file_size)}
                  </p>
                </div>
                <span className={`px-2 py-0.5 rounded text-xs ${getStatusColor(file.status)}`}>
                  {file.status}
                </span>
              </div>

              {/* Jobs (expanded) */}
              {expandedFiles[file.id] && file.jobs && file.jobs.length > 0 && (
                <div className="bg-gray-50 px-4 py-2 pl-10">
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="text-gray-500">
                        <th className="text-left py-1">Preset</th>
                        <th className="text-left py-1">Status</th>
                        <th className="text-right py-1">Elements</th>
                        <th className="text-right py-1">Score</th>
                        <th className="text-right py-1">Time</th>
                      </tr>
                    </thead>
                    <tbody>
                      {file.jobs.map((job) => (
                        <tr key={job.id} className="border-t border-gray-200">
                          <td className="py-1.5">
                            <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${
                              job.quality_preset === 'coarse' ? 'bg-yellow-100 text-yellow-700' :
                              job.quality_preset === 'fine' ? 'bg-blue-100 text-blue-700' :
                              'bg-gray-100 text-gray-700'
                            }`}>
                              {job.quality_preset}
                            </span>
                          </td>
                          <td className="py-1.5">
                            <div className="flex items-center gap-1">
                              {getStatusIcon(job.status)}
                              <span className={`${
                                job.status === 'completed' ? 'text-green-600' :
                                job.status === 'failed' ? 'text-red-600' :
                                'text-gray-600'
                              }`}>
                                {job.status}
                              </span>
                            </div>
                          </td>
                          <td className="py-1.5 text-right text-gray-600">
                            {job.element_count?.toLocaleString() || '-'}
                          </td>
                          <td className="py-1.5 text-right text-gray-600">
                            {job.score?.toFixed(2) || '-'}
                          </td>
                          <td className="py-1.5 text-right text-gray-600">
                            {formatDuration(job.processing_time)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  
                  {/* Error message if any */}
                  {file.jobs.some(j => j.error_message) && (
                    <div className="mt-2 p-2 bg-red-50 rounded text-xs text-red-600">
                      {file.jobs.find(j => j.error_message)?.error_message}
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Empty State */}
      {(!batch.files || batch.files.length === 0) && (
        <div className="px-4 py-8 text-center">
          <FileText className="w-8 h-8 mx-auto mb-2 text-gray-300" />
          <p className="text-sm text-gray-500">No files uploaded</p>
        </div>
      )}

      {/* Timestamps */}
      <div className="px-4 py-2 bg-gray-50 border-t border-gray-100 text-[10px] text-gray-500 flex items-center justify-between">
        <span>Created: {batch.created_at ? new Date(batch.created_at).toLocaleString() : '-'}</span>
        {batch.completed_at && (
          <span>Completed: {new Date(batch.completed_at).toLocaleString()}</span>
        )}
      </div>
    </div>
  )
}
