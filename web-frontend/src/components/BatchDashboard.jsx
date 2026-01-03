import { useState, useEffect } from 'react'
import {
  Play, Pause, Download, Trash2, ChevronDown, ChevronRight,
  CheckCircle, AlertCircle, Clock, Loader2, FileText, Settings,
  Box, BarChart2, Ruler, Activity, X
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
  onFileSelect,
  isLoading = false
}) {
  const [expandedFiles, setExpandedFiles] = useState({})
  const [selectedJob, setSelectedJob] = useState(null)  // For detail modal
  const [selectedFileId, setSelectedFileId] = useState(null)  // For highlighting selected file

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

  // Format quality metrics for display
  const formatMetric = (value, decimals = 3) => {
    if (value === null || value === undefined) return '-'
    return typeof value === 'number' ? value.toFixed(decimals) : value
  }

  // Job Detail Modal
  const JobDetailModal = ({ job, file, onClose }) => {
    if (!job) return null

    const metrics = job.quality_metrics || {}

    return (
      <div
        className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4"
        onClick={onClose}
      >
        <div
          className="bg-white rounded-xl shadow-2xl w-full max-w-2xl max-h-[90vh] overflow-hidden"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Modal Header */}
          <div className="px-5 py-4 bg-gradient-to-r from-blue-600 to-blue-700 text-white">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="font-semibold text-lg">
                  {file?.original_filename || 'Mesh Details'}
                </h3>
                <div className="flex items-center gap-2 mt-1">
                  <span className={`px-2 py-0.5 rounded text-xs font-medium ${job.quality_preset === 'coarse' ? 'bg-yellow-400/20 text-yellow-100' :
                      job.quality_preset === 'fine' ? 'bg-purple-400/20 text-purple-100' :
                        'bg-white/20 text-white'
                    }`}>
                    {job.quality_preset?.toUpperCase()}
                  </span>
                  <span className={`px-2 py-0.5 rounded text-xs ${job.status === 'completed' ? 'bg-green-400/20 text-green-100' :
                      job.status === 'failed' ? 'bg-red-400/20 text-red-100' :
                        'bg-white/20'
                    }`}>
                    {job.status}
                  </span>
                </div>
              </div>
              <button
                onClick={onClose}
                className="p-1 hover:bg-white/20 rounded-full transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
          </div>

          {/* Modal Content */}
          <div className="overflow-y-auto max-h-[calc(90vh-120px)]">
            {/* Summary Stats */}
            <div className="grid grid-cols-3 gap-4 p-5 bg-gray-50 border-b">
              <div className="text-center">
                <div className="text-2xl font-bold text-gray-800">
                  {job.element_count?.toLocaleString() || '-'}
                </div>
                <div className="text-xs text-gray-500 flex items-center justify-center gap-1">
                  <Box className="w-3 h-3" />
                  Elements
                </div>
              </div>
              <div className="text-center">
                <div className={`text-2xl font-bold ${(metrics.sicn_avg || 0) >= 0.7 ? 'text-green-600' :
                    (metrics.sicn_avg || 0) >= 0.4 ? 'text-yellow-600' :
                      'text-red-600'
                  }`}>
                  {formatMetric(metrics.sicn_avg)}
                </div>
                <div className="text-xs text-gray-500 flex items-center justify-center gap-1">
                  <BarChart2 className="w-3 h-3" />
                  SICN (Quality)
                </div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-gray-800">
                  {formatDuration(job.processing_time)}
                </div>
                <div className="text-xs text-gray-500 flex items-center justify-center gap-1">
                  <Clock className="w-3 h-3" />
                  Processing Time
                </div>
              </div>
            </div>

            {/* Quality Metrics */}
            {job.status === 'completed' && (
              <div className="p-5 space-y-4">
                <h4 className="text-sm font-semibold text-gray-700 flex items-center gap-2">
                  <Activity className="w-4 h-4" />
                  Quality Metrics
                </h4>

                <div className="grid grid-cols-2 gap-4">
                  {/* SICN (Scaled Jacobian) - higher is better, 0-1 scale */}
                  <div className="bg-gray-50 rounded-lg p-4">
                    <h5 className="text-xs font-medium text-gray-500 mb-2">SICN (Scaled Jacobian)</h5>
                    <div className="space-y-1 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-600">Min:</span>
                        <span className={`font-mono ${(metrics.sicn_min || 0) >= 0.4 ? 'text-green-600' :
                            (metrics.sicn_min || 0) >= 0.2 ? 'text-yellow-600' :
                              'text-red-600'
                          }`}>
                          {formatMetric(metrics.sicn_min)}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Avg:</span>
                        <span className={`font-mono ${(metrics.sicn_avg || 0) >= 0.7 ? 'text-green-600' :
                            (metrics.sicn_avg || 0) >= 0.4 ? 'text-yellow-600' :
                              'text-red-600'
                          }`}>
                          {formatMetric(metrics.sicn_avg)}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Max:</span>
                        <span className={`font-mono ${(metrics.sicn_max || 0) >= 0.9 ? 'text-green-600' :
                            (metrics.sicn_max || 0) >= 0.7 ? 'text-yellow-600' :
                              'text-red-600'
                          }`}>
                          {formatMetric(metrics.sicn_max)}
                        </span>
                      </div>
                    </div>
                    {/* Visual bar - solid color based on avg value */}
                    <div className="mt-2 h-2 bg-gray-200 rounded-full overflow-hidden">
                      <div
                        className={`h-full ${(metrics.sicn_avg || 0) >= 0.7 ? 'bg-green-500' :
                            (metrics.sicn_avg || 0) >= 0.4 ? 'bg-yellow-500' :
                              'bg-red-500'
                          }`}
                        style={{ width: `${(metrics.sicn_avg || 0) * 100}%` }}
                      />
                    </div>
                  </div>

                  {/* Gamma - higher is better, 0-1 scale */}
                  <div className="bg-gray-50 rounded-lg p-4">
                    <h5 className="text-xs font-medium text-gray-500 mb-2">Gamma (Shape Quality)</h5>
                    <div className="space-y-1 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-600">Min:</span>
                        <span className={`font-mono ${(metrics.gamma_min || 0) >= 0.5 ? 'text-green-600' :
                            (metrics.gamma_min || 0) >= 0.3 ? 'text-yellow-600' :
                              'text-red-600'
                          }`}>
                          {formatMetric(metrics.gamma_min)}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Avg:</span>
                        <span className={`font-mono ${(metrics.gamma_avg || 0) >= 0.7 ? 'text-green-600' :
                            (metrics.gamma_avg || 0) >= 0.4 ? 'text-yellow-600' :
                              'text-red-600'
                          }`}>
                          {formatMetric(metrics.gamma_avg)}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Max:</span>
                        <span className={`font-mono ${(metrics.gamma_max || 0) >= 0.9 ? 'text-green-600' :
                            (metrics.gamma_max || 0) >= 0.7 ? 'text-yellow-600' :
                              'text-red-600'
                          }`}>
                          {formatMetric(metrics.gamma_max)}
                        </span>
                      </div>
                    </div>
                    <div className="mt-2 h-2 bg-gray-200 rounded-full overflow-hidden">
                      <div
                        className={`h-full ${(metrics.gamma_avg || 0) >= 0.7 ? 'bg-green-500' :
                            (metrics.gamma_avg || 0) >= 0.4 ? 'bg-yellow-500' :
                              'bg-red-500'
                          }`}
                        style={{ width: `${(metrics.gamma_avg || 0) * 100}%` }}
                      />
                    </div>
                  </div>

                  {/* Skewness - lower is better, 0-1 scale (0 is best) */}
                  <div className="bg-gray-50 rounded-lg p-4">
                    <h5 className="text-xs font-medium text-gray-500 mb-2">Skewness (lower is better)</h5>
                    <div className="space-y-1 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-600">Min:</span>
                        <span className={`font-mono ${(metrics.skewness_min || 0) <= 0.1 ? 'text-green-600' :
                            (metrics.skewness_min || 0) <= 0.3 ? 'text-yellow-600' :
                              'text-red-600'
                          }`}>
                          {formatMetric(metrics.skewness_min)}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Avg:</span>
                        <span className={`font-mono ${(metrics.skewness_avg || 0) <= 0.25 ? 'text-green-600' :
                            (metrics.skewness_avg || 0) <= 0.5 ? 'text-yellow-600' :
                              'text-red-600'
                          }`}>
                          {formatMetric(metrics.skewness_avg)}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Max:</span>
                        <span className={`font-mono ${(metrics.skewness_max || 0) <= 0.5 ? 'text-green-600' :
                            (metrics.skewness_max || 0) <= 0.85 ? 'text-yellow-600' :
                              'text-red-600'
                          }`}>
                          {formatMetric(metrics.skewness_max)}
                        </span>
                      </div>
                    </div>
                    {/* Bar shows avg skewness - shorter is better */}
                    <div className="mt-2 h-2 bg-gray-200 rounded-full overflow-hidden">
                      <div
                        className={`h-full ${(metrics.skewness_avg || 0) <= 0.25 ? 'bg-green-500' :
                            (metrics.skewness_avg || 0) <= 0.5 ? 'bg-yellow-500' :
                              'bg-red-500'
                          }`}
                        style={{ width: `${Math.min((metrics.skewness_avg || 0) * 100, 100)}%` }}
                      />
                    </div>
                  </div>

                  {/* Aspect Ratio - closer to 1 is better, typically 1-10+ scale */}
                  <div className="bg-gray-50 rounded-lg p-4">
                    <h5 className="text-xs font-medium text-gray-500 mb-2">Aspect Ratio (closer to 1 is better)</h5>
                    <div className="space-y-1 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-600">Min:</span>
                        <span className={`font-mono ${(metrics.aspect_ratio_min || 1) <= 1.5 ? 'text-green-600' :
                            (metrics.aspect_ratio_min || 1) <= 2.5 ? 'text-yellow-600' :
                              'text-red-600'
                          }`}>
                          {formatMetric(metrics.aspect_ratio_min, 2)}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Avg:</span>
                        <span className={`font-mono ${(metrics.aspect_ratio_avg || 1) <= 2 ? 'text-green-600' :
                            (metrics.aspect_ratio_avg || 1) <= 4 ? 'text-yellow-600' :
                              'text-red-600'
                          }`}>
                          {formatMetric(metrics.aspect_ratio_avg, 2)}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Max:</span>
                        <span className={`font-mono ${(metrics.aspect_ratio_max || 1) <= 5 ? 'text-green-600' :
                            (metrics.aspect_ratio_max || 1) <= 10 ? 'text-yellow-600' :
                              'text-red-600'
                          }`}>
                          {formatMetric(metrics.aspect_ratio_max, 2)}
                        </span>
                      </div>
                    </div>
                    {/* Bar normalized: lower avg is greener */}
                    <div className="mt-2 h-2 bg-gray-200 rounded-full overflow-hidden">
                      <div
                        className={`h-full ${(metrics.aspect_ratio_avg || 1) <= 2 ? 'bg-green-500' :
                            (metrics.aspect_ratio_avg || 1) <= 4 ? 'bg-yellow-500' :
                              'bg-red-500'
                          }`}
                        style={{ width: `${Math.min(((metrics.aspect_ratio_avg || 1) - 1) / 9 * 100, 100)}%` }}
                      />
                    </div>
                  </div>
                </div>

                {/* Additional Info */}
                <div className="mt-4 pt-4 border-t border-gray-200">
                  <h5 className="text-xs font-medium text-gray-500 mb-2">Processing Details</h5>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div className="flex justify-between bg-gray-50 px-3 py-2 rounded">
                      <span className="text-gray-500">Strategy:</span>
                      <span className="font-medium text-gray-700">{job.mesh_strategy || 'Auto'}</span>
                    </div>
                    <div className="flex justify-between bg-gray-50 px-3 py-2 rounded">
                      <span className="text-gray-500">Output Size:</span>
                      <span className="font-medium text-gray-700">{formatFileSize(job.output_file_size)}</span>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Error Details */}
            {job.status === 'failed' && job.error_message && (
              <div className="p-5">
                <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                  <h5 className="text-sm font-medium text-red-800 mb-2 flex items-center gap-2">
                    <AlertCircle className="w-4 h-4" />
                    Error Details
                  </h5>
                  <pre className="text-xs text-red-700 whitespace-pre-wrap font-mono bg-red-100/50 p-3 rounded">
                    {job.error_message}
                  </pre>
                </div>
              </div>
            )}
          </div>

          {/* Modal Footer */}
          <div className="px-5 py-3 bg-gray-50 border-t flex justify-end gap-2">
            <button
              onClick={onClose}
              className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
            >
              Close
            </button>
          </div>
        </div>
      </div>
    )
  }

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
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-gray-600">
          <div className="flex items-center gap-1">
            <Settings className="w-3 h-3" />
            <span className="truncate max-w-[120px]" title={batch.mesh_strategy}>{batch.mesh_strategy}</span>
          </div>
          {batch.mesh_independence && (
            <span className="text-purple-600 whitespace-nowrap">C+M+F</span>
          )}
          <span className="whitespace-nowrap">∥{batch.parallel_limit}</span>
        </div>
      </div>

      {/* Files List */}
      {batch.files && batch.files.length > 0 && (
        <div className="max-h-80 overflow-y-auto">
          {batch.files.map((file) => (
            <div key={file.id} className={`border-b border-gray-100 last:border-0 ${selectedFileId === file.id ? 'bg-blue-50' : ''
              }`}>
              {/* File Header */}
              <div className="flex items-center gap-2 px-4 py-2 hover:bg-gray-50">
                {/* Expand/Collapse Toggle */}
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    toggleFile(file.id)
                  }}
                  className="p-0.5 hover:bg-gray-200 rounded"
                >
                  {expandedFiles[file.id]
                    ? <ChevronDown className="w-4 h-4 text-gray-400" />
                    : <ChevronRight className="w-4 h-4 text-gray-400" />
                  }
                </button>
                {getStatusIcon(file.status)}
                {/* Clickable filename for preview */}
                <div
                  className="flex-1 min-w-0 cursor-pointer hover:text-blue-600 transition-colors"
                  onClick={() => {
                    setSelectedFileId(file.id)
                    onFileSelect?.(file)
                  }}
                  title="Click to preview"
                >
                  <p className={`text-sm font-medium line-clamp-3 break-all ${selectedFileId === file.id ? 'text-blue-600' : 'text-gray-800'
                    }`}>
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
                        <th className="text-left py-1 w-16">Preset</th>
                        <th className="text-left py-1 w-6"></th>
                        <th className="text-right py-1">Elem</th>
                        <th className="text-right py-1">SICN</th>
                        <th className="text-right py-1">Time</th>
                      </tr>
                    </thead>
                    <tbody>
                      {file.jobs.map((job) => {
                        // Get SICN from quality_metrics
                        const sicnAvg = job.quality_metrics?.sicn_avg
                        return (
                          <tr
                            key={job.id}
                            className={`border-t border-gray-200 ${job.status === 'completed' || job.status === 'failed'
                                ? 'cursor-pointer hover:bg-blue-50 transition-colors'
                                : ''
                              }`}
                            onClick={() => {
                              if (job.status === 'completed' || job.status === 'failed') {
                                setSelectedJob({ job, file })
                              }
                            }}
                            title={job.status === 'completed' || job.status === 'failed' ? 'Click for details' : ''}
                          >
                            <td className="py-1.5">
                              <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${job.quality_preset === 'coarse' ? 'bg-yellow-100 text-yellow-700' :
                                  job.quality_preset === 'fine' ? 'bg-blue-100 text-blue-700' :
                                    'bg-gray-100 text-gray-700'
                                }`}>
                                {job.quality_preset}
                              </span>
                            </td>
                            <td className="py-1.5">
                              <div className="flex items-center gap-1" title={job.status}>
                                {getStatusIcon(job.status)}
                                {/* Only show text for non-final states to save space */}
                                {!['completed', 'failed'].includes(job.status) && (
                                  <span className="text-gray-600">{job.status}</span>
                                )}
                              </div>
                            </td>
                            <td className="py-1.5 text-right text-gray-600">
                              {job.element_count?.toLocaleString() || '-'}
                            </td>
                            <td className={`py-1.5 text-right ${sicnAvg >= 0.7 ? 'text-green-600' :
                                sicnAvg >= 0.4 ? 'text-yellow-600' :
                                  sicnAvg ? 'text-red-600' : 'text-gray-600'
                              }`}>
                              {sicnAvg?.toFixed(3) || '-'}
                            </td>
                            <td className="py-1.5 text-right text-gray-600">
                              {formatDuration(job.processing_time)}
                              {(job.status === 'completed' || job.status === 'failed') && (
                                <span className="ml-1 text-blue-500">→</span>
                              )}
                            </td>
                          </tr>
                        )
                      })}
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

      {/* Job Detail Modal */}
      {selectedJob && (
        <JobDetailModal
          job={selectedJob.job}
          file={selectedJob.file}
          onClose={() => setSelectedJob(null)}
        />
      )}
    </div>
  )
}
