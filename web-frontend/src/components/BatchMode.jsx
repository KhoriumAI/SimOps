import { useState, useCallback, useEffect, useRef } from 'react'
import { useAuth } from '../contexts/AuthContext'
import BatchUpload from './BatchUpload'
import BatchDashboard from './BatchDashboard'
import { useBatchPolling } from '../hooks/useWebSocket'
import {
  Plus, List, Settings, ToggleLeft, ToggleRight,
  Loader2, RefreshCw, ChevronDown
} from 'lucide-react'
import { API_BASE } from '../config'

/**
 * BatchMode Component
 * 
 * Container for batch mesh generation workflow:
 * 1. Create batch with settings
 * 2. Upload multiple files
 * 3. Start processing
 * 4. Monitor progress
 * 5. Download results
 */
export default function BatchMode({ onBatchComplete, onLog, onFileSelect }) {
  const { authFetch } = useAuth()

  // Helper to add log
  const addLog = (message) => {
    onLog?.(message)
  }

  // State
  const [currentBatchId, setCurrentBatchId] = useState(null)
  const [batches, setBatches] = useState([])
  const [selectedFiles, setSelectedFiles] = useState([])
  const [isCreating, setIsCreating] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)

  // Settings
  const [batchName, setBatchName] = useState('')
  const [meshIndependence, setMeshIndependence] = useState(false)
  const [meshStrategy, setMeshStrategy] = useState('Tet (Fast)')  // Fast strategy for batch processing
  const [curvatureAdaptive, setCurvatureAdaptive] = useState(true)
  const [showSettings, setShowSettings] = useState(false)
  const [meshStrategies, setMeshStrategies] = useState(['Tet (Fast)', 'Tetrahedral (Delaunay)'])

  // New settings for parity with App.jsx
  const [maxElementSize, setMaxElementSize] = useState(3.0)
  const [minElementSize, setMinElementSize] = useState(0.1)
  const [elementOrder, setElementOrder] = useState('1')
  const [ansysMode, setAnsysMode] = useState('None')

  // Fetch available strategies from API
  useEffect(() => {
    const fetchStrategies = async () => {
      try {
        const response = await authFetch(`${API_BASE}/strategies`)
        if (response.ok) {
          const data = await response.json()
          setMeshStrategies(data.names || ['Tet (Fast)', 'Tetrahedral (Delaunay)'])
          if (data.default) {
            setMeshStrategy(data.default)
          }
        }
      } catch (err) {
        console.error('Failed to fetch strategies:', err)
      }
    }
    fetchStrategies()
  }, [authFetch])

  // Polling for current batch
  const { batch, refresh, startPolling, stopPolling } = useBatchPolling(
    currentBatchId,
    authFetch,
    2000
  )

  // Load user's batches
  const loadBatches = useCallback(async () => {
    try {
      const response = await authFetch(`${API_BASE}/batch/list?limit=10`)
      if (response.ok) {
        const data = await response.json()
        setBatches(data.batches || [])
      }
    } catch (err) {
      console.error('Failed to load batches:', err)
    }
  }, [authFetch])

  useEffect(() => {
    loadBatches()
  }, [loadBatches])

  // Track if we've already notified about batch completion
  const notifiedBatchIdRef = useRef(null)

  // Start/stop polling based on batch status
  useEffect(() => {
    const finalStates = ['completed', 'failed', 'cancelled']

    if (batch?.status === 'processing') {
      startPolling()
    } else if (finalStates.includes(batch?.status)) {
      stopPolling()
      // Notify parent when batch completes (only once per batch)
      if (batch?.status === 'completed' && onBatchComplete && notifiedBatchIdRef.current !== batch.id) {
        notifiedBatchIdRef.current = batch.id
        onBatchComplete(batch)
      }
    }
  }, [batch?.status, batch?.id, startPolling, stopPolling, onBatchComplete])

  // Create new batch
  const createBatch = async () => {
    if (selectedFiles.length === 0) {
      setError('Please select files first')
      return
    }

    setIsCreating(true)
    setError(null)
    addLog(`[BATCH] Creating batch with ${selectedFiles.length} files...`)

    try {
      // 1. Create batch
      const batchDisplayName = batchName || `Batch ${new Date().toLocaleDateString()}`
      const createResponse = await authFetch(`${API_BASE}/batch/create`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: batchDisplayName,
          mesh_independence: meshIndependence,
          mesh_strategy: meshStrategy,
          curvature_adaptive: curvatureAdaptive,
          max_size_mm: maxElementSize,
          min_size_mm: minElementSize,
          element_order: parseInt(elementOrder),
          ansys_mode: ansysMode
        })
      })

      if (!createResponse.ok) {
        const err = await createResponse.json()
        throw new Error(err.error || 'Failed to create batch')
      }

      const { batch_id } = await createResponse.json()
      addLog(`[BATCH] Created "${batchDisplayName}" (ID: ${batch_id.slice(0, 8)})`)

      // 2. Upload files FIRST (before setting currentBatchId)
      setIsUploading(true)
      addLog(`[BATCH] Uploading ${selectedFiles.length} files...`)
      selectedFiles.forEach(f => addLog(`   • ${f.name} (${(f.size / 1024 / 1024).toFixed(1)} MB)`))

      const formData = new FormData()
      selectedFiles.forEach(file => formData.append('files', file))

      const uploadResponse = await authFetch(`${API_BASE}/batch/${batch_id}/upload`, {
        method: 'POST',
        body: formData
      })

      if (!uploadResponse.ok) {
        const err = await uploadResponse.json()
        throw new Error(err.error || 'Failed to upload files')
      }

      const uploadResult = await uploadResponse.json()
      addLog(`[BATCH] ✓ Upload complete: ${uploadResult.uploaded} files, ${uploadResult.total_jobs} jobs created`)
      if (meshIndependence) {
        addLog(`[BATCH] Independence study: Coarse + Medium + Fine = ${uploadResult.total_jobs} mesh jobs`)
      }

      // 3. NOW set currentBatchId so the polling hook fetches the batch WITH files
      setCurrentBatchId(batch_id)

      // 4. Refresh batch list
      await loadBatches()
      addLog(`[BATCH] Ready to start processing. Click "Start" to begin.`)

      // Clear form but keep the batch selected
      setSelectedFiles([])
      setBatchName('')

    } catch (err) {
      console.error('Create batch error:', err)
      addLog(`[ERROR] Batch creation failed: ${err.message}`)
      setError(err.message)
    } finally {
      setIsCreating(false)
      setIsUploading(false)
    }
  }

  // Start batch processing
  const handleStart = async () => {
    if (!currentBatchId) return
    setIsLoading(true)
    addLog(`[BATCH] Starting mesh generation...`)

    try {
      const response = await authFetch(`${API_BASE}/batch/${currentBatchId}/start`, {
        method: 'POST'
      })

      if (!response.ok) {
        const err = await response.json()
        throw new Error(err.error || 'Failed to start batch')
      }

      addLog(`[BATCH] ▶ Processing started (parallel limit: ${batch?.parallel_limit || 6})`)
      startPolling()
    } catch (err) {
      addLog(`[ERROR] Failed to start: ${err.message}`)
      setError(err.message)
    } finally {
      setIsLoading(false)
    }
  }

  // Cancel batch
  const handleCancel = async () => {
    if (!currentBatchId) return
    setIsLoading(true)
    addLog(`[BATCH] Cancelling batch processing...`)

    try {
      const response = await authFetch(`${API_BASE}/batch/${currentBatchId}/cancel`, {
        method: 'POST'
      })

      if (!response.ok) {
        const err = await response.json()
        throw new Error(err.error || 'Failed to cancel batch')
      }

      addLog(`[BATCH] ⏹ Processing cancelled`)
      await refresh()
    } catch (err) {
      addLog(`[ERROR] Cancel failed: ${err.message}`)
      setError(err.message)
    } finally {
      setIsLoading(false)
    }
  }

  // Download batch results
  const handleDownload = async () => {
    if (!currentBatchId) return
    setIsLoading(true)
    addLog(`[BATCH] Preparing download...`)

    try {
      const response = await authFetch(`${API_BASE}/batch/${currentBatchId}/download`)

      if (!response.ok) {
        const err = await response.json()
        throw new Error(err.error || 'Failed to download')
      }

      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      const filename = `batch_${currentBatchId.slice(0, 8)}.zip`
      a.download = filename
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(url)
      document.body.removeChild(a)
      addLog(`[BATCH] ✓ Downloaded ${filename}`)
    } catch (err) {
      addLog(`[ERROR] Download failed: ${err.message}`)
      setError(err.message)
    } finally {
      setIsLoading(false)
    }
  }

  // Delete batch
  const handleDelete = async () => {
    if (!currentBatchId) return
    if (!confirm('Delete this batch and all its files?')) return

    setIsLoading(true)
    addLog(`[BATCH] Deleting batch...`)

    try {
      const response = await authFetch(`${API_BASE}/batch/${currentBatchId}`, {
        method: 'DELETE'
      })

      if (!response.ok) {
        const err = await response.json()
        throw new Error(err.error || 'Failed to delete batch')
      }

      addLog(`[BATCH] 🗑 Batch deleted`)
      setCurrentBatchId(null)
      await loadBatches()
    } catch (err) {
      addLog(`[ERROR] Delete failed: ${err.message}`)
      setError(err.message)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="h-full flex flex-col overflow-hidden">
      {/* Header */}
      <div className="px-3 py-2 border-b border-gray-200 bg-gray-50">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-gray-800">Batch Processing</h2>
          <button
            onClick={loadBatches}
            className="p-1 hover:bg-gray-200 rounded text-gray-500"
            title="Refresh"
          >
            <RefreshCw className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-3 space-y-3">
        {/* Error Banner */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-2 text-xs text-red-600">
            {error}
            <button onClick={() => setError(null)} className="ml-2 underline">Dismiss</button>
          </div>
        )}

        {/* New Batch Form */}
        {!currentBatchId && (
          <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
            <div className="px-3 py-2 bg-gray-50 border-b border-gray-200">
              <h3 className="text-xs font-medium text-gray-700">Create New Batch</h3>
            </div>

            <div className="p-3 space-y-3">
              {/* Batch Name */}
              <div>
                <label className="text-[10px] uppercase text-gray-500 mb-1 block">Batch Name (optional)</label>
                <input
                  type="text"
                  value={batchName}
                  onChange={(e) => setBatchName(e.target.value)}
                  placeholder="e.g., CFD Study - Iteration 1"
                  className="w-full px-2 py-1.5 text-xs border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-blue-500"
                />
              </div>

              {/* Mesh Independence Toggle */}
              <div className="flex items-center justify-between py-2">
                <div>
                  <p className="text-xs font-medium text-gray-700">Mesh Independence Study</p>
                  <p className="text-[10px] text-gray-500">Generate Coarse, Medium, Fine for each file</p>
                </div>
                <button
                  onClick={() => setMeshIndependence(!meshIndependence)}
                  className={`p-1 rounded transition-colors ${meshIndependence ? 'text-blue-600' : 'text-gray-400'}`}
                >
                  {meshIndependence
                    ? <ToggleRight className="w-8 h-8" />
                    : <ToggleLeft className="w-8 h-8" />
                  }
                </button>
              </div>

              {/* Advanced Settings */}
              <div>
                <button
                  onClick={() => setShowSettings(!showSettings)}
                  className="flex items-center gap-1 text-xs text-gray-600 hover:text-gray-800"
                >
                  <Settings className="w-3 h-3" />
                  Advanced Settings
                  <ChevronDown className={`w-3 h-3 transition-transform ${showSettings ? 'rotate-180' : ''}`} />
                </button>

                {showSettings && (
                  <div className="mt-2 p-2 bg-gray-50 rounded space-y-2">
                    <div>
                      <label className="text-[10px] uppercase text-gray-500 mb-1 block">Strategy</label>
                      <select
                        value={meshStrategy}
                        onChange={(e) => setMeshStrategy(e.target.value)}
                        className="w-full px-2 py-1 text-xs border border-gray-300 rounded"
                      >
                        {meshStrategies.map(s => (
                          <option key={s} value={s}>{s}</option>
                        ))}
                      </select>
                    </div>
                    <label className="flex items-center gap-2 text-xs text-gray-600">
                      <input
                        type="checkbox"
                        checked={curvatureAdaptive}
                        onChange={(e) => setCurvatureAdaptive(e.target.checked)}
                        className="accent-blue-500"
                      />
                      Curvature-Adaptive
                    </label>

                    <div className="grid grid-cols-2 gap-2 pt-1 border-t border-gray-100">
                      <div>
                        <label className="text-[10px] uppercase text-gray-400 mb-1 block">Max Size</label>
                        <input
                          type="number"
                          step="0.1"
                          value={maxElementSize}
                          onChange={(e) => setMaxElementSize(parseFloat(e.target.value) || 3.0)}
                          className="w-full px-2 py-1 text-[10px] border border-gray-300 rounded"
                        />
                      </div>
                      <div>
                        <label className="text-[10px] uppercase text-gray-400 mb-1 block">Min Size</label>
                        <input
                          type="number"
                          step="0.01"
                          value={minElementSize}
                          onChange={(e) => setMinElementSize(parseFloat(e.target.value) || 0.1)}
                          className="w-full px-2 py-1 text-[10px] border border-gray-300 rounded"
                        />
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-2">
                      <div>
                        <label className="text-[10px] uppercase text-gray-400 mb-1 block">Order</label>
                        <select
                          value={elementOrder}
                          onChange={(e) => setElementOrder(e.target.value)}
                          className="w-full px-2 py-1 text-[10px] border border-gray-300 rounded"
                        >
                          <option value="1">Linear</option>
                          <option value="2">Quadratic</option>
                        </select>
                      </div>
                      <div>
                        <label className="text-[10px] uppercase text-gray-400 mb-1 block">Ansys</label>
                        <select
                          value={ansysMode}
                          onChange={(e) => setAnsysMode(e.target.value)}
                          className="w-full px-2 py-1 text-[10px] border border-gray-300 rounded"
                        >
                          <option value="None">None</option>
                          <option value="CFD (Fluent)">CFD</option>
                          <option value="FEA (Mechanical)">FEA</option>
                        </select>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* File Upload */}
              <BatchUpload
                onFilesSelected={setSelectedFiles}
                maxFiles={10}
                maxFileSize={500 * 1024 * 1024}
                disabled={isCreating || isUploading}
              />

              {/* Create Button */}
              <button
                onClick={createBatch}
                disabled={selectedFiles.length === 0 || isCreating || isUploading}
                className={`w-full px-3 py-2 rounded text-xs font-medium transition-colors flex items-center justify-center gap-2 ${selectedFiles.length > 0 && !isCreating
                  ? 'bg-blue-600 hover:bg-blue-700 text-white'
                  : 'bg-gray-200 text-gray-500 cursor-not-allowed'
                  }`}
              >
                {isCreating || isUploading ? (
                  <>
                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    {isUploading ? 'Uploading...' : 'Creating...'}
                  </>
                ) : (
                  <>
                    <Plus className="w-3.5 h-3.5" />
                    Create Batch ({selectedFiles.length} files)
                    {meshIndependence && ` → ${selectedFiles.length * 3} jobs`}
                  </>
                )}
              </button>
            </div>
          </div>
        )}

        {/* Current Batch Dashboard */}
        {currentBatchId && batch && (
          <BatchDashboard
            batch={batch}
            onStart={handleStart}
            onCancel={handleCancel}
            onDownload={handleDownload}
            onDelete={handleDelete}
            onFileSelect={onFileSelect}
            isLoading={isLoading}
          />
        )}

        {/* Loading state while batch data is being fetched */}
        {currentBatchId && !batch && (
          <div className="bg-white rounded-lg border border-gray-200 p-6 flex flex-col items-center justify-center">
            <Loader2 className="w-6 h-6 animate-spin text-blue-500 mb-2" />
            <p className="text-xs text-gray-500">Loading batch...</p>
          </div>
        )}

        {/* Back to New Batch */}
        {currentBatchId && (
          <button
            onClick={() => setCurrentBatchId(null)}
            className="w-full px-3 py-2 text-xs text-blue-600 hover:bg-blue-50 rounded transition-colors flex items-center justify-center gap-1"
          >
            <Plus className="w-3.5 h-3.5" />
            Create New Batch
          </button>
        )}

        {/* Recent Batches */}
        {batches.length > 0 && !currentBatchId && (
          <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
            <div className="px-3 py-2 bg-gray-50 border-b border-gray-200 flex items-center gap-2">
              <List className="w-3.5 h-3.5 text-gray-500" />
              <h3 className="text-xs font-medium text-gray-700">Recent Batches</h3>
            </div>
            <div className="divide-y divide-gray-100 max-h-48 overflow-y-auto">
              {batches.map((b) => (
                <button
                  key={b.id}
                  onClick={() => setCurrentBatchId(b.id)}
                  className="w-full px-3 py-2 text-left hover:bg-gray-50 transition-colors"
                >
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-medium text-gray-800 truncate">
                      {b.name || `Batch ${b.id.slice(0, 8)}`}
                    </span>
                    <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${b.status === 'completed' ? 'bg-green-100 text-green-600' :
                      b.status === 'processing' ? 'bg-blue-100 text-blue-600' :
                        b.status === 'failed' ? 'bg-red-100 text-red-600' :
                          'bg-gray-100 text-gray-600'
                      }`}>
                      {b.status}
                    </span>
                  </div>
                  <div className="text-[10px] text-gray-500 mt-0.5">
                    {b.total_files} files • {b.completed_jobs}/{b.total_jobs} jobs
                  </div>
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
