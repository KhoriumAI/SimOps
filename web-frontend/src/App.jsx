import { useState, useEffect } from 'react'
import { useAuth } from './contexts/AuthContext'
import MeshViewer from './components/MeshViewer'
import FileUpload from './components/FileUpload'
import Terminal from './components/Terminal'
import MeshTimer, { MeshTimerCompact } from './components/MeshTimer'
import BatchMode from './components/BatchMode'
import FeedbackButton from './components/FeedbackButton'
import { Download, LogOut, User, Square, ChevronDown, ChevronUp, Terminal as TerminalIcon, Copy, Clock, Layers, File, BarChart3 } from 'lucide-react'

// API base URL - uses proxy in development, full URL in production
const ALB_DNS = 'webdev-alb-1882895883.us-west-1.elb.amazonaws.com'
const API_BASE = import.meta.env.VITE_API_URL ||
  ((window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
    ? '/api'
    : (window.location.hostname.includes('s3-website')
      ? `http://${ALB_DNS}/api`
      : `http://${window.location.hostname}:5000/api`))

// Preset sizes: maps preset names to min/max element sizes
const presetSizes = {
  'Coarse': { min: 0.5, max: 10.0 },
  'Medium': { min: 0.1, max: 3.0 },
  'Fine': { min: 0.05, max: 1.0 },
  'Very Fine': { min: 0.01, max: 0.5 }
}

// Parse number expressions: supports scientific notation (1e-6) and math (39.26/2)
function parseNumberExpression(expr) {
  if (typeof expr === 'number') return expr
  const trimmed = String(expr).trim()
  if (!trimmed) return NaN
  // Handle scientific notation: 1e-6, 2.5E3
  if (/^-?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$/.test(trimmed)) {
    return parseFloat(trimmed)
  }
  // Handle simple math: 39.26/2, 10*2, 5+3
  try {
    if (/^[\d\s+\-*/().eE]+$/.test(trimmed)) {
      const result = Function('"use strict"; return (' + trimmed + ')')()
      return typeof result === 'number' && isFinite(result) ? result : NaN
    }
  } catch { }
  return parseFloat(trimmed) // Fallback
}

function App() {
  const { user, logout, authFetch } = useAuth()
  const [currentProject, setCurrentProject] = useState(null)
  const [projectStatus, setProjectStatus] = useState(null)
  const [logs, setLogs] = useState([])
  const [meshData, setMeshData] = useState(null)
  const [isPolling, setIsPolling] = useState(false)
  const [qualityPreset, setQualityPreset] = useState('Medium')
  const [maxElementSize, setMaxElementSize] = useState(3.0)
  const [minElementSize, setMinElementSize] = useState(0.1)
  const [elementOrder, setElementOrder] = useState('1')
  const [ansysMode, setAnsysMode] = useState('None')
  const [meshStrategy, setMeshStrategy] = useState('Tetrahedral (Delaunay)')
  const [curvatureAdaptive, setCurvatureAdaptive] = useState(false)
  const [geometryInfo, setGeometryInfo] = useState(null)
  const [isUploading, setIsUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [meshProgress, setMeshProgress] = useState({
    strategy: 0, '1d': 0, '2d': 0, '3d': 0,
    optimize: 0, netgen: 0, order2: 0, quality: 0
  })
  const [consoleOpen, setConsoleOpen] = useState(true)
  const [meshStartTime, setMeshStartTime] = useState(null)
  const [lastMeshDuration, setLastMeshDuration] = useState(null)
  const [isLoadingPreview, setIsLoadingPreview] = useState(false)

  // Mode: 'single' or 'batch'
  const [mode, setMode] = useState('single')

  // Visualization toggles (shared with MeshViewer)
  const [showAxes, setShowAxes] = useState(true)
  const [qualityMetric, setQualityMetric] = useState('sicn')
  const [showHistogram, setShowHistogram] = useState(false)

  const qualityPresets = ['Coarse', 'Medium', 'Fine', 'Very Fine']
  const [meshStrategies, setMeshStrategies] = useState([
    'Tetrahedral (Delaunay)',  // Fallback default
  ])

  // Fetch available strategies from backend on mount
  useEffect(() => {
    const fetchStrategies = async () => {
      try {
        const response = await fetch(`${API_BASE}/strategies`)
        if (response.ok) {
          const data = await response.json()
          setMeshStrategies(data.names || [])
          // Set default strategy if current selection isn't in the list
          if (data.default && !data.names.includes(meshStrategy)) {
            setMeshStrategy(data.default)
          }
        }
      } catch (err) {
        console.warn('Failed to fetch strategies, using defaults:', err)
      }
    }
    fetchStrategies()
  }, [])

  // Poll project status
  useEffect(() => {
    if (!currentProject) return

    const pollStatus = async () => {
      try {
        const response = await authFetch(`${API_BASE}/projects/${currentProject}/status`)
        if (!response.ok) return

        const data = await response.json()
        setProjectStatus(data)

        // Only update logs from server if there are actual generation logs
        // This preserves client-side logs (CAD preview info) when not processing
        const serverLogs = data.latest_result?.logs || []
        if (serverLogs.length > 0 && data.status === 'processing') {
          setLogs(prev => {
            // Keep CAD info logs, append server logs
            const cadLogs = prev.filter(l => l.includes('📐') || l.includes('📏') || l.includes('📦') || l.includes('Loaded CAD') || l.includes('Preview mesh') || l.startsWith('   •'))
            return [...cadLogs, '', '[INFO] Mesh Generation Progress:', ...serverLogs]
          })
          // Parse progress from logs
          parseProgressFromLogs(serverLogs)
        } else if (data.status === 'completed' && serverLogs.length > 0) {
          setLogs(prev => {
            const cadLogs = prev.filter(l => l.includes('📐') || l.includes('📏') || l.includes('📦') || l.includes('Loaded CAD') || l.includes('Preview mesh') || l.startsWith('   •'))
            return [...cadLogs, '', '[SUCCESS] Mesh generation completed!', ...serverLogs]
          })
          // Set all progress to 100% on completion
          setMeshProgress({ strategy: 100, '1d': 100, '2d': 100, '3d': 100, optimize: 100, netgen: 100, order2: 100, quality: 100 })
        }

        if (data.status === 'completed') {
          // Always fetch final mesh when completed (replaces preview)
          if (!meshData || meshData.isPreview) {
            fetchMeshData(currentProject)
          }
          // Save mesh duration
          if (meshStartTime) {
            setLastMeshDuration(Date.now() - meshStartTime)
          }
          setMeshStartTime(null)
          setIsPolling(false)
        } else if (data.status === 'error') {
          setLogs(prev => [...prev, `[ERROR] ${data.error_message || 'Mesh generation failed'}`])
          // Save duration even on error
          if (meshStartTime) {
            setLastMeshDuration(Date.now() - meshStartTime)
          }
          setMeshStartTime(null)
          setIsPolling(false)
        }
      } catch (error) {
        console.error('Failed to fetch project status:', error)
      }
    }

    if (isPolling) {
      pollStatus()
      const interval = setInterval(pollStatus, 1000)
      return () => clearInterval(interval)
    }
    // Don't auto-poll when not generating - preserves CAD preview logs
  }, [currentProject, isPolling, meshData, authFetch, meshStartTime])

  // Parse progress from log messages
  const parseProgressFromLogs = (logs) => {
    const logsText = logs.join('\n').toLowerCase()
    const progress = { ...meshProgress }

    // Strategy progress (parallel race)
    if (logsText.includes('exhaustive mesh generation') || logsText.includes('parallel race')) {
      progress.strategy = Math.min(100, progress.strategy + 5)
    }
    if (logsText.includes('winner')) progress.strategy = 100

    // 1D meshing
    if (logsText.includes('1d mesh') || logsText.includes('meshing 1d')) {
      progress['1d'] = logsText.includes('done') ? 100 : 50
    }

    // 2D meshing
    if (logsText.includes('2d mesh') || logsText.includes('meshing 2d') || logsText.includes('surface mesh')) {
      progress['2d'] = logsText.includes('done') ? 100 : 50
    }

    // 3D meshing
    if (logsText.includes('3d mesh') || logsText.includes('meshing 3d') || logsText.includes('volume mesh') || logsText.includes('tetrahedr')) {
      progress['3d'] = logsText.includes('done') || logsText.includes('complete') ? 100 : 50
    }

    // Optimization
    if (logsText.includes('optim')) {
      progress.optimize = logsText.includes('done') || logsText.includes('complete') ? 100 : 50
    }

    // Netgen
    if (logsText.includes('netgen')) {
      progress.netgen = logsText.includes('done') ? 100 : 50
    }

    // Order 2 / High order
    if (logsText.includes('order 2') || logsText.includes('high order') || logsText.includes('second order')) {
      progress.order2 = logsText.includes('done') ? 100 : 50
    }

    // Quality check
    if (logsText.includes('quality') || logsText.includes('sicn')) {
      progress.quality = logsText.includes('done') || logsText.includes('report') ? 100 : 50
    }

    setMeshProgress(progress)
  }

  const fetchMeshData = async (projectId) => {
    try {
      const response = await authFetch(`${API_BASE}/projects/${projectId}/mesh-data`)
      if (response.ok) {
        const data = await response.json()
        setMeshData(data)
      }
    } catch (error) {
      console.error('Failed to fetch mesh data:', error)
    }
  }

  const fetchCadPreview = async (projectId, filename) => {
    try {
      const response = await authFetch(`${API_BASE}/projects/${projectId}/preview`)
      if (response.ok) {
        const data = await response.json()

        // Check if the backend returned an error
        if (data.error) {
          console.warn('CAD preview error:', data.error)
          setLogs(prev => [
            ...prev,
            `[WARNING] ${data.error}`,
            `[INFO] The geometry may be too complex to preview.`,
            `[INFO] You can still try to generate a mesh.`
          ])
          // Set empty mesh data so the viewer shows something
          setMeshData({ vertices: [], colors: [], numVertices: 0, numTriangles: 0, isPreview: true })
          return
        }

        setMeshData(data)

        // Extract geometry info for display
        if (data.geometry) {
          setGeometryInfo(data.geometry)

          // Update logs with CAD geometry info
          const geo = data.geometry
          const volume = geo.total_volume > 0 ? geo.total_volume.toFixed(6) : 'N/A'
          const bboxStr = geo.bbox ? `${geo.bbox[0].toFixed(1)} x ${geo.bbox[1].toFixed(1)} x ${geo.bbox[2].toFixed(1)}` : 'N/A'

          setLogs(prev => [
            ...prev,
            `[INFO] Loaded CAD: ${filename}`,
            ``,
            `📐 CAD Geometry Info:`,
            `   • Volumes: ${geo.volumes}`,
            `   • Surfaces: ${geo.surfaces}`,
            `   • Curves: ${geo.curves}`,
            `   • Points: ${geo.points || 0}`,
            ``,
            `📏 Bounding Box: ${bboxStr} mm`,
            `📦 Volume: ${volume} mm³`,
            ``,
            `[INFO] Preview mesh: ${data.numVertices?.toLocaleString()} vertices, ${data.numTriangles?.toLocaleString()} triangles`
          ])
        } else if (data.numTriangles === 0) {
          setLogs(prev => [
            ...prev,
            `[WARNING] Could not generate preview mesh for ${filename}`,
            `[INFO] The file was loaded but visualization failed.`,
            `[INFO] You can still try to generate a mesh.`
          ])
        }
      }
    } catch (error) {
      console.error('Failed to fetch CAD preview:', error)
      setLogs(prev => [...prev, `[ERROR] Failed to load CAD preview: ${error.message}`])
    }
  }

  const handleFileUpload = async (file) => {
    const formData = new FormData()
    formData.append('file', file)

    setIsUploading(true)
    setUploadProgress(0)
    setLogs([`[INFO] Uploading ${file.name}...`])
    setMeshData(null)
    setGeometryInfo(null)

    try {
      // Simulate upload progress for better UX
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => Math.min(prev + 10, 90))
      }, 200)

      // Upload with 30-minute timeout for large files
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), 1800000) // 30 minutes

      const response = await authFetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData,
        signal: controller.signal
      })

      clearTimeout(timeoutId)
      clearInterval(progressInterval)

      if (!response.ok) {
        const error = await response.json()
        setIsUploading(false)
        setUploadProgress(0)
        alert(error.error || 'Failed to upload file')
        return
      }

      setUploadProgress(100)
      const data = await response.json()
      setCurrentProject(data.project_id)
      setProjectStatus(data)
      setLogs([`[INFO] Uploaded ${file.name}`, `[INFO] Loading CAD preview...`])
      setIsPolling(false)

      // Fetch CAD preview to show geometry before meshing
      await fetchCadPreview(data.project_id, file.name)
      setIsUploading(false)
      setUploadProgress(0)
    } catch (error) {
      console.error('Upload failed:', error)
      setIsUploading(false)
      setUploadProgress(0)

      if (error.name === 'AbortError') {
        alert('Upload timed out. File may be too large or server is taking too long to respond. Please try again.')
      } else {
        alert('Failed to upload file')
      }
    }
  }

  const handleGenerateMesh = async () => {
    if (!currentProject) return

    try {
      // Mesh generation with 30-minute timeout
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), 1800000) // 30 minutes

      const response = await authFetch(`${API_BASE}/projects/${currentProject}/generate`, {
        method: 'POST',
        body: {
          quality_preset: qualityPreset,
          target_elements: null,
          max_size_mm: maxElementSize,
          min_size_mm: minElementSize,
          element_order: parseInt(elementOrder),
          ansys_mode: ansysMode,
          mesh_strategy: meshStrategy,
          curvature_adaptive: curvatureAdaptive
        },
        signal: controller.signal
      })

      clearTimeout(timeoutId)

      if (response.ok) {
        setIsPolling(true)
        setMeshStartTime(Date.now())
        setLastMeshDuration(null)
        setLogs([`[INFO] Starting mesh generation...`])
        setMeshData(null)
        // Reset progress
        setMeshProgress({ strategy: 0, '1d': 0, '2d': 0, '3d': 0, optimize: 0, netgen: 0, order2: 0, quality: 0 })
      } else {
        const error = await response.json()
        alert(error.error || 'Failed to start mesh generation')
      }
    } catch (error) {
      console.error('Failed to start mesh generation:', error)
      alert('Failed to start mesh generation')
    }
  }

  const handleDownload = async () => {
    if (!currentProject) return

    try {
      const response = await authFetch(`${API_BASE}/projects/${currentProject}/download`)
      if (response.ok) {
        const contentType = response.headers.get('content-type')

        // Check if response is JSON (S3 presigned URL) or binary (local file)
        if (contentType && contentType.includes('application/json')) {
          // S3 presigned URL response
          const data = await response.json()
          if (data.download_url) {
            // Open presigned URL in new tab or trigger download
            const a = document.createElement('a')
            a.href = data.download_url
            a.download = data.filename || 'mesh.msh'
            a.target = '_blank'
            document.body.appendChild(a)
            a.click()
            document.body.removeChild(a)
          }
        } else {
          // Direct file download (local storage)
          const blob = await response.blob()
          const url = window.URL.createObjectURL(blob)
          const a = document.createElement('a')
          a.href = url
          a.download = projectStatus?.filename?.replace(/\.(step|stp|stl)$/i, '_mesh.msh') || 'mesh.msh'
          document.body.appendChild(a)
          a.click()
          window.URL.revokeObjectURL(url)
          document.body.removeChild(a)
        }
      }
    } catch (error) {
      console.error('Download failed:', error)
      alert('Failed to download mesh')
    }
  }

  const canGenerate = projectStatus &&
    (projectStatus.status === 'uploaded' || projectStatus.status === 'error' || projectStatus.status === 'completed')
  const canDownload = projectStatus?.status === 'completed'
  const isGenerating = projectStatus?.status === 'processing'

  // Format duration for display
  const formatDuration = (ms) => {
    const totalSeconds = Math.floor(ms / 1000)
    const minutes = Math.floor(totalSeconds / 60)
    const seconds = totalSeconds % 60
    if (minutes > 0) {
      return `${minutes}m ${seconds}s`
    }
    return `${seconds}.${Math.floor((ms % 1000) / 100)}s`
  }

  return (
    <div className="h-screen flex flex-col bg-gray-100 text-gray-900">
      {/* Compact Header Bar - Light Theme */}
      <header className="h-10 bg-white border-b border-gray-200 flex items-center justify-between px-3 text-gray-800">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <img src="/img/Khorium_logo.jpg" alt="Khorium" className="h-6 w-auto" />
            <span className="font-semibold text-sm">MeshGen</span>
          </div>

          {/* Mode Toggle */}
          <div className="flex items-center bg-gray-100 rounded-lg p-0.5">
            <button
              onClick={() => setMode('single')}
              className={`flex items-center gap-1.5 px-3 py-1 rounded-md text-xs font-medium transition-colors ${mode === 'single'
                ? 'bg-white text-blue-600 shadow-sm'
                : 'text-gray-600 hover:text-gray-800'
                }`}
            >
              <File className="w-3.5 h-3.5" />
              Single
            </button>
            <button
              onClick={() => setMode('batch')}
              className={`flex items-center gap-1.5 px-3 py-1 rounded-md text-xs font-medium transition-colors ${mode === 'batch'
                ? 'bg-white text-blue-600 shadow-sm'
                : 'text-gray-600 hover:text-gray-800'
                }`}
            >
              <Layers className="w-3.5 h-3.5" />
              Batch
            </button>
          </div>
        </div>

        {/* Center toolbar */}
        <div className="flex items-center gap-3">
          {/* Real-time Timer */}
          {(isGenerating || lastMeshDuration) && (
            <div className="flex items-center gap-2">
              {isGenerating ? (
                <MeshTimerCompact
                  isRunning={isGenerating}
                  startTime={meshStartTime}
                  status={projectStatus?.status}
                />
              ) : lastMeshDuration ? (
                <div className="flex items-center gap-1.5 text-xs text-green-600">
                  <Clock className="w-3 h-3" />
                  <span className="font-mono">{formatDuration(lastMeshDuration)}</span>
                </div>
              ) : null}
            </div>
          )}

          {canDownload && (
            <button
              onClick={handleDownload}
              className="flex items-center gap-1.5 px-3 py-1 text-xs text-white bg-blue-600 hover:bg-blue-500 rounded transition-colors"
            >
              <Download className="w-3.5 h-3.5" />
              Download
            </button>
          )}
        </div>

        {/* Right side - user */}
        <div className="flex items-center gap-2 text-xs text-gray-600">
          <User className="w-3.5 h-3.5" />
          <span>{user?.name || user?.email?.split('@')[0]}</span>
          <button
            onClick={logout}
            className="p-1 hover:bg-gray-100 rounded transition-colors text-gray-500 hover:text-gray-700"
            title="Logout"
          >
            <LogOut className="w-3.5 h-3.5" />
          </button>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Panel - Responsive Sidebar */}
        <div className={`${mode === 'batch' ? 'w-80 xl:w-96 min-w-[280px]' : 'w-56 min-w-[200px]'} max-w-[40vw] border-r border-gray-300 flex flex-col bg-gray-50 overflow-y-auto transition-all duration-300 flex-shrink-0`}>

          {/* Batch Mode */}
          {mode === 'batch' ? (
            <BatchMode
              onBatchComplete={(batch) => {
                console.log('Batch completed:', batch)
                setLogs(prev => [...prev, `[SUCCESS] Batch ${batch.name || batch.id.slice(0, 8)} completed!`])
              }}
              onLog={(msg) => setLogs(prev => [...prev, msg])}
              onFileSelect={async (file) => {
                // Fetch CAD preview for batch file
                setIsLoadingPreview(true)
                setLogs(prev => [...prev, `[INFO] Loading preview for ${file.original_filename}...`])
                try {
                  const response = await authFetch(`${API_BASE}/batch/file/${file.id}/preview`)
                  if (response.ok) {
                    const data = await response.json()
                    if (data.error) {
                      setLogs(prev => [...prev, `[WARNING] ${data.error}`])
                      setMeshData({ vertices: [], colors: [], numVertices: 0, numTriangles: 0, isPreview: true })
                    } else {
                      setMeshData(data)
                      if (data.geometry) {
                        setGeometryInfo(data.geometry)
                        const geo = data.geometry
                        setLogs(prev => [
                          ...prev,
                          `[OK] CAD Preview: ${file.original_filename}`,
                          `[INFO] Solids: ${geo.solid_count}, Faces: ${geo.face_count}, Volume: ${geo.total_volume?.toFixed(4) || 'N/A'}`
                        ])
                      }
                    }
                  } else {
                    const err = await response.json()
                    setLogs(prev => [...prev, `[ERROR] Preview failed: ${err.error || 'Unknown error'}`])
                  }
                } catch (err) {
                  console.error('Preview error:', err)
                  setLogs(prev => [...prev, `[ERROR] Failed to load preview: ${err.message}`])
                } finally {
                  setIsLoadingPreview(false)
                }
              }}
            />
          ) : (
            /* Single File Mode */
            <div className="p-3 space-y-3">
              {/* File Upload - Compact */}
              <FileUpload onFileUpload={handleFileUpload} />

              {/* Mesh Settings - Collapsible style */}
              <div className="bg-white rounded-lg border border-gray-200 text-sm">
                <div className="px-3 py-2 bg-gray-100 font-medium text-gray-700 border-b border-gray-200">
                  Mesh Settings
                </div>
                <div className="p-3 space-y-2.5">
                  <div>
                    <label className="text-gray-500 text-[10px] uppercase mb-1 block">Preset</label>
                    <select
                      value={qualityPreset}
                      onChange={(e) => {
                        const preset = e.target.value
                        setQualityPreset(preset)
                        // Update min/max sizes based on preset
                        if (presetSizes[preset]) {
                          setMinElementSize(presetSizes[preset].min)
                          setMaxElementSize(presetSizes[preset].max)
                        }
                      }}
                      className="w-full bg-white border border-gray-300 rounded px-2 py-1.5 text-xs focus:outline-none focus:ring-1 focus:ring-blue-500"
                      disabled={isGenerating}
                    >
                      {qualityPresets.map(p => <option key={p} value={p}>{p}</option>)}
                    </select>
                  </div>

                  <div className="grid grid-cols-2 gap-2">
                    <div>
                      <label className="text-gray-500 text-[10px] uppercase mb-1 block">Max Size</label>
                      <div className="relative">
                        <input
                          type="text"
                          value={maxElementSize}
                          onChange={(e) => {
                            const val = parseNumberExpression(e.target.value)
                            if (!isNaN(val) && val > 0) setMaxElementSize(val)
                          }}
                          onBlur={(e) => {
                            const val = parseNumberExpression(e.target.value)
                            if (!isNaN(val) && val > 0) setMaxElementSize(val)
                          }}
                          className="w-full bg-white border border-gray-300 rounded px-2 py-1.5 pr-8 text-xs focus:outline-none focus:ring-1 focus:ring-blue-500"
                          disabled={isGenerating}
                        />
                        <span className="absolute right-2 top-1/2 -translate-y-1/2 text-gray-400 text-[10px]">mm</span>
                      </div>
                    </div>
                    <div>
                      <label className="text-gray-500 text-[10px] uppercase mb-1 block">Min Size</label>
                      <div className="relative">
                        <input
                          type="text"
                          value={minElementSize}
                          onChange={(e) => {
                            const val = parseNumberExpression(e.target.value)
                            if (!isNaN(val) && val >= 0.01) setMinElementSize(val)
                          }}
                          onBlur={(e) => {
                            const val = parseNumberExpression(e.target.value)
                            // Enforce 0.01mm floor on blur
                            if (!isNaN(val)) setMinElementSize(Math.max(0.01, val))
                          }}
                          className="w-full bg-white border border-gray-300 rounded px-2 py-1.5 pr-8 text-xs focus:outline-none focus:ring-1 focus:ring-blue-500"
                          disabled={isGenerating}
                          placeholder="≥0.01"
                        />
                        <span className="absolute right-2 top-1/2 -translate-y-1/2 text-gray-400 text-[10px]">mm</span>
                      </div>
                    </div>
                    <div className="col-span-2">
                      <label className="text-gray-500 text-[10px] uppercase mb-1 block">Order</label>
                      <select
                        value={elementOrder}
                        onChange={(e) => setElementOrder(e.target.value)}
                        className="w-full bg-white border border-gray-300 rounded px-2 py-1.5 text-xs focus:outline-none focus:ring-1 focus:ring-blue-500"
                        disabled={isGenerating}
                      >
                        <option value="1">Linear (Tet4)</option>
                        <option value="2">Quadratic (Tet10)</option>
                      </select>
                    </div>
                  </div>

                  <div>
                    <label className="text-gray-500 text-[10px] uppercase mb-1 block">Ansys Export</label>
                    <select
                      value={ansysMode}
                      onChange={(e) => setAnsysMode(e.target.value)}
                      className="w-full bg-white border border-gray-300 rounded px-2 py-1.5 text-xs focus:outline-none focus:ring-1 focus:ring-blue-500"
                      disabled={isGenerating}
                    >
                      <option value="None">None (Only .msh)</option>
                      <option value="CFD (Fluent)">CFD (Fluent)</option>
                      <option value="FEA (Mechanical)">FEA (Mechanical)</option>
                    </select>
                  </div>

                  <div>
                    <label className="text-gray-500 text-[10px] uppercase mb-1 block">Strategy</label>
                    <select
                      value={meshStrategy}
                      onChange={(e) => setMeshStrategy(e.target.value)}
                      className="w-full bg-white border border-gray-300 rounded px-2 py-1.5 text-xs focus:outline-none focus:ring-1 focus:ring-blue-500"
                      disabled={isGenerating}
                    >
                      {meshStrategies.map(s => <option key={s} value={s}>{s}</option>)}
                    </select>
                  </div>

                  <label className="flex items-center gap-2 text-gray-600 cursor-pointer text-xs">
                    <input
                      type="checkbox"
                      checked={curvatureAdaptive}
                      onChange={(e) => setCurvatureAdaptive(e.target.checked)}
                      className="accent-blue-500"
                      disabled={isGenerating}
                    />
                    Curvature-Adaptive
                  </label>
                </div>
              </div>

              {/* Visualization Settings */}
              <div className="bg-white rounded-lg border border-gray-200 text-sm overflow-hidden">
                <div className="px-3 py-2 bg-gray-100 font-medium text-gray-700 border-b border-gray-200 flex items-center justify-between">
                  <span>Visualization</span>
                  {showHistogram && <BarChart3 className="w-3.5 h-3.5 text-blue-600 animate-pulse" />}
                </div>
                <div className="p-3 space-y-3">
                  <div className="flex flex-col gap-2">
                    <label className="flex items-center gap-2 text-gray-600 cursor-pointer text-xs">
                      <input
                        type="checkbox"
                        checked={showAxes}
                        onChange={(e) => setShowAxes(e.target.checked)}
                        className="accent-blue-500 w-3.5 h-3.5"
                      />
                      Show Axes
                    </label>
                    <label className="flex items-center gap-2 text-gray-600 cursor-pointer text-xs">
                      <input
                        type="checkbox"
                        checked={showHistogram}
                        onChange={(e) => setShowHistogram(e.target.checked)}
                        className="accent-blue-500 w-3.5 h-3.5"
                      />
                      Quality Histogram
                    </label>
                  </div>

                  <div className="pt-2 border-t border-gray-100">
                    <label className="text-gray-400 text-[10px] uppercase mb-1 block">Quality Metric</label>
                    <select
                      value={qualityMetric}
                      onChange={(e) => setQualityMetric(e.target.value)}
                      className="w-full bg-white border border-gray-300 rounded px-2 py-1.5 text-xs focus:outline-none focus:ring-1 focus:ring-blue-500"
                    >
                      <option value="sicn">SICN (Ideal=1)</option>
                      <option value="gamma">Gamma (Ideal=1)</option>
                      <option value="skewness">Skewness</option>
                      <option value="aspectRatio">Aspect Ratio</option>
                    </select>
                  </div>
                </div>
              </div>

              {/* Generate Button */}
              {currentProject && (
                <div className="space-y-2">
                  <button
                    onClick={handleGenerateMesh}
                    disabled={!canGenerate || isGenerating}
                    className={`w-full px-3 py-2 rounded text-xs font-medium transition-colors ${canGenerate && !isGenerating
                      ? 'bg-blue-600 hover:bg-blue-700 text-white'
                      : 'bg-gray-300 text-gray-500 cursor-not-allowed'
                      }`}
                  >
                    {isGenerating ? 'Generating...' : 'Generate Mesh'}
                  </button>

                  {isGenerating && (
                    <button
                      onClick={() => setIsPolling(false)}
                      className="w-full px-3 py-1.5 rounded text-xs font-medium bg-red-500 hover:bg-red-600 text-white transition-colors flex items-center justify-center gap-1"
                    >
                      <Square className="w-3 h-3 fill-current" />
                      Stop
                    </button>
                  )}
                </div>
              )}

              {/* Progress - Only when generating */}
              {isGenerating && (
                <div className="bg-white rounded border border-gray-200 p-2 text-xs">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium text-gray-700">Progress</span>
                    <MeshTimer
                      isRunning={isGenerating}
                      startTime={meshStartTime}
                      status={projectStatus?.status}
                    />
                  </div>
                  <div className="space-y-1.5">
                    {[
                      { key: 'strategy', label: 'Strategy' },
                      { key: '1d', label: '1D' },
                      { key: '2d', label: '2D' },
                      { key: '3d', label: '3D' },
                      { key: 'optimize', label: 'Optimize' },
                      { key: 'quality', label: 'Quality' }
                    ].map(({ key, label }) => (
                      <div key={key} className="flex items-center gap-2">
                        <span className="w-14 text-gray-500">{label}</span>
                        <div className="flex-1 bg-gray-200 rounded-full h-1.5">
                          <div
                            className="bg-blue-600 h-1.5 rounded-full transition-all duration-300"
                            style={{ width: `${meshProgress[key]}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Center Panel - Viewer + Console */}
        <div className="flex-1 flex flex-col">
          {/* 3D Viewer - Takes most of the space */}
          <div className="flex-1 relative min-h-0">
            <MeshViewer
              meshData={meshData}
              projectId={currentProject}
              geometryInfo={geometryInfo}
              filename={projectStatus?.filename}
              qualityMetrics={meshData?.qualityMetrics || projectStatus?.latest_result?.quality_metrics}
              status={projectStatus?.status}
              isLoading={isUploading || projectStatus?.status === 'processing' || isLoadingPreview}
              loadingProgress={isUploading ? uploadProgress : undefined}
              loadingMessage={
                isLoadingPreview ? 'Loading CAD Preview...' :
                  isUploading ? 'Uploading & Processing CAD file...' :
                    (projectStatus?.status === 'processing' ? 'Generating Mesh...' : undefined)
              }
              // Visualization props
              showAxes={showAxes}
              setShowAxes={setShowAxes}
              qualityMetric={qualityMetric}
              setQualityMetric={setQualityMetric}
              showHistogram={showHistogram}
              setShowHistogram={setShowHistogram}
            />
          </div>

          {/* Console at bottom of center area - Collapsible */}
          <div className={`border-t border-gray-600 bg-gray-900 flex flex-col transition-all duration-300 ${consoleOpen ? 'h-56' : 'h-8'}`}>
            {/* Console Header - Always visible */}
            <div
              className="h-8 flex items-center justify-between px-3 bg-gray-800 cursor-pointer hover:bg-gray-700 transition-colors flex-shrink-0"
            >
              <div
                className="flex items-center gap-2 text-gray-300 text-xs flex-1"
                onClick={() => setConsoleOpen(!consoleOpen)}
              >
                <TerminalIcon className="w-3.5 h-3.5" />
                <span className="font-medium">Console</span>
                <span className="text-gray-500">({logs.length} messages)</span>
                {consoleOpen ? <ChevronDown className="w-4 h-4 ml-1" /> : <ChevronUp className="w-4 h-4 ml-1" />}
              </div>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  const text = logs.join('\n');
                  if (navigator.clipboard && navigator.clipboard.writeText) {
                    navigator.clipboard.writeText(text).then(() => {
                      e.target.innerText = 'Copied!';
                      setTimeout(() => { e.target.innerText = 'Copy'; }, 1500);
                    }).catch(() => {
                      // Fallback for clipboard API failure
                      prompt('Copy these logs:', text);
                    });
                  } else {
                    // Fallback for older browsers
                    prompt('Copy these logs:', text);
                  }
                }}
                className="px-2 py-1 text-xs bg-blue-600 hover:bg-blue-700 text-white rounded flex items-center gap-1"
                title="Copy logs"
              >
                <Copy className="w-3 h-3" />
                Copy
              </button>
            </div>

            {/* Console Content - Only when open */}
            {consoleOpen && (
              <div className="flex-1 overflow-hidden">
                <Terminal logs={logs} noHeader={true} />
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Feedback Button - Fixed position */}
      <FeedbackButton userEmail={user?.email} />
    </div>
  )
}

export default App
