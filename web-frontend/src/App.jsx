import { useState, useEffect, useCallback } from 'react'
import MeshViewer from './components/MeshViewer'
import FileUpload from './components/FileUpload'
import ProcessStatus from './components/ProcessStatus'
import Terminal from './components/Terminal'
import QualityReport from './components/QualityReport'
import { Download, ChevronDown, ChevronUp, Settings, ArrowUp, ArrowDown } from 'lucide-react'

const API_BASE = '/api'

function App() {
  const [currentProject, setCurrentProject] = useState(null)
  const [projectStatus, setProjectStatus] = useState(null)
  const [logs, setLogs] = useState([])
  const [meshData, setMeshData] = useState(null)
  const [isPolling, setIsPolling] = useState(false)
  const [selectedStrategy, setSelectedStrategy] = useState('Tetrahedral (Delaunay)')

  // Advanced Settings State
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [workerCount, setWorkerCount] = useState(4)
  const [minSize, setMinSize] = useState(0.5)
  const [elementType, setElementType] = useState('Tet4')
  const [ansysMode, setAnsysMode] = useState('None')
  const [scoreThreshold, setScoreThreshold] = useState(0.5)
  const [exhaustiveOrder, setExhaustiveOrder] = useState([
    'Delaunay', 'Frontal', 'HXT', 'MMG3D', 'Netgen', 'Quartet'
  ])

  const strategies = [
    'Tetrahedral (Delaunay)',
    'Tetrahedral (Frontal)',
    'Tetrahedral (HXT - Parallel)',
    'Hex-Dominant (Recombined)',
    'Quad-Dominant (2D)',
    'Exhaustive (Parallel Race)'
  ]

  const moveStrategy = (index, direction) => {
    const newOrder = [...exhaustiveOrder]
    if (direction === 'up' && index > 0) {
      [newOrder[index], newOrder[index - 1]] = [newOrder[index - 1], newOrder[index]]
    } else if (direction === 'down' && index < newOrder.length - 1) {
      [newOrder[index], newOrder[index + 1]] = [newOrder[index + 1], newOrder[index]]
    }
    setExhaustiveOrder(newOrder)
  }

  // Poll project status
  useEffect(() => {
    if (!currentProject) return

    const pollStatus = async () => {
      try {
        const response = await fetch(`${API_BASE}/projects/${currentProject}/status`)
        const data = await response.json()
        setProjectStatus(data)
        setLogs(data.logs || [])

        // If completed, fetch mesh data
        if (data.status === 'completed' && !meshData) {
          fetchMeshData(currentProject)
          setIsPolling(false)
        } else if (data.status === 'error') {
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
    } else {
      pollStatus() // Fetch once even if not polling
    }
  }, [currentProject, isPolling, meshData])

  const fetchMeshData = async (projectId) => {
    try {
      const response = await fetch(`${API_BASE}/projects/${projectId}/mesh-data`)
      const data = await response.json()
      setMeshData(data)
    } catch (error) {
      console.error('Failed to fetch mesh data:', error)
    }
  }

  const handleFileUpload = async (file) => {
    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData
      })

      const data = await response.json()
      setCurrentProject(data.project_id)
      setProjectStatus(data)
      setLogs([`[INFO] Uploaded ${file.name}`])
      setMeshData(null)
      setIsPolling(false)
    } catch (error) {
      console.error('Upload failed:', error)
      alert('Failed to upload file')
    }
  }

  const handleGenerateMesh = async () => {
    if (!currentProject) return

    try {
      const payload = {
        mesh_strategy: selectedStrategy,
        worker_count: workerCount,
        min_size: minSize,
        element_order: elementType,
        ansys_mode: ansysMode,
        score_threshold: scoreThreshold,
        exhaustive_strategies: exhaustiveOrder
      }

      const response = await fetch(`${API_BASE}/projects/${currentProject}/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
      })

      if (response.ok) {
        setIsPolling(true)
        setLogs([`[INFO] Starting mesh generation...`])
      }
    } catch (error) {
      console.error('Failed to start mesh generation:', error)
      alert('Failed to start mesh generation')
    }
  }

  const handleDownload = async () => {
    if (!currentProject) return

    try {
      const response = await fetch(`${API_BASE}/projects/${currentProject}/download`)
      if (response.ok) {
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
    } catch (error) {
      console.error('Download failed:', error)
      alert('Failed to download mesh')
    }
  }

  const canGenerate = projectStatus &&
    (projectStatus.status === 'uploaded' || projectStatus.status === 'error')
  const canDownload = projectStatus?.status === 'completed'
  const isGenerating = projectStatus?.status === 'executing_mesh_generation'

  return (
    <div className="h-screen flex flex-col bg-gray-900 text-white">
      {/* Header */}
      <header className="border-b border-gray-700 bg-gray-800">
        <div className="container mx-auto px-4 py-3 flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">Khorium MeshGen</h1>
            <p className="text-sm text-gray-400">Exhaustive Mesh Generation</p>
          </div>
          {canDownload && (
            <button
              onClick={handleDownload}
              className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors"
            >
              <Download className="w-4 h-4" />
              Download Mesh
            </button>
          )}
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Panel - Controls */}
        <div className="w-96 border-r border-gray-700 flex flex-col bg-gray-800">
          <div className="p-4 space-y-4 overflow-y-auto custom-scrollbar">
            <FileUpload onFileUpload={handleFileUpload} />

            {currentProject && (
              <div className="space-y-4">
                <div className="space-y-2">
                  <label className="block text-sm font-medium text-gray-400">Meshing Strategy</label>
                  <select
                    value={selectedStrategy}
                    onChange={(e) => setSelectedStrategy(e.target.value)}
                    className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white focus:outline-none focus:border-blue-500"
                    disabled={isGenerating}
                  >
                    {strategies.map(s => (
                      <option key={s} value={s}>{s}</option>
                    ))}
                  </select>
                </div>

                {/* Advanced Settings Checkbox/Toggle */}
                <div className="border border-gray-700 rounded-lg overflow-hidden">
                  <button
                    onClick={() => setShowAdvanced(!showAdvanced)}
                    className="w-full flex items-center justify-between p-3 bg-gray-700/50 hover:bg-gray-700 transition-colors"
                  >
                    <div className="flex items-center gap-2 text-sm font-medium">
                      <Settings className="w-4 h-4" />
                      Advanced Settings
                    </div>
                    {showAdvanced ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                  </button>

                  {showAdvanced && (
                    <div className="p-3 bg-gray-800 space-y-4">
                      {/* Worker Count */}
                      <div className="space-y-1">
                        <div className="flex justify-between text-xs text-gray-400">
                          <span>Parallel Workers</span>
                          <span>{workerCount}</span>
                        </div>
                        <input
                          type="range" min="1" max="32"
                          value={workerCount}
                          onChange={(e) => setWorkerCount(parseInt(e.target.value))}
                          className="w-full h-1 bg-gray-600 rounded-lg appearance-none cursor-pointer"
                        />
                      </div>

                      {/* Element Type */}
                      <div className="space-y-1">
                        <span className="block text-xs text-gray-400">Element Type</span>
                        <div className="flex gap-2">
                          <button
                            onClick={() => setElementType('Tet4')}
                            className={`flex-1 py-1 text-xs rounded border ${elementType === 'Tet4' ? 'bg-blue-600 border-blue-600' : 'border-gray-600 hover:bg-gray-700'}`}
                          >
                            Tet4 (Linear)
                          </button>
                          <button
                            onClick={() => setElementType('Tet10')}
                            className={`flex-1 py-1 text-xs rounded border ${elementType === 'Tet10' ? 'bg-blue-600 border-blue-600' : 'border-gray-600 hover:bg-gray-700'}`}
                          >
                            Tet10 (Quad)
                          </button>
                        </div>
                      </div>

                      {/* Min Size */}
                      <div className="space-y-1">
                        <span className="block text-xs text-gray-400">Min Element Size (mm)</span>
                        <input
                          type="number" step="0.1" min="0"
                          value={minSize}
                          onChange={(e) => setMinSize(parseFloat(e.target.value))}
                          className="w-full bg-gray-900 border border-gray-600 rounded px-2 py-1 text-sm focus:outline-none focus:border-blue-500"
                        />
                      </div>

                      {/* ANSYS Export */}
                      <div className="space-y-1">
                        <span className="block text-xs text-gray-400">ANSYS Export Mode</span>
                        <select
                          value={ansysMode}
                          onChange={(e) => setAnsysMode(e.target.value)}
                          className="w-full bg-gray-900 border border-gray-600 rounded px-2 py-1 text-sm focus:outline-none focus:border-blue-500"
                        >
                          <option value="None">None</option>
                          <option value="CFD (Fluent)">ANSYS CFD (Fluent)</option>
                          <option value="FEA (Mechanical)">ANSYS FEA (Mechanical)</option>
                        </select>
                      </div>

                      {/* Exhaustive Settings */}
                      {selectedStrategy.includes('Exhaustive') && (
                        <div className="pt-2 border-t border-gray-700 mt-2 space-y-3">
                          <h4 className="text-xs font-bold text-gray-300">Exhaustive Race Config</h4>

                          <div className="space-y-1">
                            <div className="flex justify-between text-xs text-gray-400">
                              <span>Score Threshold (Lower is fast)</span>
                              <span>{scoreThreshold}</span>
                            </div>
                            <input
                              type="range" min="0.1" max="1.0" step="0.05"
                              value={scoreThreshold}
                              onChange={(e) => setScoreThreshold(parseFloat(e.target.value))}
                              className="w-full h-1 bg-gray-600 rounded-lg appearance-none cursor-pointer"
                            />
                          </div>

                          <div className="space-y-1">
                            <span className="block text-xs text-gray-400">Strategy Priority</span>
                            <div className="bg-gray-900 rounded p-1 space-y-1 max-h-32 overflow-y-auto">
                              {exhaustiveOrder.map((strat, idx) => (
                                <div key={strat} className="flex items-center justify-between bg-gray-800 px-2 py-1 rounded text-xs">
                                  <span>{strat}</span>
                                  <div className="flex gap-1">
                                    <button onClick={() => moveStrategy(idx, 'up')} disabled={idx === 0} className="hover:text-blue-400 disabled:opacity-30">
                                      <ArrowUp className="w-3 h-3" />
                                    </button>
                                    <button onClick={() => moveStrategy(idx, 'down')} disabled={idx === exhaustiveOrder.length - 1} className="hover:text-blue-400 disabled:opacity-30">
                                      <ArrowDown className="w-3 h-3" />
                                    </button>
                                  </div>
                                </div>
                              ))}
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>

                <button
                  onClick={handleGenerateMesh}
                  disabled={!canGenerate || isGenerating}
                  className={`w-full px-4 py-3 rounded-lg font-semibold transition-colors ${canGenerate && !isGenerating
                    ? 'bg-green-600 hover:bg-green-700'
                    : 'bg-gray-600 cursor-not-allowed'
                    }`}
                >
                  {isGenerating ? 'Generating...' : 'Generate Mesh'}
                </button>
              </div>
            )}
          </div>

          {currentProject && (
            <>
              <div className="flex-1 overflow-auto p-4 space-y-4 border-t border-gray-700">
                <ProcessStatus status={projectStatus} />

                {/* Iterations */}
                {projectStatus?.iterations && projectStatus.iterations.length > 0 && (
                  <div className="bg-gray-700/50 rounded-lg p-3">
                    <h3 className="text-xs font-semibold text-gray-400 uppercase mb-2">Iterations</h3>
                    <div className="flex flex-wrap gap-2">
                      {projectStatus.iterations.map((iter, idx) => (
                        <button
                          key={iter.id}
                          className={`w-8 h-8 rounded flex items-center justify-center text-sm font-bold transition-colors ${
                            // Highlight if it's the latest/current one
                            idx === projectStatus.iterations.length - 1
                              ? 'bg-blue-600 text-white'
                              : 'bg-gray-600 text-gray-300 hover:bg-gray-500'
                            }`}
                          title={`Score: ${iter.score?.toFixed(1) || 'N/A'}`}
                        >
                          {idx + 1}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {projectStatus?.quality_metrics && (
                <div className="p-4 border-t border-gray-700">
                  <QualityReport metrics={projectStatus.quality_metrics} />
                </div>
              )}
            </>
          )}

        </div>

        {/* Right Panel - 3D Viewer and Terminal */}
        <div className="flex-1 flex flex-col">
          {/* 3D Viewer */}
          <div className="flex-1 border-b border-gray-700 relative">
            <MeshViewer meshData={meshData} />
          </div>

          {/* Terminal */}
          <div className="h-48">
            <Terminal logs={logs} />
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
