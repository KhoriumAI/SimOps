import { useState, useEffect, useCallback } from 'react'
import MeshViewer from './components/MeshViewer'
import FileUpload from './components/FileUpload'
import ProcessStatus from './components/ProcessStatus'
import Terminal from './components/Terminal'
import QualityReport from './components/QualityReport'
import { Download } from 'lucide-react'

const API_BASE = '/api'

function App() {
  const [currentProject, setCurrentProject] = useState(null)
  const [projectStatus, setProjectStatus] = useState(null)
  const [logs, setLogs] = useState([])
  const [meshData, setMeshData] = useState(null)
  const [isPolling, setIsPolling] = useState(false)
  const [selectedStrategy, setSelectedStrategy] = useState('Tetrahedral (Delaunay)')

  const strategies = [
    'Tetrahedral (Delaunay)',
    'Tetrahedral (Frontal)',
    'Tetrahedral (HXT - Parallel)',
    'Hex-Dominant (Recombined)',
    'Quad-Dominant (2D)'
  ]

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
      const response = await fetch(`${API_BASE}/projects/${currentProject}/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          mesh_strategy: selectedStrategy
        })
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
        <div className="w-80 border-r border-gray-700 flex flex-col bg-gray-800">
          <div className="p-4 space-y-4">
            <FileUpload onFileUpload={handleFileUpload} />

            {currentProject && (
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
            )}

            {currentProject && (
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
            )}
          </div>

          {currentProject && (
            <>
              <div className="flex-1 overflow-auto p-4 space-y-4">
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
          <div className="flex-1 border-b border-gray-700">
            <MeshViewer meshData={meshData} />
          </div>

          {/* Terminal */}
          <div className="h-64">
            <Terminal logs={logs} />
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
