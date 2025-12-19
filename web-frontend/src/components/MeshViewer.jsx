import { useEffect, useRef, useState, useMemo } from 'react'
import { Canvas, useThree } from '@react-three/fiber'
import { OrbitControls, PerspectiveCamera } from '@react-three/drei'
import * as THREE from 'three'
import { Box, MousePointer2, Paintbrush, Scissors, BarChart3, Loader2 } from 'lucide-react'
import QualityHistogram from './QualityHistogram'

// Highlighted Face Overlay
function HighlightedFace({ meshData, selectedTag }) {
  const geometry = useMemo(() => {
    if (!meshData || selectedTag === null) return null

    // Find all triangles belonging to this entity tag
    const indices = []
    const entityTags = meshData.entityTags || []

    for (let i = 0; i < entityTags.length; i++) {
      if (entityTags[i] === selectedTag) {
        // Triangle i (vertices i*3, i*3+1, i*3+2)
        indices.push(i * 3, i * 3 + 1, i * 3 + 2)
      }
    }

    if (indices.length === 0) return null

    const subsetGeo = new THREE.BufferGeometry()

    // Extract vertices for this subset
    const allVertices = meshData.vertices
    const subsetVertices = new Float32Array(indices.length * 3)

    for (let j = 0; j < indices.length; j++) {
      const vIdx = indices[j] // index in flat array
      subsetVertices[j * 3] = allVertices[vIdx * 3]
      subsetVertices[j * 3 + 1] = allVertices[vIdx * 3 + 1]
      subsetVertices[j * 3 + 2] = allVertices[vIdx * 3 + 2]
    }

    subsetGeo.setAttribute('position', new THREE.BufferAttribute(subsetVertices, 3))
    subsetGeo.computeVertexNormals()

    // Slight offset to prevent z-fighting
    subsetGeo.translate(0, 0, 0.001)

    return subsetGeo
  }, [meshData, selectedTag])

  if (!geometry) return null

  return (
    <mesh geometry={geometry}>
      <meshBasicMaterial
        color="#ffd700"
        transparent={true}
        opacity={0.6}
        side={THREE.DoubleSide}
        depthTest={false} // Always show on top
        depthWrite={false}
      />
      <meshBasicMaterial
        color="#ffffff"
        wireframe={true}
        transparent={true}
        opacity={0.8}
        depthTest={false}
      />
    </mesh>
  )
}

function MeshObject({ meshData, clipping, showQuality, showWireframe, meshColor, customColors }) {
  const meshRef = useRef()
  const { camera, gl } = useThree()

  // Enable local clipping
  useEffect(() => {
    gl.localClippingEnabled = true
  }, [gl])

  // Create geometry from flat arrays
  const geometry = useMemo(() => {
    if (!meshData || !meshData.vertices) return null

    const geo = new THREE.BufferGeometry()

    // Vertices (Float32Array)
    const vertices = new Float32Array(meshData.vertices)
    geo.setAttribute('position', new THREE.BufferAttribute(vertices, 3))

    // Colors (Float32Array)
    // Priority: Custom (Height) > Quality (if showQuality) > None
    const sourceColors = customColors || (meshData.colors && meshData.colors.length > 0 ? meshData.colors : null)

    if (sourceColors) {
      const colors = new Float32Array(sourceColors)
      geo.setAttribute('color', new THREE.BufferAttribute(colors, 3))
    }

    geo.computeVertexNormals()
    geo.computeBoundingBox()

    return geo
  }, [meshData, customColors])

  // Auto-fit camera
  useEffect(() => {
    if (geometry && camera) {
      const boundingBox = geometry.boundingBox
      const center = new THREE.Vector3()
      boundingBox.getCenter(center)
      const size = new THREE.Vector3()
      boundingBox.getSize(size)
      const maxDim = Math.max(size.x, size.y, size.z)

      const distance = maxDim * 2
      camera.position.set(center.x + distance, center.y + distance, center.z + distance)
      camera.lookAt(center)
      camera.updateProjectionMatrix()
    }
  }, [geometry, camera])

  // Clipping planes
  const clippingPlanes = useMemo(() => {
    if (!geometry || !clipping.enabled) return []

    const bbox = geometry.boundingBox
    const center = new THREE.Vector3()
    bbox.getCenter(center)
    const size = new THREE.Vector3()
    bbox.getSize(size)

    const planes = []

    if (clipping.x) {
      const xPos = center.x + (size.x / 2) * (clipping.xValue / 50)
      planes.push(new THREE.Plane(new THREE.Vector3(-1, 0, 0), xPos))
    }
    if (clipping.y) {
      const yPos = center.y + (size.y / 2) * (clipping.yValue / 50)
      planes.push(new THREE.Plane(new THREE.Vector3(0, -1, 0), yPos))
    }
    if (clipping.z) {
      const zPos = center.z + (size.z / 2) * (clipping.zValue / 50)
      planes.push(new THREE.Plane(new THREE.Vector3(0, 0, -1), zPos))
    }

    return planes
  }, [geometry, clipping.enabled, clipping.x, clipping.y, clipping.z, clipping.xValue, clipping.yValue, clipping.zValue])

  if (!geometry) return null

  // Helper to determine if we are effectively showing vertex colors
  // Case 1: Custom Colors provided (Height mode) -> YES
  // Case 2: Show Quality is true AND mesh has colors -> YES
  // Otherwise -> NO (Solid Color)
  const shouldUseVertexColors = !!customColors || (showQuality && meshData.colors && meshData.colors.length > 0)

  return (
    <group>
      {/* Main Solid Mesh */}
      <mesh ref={meshRef} geometry={geometry} castShadow receiveShadow>
        <meshStandardMaterial
          vertexColors={shouldUseVertexColors}
          color={shouldUseVertexColors ? undefined : meshColor}
          side={THREE.DoubleSide}
          flatShading={true}
          clippingPlanes={clippingPlanes}
          clipShadows={true}
          roughness={0.5}
          metalness={0.1}
        />
      </mesh>

      {/* Wireframe Overlay */}
      {showWireframe && (
        <mesh geometry={geometry}>
          <meshBasicMaterial
            color="#ffffff"
            wireframe={true}
            opacity={0.3}
            transparent={true}
            clippingPlanes={clippingPlanes}
          />
        </mesh>
      )}

      {/* Cap for clipping */}
      {clipping.enabled && (
        <mesh geometry={geometry}>
          <meshBasicMaterial
            color="#ff5555"
            side={THREE.BackSide}
            clippingPlanes={clippingPlanes}
          />
        </mesh>
      )}
    </group>
  )
}

function GridWithAxes({ showAxes }) {
  if (!showAxes) return null
  return (
    <>
      <gridHelper args={[200, 20, '#444444', '#222222']} />
      <axesHelper args={[20]} />
    </>
  )
}

export default function MeshViewer({ meshData, qualityMetrics, filename, isLoading, loadingProgress, loadingMessage }) {
  const [clipping, setClipping] = useState({
    enabled: false,
    x: false,
    y: false,
    z: false,
    xValue: 0,
    yValue: 0,
    zValue: 0
  })

  // UI States
  const [showAxes, setShowAxes] = useState(false)
  const [showWireframe, setShowWireframe] = useState(true)
  const [showHistogram, setShowHistogram] = useState(false)

  // Tools
  const [activeTool, setActiveTool] = useState(null) // 'select', 'paint', 'clip'

  // Color/Style State
  const [meshColor, setMeshColor] = useState('#3b82f6') // Default Blue
  const [colorMode, setColorMode] = useState('solid') // 'solid', 'quality', 'height'

  // Selection State
  const [selectedTag, setSelectedTag] = useState(null)
  const [faceNames, setFaceNames] = useState({})
  const [editingName, setEditingName] = useState("")

  // Timer State
  const [elapsedTime, setElapsedTime] = useState(0)

  // Timer Effect
  useEffect(() => {
    let interval
    if (isLoading) {
      const startTime = Date.now() - (elapsedTime * 1000)
      interval = setInterval(() => {
        setElapsedTime(Math.floor((Date.now() - startTime) / 1000))
      }, 1000)
    }
    return () => clearInterval(interval)
  }, [isLoading])

  // Reset timer on new file
  useEffect(() => {
    if (isLoading && loadingProgress === 0) setElapsedTime(0)
  }, [isLoading, loadingProgress])


  const toggleTool = (tool) => {
    if (tool === 'clip') {
      const newState = !clipping.enabled
      setClipping(prev => ({ ...prev, enabled: newState }))
      setActiveTool(newState ? 'clip' : null)
    } else if (tool === 'hist') {
      setShowHistogram(prev => !prev)
    } else if (tool === 'select') {
      setActiveTool(activeTool === 'select' ? null : 'select')
      setSelectedTag(null) // Clear selection when toggling off
    } else if (tool === 'color') {
      setActiveTool(activeTool === 'color' ? null : 'color')
    } else {
      setActiveTool(activeTool === tool ? null : tool)
    }
  }

  const handleMeshClick = (e) => {
    if (activeTool !== 'select' || !meshData) return
    e.stopPropagation()

    // Get face index
    const faceIndex = e.faceIndex
    if (faceIndex !== undefined && meshData.entityTags) {
      const tag = meshData.entityTags[faceIndex]
      if (tag !== undefined) {
        setSelectedTag(tag === selectedTag ? null : tag)
        setEditingName(faceNames[tag] || "")
        console.log("Selected Face Tag:", tag)
      }
    }
  }

  const handleSaveName = () => {
    if (selectedTag !== null) {
      setFaceNames(prev => ({ ...prev, [selectedTag]: editingName }))
    }
  }

  const hasQualityData = meshData?.colors && meshData.colors.length > 0

  // Checkbox Component
  const Checkbox = ({ label, checked, onChange, disabled }) => (
    <label className={`flex items-center gap-1.5 cursor-pointer text-xs ${disabled ? 'opacity-50' : 'hover:text-white'}`}>
      <div className={`w-3.5 h-3.5 border rounded flex items-center justify-center transition-colors ${checked ? 'bg-blue-600 border-blue-600' : 'border-gray-500 bg-transparent'}`}>
        {checked && <div className="w-1.5 h-1.5 bg-white rounded-[1px]" />}
      </div>
      <span className="select-none">{label}</span>
      <input
        type="checkbox"
        className="hidden"
        checked={checked}
        onChange={e => onChange(e.target.checked)}
        disabled={disabled}
      />
    </label>
  )

  // Header Button Component
  const HeaderButton = ({ icon: Icon, label, active, onClick, disabled }) => (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`flex items-center gap-1.5 px-3 py-1 rounded text-xs transition-all ${active
        ? 'bg-blue-600 text-white shadow-sm'
        : 'bg-gray-700/50 text-gray-300 hover:bg-gray-700 hover:text-white'
        } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
    >
      {Icon && <Icon className="w-3.5 h-3.5" />}
      {label && <span>{label}</span>}
    </button>
  )

  // Computed colors for Height Gradient
  const heightColors = useMemo(() => {
    if (!meshData || !meshData.vertices || colorMode !== 'height') return null

    const count = meshData.vertices.length / 3
    const colors = new Float32Array(count * 3)
    const vertices = meshData.vertices

    // Find Y bounds
    let minY = Infinity, maxY = -Infinity
    for (let i = 1; i < vertices.length; i += 3) {
      if (vertices[i] < minY) minY = vertices[i]
      if (vertices[i] > maxY) maxY = vertices[i]
    }
    const range = maxY - minY || 1

    // Generate gradient (Blue -> Cyan -> Green -> Yellow -> Red)
    const col = new THREE.Color()
    for (let i = 0; i < count; i++) {
      const y = vertices[i * 3 + 1]
      const t = (y - minY) / range
      col.setHSL(0.66 - (t * 0.66), 1.0, 0.5) // Blue(0.66) to Red(0.0)
      colors[i * 3] = col.r
      colors[i * 3 + 1] = col.g
      colors[i * 3 + 2] = col.b
    }
    return colors
  }, [meshData, colorMode])


  return (
    <div className="w-full h-full relative bg-[#2a2b2e] flex flex-col group overflow-hidden text-gray-200 font-sans">

      {/* Top Header Bar */}
      <div className="h-12 bg-gray-800 border-b border-gray-700 flex items-center justify-between px-4 shrink-0 shadow-md z-10">

        {/* Left: File Info */}
        <div className="flex items-center gap-3">
          <div className="bg-yellow-500 text-black text-[10px] font-bold px-1.5 py-0.5 rounded-sm">
            PREVIEW
          </div>
          <div className="flex items-center gap-2 text-sm">
            <span className="font-semibold text-white">{filename || 'No File loaded'}</span>
            {meshData?.numTriangles && (
              <span className="text-gray-400 border-l border-gray-600 pl-2 ml-1 text-xs">
                {meshData.numTriangles.toLocaleString()} tris
              </span>
            )}
            {elapsedTime > 0 && isLoading && <span className="text-blue-400 font-mono text-xs border-l border-gray-600 pl-2 ml-1">⏱ {elapsedTime}s</span>}
          </div>
        </div>

        {/* Right: Controls */}
        <div className="flex items-center gap-6">

          {/* Toggles */}
          <div className="flex items-center gap-4 border-r border-gray-700 pr-6 mr-2">
            <Checkbox
              label="Axes"
              checked={showAxes}
              onChange={setShowAxes}
            />
            <Checkbox
              label="Wireframe"
              checked={showWireframe}
              onChange={setShowWireframe}
            />
            {/* Quality Checkbox logic kept for backward compat/quick toggle if needed */}
          </div>

          {/* Action Buttons */}
          <div className="flex items-center gap-2">
            <HeaderButton
              icon={Paintbrush}
              label="Color"
              active={activeTool === 'color'}
              onClick={() => toggleTool('color')}
            />
            <HeaderButton
              icon={MousePointer2}
              label="Select"
              active={activeTool === 'select'}
              onClick={() => toggleTool('select')}
            />
            <HeaderButton
              icon={BarChart3}
              label="Hist"
              active={showHistogram}
              onClick={() => setShowHistogram(!showHistogram)}
              disabled={!qualityMetrics}
            />
            <HeaderButton
              icon={Scissors}
              label="Clip"
              active={clipping.enabled}
              onClick={() => toggleTool('clip')}
            />
          </div>
        </div>
      </div>

      {/* Main 3D Viewport */}
      <div className="flex-1 relative bg-gradient-to-b from-gray-200 to-gray-300">
        {!meshData && !isLoading ? (
          <div className="absolute inset-0 flex items-center justify-center text-gray-500">
            <div className="text-center">
              <Box className="w-16 h-16 mx-auto mb-4 opacity-20" />
              <p>No mesh loaded</p>
            </div>
          </div>
        ) : (
          <>
            {meshData && (
              <Canvas shadows gl={{ localClippingEnabled: true }}>
                <PerspectiveCamera makeDefault position={[100, 100, 100]} fov={50} />
                <OrbitControls enableDamping dampingFactor={0.05} />

                <ambientLight intensity={0.6} />
                <directionalLight
                  position={[50, 50, 25]}
                  intensity={0.8}
                  castShadow
                  shadow-mapSize-width={2048}
                  shadow-mapSize-height={2048}
                />
                <directionalLight position={[-50, -50, -25]} intensity={0.4} />

                <GridWithAxes showAxes={showAxes} />

                {/* Main Mesh - Interactive for Selection */}
                <group onClick={handleMeshClick}>
                  <MeshObject
                    meshData={meshData}
                    clipping={clipping}
                    showQuality={colorMode === 'quality'}
                    showWireframe={showWireframe}
                    meshColor={meshColor}
                    customColors={colorMode === 'height' ? heightColors : null}
                  />
                </group>

                {/* Highlight Overlay */}
                {selectedTag !== null && (
                  <HighlightedFace meshData={meshData} selectedTag={selectedTag} />
                )}
              </Canvas>
            )}

            {/* Overlays */}

            {/* Color Panel */}
            {activeTool === 'color' && (
              <div className="absolute top-4 right-4 bg-gray-800/90 backdrop-blur p-4 rounded-lg shadow-xl w-64 border border-gray-700 text-gray-200 animate-in fade-in slide-in-from-top-2 z-50">
                <h3 className="text-sm font-bold text-white mb-3 flex items-center gap-2">
                  <Paintbrush className="w-4 h-4" /> Appearance
                </h3>

                <div className="space-y-4">
                  {/* Mode Selection */}
                  <div className="grid grid-cols-3 gap-1 bg-gray-700 p-1 rounded">
                    {['solid', 'quality', 'height'].map(mode => (
                      <button
                        key={mode}
                        onClick={() => setColorMode(mode)}
                        disabled={mode === 'quality' && !hasQualityData}
                        className={`text-xs py-1 px-2 rounded capitalize transition-all ${colorMode === mode
                          ? 'bg-blue-600 text-white shadow'
                          : 'hover:bg-gray-600 text-gray-400'
                          } ${mode === 'quality' && !hasQualityData ? 'opacity-30 cursor-not-allowed' : ''}`}
                      >
                        {mode}
                      </button>
                    ))}
                  </div>

                  {/* Solid Color Picker */}
                  {colorMode === 'solid' && (
                    <div className="space-y-2">
                      <label className="text-xs text-gray-400 font-medium">Mesh Base Color</label>
                      <div className="flex items-center gap-3 bg-gray-900/50 p-2 rounded border border-gray-700">
                        <input
                          type="color"
                          value={meshColor}
                          onChange={(e) => setMeshColor(e.target.value)}
                          className="bg-transparent border-0 w-8 h-8 p-0 cursor-pointer rounded"
                        />
                        <span className="text-xs font-mono text-gray-300">{meshColor}</span>
                        <div className="text-[10px] text-gray-500 ml-auto">Click to pick</div>
                      </div>
                    </div>
                  )}

                  {/* Legend for Gradients */}
                  {(colorMode === 'quality' || colorMode === 'height') && (
                    <div className="space-y-1">
                      <div className="flex justify-between text-[10px] text-gray-400 uppercase font-semibold">
                        <span>{colorMode === 'quality' ? 'Bad' : 'Low'}</span>
                        <span>{colorMode === 'quality' ? 'Good' : 'High'}</span>
                      </div>
                      <div className={`h-3 rounded-full w-full ${colorMode === 'quality' ? 'bg-gradient-to-r from-red-500 via-yellow-500 to-green-500' : 'bg-gradient-to-r from-blue-500 via-green-500 to-red-500'}`} />
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Selection Panel */}
            {activeTool === 'select' && selectedTag !== null && (
              <div className="absolute top-4 left-4 bg-gray-800/95 backdrop-blur p-4 rounded-lg shadow-xl w-60 border border-gray-700 text-gray-200 animate-in fade-in slide-in-from-left-2 z-50">
                <h3 className="text-sm font-bold text-white mb-2 flex items-center gap-2"><MousePointer2 className="w-4 h-4 text-yellow-500" /> Selected Surface</h3>
                <div className="space-y-3">
                  <div className="text-xs text-gray-400">
                    ID: <span className="font-mono text-white bg-gray-700 px-1 rounded">{selectedTag}</span>
                  </div>
                  <div className="space-y-1">
                    <label className="text-[10px] uppercase text-gray-500 font-bold">Name</label>
                    <div className="flex gap-2">
                      <input
                        type="text"
                        value={editingName}
                        onChange={(e) => setEditingName(e.target.value)}
                        className="bg-gray-900 border border-gray-600 rounded px-2 py-1 text-xs text-white w-full focus:ring-1 focus:ring-blue-500 outline-none"
                        placeholder="Enter name..."
                      />
                      <button onClick={handleSaveName} className="bg-blue-600 hover:bg-blue-500 text-white text-xs px-2 py-1 rounded">Save</button>
                    </div>
                  </div>
                  {faceNames[selectedTag] && (
                    <div className="text-green-400 text-[10px]">Name saved!</div>
                  )}
                </div>
              </div>
            )}

            {/* Histogram Overlay */}
            {qualityMetrics && showHistogram && (
              <div className="absolute top-16 left-4 shadow-xl z-40">
                <QualityHistogram
                  qualityMetrics={qualityMetrics}
                  isVisible={true}
                />
              </div>
            )}

            {/* Clipping Controls (Only when generic Clip is active) */}
            {clipping.enabled && (
              <div className="absolute top-4 right-4 bg-gray-800/90 backdrop-blur p-4 rounded-lg shadow-xl w-64 border border-gray-700 text-gray-200">
                <h3 className="text-sm font-bold text-white mb-3 flex items-center gap-2">
                  <Scissors className="w-4 h-4" /> Clipping Planes
                </h3>

                <div className="space-y-4">
                  {/* X Axis */}
                  <div className="space-y-1">
                    <div className="flex justify-between text-xs">
                      <label className="flex items-center gap-2 cursor-pointer hover:text-white">
                        <input
                          type="checkbox"
                          checked={clipping.x}
                          onChange={(e) => setClipping({ ...clipping, x: e.target.checked })}
                          className="rounded bg-gray-600 border-gray-500"
                        /> X-Axis
                      </label>
                      <span>{clipping.xValue}%</span>
                    </div>
                    <input
                      type="range" min="-50" max="50"
                      value={clipping.xValue}
                      onChange={(e) => setClipping({ ...clipping, xValue: parseInt(e.target.value) })}
                      disabled={!clipping.x}
                      className="w-full h-1 bg-gray-600 rounded-lg appearance-none cursor-pointer accent-blue-500"
                    />
                  </div>

                  {/* Y Axis */}
                  <div className="space-y-1">
                    <div className="flex justify-between text-xs">
                      <label className="flex items-center gap-2 cursor-pointer hover:text-white">
                        <input
                          type="checkbox"
                          checked={clipping.y}
                          onChange={(e) => setClipping({ ...clipping, y: e.target.checked })}
                          className="rounded bg-gray-600 border-gray-500"
                        /> Y-Axis
                      </label>
                      <span>{clipping.yValue}%</span>
                    </div>
                    <input
                      type="range" min="-50" max="50"
                      value={clipping.yValue}
                      onChange={(e) => setClipping({ ...clipping, yValue: parseInt(e.target.value) })}
                      disabled={!clipping.y}
                      className="w-full h-1 bg-gray-600 rounded-lg appearance-none cursor-pointer accent-blue-500"
                    />
                  </div>

                  {/* Z Axis */}
                  <div className="space-y-1">
                    <div className="flex justify-between text-xs">
                      <label className="flex items-center gap-2 cursor-pointer hover:text-white">
                        <input
                          type="checkbox"
                          checked={clipping.z}
                          onChange={(e) => setClipping({ ...clipping, z: e.target.checked })}
                          className="rounded bg-gray-600 border-gray-500"
                        /> Z-Axis
                      </label>
                      <span>{clipping.zValue}%</span>
                    </div>
                    <input
                      type="range" min="-50" max="50"
                      value={clipping.zValue}
                      onChange={(e) => setClipping({ ...clipping, zValue: parseInt(e.target.value) })}
                      disabled={!clipping.z}
                      className="w-full h-1 bg-gray-600 rounded-lg appearance-none cursor-pointer accent-blue-500"
                    />
                  </div>
                </div>
              </div>
            )}
          </>
        )}

        {/* Loading Overlay */}
        {isLoading && (
          <div className="absolute inset-0 bg-black/60 backdrop-blur-sm z-50 flex flex-col items-center justify-center text-white">
            <Loader2 className="w-12 h-12 animate-spin text-blue-500 mb-4" />
            <h3 className="text-lg font-medium mb-1">{loadingMessage || 'Processing...'}</h3>
            {elapsedTime > 0 && <div className="text-blue-400 font-mono text-xl mb-4">{elapsedTime}s</div>}
            {loadingProgress !== undefined && (
              <div className="w-64 space-y-2">
                <div className="h-1.5 w-full bg-gray-700 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-blue-500 transition-all duration-300"
                    style={{ width: `${loadingProgress}%` }}
                  />
                </div>
                <div className="text-xs text-gray-400 text-right">{Math.round(loadingProgress)}%</div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
