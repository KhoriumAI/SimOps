import { useEffect, useRef, useState, useMemo } from 'react'
import { Canvas, useThree } from '@react-three/fiber'
import { OrbitControls, PerspectiveCamera } from '@react-three/drei'
import * as THREE from 'three'
import { Box, MousePointer2, Paintbrush, Scissors, BarChart3, Loader2 } from 'lucide-react'
import QualityHistogram from './QualityHistogram'

function MeshObject({ meshData, clipping, showQuality, showWireframe }) {
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

    // Colors (Float32Array) - if available
    if (meshData.colors && meshData.colors.length > 0) {
      const colors = new Float32Array(meshData.colors)
      geo.setAttribute('color', new THREE.BufferAttribute(colors, 3))
    }

    geo.computeVertexNormals()
    geo.computeBoundingBox()

    return geo
  }, [meshData])

  // Auto-fit camera
  useEffect(() => {
    if (geometry && camera) {
      const boundingBox = geometry.boundingBox
      const center = new THREE.Vector3()
      boundingBox.getCenter(center)
      const size = new THREE.Vector3()
      boundingBox.getSize(size)
      const maxDim = Math.max(size.x, size.y, size.z)

      // Only adjust if this is a fresh load (heuristic: camera at default)
      // or if we want to force reset. For now, just do it on geometry change.
      const distance = maxDim * 2
      camera.position.set(center.x + distance, center.y + distance, center.z + distance)
      camera.lookAt(center)
      camera.updateProjectionMatrix()
    }
  }, [geometry, camera])

  // Clipping planes
  const clippingPlanes = useMemo(() => {
    if (!geometry || !clipping.enabled) return []

    // Only update clipping planes if values change significantly to avoid thrashing
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

  return (
    <group>
      {/* Main Solid Mesh */}
      <mesh ref={meshRef} geometry={geometry} castShadow receiveShadow>
        <meshStandardMaterial
          vertexColors={showQuality && meshData.colors && meshData.colors.length > 0}
          color={showQuality && meshData.colors && meshData.colors.length > 0 ? undefined : "#3b82f6"} // Blue color from screenshot
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
  const [showAxes, setShowAxes] = useState(false) // Default off matching screenshot
  const [showWireframe, setShowWireframe] = useState(true) // Default on matching screenshot
  const [showQuality, setShowQuality] = useState(false)
  const [showHistogram, setShowHistogram] = useState(false)

  // Tools
  const [activeTool, setActiveTool] = useState(null) // 'select', 'paint', 'clip'

  // Initialize defaults based on screenshot
  useEffect(() => {
    // If quality metrics exist, maybe auto-enable quality view? 
    // Screenshot shows Quality unchecked though.
  }, [qualityMetrics])

  const toggleTool = (tool) => {
    if (tool === 'clip') {
      const newState = !clipping.enabled
      setClipping(prev => ({ ...prev, enabled: newState }))
      setActiveTool(newState ? 'clip' : null)
    } else if (tool === 'hist') {
      setShowHistogram(prev => !prev)
    } else {
      setActiveTool(activeTool === tool ? null : tool)
    }
  }

  const hasQualityData = meshData?.colors && meshData.colors.length > 0

  // Simple check box component
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

  // Header Button
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
            <Checkbox
              label="Quality"
              checked={showQuality}
              onChange={setShowQuality}
              disabled={!hasQualityData}
            />
          </div>

          {/* Action Buttons */}
          <div className="flex items-center gap-2">
            <HeaderButton
              icon={BarChart3}
              label="Hist"
              active={showHistogram}
              onClick={() => setShowHistogram(!showHistogram)}
              disabled={!qualityMetrics}
            />
            <HeaderButton
              icon={MousePointer2}
              label="Select"
              active={activeTool === 'select'}
              onClick={() => toggleTool('select')}
            />
            <HeaderButton
              icon={Paintbrush}
              label="Paint"
              active={activeTool === 'paint'}
              onClick={() => toggleTool('paint')}
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

                <MeshObject
                  meshData={meshData}
                  clipping={clipping}
                  showQuality={showQuality}
                  showWireframe={showWireframe}
                />
              </Canvas>
            )}

            {/* Overlays */}

            {/* Axis Gizmo (Left Bottom) */}
            <div className="absolute bottom-4 left-4 pointer-events-none">
              {/* This is usually done with a ViewCube or GizmoHelper in drei, 
                  but for now we'll stick to our custom or simple overlay if needed.
                  The screenshot shows a custom colorful axis graphic. 
                  We'll skip implementing a custom WebGL gizmo for this iteration 
                  unless strictly requested, as it requires complex setup. 
              */}
            </div>

            {/* Histogram Overlay */}
            {qualityMetrics && showHistogram && (
              <div className="absolute top-4 left-4 shadow-xl">
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
