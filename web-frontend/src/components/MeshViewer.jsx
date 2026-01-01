import { useEffect, useRef, useState, useMemo, useCallback } from 'react'
import { Canvas, useThree, useFrame } from '@react-three/fiber'
import { OrbitControls, PerspectiveCamera, GizmoHelper, GizmoViewport } from '@react-three/drei'
import * as THREE from 'three'
import { Box, Loader2, MousePointer2, Tag, X, BarChart3, Scissors, Save } from 'lucide-react'
import { API_BASE } from '../config'
import QualityHistogram from './QualityHistogram'

function SliceMesh({ sliceData, clippingPlanes }) {
  const geometry = useMemo(() => {
    if (!sliceData || !sliceData.vertices || sliceData.vertices.length === 0) return null
    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.Float32BufferAttribute(sliceData.vertices, 3))
    geo.setAttribute('color', new THREE.Float32BufferAttribute(sliceData.colors, 3))
    if (sliceData.indices && sliceData.indices.length > 0) {
      geo.setIndex(sliceData.indices)
    }
    geo.computeVertexNormals()
    return geo
  }, [sliceData])

  if (!geometry) return null

  return (
    <mesh geometry={geometry}>
      <meshBasicMaterial
        vertexColors={true}
        side={THREE.DoubleSide}
        transparent={true}
        opacity={0.9}
        clippingPlanes={clippingPlanes}
      />
    </mesh>
  )
}

function FloatingProgress({ progress, visible }) {
  if (!visible) return null;

  const steps = [
    { key: 'strategy', label: 'Strategy' },
    { key: '1d', label: '1D' },
    { key: '2d', label: '2D' },
    { key: '3d', label: '3D' },
    { key: 'optimize', label: 'Optimize' },
    { key: 'quality', label: 'Quality' }
  ];

  return (
    <div className="absolute top-6 left-1/2 -translate-x-1/2 z-50 pointer-events-none w-72">
      <div className="bg-white/40 backdrop-blur-md border border-white/30 rounded-xl p-4 shadow-2xl overflow-hidden">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Loader2 className="w-4 h-4 text-blue-600 animate-spin" />
            <span className="font-semibold text-gray-800 text-sm">Generating Mesh</span>
          </div>
          <div className="text-[10px] font-mono bg-blue-500/10 text-blue-700 px-1.5 py-0.5 rounded border border-blue-500/20">
            STRATEGY RACE
          </div>
        </div>

        <div className="space-y-2.5">
          {steps.map(({ key, label }) => (
            <div key={key} className="space-y-1">
              <div className="flex justify-between text-[10px] font-medium">
                <span className="text-gray-700 uppercase tracking-tight">{label}</span>
                <span className="text-blue-700 font-bold">{Math.round(progress[key] || 0)}%</span>
              </div>
              <div className="h-1.5 w-full bg-gray-200/50 rounded-full overflow-hidden border border-white/20">
                <div
                  className="h-full bg-gradient-to-r from-blue-500 to-indigo-600 transition-all duration-700 ease-out"
                  style={{ width: `${progress[key] || 0}%` }}
                />
              </div>
            </div>
          ))}
        </div>

        <div className="mt-4 flex items-center gap-2">
          <div className="flex-1 h-px bg-gradient-to-r from-transparent via-gray-400/30 to-transparent" />
          <span className="text-[9px] text-gray-500 font-medium">COMPUTING ON AWS</span>
          <div className="flex-1 h-px bg-gradient-to-r from-transparent via-gray-400/30 to-transparent" />
        </div>
      </div>
    </div>
  );
}

function AxesIndicator({ visible }) {
  if (!visible) return null
  return (
    <GizmoHelper alignment="bottom-left" margin={[30, 30]}>
      <GizmoViewport
        axisColors={['#ff4444', '#44ff44', '#4444ff']}
        labelColor="white"
        scale={20}
      />
    </GizmoHelper>
  )
}

function MeshObject({ meshData, sliceData, clipping, showQuality, showWireframe, wireframeColor, wireframeOpacity, wireframeScale, meshColor, opacity, metalness, roughness, colorMode, gradientColors, onFaceSelect, selectionMode, selectedFaces }) {
  const meshRef = useRef()
  const { camera, gl, raycaster, pointer } = useThree()

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

    // Colors (Float32Array) - if available (for quality mode)
    // Support multiple color sets if available
    const activeColors = (meshData.qualityColors && showQuality) ?
      meshData.qualityColors[showQuality === true ? 'sicn' : showQuality] :
      meshData.colors;

    if (activeColors && activeColors.length > 0) {
      const colors = new Float32Array(activeColors)
      geo.setAttribute('color', new THREE.BufferAttribute(colors, 3))
    }

    geo.computeVertexNormals()
    geo.computeBoundingBox()

    // Position geometry so it sits ON the XY plane at z=0 
    // and is centered on X and Y
    const offset = new THREE.Vector3();
    geo.boundingBox.getCenter(offset).negate();
    offset.z = -geo.boundingBox.min.z;
    geo.translate(offset.x, offset.y, offset.z);
    geo.computeBoundingBox();

    return geo
  }, [meshData, showQuality])

  // Create gradient colors based on height (Z position in original coords)
  const gradientGeometry = useMemo(() => {
    if (!geometry) return null

    const geo = new THREE.BufferGeometry()

    // Copy position attribute
    geo.setAttribute('position', geometry.attributes.position.clone())

    // Copy normals if available
    if (geometry.attributes.normal) {
      geo.setAttribute('normal', geometry.attributes.normal.clone())
    } else {
      geo.computeVertexNormals()
    }

    const positions = geo.attributes.position.array
    const vertexCount = positions.length / 3

    // Find min/max Z (height in original CAD coords)
    let minZ = Infinity, maxZ = -Infinity
    for (let i = 0; i < vertexCount; i++) {
      const z = positions[i * 3 + 2] // Z coordinate
      minZ = Math.min(minZ, z)
      maxZ = Math.max(maxZ, z)
    }

    // Create gradient colors
    const colors = new Float32Array(vertexCount * 3)
    const range = maxZ - minZ || 1

    // Parse gradient colors
    const color1 = new THREE.Color(gradientColors?.start || '#4a9eff')
    const color2 = new THREE.Color(gradientColors?.end || '#ff6b6b')

    for (let i = 0; i < vertexCount; i++) {
      const z = positions[i * 3 + 2]
      const t = (z - minZ) / range // Normalized 0-1

      // Interpolate between colors
      colors[i * 3] = color1.r + (color2.r - color1.r) * t
      colors[i * 3 + 1] = color1.g + (color2.g - color1.g) * t
      colors[i * 3 + 2] = color1.b + (color2.b - color1.b) * t
    }

    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3))
    geo.computeBoundingBox()

    // No need to translate again, it already has geometry's positions

    return geo
  }, [geometry, gradientColors])

  // Auto-fit camera (accounting for Z-up to Y-up rotation)
  useEffect(() => {
    if (geometry && camera) {
      const size = new THREE.Vector3()
      geometry.boundingBox.getSize(size)
      const maxDim = Math.max(size.x, size.y, size.z)

      // The model is centered on XY, and its base is at world-Y=0 (after rotation).
      // In Three.js world space (after MeshObject group rotation):
      // Center of object is [0, size.z/2, 0] since original Z is now world Y.
      const viewerCenter = new THREE.Vector3(0, size.z / 2, 0)

      // Position camera at a good distance
      const distance = maxDim * 1.5
      camera.position.set(distance, distance * 0.8, distance)
      camera.lookAt(viewerCenter)
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
      // X plane: Normal (-1, 0, 0) to cut from right
      const xPos = center.x + (size.x / 2) * (clipping.xValue / 50)
      planes.push(new THREE.Plane(new THREE.Vector3(-1, 0, 0), xPos))
    }
    if (clipping.y) {
      // Y plane
      const yPos = center.y + (size.y / 2) * (clipping.yValue / 50)
      planes.push(new THREE.Plane(new THREE.Vector3(0, -1, 0), yPos))
    }
    if (clipping.z) {
      // Z plane
      const zPos = center.z + (size.z / 2) * (clipping.zValue / 50)
      planes.push(new THREE.Plane(new THREE.Vector3(0, 0, -1), zPos))
    }

    return planes
  }, [geometry, clipping])

  // Handle face click for selection
  const handleClick = useCallback((event) => {
    if (!selectionMode || !meshRef.current || !onFaceSelect) return

    event.stopPropagation()

    const intersects = raycaster.intersectObject(meshRef.current)
    if (intersects.length > 0) {
      const faceIndex = intersects[0].faceIndex
      const point = intersects[0].point
      const normal = intersects[0].face?.normal

      const isFloodFill = event.shiftKey || event.metaKey || event.ctrlKey
      onFaceSelect({
        faceIndex,
        point: { x: point.x, y: point.y, z: point.z },
        normal: normal ? { x: normal.x, y: normal.y, z: normal.z } : null,
      }, isFloodFill)
    }
  }, [selectionMode, onFaceSelect, raycaster])

  // Create highlight geometry for selected faces
  const selectedFaceGeometry = useMemo(() => {
    if (!selectedFaces || selectedFaces.length === 0 || !geometry) return null

    const positions = geometry.attributes.position.array
    const highlightPositions = []

    selectedFaces.forEach(face => {
      const idx = face.faceIndex * 9 // 3 vertices * 3 coords
      if (idx + 8 < positions.length) {
        for (let i = 0; i < 9; i++) {
          highlightPositions.push(positions[idx + i])
        }
      }
    })

    if (highlightPositions.length === 0) return null

    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.Float32BufferAttribute(highlightPositions, 3))
    geo.computeVertexNormals()
    return geo
  }, [selectedFaces, geometry])

  if (!geometry) return null

  // Determine which geometry and colors to use
  const useVertexColors = colorMode === 'quality' && meshData.colors && meshData.colors.length > 0
  const useGradient = colorMode === 'gradient' && gradientGeometry
  const activeGeometry = useGradient ? gradientGeometry : geometry

  return (
    // Rotate from Z-up (CAD) to Y-up (Three.js) coordinate system
    <group rotation={[-Math.PI / 2, 0, 0]}>
      <mesh
        key={`mesh-${colorMode}-${gradientColors?.start}-${gradientColors?.end}`}
        ref={meshRef}
        geometry={activeGeometry}
        frustumCulled={true}
        onClick={handleClick}
      >
        <meshStandardMaterial
          vertexColors={useVertexColors || useGradient}
          color={useVertexColors || useGradient ? undefined : meshColor}
          side={THREE.DoubleSide}
          flatShading={true}
          clippingPlanes={clippingPlanes}
          clipShadows={true}
          roughness={roughness}
          metalness={metalness}
          opacity={opacity}
          transparent={opacity < 1 || showQuality}
        />
      </mesh>

      {/* Highlight selected faces */}
      {selectedFaceGeometry && (
        <mesh geometry={selectedFaceGeometry}>
          <meshBasicMaterial
            color="#00ff00"
            side={THREE.DoubleSide}
            transparent
            opacity={0.6}
            depthTest={false}
          />
        </mesh>
      )}

      {/* Wireframe overlay - shows ALL triangle edges */}
      {showWireframe && (
        <group scale={[wireframeScale, wireframeScale, wireframeScale]}>
          <lineSegments>
            <wireframeGeometry args={[activeGeometry]} />
            <lineBasicMaterial
              color={wireframeColor}
              opacity={wireframeOpacity}
              transparent={true}
              linewidth={1}
              clippingPlanes={clippingPlanes}
            />
          </lineSegments>
        </group>
      )}

      {/* Volumetric Quality Slice */}
      {clipping.enabled && clipping.showQualitySlice && sliceData && (
        <SliceMesh sliceData={sliceData} clippingPlanes={clippingPlanes} />
      )}
    </group>
  )
}

export default function MeshViewer({
  meshData,
  projectId,
  geometryInfo,
  filename,
  qualityMetrics,
  status,
  showHistogram,
  setShowHistogram,
  // Mesh progress from App
  meshProgress,
  loadingStartTime
}) {
  // Derive wireframe visibility: ON for completed meshes, OFF for CAD preview
  const showWireframe = meshData && !meshData.isPreview && status === 'completed'
  // Derive quality coloring: ON for completed meshes with quality data
  const showQuality = status === 'completed' && meshData?.colors?.length > 0
  const [clipping, setClipping] = useState({
    enabled: false,
    showQualitySlice: true,
    x: false,
    y: false,
    z: false,
    xValue: 0,
    yValue: 0,
    zValue: 0
  })

  const [sliceData, setSliceData] = useState(null)
  const [isSlicing, setIsSlicing] = useState(false)

  // Fetch slice from backend when clipping changes
  useEffect(() => {
    if (!clipping.enabled || !clipping.showQualitySlice || !projectId || !meshData) {
      setSliceData(null)
      return
    }

    // Determine active axis
    let activeAxis = null
    let value = 0
    if (clipping.x) { activeAxis = 'x'; value = clipping.xValue }
    else if (clipping.y) { activeAxis = 'y'; value = clipping.yValue }
    else if (clipping.z) { activeAxis = 'z'; value = clipping.zValue }

    if (!activeAxis) {
      setSliceData(null)
      return
    }

    const timer = setTimeout(async () => {
      setIsSlicing(true)
      try {
        const token = localStorage.getItem('token')
        const response = await fetch(`${API_BASE}/projects/${projectId}/slice`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`
          },
          body: JSON.stringify({
            axis: activeAxis,
            offset: value
          })
        })

        if (response.ok) {
          const data = await response.json()
          setSliceData(data.mesh)
        }
      } catch (err) {
        console.error("Failed to fetch slice:", err)
      } finally {
        setIsSlicing(false)
      }
    }, 500) // Debounce 500ms

    return () => clearTimeout(timer)
  }, [clipping.enabled, clipping.showQualitySlice, clipping.x, clipping.y, clipping.z, clipping.xValue, clipping.yValue, clipping.zValue, projectId, meshData])

  // States are now props from App.jsx or kept here for local UI
  const [showControls, setShowControls] = useState(false)
  const [showPaintPanel, setShowPaintPanel] = useState(false)
  const [showWireframePanel, setShowWireframePanel] = useState(false)

  // Wireframe options
  const [wireframeColor, setWireframeColor] = useState('#000000')
  const [wireframeOpacity, setWireframeOpacity] = useState(0.6)
  const [wireframeScale, setWireframeScale] = useState(1.0) // Scale factor for mesh when viewing wireframe

  // Paint/Color options
  const [meshColor, setMeshColor] = useState('#4a9eff')
  const [colorMode, setColorMode] = useState('solid') // 'solid', 'quality', 'gradient'
  const [opacity, setOpacity] = useState(1.0)
  const [metalness, setMetalness] = useState(0.1)
  const [roughness, setRoughness] = useState(0.5)
  const [gradientColors, setGradientColors] = useState({ start: '#4a9eff', end: '#ff6b6b' })

  // Face selection state
  const [selectionMode, setSelectionMode] = useState(false)
  const [selectedFaces, setSelectedFaces] = useState([])
  const [boundaryZones, setBoundaryZones] = useState({}) // { name: [indices] }
  const [showFacePanel, setShowFacePanel] = useState(false)
  const [pendingFaceName, setPendingFaceName] = useState('')
  const [isSavingZones, setIsSavingZones] = useState(false)

  const hasQualityData = (meshData?.colors && meshData.colors.length > 0) || meshData?.hasQualityData
  const isCompleted = status === 'completed'

  // Fetch zones on load
  useEffect(() => {
    if (projectId && isCompleted) {
      const fetchZones = async () => {
        try {
          const token = localStorage.getItem('token')
          const response = await fetch(`${API_BASE}/projects/${projectId}/boundary-zones`, {
            headers: { 'Authorization': `Bearer ${token}` }
          })
          if (response.ok) {
            const data = await response.json()
            setBoundaryZones(data)
          }
        } catch (err) {
          console.error("Failed to fetch boundary zones:", err)
        }
      }
      fetchZones()
    }
  }, [projectId, isCompleted])

  // Build adjacency map for flood fill
  const adjacency = useMemo(() => {
    if (!meshData || !meshData.vertices || meshData.vertices.length === 0) return null

    console.log("[FLOOD-FILL] Building adjacency map...")
    const startTime = performance.now()
    const vertices = meshData.vertices
    const numFaces = vertices.length / 9
    const nodeToFaces = new Map()

    // 1. Group faces by their vertices
    for (let i = 0; i < numFaces; i++) {
      for (let v = 0; v < 3; v++) {
        const x = vertices[i * 9 + v * 3].toFixed(5)
        const y = vertices[i * 9 + v * 3 + 1].toFixed(5)
        const z = vertices[i * 9 + v * 3 + 2].toFixed(5)
        const key = `${x},${y},${z}`

        if (!nodeToFaces.has(key)) nodeToFaces.set(key, [])
        nodeToFaces.get(key).push(i)
      }
    }

    // 2. Identify neighbors (sharing at least 2 vertices = edge adjacency)
    const neighbors = Array.from({ length: numFaces }, () => [])
    const facePairCounts = new Map() // (i,j) -> count

    for (const faces of nodeToFaces.values()) {
      if (faces.length < 2) continue
      for (let a = 0; a < faces.length; a++) {
        for (let b = a + 1; b < faces.length; b++) {
          const f1 = faces[a]
          const f2 = faces[b]
          const pairKey = f1 < f2 ? `${f1}_${f2}` : `${f2}_${f1}`
          const count = (facePairCounts.get(pairKey) || 0) + 1
          facePairCounts.set(pairKey, count)

          if (count === 2) { // Shared an edge
            neighbors[f1].push(f2)
            neighbors[f2].push(f1)
          }
        }
      }
    }

    console.log(`[FLOOD-FILL] Adjacency built in ${(performance.now() - startTime).toFixed(1)}ms. ${numFaces} faces.`)
    return neighbors
  }, [meshData])

  // Flood fill algorithm
  const performFloodFill = useCallback((startFaceIndex) => {
    if (!adjacency) return [startFaceIndex]

    const visited = new Set()
    const queue = [startFaceIndex]
    visited.add(startFaceIndex)

    // We only flood fill across surfaces (same entity_tag if available)
    const targetEntity = meshData?.entity_tags ? meshData.entity_tags[startFaceIndex] : null

    let iterations = 0
    while (queue.length > 0 && iterations < 50000) { // Safety limit
      const current = queue.shift()
      iterations++

      const neighbors = adjacency[current]
      for (const next of neighbors) {
        if (!visited.has(next)) {
          // Check entity constraint
          if (targetEntity === null || (meshData?.entity_tags && meshData.entity_tags[next] === targetEntity)) {
            visited.add(next)
            queue.push(next)
          }
        }
      }
    }

    return Array.from(visited)
  }, [adjacency, meshData])

  // Handle face selection
  const handleFaceSelect = useCallback((faceData, isFloodFill = false) => {
    if (!selectionMode) return

    let indicesToSelect = [faceData.faceIndex]
    if (isFloodFill) {
      indicesToSelect = performFloodFill(faceData.faceIndex)
    }

    setSelectedFaces(prev => {
      // For now, toggle the whole group
      const firstIndex = faceData.faceIndex
      const alreadySelected = prev.find(f => f.faceIndex === firstIndex)

      if (alreadySelected) {
        // Deselect the group (complex to find the exact group, just deselect everything for now or the overlap)
        const toDeselectSet = new Set(indicesToSelect)
        return prev.filter(f => !toDeselectSet.has(f.faceIndex))
      }

      // Add new selections
      const newSelections = indicesToSelect.map(idx => ({
        faceIndex: idx,
        // Optional: reconstruct position/normal if needed, or just use idx
      }))

      return [...prev, ...newSelections]
    })
    setShowFacePanel(true)
  }, [selectionMode, performFloodFill])

  // Save face name and sync to backend
  const saveFaceName = useCallback(async () => {
    if (selectedFaces.length > 0 && pendingFaceName.trim()) {
      const name = pendingFaceName.trim()
      const indices = selectedFaces.map(f => f.faceIndex)

      const newZones = { ...boundaryZones }
      newZones[name] = [...(newZones[name] || []), ...indices]

      setBoundaryZones(newZones)
      setIsSavingZones(true)

      try {
        const token = localStorage.getItem('token')
        await fetch(`${API_BASE}/projects/${projectId}/boundary-zones`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`
          },
          body: JSON.stringify(newZones)
        })
      } catch (err) {
        console.error("Failed to save boundary zones:", err)
      } finally {
        setIsSavingZones(false)
        setPendingFaceName('')
        setSelectedFaces([])
        setShowFacePanel(false)
      }
    }
  }, [selectedFaces, pendingFaceName, boundaryZones, projectId])

  // Clear selection
  const clearSelection = useCallback(() => {
    setSelectedFaces([])
    setShowFacePanel(false)
    setPendingFaceName('')
  }, [])

  // Gradient presets
  const gradientPresets = [
    { name: 'Blue to Red', start: '#4a9eff', end: '#ff6b6b' },
    { name: 'Green to Yellow', start: '#22c55e', end: '#facc15' },
    { name: 'Purple to Pink', start: '#8b5cf6', end: '#ec4899' },
    { name: 'Cyan to Blue', start: '#06b6d4', end: '#3b82f6' },
    { name: 'Orange to Red', start: '#f97316', end: '#dc2626' },
    { name: 'Rainbow', start: '#22d3ee', end: '#f43f5e' },
  ]

  // Preset color palette
  const colorPresets = [
    { name: 'Blue', color: '#4a9eff' },
    { name: 'Green', color: '#4ade80' },
    { name: 'Orange', color: '#fb923c' },
    { name: 'Red', color: '#f87171' },
    { name: 'Purple', color: '#a78bfa' },
    { name: 'Cyan', color: '#22d3ee' },
    { name: 'Gold', color: '#fbbf24' },
    { name: 'Steel', color: '#94a3b8' },
  ]

  return (
    <div className="w-full h-full relative bg-gradient-to-br from-gray-200 to-gray-300">
      {/* Empty State */}
      {!meshData && (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center text-gray-500">
            <Box className="w-16 h-16 mx-auto mb-3 opacity-30" />
            <p className="text-sm">Open a CAD file to begin</p>
          </div>
        </div>
      )}

      {meshData && (
        <>
          {/* Floating Action Buttons - Bottom Left */}
          <div className="absolute top-2 left-2 z-20 flex gap-2">
            <button
              onClick={() => setShowPaintPanel(!showPaintPanel)}
              className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all shadow-lg flex items-center gap-1.5 ${showPaintPanel ? 'bg-blue-600 text-white' : 'bg-gray-800/90 text-gray-300 hover:bg-gray-700'}`}
            >
              <div
                className="w-2.5 h-2.5 rounded-full border border-white/20"
                style={{ backgroundColor: meshColor }}
              ></div>
              Paint
            </button>
            <button
              onClick={() => {
                if (!showControls) {
                  // Opening panel - auto-enable with X axis
                  setShowControls(true)
                  setClipping({ enabled: true, showQualitySlice: true, x: true, y: false, z: false, xValue: 0, yValue: 0, zValue: 0 })
                } else {
                  // Closing panel - disable clipping
                  setShowControls(false)
                  setClipping({ enabled: false, showQualitySlice: true, x: false, y: false, z: false, xValue: 0, yValue: 0, zValue: 0 })
                }
              }}
              className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all shadow-lg flex items-center gap-1.5 ${showControls ? 'bg-blue-600 text-white' : 'bg-gray-800/90 text-gray-300 hover:bg-gray-700'}`}
            >
              <Scissors className="w-3.5 h-3.5" />
              Section View
            </button>
            <button
              onClick={() => { setSelectionMode(!selectionMode); if (selectionMode) clearSelection(); }}
              className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all shadow-lg flex items-center gap-1.5 ${selectionMode ? 'bg-green-600 text-white' : 'bg-gray-800/90 text-gray-300 hover:bg-gray-700'}`}
            >
              <MousePointer2 className="w-3.5 h-3.5" />
              Select
            </button>
          </div>

          {/* Paint Panel */}
          {showPaintPanel && (
            <div className="absolute top-10 left-3 bg-gray-900/95 backdrop-blur rounded p-3 z-10 text-xs text-gray-300 w-48">
              <div className="font-medium text-white mb-2 text-sm">🎨 Paint Options</div>

              {/* Color Mode */}
              <div className="mb-3">
                <label className="text-gray-400 text-[10px] uppercase">Color Mode</label>
                <select
                  value={colorMode}
                  onChange={(e) => setColorMode(e.target.value)}
                  className="w-full mt-1 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs"
                >
                  <option value="solid">Solid Color</option>
                  {hasQualityData && <option value="quality">Quality Colors</option>}
                  <option value="gradient">Gradient</option>
                </select>
              </div>

              {/* Color Picker - Solid */}
              {colorMode === 'solid' && (
                <div className="mb-3">
                  <label className="text-gray-400 text-[10px] uppercase">Mesh Color</label>
                  <div className="flex items-center gap-2 mt-1">
                    <input
                      type="color"
                      value={meshColor}
                      onChange={(e) => setMeshColor(e.target.value)}
                      className="w-8 h-8 rounded cursor-pointer border-0"
                    />
                    <span className="text-gray-400">{meshColor}</span>
                  </div>

                  {/* Preset Colors */}
                  <div className="flex flex-wrap gap-1 mt-2">
                    {colorPresets.map(preset => (
                      <button
                        key={preset.name}
                        onClick={() => setMeshColor(preset.color)}
                        className={`w-5 h-5 rounded border-2 transition-transform hover:scale-110 ${meshColor === preset.color ? 'border-white' : 'border-transparent'}`}
                        style={{ backgroundColor: preset.color }}
                        title={preset.name}
                      />
                    ))}
                  </div>
                </div>
              )}

              {/* Gradient Controls */}
              {colorMode === 'gradient' && (
                <div className="mb-3">
                  <label className="text-gray-400 text-[10px] uppercase mb-2 block">Height Gradient</label>

                  {/* Gradient Preview */}
                  <div
                    className="h-6 rounded mb-2 border border-gray-600"
                    style={{ background: `linear-gradient(to right, ${gradientColors.start}, ${gradientColors.end})` }}
                  />

                  {/* Start/End Color Pickers */}
                  <div className="flex gap-2 mb-2">
                    <div className="flex-1">
                      <label className="text-gray-500 text-[9px]">Bottom</label>
                      <input
                        type="color"
                        value={gradientColors.start}
                        onChange={(e) => setGradientColors(prev => ({ ...prev, start: e.target.value }))}
                        className="w-full h-6 rounded cursor-pointer border-0"
                      />
                    </div>
                    <div className="flex-1">
                      <label className="text-gray-500 text-[9px]">Top</label>
                      <input
                        type="color"
                        value={gradientColors.end}
                        onChange={(e) => setGradientColors(prev => ({ ...prev, end: e.target.value }))}
                        className="w-full h-6 rounded cursor-pointer border-0"
                      />
                    </div>
                  </div>

                  {/* Gradient Presets */}
                  <label className="text-gray-500 text-[9px]">Presets</label>
                  <div className="flex flex-wrap gap-1 mt-1">
                    {gradientPresets.map(preset => (
                      <button
                        key={preset.name}
                        onClick={() => setGradientColors({ start: preset.start, end: preset.end })}
                        className="w-8 h-4 rounded border border-gray-600 transition-transform hover:scale-110"
                        style={{ background: `linear-gradient(to right, ${preset.start}, ${preset.end})` }}
                        title={preset.name}
                      />
                    ))}
                  </div>
                </div>
              )}

              {/* Opacity */}
              <div className="mb-3">
                <div className="flex justify-between text-[10px]">
                  <label className="text-gray-400 uppercase">Opacity</label>
                  <span>{Math.round(opacity * 100)}%</span>
                </div>
                <input
                  type="range"
                  min="0.1" max="1" step="0.1"
                  value={opacity}
                  onChange={(e) => setOpacity(parseFloat(e.target.value))}
                  className="w-full h-1 mt-1 bg-gray-700 rounded appearance-none cursor-pointer accent-blue-500"
                />
              </div>

              {/* Metalness */}
              <div className="mb-3">
                <div className="flex justify-between text-[10px]">
                  <label className="text-gray-400 uppercase">Metalness</label>
                  <span>{Math.round(metalness * 100)}%</span>
                </div>
                <input
                  type="range"
                  min="0" max="1" step="0.1"
                  value={metalness}
                  onChange={(e) => setMetalness(parseFloat(e.target.value))}
                  className="w-full h-1 mt-1 bg-gray-700 rounded appearance-none cursor-pointer accent-blue-500"
                />
              </div>

              {/* Roughness */}
              <div className="mb-2">
                <div className="flex justify-between text-[10px]">
                  <label className="text-gray-400 uppercase">Roughness</label>
                  <span>{Math.round(roughness * 100)}%</span>
                </div>
                <input
                  type="range"
                  min="0" max="1" step="0.1"
                  value={roughness}
                  onChange={(e) => setRoughness(parseFloat(e.target.value))}
                  className="w-full h-1 mt-1 bg-gray-700 rounded appearance-none cursor-pointer accent-blue-500"
                />
              </div>

              {/* Reset Button */}
              <button
                onClick={() => { setMeshColor('#4a9eff'); setOpacity(1); setMetalness(0.1); setRoughness(0.5); setColorMode('solid'); setGradientColors({ start: '#4a9eff', end: '#ff6b6b' }); }}
                className="w-full mt-2 py-1 bg-gray-700 hover:bg-gray-600 rounded text-[10px] transition-colors"
              >
                Reset to Default
              </button>
            </div>
          )}

          {/* Wireframe Settings Panel */}
          {showWireframe && showWireframePanel && (
            <div className="absolute top-10 left-3 bg-gray-900/95 backdrop-blur rounded p-3 z-10 text-xs text-gray-300 w-52">
              <div className="font-medium text-white mb-3 text-sm">🔲 Wireframe Settings</div>

              {/* Wireframe Color */}
              <div className="mb-3">
                <label className="text-[10px] text-gray-400 uppercase mb-1 block">Color</label>
                <div className="flex gap-1">
                  {['#000000', '#ffffff', '#ff0000', '#00ff00', '#0066ff', '#ffff00'].map(color => (
                    <button
                      key={color}
                      onClick={() => setWireframeColor(color)}
                      className={`w-6 h-6 rounded border-2 transition-all ${wireframeColor === color ? 'border-blue-400 scale-110' : 'border-gray-600'}`}
                      style={{ backgroundColor: color }}
                    />
                  ))}
                </div>
              </div>

              {/* Custom Color */}
              <div className="mb-3">
                <label className="text-[10px] text-gray-400 uppercase mb-1 block">Custom Color</label>
                <input
                  type="color"
                  value={wireframeColor}
                  onChange={(e) => setWireframeColor(e.target.value)}
                  className="w-full h-6 rounded cursor-pointer"
                />
              </div>

              {/* Wireframe Opacity */}
              <div className="mb-3">
                <div className="flex justify-between text-[10px] mb-1">
                  <label className="text-gray-400 uppercase">Opacity</label>
                  <span>{Math.round(wireframeOpacity * 100)}%</span>
                </div>
                <input
                  type="range"
                  min="0.1" max="1" step="0.1"
                  value={wireframeOpacity}
                  onChange={(e) => setWireframeOpacity(parseFloat(e.target.value))}
                  className="w-full h-1 bg-gray-700 rounded appearance-none cursor-pointer accent-blue-500"
                />
              </div>

              {/* Wireframe Scale (Explode) */}
              <div className="mb-3">
                <div className="flex justify-between text-[10px] mb-1">
                  <label className="text-gray-400 uppercase">Scale (Explode)</label>
                  <span>{wireframeScale.toFixed(2)}x</span>
                </div>
                <input
                  type="range"
                  min="1" max="1.5" step="0.01"
                  value={wireframeScale}
                  onChange={(e) => setWireframeScale(parseFloat(e.target.value))}
                  className="w-full h-1 bg-gray-700 rounded appearance-none cursor-pointer accent-blue-500"
                />
                <p className="text-[9px] text-gray-500 mt-1">Scale &gt; 1 to "explode" wireframe outward</p>
              </div>

              {/* Reset Wireframe */}
              <button
                onClick={() => { setWireframeColor('#000000'); setWireframeOpacity(0.6); setWireframeScale(1.0); }}
                className="w-full py-1 bg-gray-700 hover:bg-gray-600 rounded text-[10px] transition-colors"
              >
                Reset Wireframe
              </button>
            </div>
          )}

          {/* Quality Metrics - Top Right (only show when mesh is completed) */}
          {isCompleted && qualityMetrics && !showPaintPanel && (
            <div className="absolute top-10 right-3 bg-gray-900/90 backdrop-blur rounded p-2.5 z-10 text-[10px] text-gray-300 min-w-[180px]">
              <div className="font-medium text-white mb-2 text-xs">Quality Metrics</div>

              {/* SICN */}
              <div className="mb-2 pb-2 border-b border-gray-700">
                <div className="text-[9px] text-gray-500 uppercase mb-1">SICN (Shape Quality)</div>
                <div className="grid grid-cols-3 gap-1 text-center">
                  <div>
                    <div className="text-[8px] text-gray-500">Min</div>
                    <div className={(qualityMetrics.sicn_min ?? qualityMetrics.min_sicn) < 0.1 ? 'text-red-400 font-medium' : 'text-green-400 font-medium'}>
                      {(qualityMetrics.sicn_min ?? qualityMetrics.min_sicn)?.toFixed(3) || 'N/A'}
                    </div>
                  </div>
                  <div>
                    <div className="text-[8px] text-gray-500">Avg</div>
                    <div className="text-blue-400 font-medium">{(qualityMetrics.sicn_avg ?? qualityMetrics.avg_sicn)?.toFixed(3) || 'N/A'}</div>
                  </div>
                  <div>
                    <div className="text-[8px] text-gray-500">Max</div>
                    <div className="text-green-400 font-medium">{(qualityMetrics.sicn_max ?? qualityMetrics.max_sicn)?.toFixed(3) || 'N/A'}</div>
                  </div>
                </div>
              </div>

              {/* Gamma */}
              {((qualityMetrics.gamma_min !== undefined || qualityMetrics.gamma_avg !== undefined)) && (
                <div className="mb-2 pb-2 border-b border-gray-700">
                  <div className="text-[9px] text-gray-500 uppercase mb-1">Gamma (Radius Ratio)</div>
                  <div className="grid grid-cols-3 gap-1 text-center">
                    <div>
                      <div className="text-[8px] text-gray-500">Min</div>
                      <div className="text-yellow-400 font-medium">{qualityMetrics.gamma_min?.toFixed(3) || 'N/A'}</div>
                    </div>
                    <div>
                      <div className="text-[8px] text-gray-500">Avg</div>
                      <div className="text-yellow-400 font-medium">{qualityMetrics.gamma_avg?.toFixed(3) || 'N/A'}</div>
                    </div>
                    <div>
                      <div className="text-[8px] text-gray-500">Max</div>
                      <div className="text-yellow-400 font-medium">{qualityMetrics.gamma_max?.toFixed(3) || 'N/A'}</div>
                    </div>
                  </div>
                </div>
              )}

              {/* Element Count */}
              <div className="flex justify-between">
                <span className="text-gray-400">Elements:</span>
                <span className="font-medium">{(qualityMetrics.total_elements ?? qualityMetrics.element_count)?.toLocaleString() || 'N/A'}</span>
              </div>
              {qualityMetrics.poor_elements > 0 && (
                <div className="flex justify-between text-yellow-400 mt-0.5">
                  <span>Poor (&lt;0.1):</span>
                  <span>{qualityMetrics.poor_elements} ({((qualityMetrics.poor_elements / (qualityMetrics.total_elements ?? qualityMetrics.element_count)) * 100).toFixed(1)}%)</span>
                </div>
              )}
            </div>
          )}

          {/* Quality Histogram Panel */}
          {showHistogram && (
            <div className="absolute bottom-3 left-3 z-10">
              <QualityHistogram
                histogramData={meshData?.histogramData}
                qualityMetrics={qualityMetrics ? {
                  min_sicn: qualityMetrics.sicn_min ?? qualityMetrics.min_sicn ?? qualityMetrics.minSICN,
                  avg_sicn: qualityMetrics.sicn_avg ?? qualityMetrics.avg_sicn ?? qualityMetrics.avgSICN,
                  max_sicn: qualityMetrics.sicn_max ?? qualityMetrics.max_sicn ?? qualityMetrics.maxSICN,
                  element_count: qualityMetrics.total_elements ?? qualityMetrics.element_count ?? qualityMetrics.totalElements,
                } : null}
                isVisible={showHistogram}
              />
            </div>
          )}

          {/* Face Selection Panel */}
          {showFacePanel && selectedFaces.length > 0 && (
            <div className="absolute bottom-3 right-3 bg-gray-900/95 backdrop-blur rounded-lg p-3 z-10 text-xs text-gray-300 w-56 shadow-xl border border-gray-700">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <Tag className="w-4 h-4 text-green-400" />
                  <span className="font-medium text-white text-sm">Name Face</span>
                </div>
                <button onClick={clearSelection} className="text-gray-400 hover:text-white">
                  <X className="w-4 h-4" />
                </button>
              </div>

              <div className="text-[10px] text-gray-400 mb-2">
                {selectedFaces.length} face{selectedFaces.length > 1 ? 's' : ''} selected
              </div>

              <input
                type="text"
                value={pendingFaceName}
                onChange={(e) => setPendingFaceName(e.target.value)}
                placeholder="Enter face name..."
                className="w-full bg-gray-800 border border-gray-600 rounded px-2 py-1.5 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-green-500 mb-2"
                onKeyDown={(e) => e.key === 'Enter' && saveFaceName()}
              />

              <div className="flex gap-2">
                <button
                  onClick={saveFaceName}
                  disabled={!pendingFaceName.trim() || isSavingZones}
                  className="flex-1 py-1.5 bg-green-600 hover:bg-green-500 disabled:bg-gray-600 disabled:cursor-not-allowed rounded text-white text-[11px] font-medium transition-colors flex items-center justify-center gap-1"
                >
                  {isSavingZones ? <Loader2 className="w-3 h-3 animate-spin" /> : <Save className="w-3 h-3" />}
                  Save Zone
                </button>
                <button
                  onClick={clearSelection}
                  className="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 rounded text-white text-[11px] transition-colors"
                >
                  Cancel
                </button>
              </div>
            </div>
          )}

          {/* Named Faces List */}
          {Object.keys(faceNames).length > 0 && !showFacePanel && (
            <div className="absolute bottom-3 right-3 bg-gray-900/95 backdrop-blur rounded-lg p-2 z-10 text-xs text-gray-300 max-w-48 shadow-xl border border-gray-700">
              <div className="flex items-center gap-1.5 mb-1.5 text-[10px] text-gray-400">
                <Tag className="w-3 h-3" />
                <span>Named Faces ({Object.keys(faceNames).length})</span>
              </div>
              <div className="space-y-1 max-h-32 overflow-y-auto">
                {Object.entries(faceNames).map(([faceIndex, name]) => (
                  <div key={faceIndex} className="flex items-center justify-between bg-gray-800 rounded px-2 py-1">
                    <span className="text-white truncate">{name}</span>
                    <button
                      onClick={() => {
                        const newNames = { ...faceNames }
                        delete newNames[faceIndex]
                        setFaceNames(newNames)
                      }}
                      className="text-gray-500 hover:text-red-400 ml-2"
                    >
                      <X className="w-3 h-3" />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Selection Mode Indicator */}
          {selectionMode && (
            <div className="absolute bottom-3 left-1/2 -translate-x-1/2 bg-green-600 text-white px-4 py-2 rounded-full text-xs font-medium flex items-center gap-2 shadow-lg z-10">
              <MousePointer2 className="w-4 h-4" />
              Click on mesh to select faces
            </div>
          )}

          <Canvas
            gl={{
              localClippingEnabled: true,
              antialias: true,
              powerPreference: "high-performance",
            }}
            dpr={[1, 1.5]} // Limit pixel ratio for performance
            performance={{ min: 0.5 }} // Adaptive performance
            className="!pt-8"
          >
            <PerspectiveCamera makeDefault position={[100, 100, 100]} fov={45} near={0.1} far={10000} />
            <OrbitControls
              enableDamping={false}
              dampingFactor={0}
              rotateSpeed={0.8}
              panSpeed={0.8}
              zoomSpeed={2.4}
              enabled={!selectionMode}
              makeDefault
            />

            <ambientLight intensity={0.7} />
            <directionalLight position={[50, 50, 25]} intensity={0.6} />
            <directionalLight position={[-50, -50, -25]} intensity={0.3} />
            <hemisphereLight intensity={0.3} />

            <MeshObject
              meshData={meshData}
              sliceData={sliceData}
              clipping={clipping}
              showQuality={showQuality ? qualityMetric : false}
              showWireframe={showWireframe}
              wireframeColor={wireframeColor}
              wireframeOpacity={wireframeOpacity}
              wireframeScale={wireframeScale}
              meshColor={meshColor}
              opacity={opacity}
              metalness={metalness}
              roughness={roughness}
              colorMode={colorMode}
              gradientColors={gradientColors}
              onFaceSelect={handleFaceSelect}
              selectionMode={selectionMode}
              selectedFaces={selectedFaces}
            />

            <gridHelper args={[200, 20, '#aaaaaa', '#dddddd']} />
            <AxesIndicator visible={showAxes} />
          </Canvas>

          {/* Section View Controls Panel */}
          {showControls && (
            <div className="absolute top-10 right-3 bg-gray-900/90 backdrop-blur p-3 rounded z-10 text-xs text-gray-300 w-48">
              <div className="font-medium text-white mb-2">Section View</div>

              <div className="mb-3 pb-2 border-b border-gray-800">
                <label className="flex items-center gap-2 cursor-pointer text-blue-400 font-medium">
                  <input
                    type="checkbox"
                    checked={clipping.showQualitySlice}
                    onChange={(e) => setClipping({ ...clipping, showQualitySlice: e.target.checked })}
                    className="accent-blue-500 w-3 h-3"
                  />
                  View Quality Slice
                </label>
                {isSlicing && <span className="text-[9px] text-gray-500 italic block mt-1 animate-pulse">Computing section...</span>}
              </div>

              <div className="space-y-2">
                {[
                  { axis: 'x', label: 'X', color: 'red' },
                  { axis: 'y', label: 'Y', color: 'green' },
                  { axis: 'z', label: 'Z', color: 'blue' }
                ].map(({ axis, label }) => (
                  <div key={axis} className="space-y-0.5">
                    <div className="flex justify-between">
                      <label className="flex items-center gap-1">
                        <input
                          type="checkbox"
                          checked={clipping[axis]}
                          onChange={(e) => setClipping({ ...clipping, [axis]: e.target.checked })}
                          className="accent-blue-500 w-3 h-3"
                        />
                        {label}
                      </label>
                      <span className="text-gray-500">{clipping[`${axis}Value`]}%</span>
                    </div>
                    <input
                      type="range" min="-50" max="50"
                      value={clipping[`${axis}Value`]}
                      onChange={(e) => setClipping({ ...clipping, [`${axis}Value`]: parseInt(e.target.value) })}
                      disabled={!clipping[axis]}
                      className="w-full h-1 bg-gray-700 rounded appearance-none cursor-pointer accent-blue-500"
                    />
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}
