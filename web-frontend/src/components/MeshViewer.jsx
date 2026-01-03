import { useEffect, useRef, useState, useMemo, useCallback } from 'react'
import { Canvas, useThree, useFrame } from '@react-three/fiber'
import { OrbitControls, PerspectiveCamera, GizmoHelper, GizmoViewport, Edges } from '@react-three/drei'
import * as THREE from 'three'
import { Box, Loader2, MousePointer2, Tag, X, BarChart3, Scissors, Save, Lightbulb, Hexagon } from 'lucide-react'
import { API_BASE } from '../config'
import QualityHistogram from './QualityHistogram'

/**
 * Compute smooth normals with angle-threshold-based splitting.
 * Adjacent faces share normals only if their angle difference < threshold.
 * This preserves chamfers/sharp edges while smoothing coarse tessellations.
 * 
 * @param {Float32Array} vertices - Flat vertex array (non-indexed, 9 floats per tri)
 * @param {number} angleThresholdDegrees - Faces with angle > this get split normals (default 30°)
 * @returns {Float32Array} - Smooth normals array matching vertex positions
 */
function computeSmoothNormalsWithThreshold(vertices, angleThresholdDegrees = 30) {
  const numTriangles = vertices.length / 9
  if (numTriangles === 0) return new Float32Array(0)

  const threshold = Math.cos((angleThresholdDegrees * Math.PI) / 180)

  // Step 1: Compute face normals
  const faceNormals = new Float32Array(numTriangles * 3)
  const v1 = new THREE.Vector3(), v2 = new THREE.Vector3(), v3 = new THREE.Vector3()
  const e1 = new THREE.Vector3(), e2 = new THREE.Vector3(), fn = new THREE.Vector3()

  for (let i = 0; i < numTriangles; i++) {
    const idx = i * 9
    v1.set(vertices[idx], vertices[idx + 1], vertices[idx + 2])
    v2.set(vertices[idx + 3], vertices[idx + 4], vertices[idx + 5])
    v3.set(vertices[idx + 6], vertices[idx + 7], vertices[idx + 8])
    e1.subVectors(v2, v1)
    e2.subVectors(v3, v1)
    fn.crossVectors(e1, e2).normalize()
    faceNormals[i * 3] = fn.x
    faceNormals[i * 3 + 1] = fn.y
    faceNormals[i * 3 + 2] = fn.z
  }

  // Step 2: Build vertex -> face adjacency map
  const vertexToFaces = new Map()
  const precision = 5

  for (let faceIdx = 0; faceIdx < numTriangles; faceIdx++) {
    for (let v = 0; v < 3; v++) {
      const idx = faceIdx * 9 + v * 3
      const key = `${vertices[idx].toFixed(precision)},${vertices[idx + 1].toFixed(precision)},${vertices[idx + 2].toFixed(precision)}`
      if (!vertexToFaces.has(key)) vertexToFaces.set(key, [])
      vertexToFaces.get(key).push({ faceIdx, vertexIdx: v })
    }
  }

  // Step 3: Compute smoothed normals with angle-based grouping
  const smoothNormals = new Float32Array(vertices.length)
  const tempNormal = new THREE.Vector3()
  const baseFaceNormal = new THREE.Vector3()
  const adjacentFaceNormal = new THREE.Vector3()

  for (let faceIdx = 0; faceIdx < numTriangles; faceIdx++) {
    baseFaceNormal.set(
      faceNormals[faceIdx * 3],
      faceNormals[faceIdx * 3 + 1],
      faceNormals[faceIdx * 3 + 2]
    )

    for (let v = 0; v < 3; v++) {
      const idx = faceIdx * 9 + v * 3
      const key = `${vertices[idx].toFixed(precision)},${vertices[idx + 1].toFixed(precision)},${vertices[idx + 2].toFixed(precision)}`

      tempNormal.set(0, 0, 0)
      const adjacentFaces = vertexToFaces.get(key) || []

      for (const { faceIdx: adjFaceIdx } of adjacentFaces) {
        adjacentFaceNormal.set(
          faceNormals[adjFaceIdx * 3],
          faceNormals[adjFaceIdx * 3 + 1],
          faceNormals[adjFaceIdx * 3 + 2]
        )

        // Only include if angle is below threshold
        if (baseFaceNormal.dot(adjacentFaceNormal) >= threshold) {
          tempNormal.add(adjacentFaceNormal)
        }
      }

      // Fallback to face normal if no smooth neighbors
      if (tempNormal.lengthSq() < 0.001) {
        tempNormal.copy(baseFaceNormal)
      }

      tempNormal.normalize()
      smoothNormals[idx] = tempNormal.x
      smoothNormals[idx + 1] = tempNormal.y
      smoothNormals[idx + 2] = tempNormal.z
    }
  }

  return smoothNormals
}

function SliceMesh({ sliceData, clippingPlanes, renderOffset, activeAxis }) {
  // Debug log to verify data reception - ensure this runs
  useEffect(() => {
    if (sliceData && sliceData.vertices) {
      console.log(`[SliceMesh] Data Loaded: ${sliceData.vertices.length / 3} vertices. Displaying...`);
    } else if (sliceData) {
      console.log("[SliceMesh] sliceData present but empty/invalid:", sliceData);
    }
  }, [sliceData]);

  const geometry = useMemo(() => {
    if (!sliceData || !sliceData.vertices || sliceData.vertices.length === 0) return null
    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.Float32BufferAttribute(sliceData.vertices, 3))
    geo.setAttribute('color', new THREE.Float32BufferAttribute(sliceData.colors, 3))
    if (sliceData.indices && sliceData.indices.length > 0) {
      geo.setIndex(sliceData.indices)
    }
    geo.computeVertexNormals()

    // Apply centering offset to align with main mesh
    if (renderOffset) {
      geo.translate(renderOffset.x, renderOffset.y, renderOffset.z)
    }

    return geo
  }, [sliceData, renderOffset])

  // Filter out the plane that corresponds to the activeAxis
  const filteredPlanes = useMemo(() => {
    if (!clippingPlanes || !activeAxis) return clippingPlanes

    return clippingPlanes.filter(plane => {
      if (activeAxis === 'x' && Math.abs(plane.normal.x + 1) < 0.01) return false
      if (activeAxis === 'y' && Math.abs(plane.normal.z + 1) < 0.01) return false
      if (activeAxis === 'z' && Math.abs(plane.normal.y + 1) < 0.01) return false
      return true
    })
  }, [clippingPlanes, activeAxis])

  if (!geometry) return null

  // Use MeshBasicMaterial to ignore lighting issues. 
  // Disable depth testing to force it to show up (might look weird but verifies existence).
  return (
    <group>
      {/* Main Slice Surface */}
      <mesh geometry={geometry}>
        <meshBasicMaterial
          vertexColors={true}
          side={THREE.DoubleSide}
          transparent={false}
          opacity={1.0}
          clippingPlanes={filteredPlanes}
          depthTest={true} // Keep depth test for correct volume sorting
        />
      </mesh>

      {/* Debug Wireframe to make it super obvious */}
      {/* 
      <mesh geometry={geometry}>
        <meshBasicMaterial 
            color="red" 
            wireframe={true} 
            side={THREE.DoubleSide} 
            clippingPlanes={filteredPlanes}
        />
      </mesh>
      */}
    </group>
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

function Lights({ dynamic }) {
  const { camera } = useThree()
  const lightRef = useRef()

  useFrame(() => {
    if (dynamic && lightRef.current) {
      lightRef.current.position.copy(camera.position)
    }
  })

  if (dynamic) {
    return (
      <>
        <ambientLight intensity={0.4} />
        {/* Headlight attached to camera - updates every frame via useFrame */}
        <directionalLight
          ref={lightRef}
          position={[camera.position.x, camera.position.y, camera.position.z]}
          intensity={1.2}
          castShadow={false}
        />
        {/* Fill light */}
        <directionalLight position={[-50, -50, -25]} intensity={0.2} />
      </>
    )
  }

  return (
    <>
      <ambientLight intensity={0.7} />
      <directionalLight position={[50, 50, 25]} intensity={0.6} />
      <directionalLight position={[-50, -50, -25]} intensity={0.3} />
      <hemisphereLight intensity={0.3} />
    </>
  )
}

function MeshObject({ meshData, sliceData, clipping, showQuality, showWireframe, wireframeColor, wireframeOpacity, wireframeScale, meshColor, opacity, metalness, roughness, colorMode, gradientColors, onFaceSelect, selectionMode, selectedFaces, boundaryZones, showEdges, edgeThreshold }) {
  const meshRef = useRef()
  const { camera, gl, raycaster, pointer } = useThree()

  // Enable local clipping
  useEffect(() => {
    gl.localClippingEnabled = true
  }, [gl])

  // Create geometry from flat arrays and calculate centering offset
  const { geometry, renderOffset } = useMemo(() => {
    if (!meshData || !meshData.vertices) return { geometry: null, renderOffset: null }

    const geo = new THREE.BufferGeometry()

    // Vertices (Float32Array)
    const vertices = new Float32Array(meshData.vertices)
    geo.setAttribute('position', new THREE.BufferAttribute(vertices, 3))

    // Colors (Float32Array) - if available (for quality mode)
    const activeColors = (meshData.qualityColors && showQuality) ?
      meshData.qualityColors[showQuality === true ? 'sicn' : showQuality] :
      meshData.colors;

    if (activeColors && activeColors.length > 0) {
      const colors = new Float32Array(activeColors)
      geo.setAttribute('color', new THREE.BufferAttribute(colors, 3))
    }

    // Use smooth normals for preview meshes (preserves chamfers, smooths coarse tessellation)
    // Use standard vertex normals for completed meshes (shows mesh structure)
    if (meshData.isPreview) {
      const smoothNormals = computeSmoothNormalsWithThreshold(meshData.vertices, 30)
      geo.setAttribute('normal', new THREE.BufferAttribute(smoothNormals, 3))
    } else {
      geo.computeVertexNormals()
    }
    geo.computeBoundingBox()

    // Position geometry so it sits ON the XY plane at z=0 
    // and is centered on X and Y
    const offset = new THREE.Vector3();
    geo.boundingBox.getCenter(offset).negate();
    offset.z = -geo.boundingBox.min.z;
    geo.translate(offset.x, offset.y, offset.z);
    geo.computeBoundingBox();

    return { geometry: geo, renderOffset: offset }
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
      // X plane: Normal (-1, 0, 0) to cut from right (max -> min)
      // World X matches Original X
      const xPos = (center.x + size.x / 2) - (clipping.xValue / 100) * size.x
      planes.push(new THREE.Plane(new THREE.Vector3(-1, 0, 0), xPos))
    }
    if (clipping.y) {
      // Original Y is World Z in the view (due to -PI/2 X rotation)
      // Normal (0, 0, -1) clips against Original Y
      const yPos = (center.z + size.z / 2) - (clipping.yValue / 100) * size.z
      planes.push(new THREE.Plane(new THREE.Vector3(0, 0, -1), yPos))
    }
    if (clipping.z) {
      // Original Z is World Y in the view (due to -PI/2 X rotation)
      // Normal (0, -1, 0) clips against Original Z
      const zPos = (center.y + size.y / 2) - (clipping.zValue / 100) * size.y
      planes.push(new THREE.Plane(new THREE.Vector3(0, -1, 0), zPos))
    }

    return planes
  }, [geometry, clipping])

  // Track mousedown time to distinguish clicks from drags
  const mouseDownTime = useRef(0)

  const handlePointerDown = useCallback(() => {
    mouseDownTime.current = Date.now()
  }, [])

  // Handle face click for selection - now works without selectionMode
  const handleClick = useCallback((event) => {
    if (!meshRef.current || !onFaceSelect) return

    // Ignore if this was a drag operation (click held for >200ms)
    const clickDuration = Date.now() - mouseDownTime.current
    if (clickDuration > 200) return

    event.stopPropagation()

    const intersects = raycaster.intersectObject(meshRef.current)
    if (intersects.length > 0) {
      const faceIndex = intersects[0].faceIndex
      const point = intersects[0].point
      const normal = intersects[0].face?.normal

      // Default to flood fill (logical face selection)
      const isFloodFill = !event.altKey
      onFaceSelect({
        faceIndex,
        point: { x: point.x, y: point.y, z: point.z },
        normal: normal ? { x: normal.x, y: normal.y, z: normal.z } : null,
      }, isFloodFill)
    }
  }, [onFaceSelect, raycaster])

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

  // Create geometries for saved boundary zones
  const zoneGeometries = useMemo(() => {
    if (!boundaryZones || Object.keys(boundaryZones).length === 0 || !geometry) return []

    const geometries = []
    const positions = geometry.attributes.position.array

    Object.entries(boundaryZones).forEach(([name, indices]) => {
      const highlightPositions = []
      // indices is an array of face indices
      indices.forEach(faceIndex => {
        const idx = faceIndex * 9
        if (idx + 8 < positions.length) {
          for (let i = 0; i < 9; i++) highlightPositions.push(positions[idx + i])
        }
      })

      if (highlightPositions.length > 0) {
        const geo = new THREE.BufferGeometry()
        geo.setAttribute('position', new THREE.Float32BufferAttribute(highlightPositions, 3))
        geo.computeVertexNormals()
        geometries.push({ name, geometry: geo })
      }
    })

    return geometries
  }, [boundaryZones, geometry])

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
        onPointerDown={handlePointerDown}
      >
        <meshStandardMaterial
          vertexColors={useVertexColors || useGradient}
          color={useVertexColors || useGradient ? undefined : meshColor}
          side={THREE.DoubleSide}
          flatShading={!meshData?.isPreview}
          clippingPlanes={clippingPlanes}
          clipShadows={true}
          roughness={roughness}
          metalness={metalness}
          opacity={opacity}
          transparent={opacity < 1 || showQuality}
        />
      </mesh>

      {/* Render persistent boundary zones */}
      {zoneGeometries.map(zone => (
        <mesh key={zone.name} geometry={zone.geometry}>
          <meshBasicMaterial
            color="#3b82f6" // Blue-500
            side={THREE.DoubleSide}
            transparent
            opacity={0.5}
            depthTest={false} // Overlay behavior
          />
        </mesh>
      ))}

      {/* Highlight selected faces (on top of zones) */}
      {selectedFaceGeometry && (
        <mesh geometry={selectedFaceGeometry}>
          <meshBasicMaterial
            color="#22c55e" // Green-500
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
        <SliceMesh
          sliceData={sliceData}
          clippingPlanes={clippingPlanes}
          renderOffset={renderOffset}
          activeAxis={clipping.x ? 'x' : clipping.y ? 'y' : clipping.z ? 'z' : null}
        />
      )}

      {/* Sharp Edges Overlay */}
      {showEdges && (
        <Edges
          geometry={activeGeometry}
          threshold={edgeThreshold}
          color="#1a1a1a"
          opacity={0.8}
          transparent
        />
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
  showAxes,
  setShowAxes,
  qualityMetric,
  setQualityMetric,
  colorMode,
  setColorMode,
  // Mesh progress from App
  meshProgress,
  loadingStartTime
}) {
  // Derive wireframe visibility: ON for completed meshes, OFF for CAD preview
  const showWireframe = meshData && !meshData.isPreview && status === 'completed'
  // Derive quality coloring: ON for completed meshes with quality data
  const hasQualityData = (meshData?.colors && meshData.colors.length > 0) || (meshData?.qualityColors && Object.keys(meshData.qualityColors).length > 0) || meshData?.hasQualityData
  const showQuality = status === 'completed' && hasQualityData

  // Clear slice data when project changes to prevent ghosting
  useEffect(() => {
    setSliceData(null)
  }, [projectId])

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
  // colorMode lifted to App.jsx
  const [opacity, setOpacity] = useState(1.0)
  const [metalness, setMetalness] = useState(0.1)
  const [roughness, setRoughness] = useState(0.5)
  const [gradientColors, setGradientColors] = useState({ start: '#4a9eff', end: '#ff6b6b' })

  // Lighting & Visibility
  const [dynamicLighting, setDynamicLighting] = useState(true)
  const [showEdges, setShowEdges] = useState(false) // Default off until user enables
  const [edgeThreshold, setEdgeThreshold] = useState(15)

  // Face selection state (selectionMode removed - always enabled)
  const selectionMode = true
  const [selectedFaces, setSelectedFaces] = useState([])
  const [boundaryZones, setBoundaryZones] = useState({}) // { name: [indices] }
  const [showFacePanel, setShowFacePanel] = useState(false)
  const [pendingFaceName, setPendingFaceName] = useState('')
  const [isSavingZones, setIsSavingZones] = useState(false)


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
    if (!adjacency || !meshData) return [startFaceIndex]

    const visited = new Set()
    const queue = [startFaceIndex]
    visited.add(startFaceIndex)

    // Check for entity tags (CAD faces)
    const hasTags = meshData.entity_tags && meshData.entity_tags.length > 0
    const targetEntity = hasTags ? meshData.entity_tags[startFaceIndex] : null

    // For geometric flood fill (if no tags available for this face)
    // Reuse vectors to avoid GC pressure in loop
    const v1 = new THREE.Vector3(), v2 = new THREE.Vector3(), v3 = new THREE.Vector3()
    const e1 = new THREE.Vector3(), e2 = new THREE.Vector3()
    const n1 = new THREE.Vector3(), n2 = new THREE.Vector3()
    const CREASE_THRESHOLD = 0.8 // Stop at angles > ~37 deg

    const vertices = meshData.vertices
    const getNormal = (idx, target) => {
      const i = idx * 9
      v1.set(vertices[i], vertices[i + 1], vertices[i + 2])
      v2.set(vertices[i + 3], vertices[i + 4], vertices[i + 5])
      v3.set(vertices[i + 6], vertices[i + 7], vertices[i + 8])
      e1.subVectors(v2, v1)
      e2.subVectors(v3, v1)
      target.crossVectors(e1, e2).normalize()
    }

    let iterations = 0
    while (queue.length > 0 && iterations < 50000) { // Safety limit
      const current = queue.shift()
      iterations++

      // If doing geometric fill, compute current normal
      if (!hasTags || targetEntity == null) {
        getNormal(current, n1)
      }

      const neighbors = adjacency[current]
      for (const next of neighbors) {
        if (!visited.has(next)) {
          let shouldAdd = false

          if (hasTags && targetEntity != null) {
            // Use Entity Tags if available
            shouldAdd = (meshData.entity_tags[next] === targetEntity)
          } else {
            // Use Geometric Crease Angle
            getNormal(next, n2)
            if (n1.dot(n2) >= CREASE_THRESHOLD) {
              shouldAdd = true
            }
          }

          if (shouldAdd) {
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
  }, [performFloodFill])

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
                  // Opening panel - auto-enable with X axis at 50% (midplane)
                  setShowControls(true)
                  setClipping({ enabled: true, showQualitySlice: true, x: true, y: false, z: false, xValue: 50, yValue: 50, zValue: 50 })
                } else {
                  // Closing panel - disable clipping
                  setShowControls(false)
                  setClipping({ enabled: false, showQualitySlice: true, x: false, y: false, z: false, xValue: 50, yValue: 50, zValue: 50 })
                }
              }}
              className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all shadow-lg flex items-center gap-1.5 ${showControls ? 'bg-blue-600 text-white' : 'bg-gray-800/90 text-gray-300 hover:bg-gray-700'}`}
            >
              <Scissors className="w-3.5 h-3.5" />
              Section View
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

              <div className="mt-3 border-t border-gray-700 pt-3">
                <div className="font-medium text-white mb-2 text-sm">💡 Lighting & Visibility</div>

                {/* Dynamic Lighting Toggle */}
                <div className="mb-2">
                  <label className="flex items-center gap-2 cursor-pointer text-gray-300">
                    <input
                      type="checkbox"
                      checked={dynamicLighting}
                      onChange={(e) => setDynamicLighting(e.target.checked)}
                      className="accent-blue-500 w-3 h-3"
                    />
                    <div className="flex items-center gap-1.5">
                      <Lightbulb className={`w-3 h-3 ${dynamicLighting ? 'text-yellow-400 fill-yellow-400' : 'text-gray-500'}`} />
                      Dynamic Lighting (Headlight)
                    </div>
                  </label>
                  <p className="text-[9px] text-gray-500 ml-5 mt-0.5">Light follows camera to see all details</p>
                </div>

                {/* Sharp Edges Toggle */}
                <div className="mb-2">
                  <label className="flex items-center gap-2 cursor-pointer text-gray-300">
                    <input
                      type="checkbox"
                      checked={showEdges}
                      onChange={(e) => setShowEdges(e.target.checked)}
                      className="accent-blue-500 w-3 h-3"
                    />
                    <div className="flex items-center gap-1.5">
                      <Hexagon className={`w-3 h-3 ${showEdges ? 'text-blue-400' : 'text-gray-500'}`} />
                      Show Sharp Edges
                    </div>
                  </label>
                </div>

                {/* Edge Threshold Slider */}
                {showEdges && (
                  <div className="ml-5 mb-2">
                    <div className="flex justify-between text-[10px] mb-1">
                      <label className="text-gray-400">Angle Threshold</label>
                      <span>{edgeThreshold}°</span>
                    </div>
                    <input
                      type="range"
                      min="1" max="90" step="1"
                      value={edgeThreshold}
                      onChange={(e) => setEdgeThreshold(parseInt(e.target.value))}
                      className="w-full h-1 bg-gray-700 rounded appearance-none cursor-pointer accent-blue-500"
                    />
                  </div>
                )}
              </div>
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

          {/* View Options & Quality Metrics - Top Right */}
          {(!showPaintPanel && !showControls) && (
            <div className="absolute top-10 right-3 flex flex-col gap-2 z-10 min-w-[180px]">

              {/* Main Info Panel */}
              <div className="bg-gray-900/90 backdrop-blur rounded p-2.5 text-[10px] text-gray-300 shadow-xl border border-gray-700">

                {/* Header / Visualization Toggles */}
                <div className="mb-2 pb-2 border-b border-gray-700 space-y-2">
                  <div className="font-medium text-white text-xs flex items-center gap-2">
                    <span>View Options</span>
                  </div>

                  <label className="flex items-center gap-2 cursor-pointer hover:text-white transition-colors">
                    <input
                      type="checkbox"
                      checked={showAxes}
                      onChange={(e) => setShowAxes(e.target.checked)}
                      className="accent-blue-500 w-3 h-3 cursor-pointer"
                    />
                    Show Axes
                  </label>

                  {/* Only show Histogram toggle if metrics are available (Mesh Completed) */}
                  {isCompleted && qualityMetrics && (
                    <label className="flex items-center gap-2 cursor-pointer hover:text-white transition-colors">
                      <input
                        type="checkbox"
                        checked={showHistogram}
                        onChange={(e) => setShowHistogram(e.target.checked)}
                        className="accent-blue-500 w-3 h-3 cursor-pointer"
                      />
                      Show Quality Histogram
                    </label>
                  )}
                </div>

                {/* Quality Metrics Content (Only when mesh is completed) */}
                {isCompleted && qualityMetrics ? (
                  <>
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
                  </>
                ) : (
                  /* Placeholder or info when no metrics available (e.g. CAD preview or processing) */
                  <div className="text-[9px] text-gray-500 italic text-center py-1">
                    {status === 'processing' ? 'Generating mesh...' : 'No mesh metrics available'}
                  </div>
                )}
              </div>
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

          {/* Named Faces List (Boundary Zones) */}
          {Object.keys(boundaryZones).length > 0 && !showFacePanel && (
            <div className="absolute bottom-3 right-3 bg-gray-900/95 backdrop-blur rounded-lg p-2 z-10 text-xs text-gray-300 max-w-48 shadow-xl border border-gray-700">
              <div className="flex items-center gap-1.5 mb-1.5 text-[10px] text-gray-400">
                <Tag className="w-3 h-3" />
                <span>Boundary Zones ({Object.keys(boundaryZones).length})</span>
              </div>
              <div className="space-y-1 max-h-32 overflow-y-auto">
                {Object.entries(boundaryZones).map(([name, indices]) => (
                  <div
                    key={name}
                    className="flex items-center justify-between bg-gray-800 rounded px-2 py-1 cursor-pointer hover:bg-gray-700 transition-colors group"
                    onClick={() => {
                      // Recall selection
                      const newSelection = indices.map(idx => ({ faceIndex: idx }))
                      setSelectedFaces(newSelection)
                      setPendingFaceName(name)
                      setShowFacePanel(true)
                    }}
                  >
                    <div className="flex flex-col overflow-hidden max-w-[100px]">
                      <span className="text-white truncate" title={name}>{name}</span>
                      <span className="text-[9px] text-gray-500">{indices.length} faces</span>
                    </div>
                    <button
                      onClick={(e) => {
                        e.stopPropagation() // Prevent selecting when deleting
                        if (!window.confirm(`Delete zone "${name}"?`)) return

                        const newZones = { ...boundaryZones }
                        delete newZones[name]
                        setBoundaryZones(newZones)

                        // Also clear from active selection if currently viewing it
                        if (pendingFaceName === name) {
                          setPendingFaceName('')
                          setShowFacePanel(false)
                          setSelectedFaces([])
                        }

                        // ... fetch call
                        const deleteZone = async () => {
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
                            console.error("Failed to delete zone:", err)
                          }
                        }
                        deleteZone()
                      }}
                      className="text-gray-500 hover:text-red-400 ml-2 flex-shrink-0 opacity-0 group-hover:opacity-100 transition-opacity"
                    >
                      <X className="w-3 h-3" />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Selection Mode Indicator */}


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
              makeDefault
            />

            <Lights dynamic={dynamicLighting} />

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
              boundaryZones={boundaryZones}
              showEdges={showEdges}
              edgeThreshold={edgeThreshold}
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
                      type="range" min="0" max="100"
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
