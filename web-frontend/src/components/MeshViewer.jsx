import { useEffect, useRef, useState, useMemo } from 'react'
import { Canvas, useThree } from '@react-three/fiber'
import { OrbitControls, PerspectiveCamera } from '@react-three/drei'
import * as THREE from 'three'
import { Box, Scissors, Palette, Layers, Eye, EyeOff } from 'lucide-react'

function MeshObject({ meshData, clipping, showQuality, activeMetric, qualityRange }) {
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

    // Colors
    // If we have specific metric data, we might want to re-calculate colors on the client side
    // But for performance with large meshes, we should use the pre-calculated ones or attributes.
    // However, since we want dynamic switching, we should re-compute colors here if activeMetric changes.
    // BUT modifying geometry attributes triggers re-upload to GPU. 

    // Let's use the colors passed from backend as default, and override if we have metrics.
    let colors = null;

    if (showQuality && meshData.metrics && meshData.metrics[activeMetric]) {
      const metricValues = meshData.metrics[activeMetric];
      const newColors = [];

      // Determine normalization range (0-1 usually, but some like Aspect Ratio can be large)
      let minVal = 0;
      let maxVal = 1;

      if (activeMetric === 'aspect_ratio') {
        maxVal = 20; // Cap at 20 for coloring
      } else if (activeMetric === 'skewness') {
        // 0 is good, 1 is bad usually? 
        // Skewness: 0=Equilateral, 1=Degenerate.
      }

      // Helper for HSL to RGB or simple generic gradient
      const getColor = (val) => {
        // Normalize
        let n = (val - minVal) / (maxVal - minVal);
        n = Math.max(0, Math.min(1, n));

        // Invert for metrics where Lower is Better (Skewness)
        if (activeMetric === 'skewness' || activeMetric === 'aspect_ratio') {
          n = 1.0 - n;
        }

        // Gradient: Red(0) -> Green(1)
        const r = 1.0 - n;
        const g = n;
        const b = 0.0;
        return [r, g, b];
      };

      for (let i = 0; i < metricValues.length; i++) {
        // Filter by range
        const val = metricValues[i];

        // Handling range filtering: We can't easily "hide" triangles in a single buffer geometry without index manipulation or discard shader.
        // For now, let's just color them gray if out of range? 
        // Or set alpha?

        let c = [0.5, 0.5, 0.5]; // Default gray

        // Check filtering (normalized or absolute?)
        // Let's assume range slider provides 0-1 values.
        // And we map metric to 0-1.

        let norm = val;
        if (activeMetric === 'aspect_ratio') norm = Math.min(val, 20) / 20;

        if (norm >= qualityRange[0] && norm <= qualityRange[1]) {
          c = getColor(val);
        } else {
          c = [0.1, 0.1, 0.1]; // Dark gray/black for filtered out
        }

        // Push 3 times for RGB
        newColors.push(c[0], c[1], c[2]);
      }
      colors = new Float32Array(newColors);
    } else if (meshData.colors && meshData.colors.length > 0) {
      colors = new Float32Array(meshData.colors)
    }

    if (colors) {
      geo.setAttribute('color', new THREE.BufferAttribute(colors, 3))
    }

    geo.computeVertexNormals()
    geo.computeBoundingBox()

    return geo
  }, [meshData, showQuality, activeMetric, qualityRange])

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
  }, [geometry, clipping])

  if (!geometry) return null

  return (
    <group>
      <mesh ref={meshRef} geometry={geometry} castShadow receiveShadow>
        <meshStandardMaterial
          vertexColors={!!geometry.getAttribute('color')}
          color={geometry.getAttribute('color') ? undefined : "#4a9eff"}
          side={THREE.DoubleSide}
          flatShading={true}
          clippingPlanes={clippingPlanes}
          clipShadows={true}
          roughness={0.5}
          metalness={0.1}
        />
      </mesh>

      {/* Wireframe overlay */}
      <mesh geometry={geometry}>
        <meshBasicMaterial
          color="#000000"
          wireframe={true}
          opacity={0.1}
          transparent={true}
          clippingPlanes={clippingPlanes}
        />
      </mesh>

      {/* Capping for clipping */}
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

function GridAdjuster({ meshData }) {
  // Determine grid position based on mesh bounding box
  const pos = useMemo(() => {
    if (!meshData || !meshData.vertices) return [0, -50, 0];

    let minY = Infinity;
    const v = meshData.vertices;
    for (let i = 1; i < v.length; i += 3) {
      if (v[i] < minY) minY = v[i];
    }

    // Add a small buffer
    return [0, minY - 1, 0];
  }, [meshData]);

  return <gridHelper position={pos} args={[200, 20, '#444444', '#222222']} />;
}

export default function MeshViewer({ meshData }) {
  const [clipping, setClipping] = useState({
    enabled: false,
    x: false,
    y: false,
    z: false,
    xValue: 0,
    yValue: 0,
    zValue: 0
  })

  const [showQuality, setShowQuality] = useState(false)
  const [showControls, setShowControls] = useState(false)

  // Advanced Visualization Params
  const [activeMetric, setActiveMetric] = useState('sicn')
  const [qualityRange, setQualityRange] = useState([0, 1]) // 0 to 1 normalized

  const hasQualityData = meshData?.metrics && Object.keys(meshData.metrics).length > 0

  return (
    <div className="w-full h-full relative bg-gray-950 group">
      {!meshData && (
        <div className="absolute inset-0 flex items-center justify-center text-gray-500">
          <div className="text-center">
            <Box className="w-16 h-16 mx-auto mb-4 opacity-20" />
            <p>No mesh loaded</p>
            <p className="text-sm">Upload a CAD file to view</p>
          </div>
        </div>
      )}

      {meshData && (
        <>
          <Canvas shadows gl={{ localClippingEnabled: true }}>
            <PerspectiveCamera makeDefault position={[100, 100, 100]} fov={50} />
            {/* DISABLED DAMPING for fixed movement */}
            <OrbitControls enableDamping={false} />

            <ambientLight intensity={0.6} />
            <directionalLight
              position={[50, 50, 25]}
              intensity={0.8}
              castShadow
              shadow-mapSize-width={2048}
              shadow-mapSize-height={2048}
            />
            <directionalLight position={[-50, -50, -25]} intensity={0.4} />

            <MeshObject
              meshData={meshData}
              clipping={clipping}
              showQuality={showQuality}
              activeMetric={activeMetric}
              qualityRange={qualityRange}
            />

            <GridAdjuster meshData={meshData} />
          </Canvas>

          {/* Floating Controls */}
          <div className="absolute top-4 right-4 flex flex-col gap-2">
            <button
              onClick={() => setShowControls(!showControls)}
              className="bg-gray-800 p-2 rounded text-white hover:bg-gray-700 transition-colors shadow-lg"
              title="View Controls"
            >
              <Layers className="w-5 h-5" />
            </button>

            {hasQualityData && (
              <button
                onClick={() => setShowQuality(!showQuality)}
                className={`p-2 rounded transition-colors shadow-lg ${showQuality ? 'bg-green-600 text-white' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'}`}
                title="Toggle Quality Colors"
              >
                <Palette className="w-5 h-5" />
              </button>
            )}
          </div>

          {/* Controls Panel */}
          {showControls && (
            <div className="absolute top-4 right-16 bg-gray-800/90 backdrop-blur p-4 rounded-lg shadow-xl w-72 border border-gray-700 max-h-[80vh] overflow-y-auto">

              {/* Quality Settings */}
              {showQuality && hasQualityData && (
                <div className="mb-6 border-b border-gray-700 pb-4">
                  <h3 className="text-sm font-bold text-white mb-3 flex items-center gap-2">
                    <Palette className="w-4 h-4" /> Quality Metric
                  </h3>
                  <div className="space-y-3">
                    <select
                      value={activeMetric}
                      onChange={(e) => setActiveMetric(e.target.value)}
                      className="w-full bg-gray-900 border border-gray-600 rounded px-2 py-1 text-xs text-white"
                    >
                      <option value="sicn">SICN (Signed Inv Cond Num)</option>
                      <option value="gamma">Gamma</option>
                      <option value="skewness">Skewness</option>
                      <option value="aspect_ratio">Aspect Ratio</option>
                    </select>

                    {/* Range Slider */}
                    <div className="space-y-1">
                      <div className="flex justify-between text-xs text-gray-400">
                        <span>Range Filter</span>
                        <span>{qualityRange[0].toFixed(2)} - {qualityRange[1].toFixed(2)}</span>
                      </div>
                      <div className="flex gap-2 items-center">
                        <input
                          type="range" min="0" max="1" step="0.05"
                          value={qualityRange[0]}
                          onChange={(e) => setQualityRange([Math.min(parseFloat(e.target.value), qualityRange[1]), qualityRange[1]])}
                          className="w-full h-1 bg-gray-600 rounded-lg appearance-none cursor-pointer"
                        />
                        <input
                          type="range" min="0" max="1" step="0.05"
                          value={qualityRange[1]}
                          onChange={(e) => setQualityRange([qualityRange[0], Math.max(parseFloat(e.target.value), qualityRange[0])])}
                          className="w-full h-1 bg-gray-600 rounded-lg appearance-none cursor-pointer"
                        />
                      </div>
                    </div>
                  </div>
                </div>
              )}

              <h3 className="text-sm font-bold text-white mb-3 flex items-center gap-2">
                <Scissors className="w-4 h-4" /> Clipping
              </h3>

              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-300">Enable Clipping</span>
                  <input
                    type="checkbox"
                    checked={clipping.enabled}
                    onChange={(e) => setClipping({ ...clipping, enabled: e.target.checked })}
                    className="accent-blue-500"
                  />
                </div>

                {clipping.enabled && (
                  <>
                    {/* X Axis */}
                    <div className="space-y-1">
                      <div className="flex justify-between text-xs">
                        <label className="flex items-center gap-2">
                          <input
                            type="checkbox"
                            checked={clipping.x}
                            onChange={(e) => setClipping({ ...clipping, x: e.target.checked })}
                          /> X-Axis
                        </label>
                        <span>{clipping.xValue}%</span>
                      </div>
                      <input
                        type="range" min="-50" max="50"
                        value={clipping.xValue}
                        onChange={(e) => setClipping({ ...clipping, xValue: parseInt(e.target.value) })}
                        disabled={!clipping.x}
                        className="w-full h-1 bg-gray-600 rounded-lg appearance-none cursor-pointer"
                      />
                    </div>

                    {/* Y Axis */}
                    <div className="space-y-1">
                      <div className="flex justify-between text-xs">
                        <label className="flex items-center gap-2">
                          <input
                            type="checkbox"
                            checked={clipping.y}
                            onChange={(e) => setClipping({ ...clipping, y: e.target.checked })}
                          /> Y-Axis
                        </label>
                        <span>{clipping.yValue}%</span>
                      </div>
                      <input
                        type="range" min="-50" max="50"
                        value={clipping.yValue}
                        onChange={(e) => setClipping({ ...clipping, yValue: parseInt(e.target.value) })}
                        disabled={!clipping.y}
                        className="w-full h-1 bg-gray-600 rounded-lg appearance-none cursor-pointer"
                      />
                    </div>

                    {/* Z Axis */}
                    <div className="space-y-1">
                      <div className="flex justify-between text-xs">
                        <label className="flex items-center gap-2">
                          <input
                            type="checkbox"
                            checked={clipping.z}
                            onChange={(e) => setClipping({ ...clipping, z: e.target.checked })}
                          /> Z-Axis
                        </label>
                        <span>{clipping.zValue}%</span>
                      </div>
                      <input
                        type="range" min="-50" max="50"
                        value={clipping.zValue}
                        onChange={(e) => setClipping({ ...clipping, zValue: parseInt(e.target.value) })}
                        disabled={!clipping.z}
                        className="w-full h-1 bg-gray-600 rounded-lg appearance-none cursor-pointer"
                      />
                    </div>
                  </>
                )}
              </div>
            </div>
          )}

          <div className="absolute bottom-4 left-4 bg-black/70 px-3 py-2 rounded text-xs pointer-events-none">
            <div>Vertices: {meshData.numVertices?.toLocaleString()}</div>
            <div>Triangles: {meshData.numTriangles?.toLocaleString()}</div>
          </div>
        </>
      )}
    </div>
  )
}
