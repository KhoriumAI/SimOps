import { useEffect, useRef, useState, useMemo } from 'react'
import { Canvas, useThree } from '@react-three/fiber'
import { OrbitControls, PerspectiveCamera } from '@react-three/drei'
import * as THREE from 'three'
import { Cube, Scissors, Palette, Layers } from 'lucide-react'

function MeshObject({ meshData, clipping, showQuality }) {
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

  if (!geometry) return null

  return (
    <group>
      <mesh ref={meshRef} geometry={geometry} castShadow receiveShadow>
        <meshStandardMaterial
          vertexColors={showQuality && meshData.colors && meshData.colors.length > 0}
          color={showQuality && meshData.colors && meshData.colors.length > 0 ? undefined : "#4a9eff"}
          side={THREE.DoubleSide}
          flatShading={true} // Use flat shading for that CAD/Mesh look
          clippingPlanes={clippingPlanes}
          clipShadows={true}
          roughness={0.5}
          metalness={0.1}
        />
      </mesh>

      {/* Wireframe overlay - faint */}
      <mesh geometry={geometry}>
        <meshBasicMaterial
          color="#000000"
          wireframe={true}
          opacity={0.1}
          transparent={true}
          clippingPlanes={clippingPlanes}
        />
      </mesh>

      {/* Cap for clipping (Simplified: just show backface with different color) */}
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

export default function MeshViewer({ meshData }) {
  const [clipping, setClipping] = useState({
    enabled: false,
    x: false,
    y: false,
    z: false,
    xValue: 0, // -50 to 50
    yValue: 0,
    zValue: 0
  })

  const [showQuality, setShowQuality] = useState(false)
  const [showControls, setShowControls] = useState(false)

  const hasQualityData = meshData?.colors && meshData.colors.length > 0

  return (
    <div className="w-full h-full relative bg-gray-950 group">
      {!meshData && (
        <div className="absolute inset-0 flex items-center justify-center text-gray-500">
          <div className="text-center">
            <Cube className="w-16 h-16 mx-auto mb-4 opacity-20" />
            <p>No mesh loaded</p>
            <p className="text-sm">Upload a CAD file and generate a mesh to view</p>
          </div>
        </div>
      )}

      {meshData && (
        <>
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

            <MeshObject
              meshData={meshData}
              clipping={clipping}
              showQuality={showQuality}
            />

            <gridHelper args={[200, 20, '#444444', '#222222']} />
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
            <div className="absolute top-4 right-16 bg-gray-800/90 backdrop-blur p-4 rounded-lg shadow-xl w-64 border border-gray-700">
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
