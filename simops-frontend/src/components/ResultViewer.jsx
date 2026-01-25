import React, { useRef, useMemo, useState, useEffect, Suspense } from 'react'
import { Canvas, useThree, useFrame, useLoader } from '@react-three/fiber'
import { OrbitControls, PerspectiveCamera, GizmoHelper, GizmoViewport, Environment, Center, Html } from '@react-three/drei'
import { EffectComposer, SSAO, Bloom } from '@react-three/postprocessing'
import * as THREE from 'three'
import { VTKLoader, STLLoader } from 'three-stdlib'
import { Box, Loader2, MousePointer2, Tag, X, BarChart3, Settings2, Maximize, Minimize, Ruler, AlertTriangle } from 'lucide-react'

class ErrorBoundary extends React.Component {
    constructor(props) {
        super(props)
        this.state = { hasError: false, error: null }
        this.handleCopy = this.handleCopy.bind(this)
    }
    static getDerivedStateFromError(error) {
        return { hasError: true, error }
    }
    handleCopy() {
        const message = this.state.error?.message || 'Rendering Error'
        if (navigator?.clipboard?.writeText) {
            navigator.clipboard.writeText(message)
        }
    }
    render() {
        if (this.state.hasError) {
            // Check for specific error to supress
            if (this.state.error?.message?.includes && this.state.error.message.includes("Unsupported DATASET type") && this.props.fallback) {
                return this.props.fallback
            }

            return (
                <Html center>
                    <div className="flex flex-col items-center justify-center h-full text-red-500 p-4 bg-black/90 w-96 rounded border border-red-500/50">
                        <AlertTriangle className="w-8 h-8 mb-2" />
                        <p className="font-bold">Rendering Error</p>
                        <p className="text-xs font-mono break-all">{this.state.error?.message}</p>
                        <button
                            type="button"
                            onClick={this.handleCopy}
                            className="mt-3 px-2 py-1 text-[10px] font-mono text-red-200 border border-red-500/40 rounded hover:bg-red-500/10"
                        >
                            Copy Error
                        </button>
                    </div>
                </Html>
            )
        }
        return this.props.children
    }
}

/**
 * ResultViewer (Engineering Edition)
 * Features:
 * - CameraControls (Arcball/Map style interaction)
 * - MatCap rendering for surface fidelity
 * - Exact Scale Bar
 * - Gradient Background via CSS
 */

function EngineeringMesh({ url, wireframe, opacity, useMatcap }) {
    // Determine loader based on extension (simple heuristic)
    const isStl = url?.toLowerCase().endsWith('.stl')
    const isVtk = url?.toLowerCase().endsWith('.vtk') || url?.toLowerCase().endsWith('.vtu')
    const isMsh = url?.toLowerCase().endsWith('.msh')

    // Safety Check: If not a supported mesh format, don't attempt to load
    if (!isStl && !isVtk) {
        if (isMsh) {
            console.warn("EngineeringMesh: Raw .msh files cannot be loaded directly. Expected a converted .vtk preview.")
        } else {
            console.warn("EngineeringMesh: Unsupported format for direct loading:", url)
        }
        return null
    }

    // Determine loader: .stl uses STLLoader, .vtk/.vtu uses VTKLoader
    const Loader = isStl ? STLLoader : VTKLoader

    const geometry = useLoader(Loader, url)

    // Colormap state
    const [stats, setStats] = useState({ min: 0, max: 0, hasTemp: false })

    useMemo(() => {
        if (geometry) {
            geometry.computeBoundingBox()
            geometry.center()
            geometry.computeVertexNormals()

            // Thermal Color Mapping Logic
            // 1. Find temperature attribute
            let tempAttr = geometry.attributes.T || geometry.attributes.temperature || geometry.attributes.Temperature

            // If not found in standard attributes, check userData (common in some loaders)
            if (!tempAttr && geometry.userData?.pointData) {
                // Try to find it in pointData
                // Note: VTKLoader might put data in geometry.attributes directly if it's a BufferGeometry
            }

            if (tempAttr) {
                const count = tempAttr.count
                let min = Infinity, max = -Infinity

                // 2. Compute Range
                for (let i = 0; i < count; i++) {
                    const val = tempAttr.getX(i)
                    if (val < min) min = val
                    if (val > max) max = val
                }

                // 3. Generate Colors
                // Simple Blue-Red Colormap (Jet-like)
                const colors = []
                const color = new THREE.Color()

                for (let i = 0; i < count; i++) {
                    const val = tempAttr.getX(i)
                    const t = (val - min) / (max - min || 1)

                    // Lerp from Blue (0,0,1) to Red (1,0,0) via Green/Yellow?
                    // Simple 3-stop: Blue -> Green -> Red
                    // Or standard HSV sweep. Let's do a simple HSL sweep: 240 (blue) -> 0 (red)
                    // H: 0.66 -> 0.0
                    color.setHSL(0.66 * (1.0 - t), 1.0, 0.5)

                    colors.push(color.r, color.g, color.b)
                }

                geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3))
                setStats({ min, max, hasTemp: true })
            }
        }
    }, [geometry])

    return (
        <mesh castShadow receiveShadow>
            <primitive object={geometry} />
            <meshStandardMaterial
                vertexColors={stats.hasTemp}
                color={stats.hasTemp ? undefined : "#e5e7eb"}
                roughness={0.4}
                metalness={0.1}
                side={THREE.DoubleSide}
            />

            {/* Wireframe Overlay */}
            {wireframe && (
                <mesh>
                    <primitive object={geometry} />
                    <meshBasicMaterial color="#00ffff" wireframe transparent opacity={0.2} />
                </mesh>
            )}
        </mesh>
    )
}

function GridBackground() {
    return (
        <gridHelper args={[100, 100, '#333333', '#1a1a1a']} position={[0, -5, 0]} />
    )
}

function ScaleBar() {
    return (
        <div className="absolute bottom-4 right-16 flex flex-col items-end pointer-events-none z-10">
            <div className="border-b-2 border-r-2 border-white/50 w-24 h-2 mb-1" />
            <span className="text-[10px] font-mono text-white/70">100 mm</span>
        </div>
    )
}

function ContextMenu({ x, y, onClose }) {
    return (
        <div
            style={{ top: y, left: x }}
            className="fixed bg-popover border border-border rounded shadow-xl py-1 z-50 min-w-[120px]"
        >
            <button className="w-full text-left px-3 py-1 text-xs hover:bg-muted text-foreground" onClick={onClose}>Inspect Point</button>
            <button className="w-full text-left px-3 py-1 text-xs hover:bg-muted text-foreground" onClick={onClose}>Set Center</button>
            <div className="h-px bg-border my-1" />
            <button className="w-full text-left px-3 py-1 text-xs hover:bg-muted text-destructive" onClick={onClose}>Hide Part</button>
        </div>
    )
}

export default function ResultViewer({ meshData, filename, isLoading, loadingMessage, meshUrl, previewUrl, consoleHeight = 150, consoleOpen = true }) {
    const [wireframe, setWireframe] = useState(false)
    const [opacity, setOpacity] = useState(1.0)
    const [useMatcap, setUseMatcap] = useState(true)
    const [menuPos, setMenuPos] = useState(null) // {x, y}
    const consoleOffset = consoleOpen ? consoleHeight : 32

    // const cameraControlsRef = useRef()

    const handleContextMenu = (e) => {
        e.preventDefault()
        setMenuPos({ x: e.clientX, y: e.clientY })
    }

    useEffect(() => {
        const closeMenu = () => setMenuPos(null)
        window.addEventListener('click', closeMenu)
        return () => window.removeEventListener('click', closeMenu)
    }, [])

    return (
        <div
            className="w-full h-full relative overflow-hidden bg-gradient-to-br from-[#2b2b2b] to-[#1a1a1a]"
            onContextMenu={handleContextMenu}
        >
            {/* Corner Loading Indicator - No Overlay */}
            {isLoading && (
                <div className="absolute top-4 right-4 z-50 flex items-center gap-2 bg-card/95 backdrop-blur border border-primary/30 rounded-lg px-3 py-2 shadow-xl">
                    <Loader2 className="w-4 h-4 animate-spin text-primary" />
                    <span className="text-xs font-mono text-foreground">
                        {loadingMessage || 'PROCESSING...'}
                    </span>
                </div>
            )}

            {/* Canvas */}
            <Canvas shadows dpr={[1, 2]} gl={{ preserveDrawingBuffer: true }}>
                <PerspectiveCamera makeDefault position={[50, 50, 50]} fov={45} up={[0, 0, 1]} />

                {/* Engineering Controls: Zero Damping for "Mechanical" feel, Z-up orientation */}
                <OrbitControls
                    makeDefault
                    enableDamping={false}
                    rotateSpeed={0.8}
                    zoomSpeed={1.2}
                    up={[0, 0, 1]}
                />

                {/* Lighting Environment */}
                <Environment preset="city" />
                <ambientLight intensity={0.5} />
                <directionalLight position={[10, 10, 5]} intensity={1} castShadow />

                <Center>
                    <ErrorBoundary fallback={
                        previewUrl ? (
                            <EngineeringMesh
                                url={previewUrl}
                                wireframe={wireframe}
                                opacity={opacity}
                                useMatcap={useMatcap}
                            />
                        ) : null
                    } key={meshUrl}>
                        <Suspense fallback={null}>
                            {meshUrl || previewUrl ? (
                                <EngineeringMesh
                                    url={meshUrl || previewUrl}
                                    wireframe={wireframe}
                                    opacity={opacity}
                                    useMatcap={useMatcap}
                                />
                            ) : (
                                null
                            )}
                        </Suspense>
                    </ErrorBoundary>
                </Center>

                {/* Visual Helpers - Grid removed per user request */}
                {/* <GridBackground /> */}
                <GizmoHelper alignment="bottom-right" margin={[80, 200]}>
                    <GizmoViewport
                        axisColors={['#ef4444', '#22c55e', '#3b82f6']}
                        labelColor="white"
                        axisHeadScale={0.75}
                        labels={['X', 'Y', 'Z']}
                    />
                </GizmoHelper>

                {/* Post Processing for "Polish" */}
                {/* <EffectComposer disableNormalPass={false}>
                    <SSAO radius={0.1} intensity={15} luminanceInfluence={0.5} color="black" />
                </EffectComposer> */}
            </Canvas>

            {/* Context Menu */}
            {menuPos && <ContextMenu x={menuPos.x} y={menuPos.y} onClose={() => setMenuPos(null)} />}

            {/* HUD / Overlay */}
            <div className="absolute top-2 left-2 flex flex-col gap-2 pointer-events-none">
                {filename && (
                    <div className="pointer-events-auto bg-card/90 backdrop-blur border-l-2 border-primary p-2 shadow-lg min-w-[180px]">
                        <div className="flex items-center gap-2">
                            <Box className="w-3 h-3 text-primary" />
                            <span className="text-[10px] font-bold text-foreground uppercase tracking-wider">{filename}</span>
                        </div>
                    </div>
                )}
            </div>

            <ScaleBar />
        </div>
    )
}
