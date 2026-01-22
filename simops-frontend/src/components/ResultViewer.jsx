import React, { useRef, useMemo, useState, useEffect, Suspense } from 'react'
import { Canvas, useThree, useFrame, useLoader } from '@react-three/fiber'
import { OrbitControls, PerspectiveCamera, GizmoHelper, GizmoViewport, Environment, Center, Html } from '@react-three/drei'
import { EffectComposer, SSAO, Bloom } from '@react-three/postprocessing'
import * as THREE from 'three'
import { VTKLoader, STLLoader } from 'three-stdlib'
import { Box, Loader2, MousePointer2, Tag, X, BarChart3, Settings2, Maximize, Minimize, Ruler, AlertTriangle } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'

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
    // Matcap: 'Matcap_Gray' provides a good neutral engineering look
    // Matcap removed for offline stability. Using standard material.

    useMemo(() => {
        if (geometry) {
            geometry.computeBoundingBox()
            geometry.center()
            geometry.computeVertexNormals()
        }
    }, [geometry])

    return (
        <mesh castShadow receiveShadow>
            {/* Clone geometry to allow independent wireframe mesh if needed */}
            <primitive object={geometry} />
            <meshStandardMaterial
                color="#e5e7eb"
                roughness={0.4}
                metalness={0.6}
                transparent={opacity < 1}
                opacity={opacity}
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
            {/* Loading Overlay */}
            <AnimatePresence>
                {isLoading && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        style={{ bottom: consoleOffset }}
                        className="absolute inset-x-0 top-0 z-50 bg-background/80 flex flex-col items-center justify-center text-foreground"
                    >
                        <Loader2 className="w-8 h-8 animate-spin text-primary mb-2" />
                        <p className="text-xs font-mono animate-pulse text-muted-foreground">{loadingMessage || 'PROCESSING GEOMETRY...'}</p>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Canvas */}
            <Canvas shadows dpr={[1, 2]} gl={{ preserveDrawingBuffer: true }}>
                <PerspectiveCamera makeDefault position={[50, 50, 50]} fov={45} />

                {/* Engineering Controls: Zero Damping for "Mechanical" feel */}
                <OrbitControls
                    makeDefault
                    enableDamping={false}
                    rotateSpeed={0.8}
                    zoomSpeed={1.2}
                />

                {/* Lighting Environment */}
                <Environment preset="city" />
                <ambientLight intensity={0.5} />
                <directionalLight position={[10, 10, 5]} intensity={1} castShadow />

                <Center>
                    <ErrorBoundary>
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
                    <GizmoViewport axisColors={['#ef4444', '#22c55e', '#3b82f6']} labelColor="white" />
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
                        <div className="flex items-center gap-2 mb-1">
                            <Box className="w-3 h-3 text-primary" />
                            <span className="text-[10px] font-bold text-foreground uppercase tracking-wider">{filename}</span>
                        </div>
                        <div className="text-[9px] font-mono text-muted-foreground">
                            {meshUrl ? 'ACTIVE MESH' : 'NO GEOMETRY'}
                        </div>
                    </div>
                )}
            </div>

            <ScaleBar />
        </div>
    )
}
