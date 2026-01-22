import React, { useState, useEffect, useRef } from 'react'
import { Activity, Box, Settings, Play, CheckCircle, ChevronRight, Terminal, Upload, AlertCircle, Monitor, HelpCircle, Copy } from 'lucide-react'
import FileUpload from './components/FileUpload'
import CADDropZone from './components/CADImport/CADDropZone'
import ResultViewer from './components/ResultViewer.jsx'
import SmartInput from './components/SmartInput'

/**
 * SimOps Engineering Workbench
 * Refactored for "Zero Page Scroll" and Input Fidelity.
 */
function App() {
    // Workflow State
    const [activeStep, setActiveStep] = useState('upload')
    const [file, setFile] = useState(null)
    const [uploadedFilename, setUploadedFilename] = useState(null)
    const [previewUrl, setPreviewUrl] = useState(null)
    const [isProcessing, setIsProcessing] = useState(false)
    const [statusMessage, setStatusMessage] = useState("Ready")
    const [simResults, setSimResults] = useState(null)

    // Layout Persistence
    const [leftPanelWidth, setLeftPanelWidth] = useState(320)
    const [consoleHeight, setConsoleHeight] = useState(150)
    const [consoleOpen, setConsoleOpen] = useState(true)
    const [isBackendReady, setIsBackendReady] = useState(false)
    const [openfoamAvailable, setOpenfoamAvailable] = useState(false)
    const [consoleLogs, setConsoleLogs] = useState([{ type: 'info', text: 'SimOps [Version 2.0.0-Eng]' }])
    const [consoleCopied, setConsoleCopied] = useState(false)
    const consoleRef = useRef(null)
    const abortRef = useRef(null)
    const timeoutsRef = useRef([])
    const stopRequestedRef = useRef(false)

    // Physics Parameters (Deterministic)
    const [power, setPower] = useState(50) // Watts
    const [ambientTemp, setAmbientTemp] = useState(293.15) // Kelvin
    const [convection, setConvection] = useState(20) // W/m2K
    const [material, setMaterial] = useState('Aluminum')
    const [simMode, setSimMode] = useState('steady_state') // steady_state or transient
    const [timestep, setTimestep] = useState(0.1) // seconds
    const [duration, setDuration] = useState(10.0) // seconds
    const [initialTemp, setInitialTemp] = useState(293.15) // K
    const [iterations, setIterations] = useState(1000)
    const [tolerance, setTolerance] = useState(1e-3)
    const [writeInterval, setWriteInterval] = useState(50)
    const [colormap, setColormap] = useState('jet')

    // Force Dark Mode for Engineering Feel
    useEffect(() => {
        document.documentElement.classList.add('dark');

        let checkCount = 0;
        const interval = setInterval(async () => {
            checkCount++;
            try {
                const res = await fetch('http://localhost:8000/api/health')
                if (res.ok) {
                    setIsBackendReady(true)
                    addLog("Backend connected.", 'info')
                    clearInterval(interval)

                    // Check OpenFOAM availability
                    try {
                        const diagRes = await fetch('http://localhost:8000/api/diagnostics')
                        if (diagRes.ok) {
                            const diagData = await diagRes.json()
                            setOpenfoamAvailable(diagData.openfoam_available || false)

                            if (diagData.openfoam_available) {
                                addLog("OpenFOAM detected - all solvers available.", 'success')
                            } else {
                                addLog("OpenFOAM not found - using builtin solver only.", 'info')
                                addLog("   (To use OpenFOAM: install WSL and OpenFOAM)", 'info')
                            }
                        }
                    } catch (e) {
                        addLog("Could not check OpenFOAM status.", 'info')
                    }
                }
            } catch (e) {
                if (checkCount % 5 === 0) addLog("Waiting for backend sidecar...", 'info')
            }
            if (checkCount > 60) clearInterval(interval); // Timeout after 60s
        }, 1000)
        return () => clearInterval(interval)
    }, []);

    // Auto-scroll console to bottom when new logs are added
    useEffect(() => {
        if (consoleRef.current) {
            consoleRef.current.scrollTop = consoleRef.current.scrollHeight
        }
    }, [consoleLogs])

    const clearPendingTimeouts = () => {
        timeoutsRef.current.forEach((timeoutId) => clearTimeout(timeoutId))
        timeoutsRef.current = []
    }

    const waitWithAbort = (ms, signal) => {
        return new Promise((resolve, reject) => {
            if (signal?.aborted) {
                reject(new DOMException('Aborted', 'AbortError'))
                return
            }
            const timeoutId = setTimeout(() => resolve(), ms)
            timeoutsRef.current.push(timeoutId)
            if (signal) {
                signal.addEventListener(
                    'abort',
                    () => {
                        clearTimeout(timeoutId)
                        reject(new DOMException('Aborted', 'AbortError'))
                    },
                    { once: true }
                )
            }
        })
    }

    const addLog = (msg, type = 'info') => {
        console.log(msg)
        setStatusMessage(msg)
        setConsoleLogs(prev => [...prev, { type, text: msg }])
    }

    const getLogDisplay = (log) => {
        let colorClass = 'text-gray-400'
        let prefix = ''

        if (log.type === 'info') {
            colorClass = 'text-gray-500'
        } else if (log.type === 'step') {
            colorClass = 'text-blue-400'
            prefix = '>> '
        } else if (log.type === 'success') {
            colorClass = 'text-green-400'
            prefix = log.text.startsWith('Step') ? '>> ' : '   '
        } else if (log.type === 'error') {
            colorClass = 'text-red-400'
            prefix = '!! '
        }

        return { colorClass, prefix }
    }

    const copyConsole = async (event) => {
        event.stopPropagation()
        const consoleText = consoleLogs
            .map((log) => {
                const { prefix } = getLogDisplay(log)
                return `${prefix}${log.text}`
            })
            .join('\n')

        try {
            await navigator.clipboard.writeText(consoleText)
            setConsoleCopied(true)
            setTimeout(() => setConsoleCopied(false), 1500)
        } catch (e) {
            setConsoleCopied(false)
        }
    }

    const handleFileSelect = async (selectedFile) => {
        if (!isBackendReady) {
            throw new Error("Backend not ready yet. Please wait.")
        }
        setFile(selectedFile)
        if (selectedFile) {
            addLog(`Uploading ${selectedFile.name}...`, 'info')
            setIsProcessing(true)

            const formData = new FormData()
            formData.append('files', selectedFile)

            const res = await fetch('http://localhost:8000/api/upload', {
                method: 'POST',
                body: formData
            })

            if (!res.ok) {
                setIsProcessing(false)
                throw new Error(`Upload failed: ${res.statusText}`)
            }

            const data = await res.json()
            if (data.error) {
                setIsProcessing(false)
                throw new Error(data.error)
            }

            if (data.saved_as) {
                setUploadedFilename(data.saved_as)

                // Log any STL generation errors for debugging
                if (data.stl_generation_error) {
                    addLog(`Warning: Preview generation failed: ${data.stl_generation_error}`, 'error')
                }

                // Set preview URL (prioritize the generated preview over the raw file)
                if (data.preview_url) {
                    setPreviewUrl(`http://localhost:8000${data.preview_url}`)
                    addLog(`Loaded mesh: ${data.saved_as}`, 'success')
                } else if (data.url && selectedFile.name.toLowerCase().endsWith('.msh')) {
                    // Only use raw URL if it's already a viewable mesh format
                    setPreviewUrl(`http://localhost:8000${data.url}`)
                    addLog(`Loaded mesh: ${data.saved_as}`, 'success')
                } else {
                    addLog(`Upload complete: ${data.saved_as} (no preview available)`, 'info')
                }

                setIsProcessing(false)
            }
        }
    }

    const stopSimulation = () => {
        if (abortRef.current) {
            stopRequestedRef.current = true
            abortRef.current.abort()
            abortRef.current = null
        }
        clearPendingTimeouts()
        setIsProcessing(false)
    }

    const runSimulation = async () => {
        if (!uploadedFilename) return
        setIsProcessing(true)
        setActiveStep('result')
        clearPendingTimeouts()
        stopRequestedRef.current = false
        abortRef.current = new AbortController()
        const { signal } = abortRef.current

        try {
            // Build config directly from UI inputs (deterministic, no AI)
            const config = {
                heat_source_power: power,
                ambient_temperature: ambientTemp,
                initial_temperature: initialTemp,
                convection_coefficient: convection,
                material: material,
                simulation_type: simMode,
                time_step: timestep,
                duration: duration,
                max_iterations: iterations,
                tolerance: tolerance,
                write_interval: writeInterval,
                colormap: colormap,
                solver: openfoamAvailable ? 'openfoam' : 'builtin'  // Use OpenFOAM if available, otherwise builtin
            }

            // Explicitly tell user which solver is being used
            const solverName = openfoamAvailable ? 'OpenFOAM' : 'Builtin (Python)'
            addLog(`Solver: ${solverName}`, 'info')

            if (!openfoamAvailable) {
                addLog("   (OpenFOAM not available - using fast Python solver)", 'info')
            }

            addLog(`Starting simulation: ${power}W heat source, ${ambientTemp}K ambient`, 'info')
            addLog(`Step 1: Configuring solver (${simMode})`, 'step')

            // Simulate setup delay
            await waitWithAbort(500, signal)
            addLog(`Step 2: Initializing mesh (${uploadedFilename})`, 'step')

            await waitWithAbort(500, signal)
            addLog(`Step 3: Setting boundary conditions`, 'step')

            const simRes = await fetch('http://localhost:8000/api/simulate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    filename: uploadedFilename,
                    config: config
                }),
                signal
            })

            // Check response status BEFORE parsing JSON
            if (!simRes.ok) {
                let errorMsg = `Simulation failed with status ${simRes.status}`
                try {
                    const errorData = await simRes.json()
                    errorMsg = errorData.error || errorMsg
                } catch {
                    const text = await simRes.text()
                    errorMsg = text || errorMsg
                }
                throw new Error(errorMsg)
            }

            const simData = await simRes.json()

            if (simData.status === 'success') {
                setSimResults(simData.results)
                const resultIterations = simData.results.iterations_run ?? simData.results.iterations
                const maxIters = simData.results.max_iterations ?? iterations
                const resultTolerance = simData.results.tolerance ?? tolerance
                const finalResidual = simData.results.final_residual
                const converged = simData.results.converged !== false

                if (!converged) {
                    addLog(`Step 4: Solving equations (failed to converge in ${resultIterations ?? maxIters} iterations)`, 'error')
                    if (Number.isFinite(finalResidual) && Number.isFinite(resultTolerance)) {
                        addLog(`   Final residual: ${finalResidual.toExponential(2)} (tol ${resultTolerance.toExponential(2)})`, 'error')
                    } else if (Number.isFinite(finalResidual)) {
                        addLog(`   Final residual: ${finalResidual.toExponential(2)}`, 'error')
                    }
                } else if (Number.isFinite(resultIterations)) {
                    addLog(`Step 4: Solving equations (converged in ${resultIterations} iterations)`, 'success')
                    if (Number.isFinite(resultTolerance)) {
                        addLog(`   Convergence: ${resultTolerance.toExponential(2)}`, 'success')
                    }
                } else {
                    addLog(`Step 4: Solving equations (direct solve)`, 'success')
                }
                addLog(`   Max Temperature: ${simData.results.max_temperature_C?.toFixed(1) || simData.results.max_temp?.toFixed(1)}°C`, 'success')
                addLog(`Step 5: Generating visualization`, 'success')
                addLog(`Simulation completed successfully!`, 'success')

                // Auto-download and open PDF if available
                if (simData.results.pdf_url) {
                    const pdfUrl = `http://localhost:8000${simData.results.pdf_url}`
                    addLog(`Opening PDF report...`, 'info')

                    // Open PDF in new tab
                    window.open(pdfUrl, '_blank')

                    // Also trigger download
                    const link = document.createElement('a')
                    link.href = pdfUrl
                    link.download = `SimOps_Report_${Date.now()}.pdf`
                    document.body.appendChild(link)
                    link.click()
                    document.body.removeChild(link)

                    addLog(`   PDF opened in new tab and downloaded`, 'success')
                } else if (simData.results.report_url) {
                    // Fallback to HTML report
                    const reportUrl = `http://localhost:8000${simData.results.report_url}`
                    window.open(reportUrl, '_blank')
                    addLog(`   HTML report opened in new tab`, 'success')
                }

                // Note about VTK viewer limitation with builtin solver
                if (!openfoamAvailable && simData.results.vtk_url) {
                    addLog(`   Note: 3D viewer not supported for builtin solver results`, 'info')
                    addLog(`   (Use PDF report for visualization)`, 'info')
                }
            } else {
                throw new Error(simData.error || "Simulation failed")
            }

        } catch (e) {
            if (e?.name === 'AbortError') {
                addLog(stopRequestedRef.current ? 'Simulation stopped by user.' : 'Simulation cancelled.', 'info')
            } else {
                addLog(`Error: ${e.message}`, 'error')
            }
        } finally {
            setIsProcessing(false)
            abortRef.current = null
            clearPendingTimeouts()
            stopRequestedRef.current = false
        }
    }

    return (
        <div className="h-screen w-screen flex flex-col bg-background text-foreground overflow-hidden select-none">

            {/* 1. Header (Dense, 48px) */}
            <header className="h-12 bg-card border-b border-border flex items-center justify-between px-4 shrink-0">
                <div className="flex items-center gap-3">
                    <div className="w-6 h-6 bg-primary rounded flex items-center justify-center">
                        <Activity className="w-4 h-4 text-primary-foreground" />
                    </div>
                    <span className="font-bold text-sm tracking-tight">SimOps <span className="text-muted-foreground font-normal">Engineer</span></span>
                </div>

                <div className="flex items-center gap-2">
                    <div className="flex items-center bg-muted/30 rounded p-0.5 border border-border">
                        {['upload', 'simulate', 'result'].map(step => (
                            <button
                                key={step}
                                onClick={() => setActiveStep(step)}
                                className={`px-3 py-1 text-[10px] uppercase font-medium rounded transition-colors ${activeStep === step ? 'bg-primary text-primary-foreground shadow-sm' : 'text-muted-foreground hover:text-foreground'}`}
                            >
                                {step}
                            </button>
                        ))}
                    </div>
                </div>
            </header>

            {/* 2. Main Docking Layout */}
            <div className="flex-1 flex overflow-hidden">

                {/* Left Panel: Tree & Setup */}
                <div style={{ width: leftPanelWidth }} className="flex flex-col border-r border-border bg-card shrink-0 overflow-hidden">
                    <div className="p-2 border-b border-border bg-muted/10">
                        <h3 className="text-[10px] font-bold uppercase text-muted-foreground tracking-wider">Project Explorer</h3>
                    </div>

                    <div className="flex-1 overflow-y-auto overflow-x-hidden p-2 space-y-4 min-w-0">
                        <section>
                            <div className="text-[10px] font-medium text-foreground mb-2 flex items-center gap-2">
                                <Box className="w-3 h-3 text-primary" /> Mesh
                            </div>
                            <FileUpload compact onFileSelect={handleFileSelect} selectedFile={file} isUploading={isProcessing} />
                        </section>

                        <section className={`transition-opacity ${!file ? 'opacity-50 pointer-events-none' : ''}`}>
                            <div className="text-[10px] font-medium text-foreground mb-2 flex items-center gap-2">
                                <Settings className="w-3 h-3 text-primary" /> Physics Setup
                            </div>

                            <div className="space-y-3 pl-2 border-l border-border ml-1.5 mb-6">
                                <div className="flex flex-col gap-1 mb-2">
                                    <label className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Simulation Mode</label>
                                    <div className="flex bg-muted/30 rounded p-0.5 border border-border">
                                        <button
                                            onClick={() => setSimMode('steady_state')}
                                            className={`flex-1 py-1 text-[9px] uppercase font-bold rounded transition-all ${simMode === 'steady_state' ? 'bg-primary text-primary-foreground shadow-sm' : 'text-muted-foreground hover:text-foreground'}`}
                                        >
                                            Steady
                                        </button>
                                        <button
                                            onClick={() => setSimMode('transient')}
                                            className={`flex-1 py-1 text-[9px] uppercase font-bold rounded transition-all ${simMode === 'transient' ? 'bg-primary text-primary-foreground shadow-sm' : 'text-muted-foreground hover:text-foreground'}`}
                                        >
                                            Transient
                                        </button>
                                    </div>
                                </div>

                                <div className="flex flex-col gap-1">
                                    <label className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium flex items-center gap-1.5">
                                        Material
                                        <div className="relative group/tip cursor-help">
                                            <HelpCircle className="w-2.5 h-2.5 opacity-40 hover:opacity-80 transition-opacity" />
                                            <div className="absolute left-full ml-2 top-0 px-2 py-1 bg-popover text-popover-foreground text-[10px] rounded border border-border shadow-xl opacity-0 group-hover/tip:opacity-100 pointer-events-none transition-opacity whitespace-normal w-48 z-50">
                                                Defines the physical properties (k, rho, Cp).
                                            </div>
                                        </div>
                                    </label>
                                    <select
                                        value={material}
                                        onChange={(e) => setMaterial(e.target.value)}
                                        className="w-full bg-input/50 border border-border rounded px-2 py-1 text-xs font-mono text-foreground focus:border-primary focus:ring-1 focus:ring-primary outline-none transition-all"
                                    >
                                        <option value="Aluminum">Aluminum 6061</option>
                                        <option value="Copper">Copper (Pure)</option>
                                        <option value="Steel">Stainless Steel 304</option>
                                        <option value="Silicon">Silicon (Polished)</option>
                                    </select>
                                </div>

                                <SmartInput
                                    label="Heat Power"
                                    tooltip="Total thermal energy generated by the source."
                                    value={power}
                                    onChange={setPower}
                                    units="W"
                                    min={0}
                                />
                                <SmartInput
                                    label="Ambient Temp"
                                    tooltip="Temperature of the surrounding environment."
                                    value={ambientTemp}
                                    onChange={setAmbientTemp}
                                    units="K"
                                    min={0}
                                />
                                <SmartInput
                                    label="Initial Temp"
                                    tooltip="Starting temperature of the solid components at T=0."
                                    value={initialTemp}
                                    onChange={setInitialTemp}
                                    units="K"
                                    min={0}
                                />
                                <SmartInput
                                    label="Convection (h)"
                                    tooltip="Heat transfer coefficient to the fluid. Higher is better cooling."
                                    value={convection}
                                    onChange={setConvection}
                                    units="W/m²K"
                                    min={0}
                                />
                            </div>

                            <div className="text-[10px] font-medium text-foreground mb-2 flex items-center gap-2">
                                <Activity className="w-3 h-3 text-primary" /> Solver & Output
                            </div>

                            <div className="space-y-3 pl-2 border-l border-border ml-1.5">
                                {simMode === 'transient' && (
                                    <>
                                        <SmartInput
                                            label="Time Step"
                                            tooltip="Temporal resolution of the simulation."
                                            value={timestep}
                                            onChange={setTimestep}
                                            units="s"
                                            step={0.01}
                                            min={0.001}
                                        />
                                        <SmartInput
                                            label="Run Duration"
                                            tooltip="Total real-time duration of the process."
                                            value={duration}
                                            onChange={setDuration}
                                            units="s"
                                            min={timestep}
                                        />
                                    </>
                                )}

                                <div className="flex flex-col gap-1">
                                    <label className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium flex items-center gap-1.5">
                                        Palette
                                        <div className="relative group/tip cursor-help">
                                            <HelpCircle className="w-2.5 h-2.5 opacity-40 hover:opacity-80 transition-opacity" />
                                            <div className="absolute left-full ml-2 top-0 px-2 py-1 bg-popover text-popover-foreground text-[10px] rounded border border-border shadow-xl opacity-0 group-hover/tip:opacity-100 pointer-events-none transition-opacity whitespace-normal w-48 z-50">
                                                Visual color mapping for temperature results.
                                            </div>
                                        </div>
                                    </label>
                                    <select
                                        value={colormap}
                                        onChange={(e) => setColormap(e.target.value)}
                                        className="w-full bg-input/50 border border-border rounded px-2 py-1 text-xs font-mono text-foreground focus:border-primary focus:ring-1 focus:ring-primary outline-none transition-all"
                                    >
                                        <option value="jet">Jet (Standard)</option>
                                        <option value="viridis">Viridis (Perceptual)</option>
                                        <option value="inferno">Inferno (Heat)</option>
                                        <option value="magma">Magma</option>
                                    </select>
                                </div>

                                <SmartInput
                                    label="Write Freq"
                                    tooltip="Saves results every N iterations/steps."
                                    value={writeInterval}
                                    onChange={setWriteInterval}
                                    step={10}
                                    min={1}
                                />
                                <SmartInput
                                    label="Iterations"
                                    tooltip="Max solver loops to reach convergence."
                                    value={iterations}
                                    onChange={setIterations}
                                    step={100}
                                    min={10}
                                />
                                <SmartInput
                                    label="Tolerance"
                                    tooltip="Numerical precision threshold for convergence."
                                    value={tolerance}
                                    onChange={setTolerance}
                                    step={1e-7}
                                    min={1e-12}
                                />
                            </div>
                        </section>


                    </div>

                    <div className="p-3 border-t border-border bg-muted/10">
                        <div className="flex gap-2">
                            <button
                                onClick={runSimulation}
                                disabled={!file || isProcessing}
                                className={`flex-1 py-1.5 rounded text-xs font-medium flex items-center justify-center gap-2 ${!file || isProcessing ? 'bg-muted text-muted-foreground' : 'bg-primary text-primary-foreground hover:bg-primary/90'}`}
                            >
                                {isProcessing ? <span className="animate-spin">⟳</span> : <Play className="w-3 h-3 fill-current" />}
                                Execute Solve
                            </button>
                            <button
                                onClick={stopSimulation}
                                disabled={!isProcessing}
                                className={`px-3 py-1.5 rounded text-xs font-medium border ${isProcessing ? 'border-red-500 text-red-300 hover:bg-red-500/10' : 'border-white/10 text-muted-foreground'}`}
                            >
                                Stop
                            </button>
                        </div>
                    </div>
                </div>

                {/* Center Panel: Viewport */}
                <div className="flex-1 flex flex-col relative min-w-0 bg-gradient-to-br from-[#2b2b2b] to-[#1a1a1a]">
                    {!file && activeStep === 'upload' && (
                        <div className="absolute inset-0 flex items-center justify-center z-10 p-10">
                            <CADDropZone onFileAccepted={handleFileSelect} />
                        </div>
                    )}

                    {(file || activeStep !== 'upload') && (
                        <ResultViewer
                            filename={file?.name}
                            isLoading={isProcessing}
                            loadingMessage={statusMessage}
                            meshUrl={simResults?.vtk_url && openfoamAvailable ? `http://localhost:8000${simResults.vtk_url}` : null}
                            previewUrl={previewUrl}
                            consoleHeight={consoleHeight}
                            consoleOpen={consoleOpen}
                        />
                    )}

                    {/* Bottom Console Pane (Collapsible) */}
                    <div
                        style={{ height: consoleOpen ? consoleHeight : 32 }}
                        className="absolute bottom-0 left-0 right-0 bg-black/90 border-t border-border flex flex-col transition-all duration-200 z-20"
                    >
                        <div
                            className="h-8 bg-black flex items-center justify-between px-3 cursor-pointer hover:bg-white/5 border-b border-white/10"
                            onClick={() => setConsoleOpen(!consoleOpen)}
                        >
                            <div className="flex items-center gap-2 text-xs font-mono text-muted-foreground">
                                <Terminal className="w-3 h-3" /> CONSOLE
                            </div>
                            <div className="flex items-center gap-2">
                                <div className="text-[10px] font-mono text-muted-foreground truncate max-w-[420px] opacity-70">
                                    {statusMessage}
                                </div>
                                <button
                                    type="button"
                                    onClick={copyConsole}
                                    className="flex items-center gap-1 text-[10px] font-mono text-muted-foreground hover:text-foreground px-2 py-0.5 border border-white/10 rounded bg-white/5"
                                    title="Copy console log"
                                >
                                    <Copy className="w-3 h-3" />
                                    {consoleCopied ? 'Copied' : 'Copy'}
                                </button>
                            </div>
                        </div>

                        <div ref={consoleRef} className="flex-1 overflow-y-auto p-2 font-mono text-[11px] text-gray-300 space-y-0.5">
                            {consoleLogs.map((log, idx) => {
                                const { colorClass, prefix } = getLogDisplay(log)

                                return (
                                    <div key={idx} className={colorClass}>
                                        {prefix}{log.text}
                                    </div>
                                )
                            })}
                            {simResults?.report_url && (
                                <div className="text-blue-400 mt-2">
                                    {'   '} <a href={`http://localhost:8000${simResults.report_url}`} target="_blank" rel="noreferrer" className="hover:underline underline-offset-4 decoration-blue-400/30">View Analysis Report</a>
                                </div>
                            )}
                        </div>
                    </div>
                </div>

            </div >
        </div >
    )
}

export default App

