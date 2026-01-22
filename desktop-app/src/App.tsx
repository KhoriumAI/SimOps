import React, { useEffect, useState } from 'react';
import { invoke } from '@tauri-apps/api/tauri';
import { checkUpdate, installUpdate } from '@tauri-apps/api/updater';
import { relaunch } from '@tauri-apps/api/process';
import {
    RefreshCw,
    Download,
    CheckCircle,
    AlertCircle,
    Activity,
    Settings,
    HardDrive,
    Cpu,
    Terminal,
    ExternalLink
} from 'lucide-react';

interface SystemHealth {
    docker_installed: bool;
    docker_running: bool;
    wsl2_available: bool;
    disk_space_gb: number;
    recommended_action: string | null;
}

const App = () => {
    // Update state
    const [updateStatus, setUpdateStatus] = useState('idle'); // idle, checking, available, updating, current, error
    const [version, setVersion] = useState('');

    // System Health state
    const [health, setHealth] = useState<SystemHealth | null>(null);
    const [healthLoading, setHealthLoading] = useState(true);

    // Backend state
    const [backendStatus, setBackendStatus] = useState<'checking' | 'running' | 'offline'>('checking');

    const [error, setError] = useState('');

    const checkForUpdates = async () => {
        setUpdateStatus('checking');
        try {
            const { shouldUpdate, manifest } = await checkUpdate();
            if (shouldUpdate) {
                setVersion(manifest?.version || 'new version');
                setUpdateStatus('available');
            } else {
                setUpdateStatus('current');
            }
        } catch (e) {
            console.error(e);
            setUpdateStatus('error');
        }
    };

    const runSystemCheck = async () => {
        setHealthLoading(true);
        try {
            const result: SystemHealth = await invoke('check_system_health');
            setHealth(result);
            setHealthLoading(false);
        } catch (e) {
            setError(e.toString());
            setHealthLoading(false);
        }
    };

    const checkBackend = async () => {
        setBackendStatus('checking');
        try {
            const response: string = await invoke('check_backend_health');
            setBackendStatus('running');
        } catch (e) {
            setBackendStatus('offline');
        }
    };

    const tryStartBackend = async () => {
        try {
            await invoke('start_backend');
            checkBackend();
        } catch (e) {
            setError(e.toString());
        }
    };

    const runUpdate = async () => {
        setUpdateStatus('updating');
        try {
            await installUpdate();
            await relaunch();
        } catch (e) {
            setError(e.toString());
            setUpdateStatus('error');
        }
    };

    useEffect(() => {
        checkForUpdates();
        runSystemCheck();

        const interval = setInterval(() => {
            checkBackend();
        }, 3000);

        return () => clearInterval(interval);
    }, []);

    const isReady = health?.docker_installed && health?.docker_running && backendStatus === 'running';

    return (
        <div className="min-h-screen bg-slate-900 text-slate-100 flex flex-col items-center justify-center p-6 font-sans">
            <div className="max-w-xl w-full bg-slate-800 rounded-3xl shadow-2xl p-8 border border-slate-700/50 backdrop-blur-xl">
                <div className="flex justify-between items-start mb-8">
                    <div>
                        <h1 className="text-4xl font-black bg-gradient-to-r from-cyan-400 via-blue-500 to-indigo-600 bg-clip-text text-transparent">
                            SimOps
                        </h1>
                        <p className="text-slate-400 font-medium mt-1">Desktop Operations Hub</p>
                    </div>
                    <div className="flex flex-col items-end">
                        <span className="text-xs font-mono text-slate-500 uppercase tracking-widest">Version</span>
                        <span className="text-sm font-bold text-cyan-500/80">0.1.0-alpha</span>
                    </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-8">
                    {/* System Health Block */}
                    <div className="bg-slate-900/50 rounded-2xl p-5 border border-slate-700/30">
                        <div className="flex items-center space-x-2 mb-4 text-slate-300">
                            <Activity size={18} className="text-cyan-400" />
                            <h2 className="font-bold text-sm uppercase tracking-wider">Environment</h2>
                        </div>

                        <div className="space-y-3">
                            <StatusItem
                                label="Docker Engine"
                                status={healthLoading ? 'loading' : (health?.docker_installed ? (health?.docker_running ? 'ok' : 'warn') : 'error')}
                                text={healthLoading ? 'Detecting...' : (health?.docker_installed ? (health?.docker_running ? 'Running' : 'Stopped') : 'Missing')}
                            />
                            <StatusItem
                                label="WSL2 Subsystem"
                                status={healthLoading ? 'loading' : (health?.wsl2_available ? 'ok' : 'info')}
                                text={healthLoading ? 'Checking...' : (health?.wsl2_available ? 'Available' : 'N/A')}
                            />
                            <StatusItem
                                label="Disk Space"
                                status={healthLoading ? 'loading' : (health && health.disk_space_gb > 10 ? 'ok' : 'error')}
                                text={healthLoading ? 'Calculating...' : `${health?.disk_space_gb.toFixed(1)} GB Free`}
                            />
                        </div>
                    </div>

                    {/* Backend Services Block */}
                    <div className="bg-slate-900/50 rounded-2xl p-5 border border-slate-700/30">
                        <div className="flex items-center space-x-2 mb-4 text-slate-300">
                            <Terminal size={18} className="text-indigo-400" />
                            <h2 className="font-bold text-sm uppercase tracking-wider">Services</h2>
                        </div>

                        <div className="space-y-3">
                            <StatusItem
                                label="Flask API"
                                status={backendStatus === 'checking' ? 'loading' : (backendStatus === 'running' ? 'ok' : 'error')}
                                text={backendStatus === 'checking' ? 'Connecting...' : (backendStatus === 'running' ? 'Connected' : 'Disconnected')}
                            />
                            <StatusItem
                                label="Compute Provider"
                                status="ok"
                                text="Local / Cloud"
                            />
                            <div className="mt-4">
                                {backendStatus === 'offline' && (
                                    <button
                                        onClick={tryStartBackend}
                                        className="text-xs bg-indigo-500/20 hover:bg-indigo-500/30 text-indigo-300 px-3 py-1.5 rounded-lg border border-indigo-500/30 transition-colors w-full"
                                    >
                                        Restart API Server
                                    </button>
                                )}
                            </div>
                        </div>
                    </div>
                </div>

                {/* Important Alerts */}
                {health?.recommended_action && (
                    <div className="mb-6 p-4 bg-amber-900/20 border border-amber-500/30 rounded-2xl flex items-start space-x-3">
                        <AlertCircle className="text-amber-500 shrink-0 mt-0.5" size={20} />
                        <div>
                            <p className="text-amber-200 text-sm font-medium leading-relaxed whitespace-pre-line">
                                {health.recommended_action}
                            </p>
                        </div>
                    </div>
                )}

                {/* Main Action Area */}
                <div className="space-y-4">
                    {updateStatus === 'available' ? (
                        <button
                            onClick={runUpdate}
                            className="w-full py-4 bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-400 hover:to-emerald-500 rounded-2xl font-bold text-lg transition-all shadow-xl shadow-emerald-900/20 active:scale-[0.98] flex items-center justify-center space-x-3"
                        >
                            <Download size={24} />
                            <span>Update to v{version} & Launch</span>
                        </button>
                    ) : isReady ? (
                        <button
                            onClick={() => window.location.href = 'http://localhost:5000'} // Or whatever entry point
                            className="w-full py-4 bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 rounded-2xl font-bold text-lg transition-all shadow-xl shadow-cyan-900/20 active:scale-[0.98] flex items-center justify-center space-x-3"
                        >
                            <ExternalLink size={24} />
                            <span>Launch SimOps Dashboard</span>
                        </button>
                    ) : (
                        <div className="w-full py-4 bg-slate-700/50 rounded-2xl font-bold text-lg text-slate-400 flex items-center justify-center space-x-3 cursor-not-allowed border border-slate-600/30">
                            <span className="opacity-50">Initializing Environment...</span>
                        </div>
                    )}

                    <button
                        onClick={() => { runSystemCheck(); checkForUpdates(); }}
                        className="w-full py-2.5 text-slate-400 hover:text-slate-200 text-sm font-medium transition-colors flex items-center justify-center space-x-2"
                    >
                        <RefreshCw size={14} className={healthLoading ? "animate-spin" : ""} />
                        <span>Force Refresh Status</span>
                    </button>
                </div>

                <div className="mt-8 pt-6 border-t border-slate-700/50 text-slate-500 text-[10px] text-center uppercase tracking-[0.2em] font-bold">
                    Khorium AI • Engineering Phase 2 • Task 06
                </div>
            </div>
        </div>
    );
};

const StatusItem = ({ label, status, text }: { label: string, status: 'ok' | 'warn' | 'error' | 'loading' | 'info', text: string }) => {
    return (
        <div className="flex justify-between items-center">
            <span className="text-xs font-medium text-slate-400">{label}</span>
            <div className="flex items-center space-x-2">
                <span className="text-[11px] font-bold text-slate-300">{text}</span>
                <div className={`w-2 h-2 rounded-full ${status === 'ok' ? 'bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.5)]' :
                        status === 'warn' ? 'bg-amber-500 shadow-[0_0_8px_rgba(245,158,11,0.5)]' :
                            status === 'error' ? 'bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.5)]' :
                                status === 'loading' ? 'bg-cyan-500 animate-pulse' :
                                    'bg-slate-500'
                    }`} />
            </div>
        </div>
    );
}

export default App;
