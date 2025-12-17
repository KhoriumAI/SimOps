import { useState, useEffect } from 'react'
import { Clock, Play, Pause, Check, AlertCircle } from 'lucide-react'

/**
 * MeshTimer Component
 * Shows real-time elapsed time during mesh generation
 */
export default function MeshTimer({ isRunning, startTime, status, onComplete }) {
  const [elapsed, setElapsed] = useState(0)
  const [displayTime, setDisplayTime] = useState('00:00')

  // Update elapsed time every 100ms for smooth display
  useEffect(() => {
    if (!isRunning || !startTime) {
      return
    }

    const updateTimer = () => {
      const now = Date.now()
      const elapsedMs = now - startTime
      setElapsed(elapsedMs)
      setDisplayTime(formatTime(elapsedMs))
    }

    // Initial update
    updateTimer()

    // Update every 100ms
    const interval = setInterval(updateTimer, 100)

    return () => clearInterval(interval)
  }, [isRunning, startTime])

  // Reset when a new generation starts
  useEffect(() => {
    if (isRunning && startTime) {
      setElapsed(0)
      setDisplayTime('00:00')
    }
  }, [startTime])

  // Format milliseconds to MM:SS.ms
  function formatTime(ms) {
    const totalSeconds = Math.floor(ms / 1000)
    const minutes = Math.floor(totalSeconds / 60)
    const seconds = totalSeconds % 60
    const tenths = Math.floor((ms % 1000) / 100)
    
    if (minutes > 0) {
      return `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}.${tenths}`
    }
    return `${seconds.toString().padStart(2, '0')}.${tenths}s`
  }

  // Get status icon and color
  const getStatusDisplay = () => {
    switch (status) {
      case 'processing':
        return {
          icon: Play,
          color: 'text-blue-400',
          bgColor: 'bg-blue-500/20',
          borderColor: 'border-blue-500/50',
          label: 'Running',
          pulse: true,
        }
      case 'completed':
        return {
          icon: Check,
          color: 'text-green-400',
          bgColor: 'bg-green-500/20',
          borderColor: 'border-green-500/50',
          label: 'Complete',
          pulse: false,
        }
      case 'error':
        return {
          icon: AlertCircle,
          color: 'text-red-400',
          bgColor: 'bg-red-500/20',
          borderColor: 'border-red-500/50',
          label: 'Error',
          pulse: false,
        }
      default:
        return {
          icon: Clock,
          color: 'text-gray-400',
          bgColor: 'bg-gray-500/20',
          borderColor: 'border-gray-500/50',
          label: 'Ready',
          pulse: false,
        }
    }
  }

  const statusDisplay = getStatusDisplay()
  const Icon = statusDisplay.icon

  // Don't show if not running and no elapsed time
  if (!isRunning && elapsed === 0 && status !== 'completed' && status !== 'error') {
    return null
  }

  return (
    <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg ${statusDisplay.bgColor} border ${statusDisplay.borderColor} backdrop-blur`}>
      {/* Status Icon */}
      <div className={`${statusDisplay.color} ${statusDisplay.pulse ? 'animate-pulse' : ''}`}>
        <Icon className="w-4 h-4" />
      </div>

      {/* Timer Display */}
      <div className="flex flex-col">
        <div className="flex items-center gap-2">
          <span className={`font-mono text-sm font-bold ${statusDisplay.color}`}>
            {displayTime}
          </span>
          {isRunning && (
            <span className="flex gap-0.5">
              <span className="w-1 h-1 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
              <span className="w-1 h-1 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
              <span className="w-1 h-1 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
            </span>
          )}
        </div>
        <span className={`text-[10px] ${statusDisplay.color} opacity-75`}>
          {statusDisplay.label}
        </span>
      </div>
    </div>
  )
}

/**
 * Compact version for toolbar
 */
export function MeshTimerCompact({ isRunning, startTime, status }) {
  const [displayTime, setDisplayTime] = useState('00:00')

  useEffect(() => {
    if (!isRunning || !startTime) return

    const updateTimer = () => {
      const elapsedMs = Date.now() - startTime
      const totalSeconds = Math.floor(elapsedMs / 1000)
      const minutes = Math.floor(totalSeconds / 60)
      const seconds = totalSeconds % 60
      setDisplayTime(`${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`)
    }

    updateTimer()
    const interval = setInterval(updateTimer, 1000)
    return () => clearInterval(interval)
  }, [isRunning, startTime])

  if (!isRunning) return null

  return (
    <div className="flex items-center gap-1.5 text-xs">
      <Clock className="w-3 h-3 text-blue-400 animate-pulse" />
      <span className="font-mono text-blue-400">{displayTime}</span>
    </div>
  )
}
