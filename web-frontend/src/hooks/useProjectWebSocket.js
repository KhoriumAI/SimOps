import { useEffect, useRef, useCallback, useState } from 'react'
import { io } from 'socket.io-client'
import { WS_URL } from '../config'

/**
 * WebSocket hook for real-time project status and log streaming
 * Replaces polling with event-driven updates
 */
export function useProjectWebSocket(projectId, onStatusUpdate, onLogLine, onJobCompleted, onJobFailed) {
  const socketRef = useRef(null)
  const [connected, setConnected] = useState(false)
  const pendingSubscriptionsRef = useRef(new Set()) // Track pending log subscriptions

  // Store handlers in refs so they can be updated without recreating the connection
  const handlersRef = useRef({ onLogLine, onJobCompleted, onJobFailed, projectId })
  
  // Update handlers ref when they change
  useEffect(() => {
    handlersRef.current = { onLogLine, onJobCompleted, onJobFailed, projectId }
  }, [onLogLine, onJobCompleted, onJobFailed, projectId])

  const connect = useCallback(() => {
    if (!projectId || socketRef.current?.connected) return

    try {
      // Disconnect existing socket if any
      if (socketRef.current) {
        socketRef.current.removeAllListeners()
        socketRef.current.disconnect()
      }

      socketRef.current = io(WS_URL, {
        transports: ['websocket', 'polling'],
        reconnection: true,
        reconnectionAttempts: 5,
        reconnectionDelay: 1000,
        timeout: 10000,
      })

      socketRef.current.on('connect', () => {
        console.log('[WS] Connected to project WebSocket')
        setConnected(true)

        const currentProjectId = handlersRef.current.projectId
        // Subscribe to project updates
        if (currentProjectId) {
          socketRef.current.emit('subscribe_project', { project_id: currentProjectId })
        }

        // Retry any pending log subscriptions
        if (pendingSubscriptionsRef.current.size > 0) {
          console.log(`[WS] Retrying ${pendingSubscriptionsRef.current.size} pending log subscriptions`)
          pendingSubscriptionsRef.current.forEach(jobId => {
            console.log('[WS] Retrying subscription for job:', jobId)
            socketRef.current.emit('subscribe_logs', {
              job_id: jobId,
              project_id: currentProjectId
            })
          })
          pendingSubscriptionsRef.current.clear()
        }
      })

      socketRef.current.on('disconnect', (reason) => {
        console.log('[WS] Disconnected:', reason)
        setConnected(false)
      })

      socketRef.current.on('connect_error', (err) => {
        console.error('[WS] Connection error:', err.message)
        setConnected(false)
      })

      // Listen for log lines - use handlersRef to get latest handlers
      socketRef.current.on('log_line', (data) => {
        const handlers = handlersRef.current
        console.log('[WS] Received log_line event:', {
          job_id: data.job_id,
          message_preview: data.message?.substring(0, 50),
          has_handler: !!handlers.onLogLine
        })
        if (data.job_id && handlers.onLogLine) {
          console.log('[WS] Calling onLogLine handler')
          handlers.onLogLine(data.message, data.timestamp)
        } else {
          if (!data.job_id) {
            console.warn('[WS] log_line event missing job_id')
          }
          if (!handlers.onLogLine) {
            console.warn('[WS] log_line handler (onLogLine) is not set')
          }
        }
      })

      // Listen for job completion
      socketRef.current.on('job_completed', (data) => {
        const handlers = handlersRef.current
        console.log('[WS] Received job_completed:', data.job_id)
        if (data.project_id === handlers.projectId && handlers.onJobCompleted) {
          handlers.onJobCompleted(data)
        }
      })

      // Listen for job failure
      socketRef.current.on('job_failed', (data) => {
        const handlers = handlersRef.current
        console.log('[WS] Received job_failed:', data.job_id)
        if (data.project_id === handlers.projectId && handlers.onJobFailed) {
          handlers.onJobFailed(data)
        }
      })

      // Listen for subscription confirmation
      socketRef.current.on('subscribed_project', (data) => {
        console.log('[WS] Subscribed to project:', data.project_id)
      })

      // Listen for log subscription confirmation
      socketRef.current.on('subscribed', (data) => {
        console.log('[WS] Subscribed to logs:', data.job_id, 'type:', data.type)
        pendingSubscriptionsRef.current.delete(data.job_id)
      })

      socketRef.current.on('error', (error) => {
        console.error('[WS] Error:', error)
      })

    } catch (error) {
      console.error('[WS] Failed to connect:', error)
    }
  }, [projectId]) // Only depend on projectId

  const subscribeToLogs = useCallback((jobId) => {
    if (!jobId) {
      console.warn('[WS] Cannot subscribe: no jobId provided')
      return
    }

    if (socketRef.current?.connected) {
      console.log('[WS] Subscribing to logs for job:', jobId)
      socketRef.current.emit('subscribe_logs', {
        job_id: jobId,
        project_id: projectId
      })
      pendingSubscriptionsRef.current.add(jobId)
    } else {
      console.log('[WS] Socket not connected yet, queueing subscription for job:', jobId)
      pendingSubscriptionsRef.current.add(jobId)
      // If socket exists but not connected, wait a bit and retry
      if (socketRef.current) {
        const checkConnection = setInterval(() => {
          if (socketRef.current?.connected) {
            clearInterval(checkConnection)
            console.log('[WS] Socket connected, subscribing to logs for job:', jobId)
            socketRef.current.emit('subscribe_logs', {
              job_id: jobId,
              project_id: projectId
            })
          }
        }, 100)
        // Clear interval after 5 seconds to avoid infinite loop
        setTimeout(() => clearInterval(checkConnection), 5000)
      }
    }
  }, [projectId])

  const unsubscribeFromLogs = useCallback((jobId) => {
    if (socketRef.current?.connected && jobId) {
      console.log('[WS] Unsubscribing from logs for job:', jobId)
      socketRef.current.emit('unsubscribe_logs', { job_id: jobId })
      pendingSubscriptionsRef.current.delete(jobId)
    }
  }, [])

  useEffect(() => {
    if (!projectId) return

    // Only connect if not already connected
    if (!socketRef.current || !socketRef.current.connected) {
      connect()
    }

    return () => {
      // Only disconnect if projectId actually changes (not when callbacks change)
      if (socketRef.current) {
        console.log('[WS] Cleaning up WebSocket connection')
        socketRef.current.disconnect()
        socketRef.current = null
        setConnected(false)
        pendingSubscriptionsRef.current.clear()
      }
    }
  }, [projectId]) // Only depend on projectId, not connect

  return {
    connected,
    subscribeToLogs,
    unsubscribeFromLogs,
    socket: socketRef.current
  }
}


