import { useState, useEffect, useCallback, useRef } from 'react'
import { io } from 'socket.io-client'
import { API_BASE, WS_URL } from '../config'

/**
 * useWebSocket Hook
 * 
 * Manages WebSocket connection for real-time batch progress updates.
 * Falls back to polling if WebSocket is unavailable.
 */
export function useWebSocket(batchId, onProgress) {
  const [connected, setConnected] = useState(false)
  const [error, setError] = useState(null)
  const socketRef = useRef(null)
  const reconnectAttempts = useRef(0)
  const maxReconnectAttempts = 5

  const connect = useCallback(() => {
    if (!batchId) return

    try {
      socketRef.current = io(WS_URL, {
        transports: ['websocket', 'polling'],
        reconnection: true,
        reconnectionAttempts: maxReconnectAttempts,
        reconnectionDelay: 1000,
        reconnectionDelayMax: 5000,
        timeout: 10000,
      })

      socketRef.current.on('connect', () => {
        console.log('[WS] Connected')
        setConnected(true)
        setError(null)
        reconnectAttempts.current = 0

        // Join batch room for updates
        socketRef.current.emit('join_batch', { batch_id: batchId })
      })

      socketRef.current.on('disconnect', (reason) => {
        console.log('[WS] Disconnected:', reason)
        setConnected(false)
      })

      socketRef.current.on('connect_error', (err) => {
        console.log('[WS] Connection error:', err.message)
        setError(err.message)
        reconnectAttempts.current++

        if (reconnectAttempts.current >= maxReconnectAttempts) {
          console.log('[WS] Max reconnect attempts reached, falling back to polling')
        }
      })

      // Listen for job progress updates
      socketRef.current.on('job_progress', (data) => {
        console.log('[WS] Job progress:', data)
        onProgress?.(data)
      })

      // Listen for batch status updates
      socketRef.current.on('batch_status', (data) => {
        console.log('[WS] Batch status:', data)
        onProgress?.({ ...data, type: 'batch_status' })
      })

    } catch (err) {
      console.error('[WS] Setup error:', err)
      setError(err.message)
    }
  }, [batchId, onProgress])

  const disconnect = useCallback(() => {
    if (socketRef.current) {
      socketRef.current.disconnect()
      socketRef.current = null
    }
    setConnected(false)
  }, [])

  useEffect(() => {
    if (batchId) {
      connect()
    }

    return () => {
      disconnect()
    }
  }, [batchId, connect, disconnect])

  return {
    connected,
    error,
    reconnect: connect,
    disconnect
  }
}

/**
 * useBatchWebSocket Hook
 * 
 * Replaces useBatchPolling with a WebSocket-based system.
 * Fetches initial state and then listens for real-time updates.
 */
export function useBatchWebSocket(batchId, authFetch) {
  const [batch, setBatch] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [connected, setConnected] = useState(false)
  const socketRef = useRef(null)

  const fetchBatch = useCallback(async () => {
    if (!batchId || !authFetch) return

    try {
      const response = await authFetch(`${API_BASE}/batch/${batchId}?include_files=true&include_jobs=true`)
      if (response.ok) {
        const data = await response.json()
        setBatch(data)
        setError(null)
      } else {
        const err = await response.json()
        setError(err.error || 'Failed to fetch batch')
      }
    } catch (err) {
      console.error('[WS-Batch] Initial fetch error:', err.message)
      setError(err.message)
    }
  }, [batchId, authFetch])

  // Initial fetch when batchId changes
  useEffect(() => {
    if (batchId && authFetch) {
      setLoading(true)
      fetchBatch().finally(() => setLoading(false))
    } else {
      setBatch(null)
    }
  }, [batchId, authFetch, fetchBatch])

  // WebSocket connection
  useEffect(() => {
    if (!batchId) return

    const socket = io(WS_URL, {
      transports: ['websocket', 'polling'],
      reconnection: true,
    })

    socketRef.current = socket

    socket.on('connect', () => {
      console.log('[WS-Batch] Connected')
      setConnected(true)
      socket.emit('join_batch', { batch_id: batchId })
    })

    socket.on('disconnect', () => {
      console.log('[WS-Batch] Disconnected')
      setConnected(false)
    })

    // Listen for job progress updates
    socket.on('job_progress', (data) => {
      if (data.batch_id === batchId) {
        console.log('[WS-Batch] Job progress:', data)
        // Optionally update local batch state if it contains jobs
        setBatch(prev => {
          if (!prev || !prev.jobs) return prev
          return {
            ...prev,
            jobs: prev.jobs.map(j => j.id === data.job_id ? { ...j, status: data.status, progress: data.progress } : j)
          }
        })
      }
    })

    // Listen for batch status updates
    socket.on('batch_status', (data) => {
      if (data.batch_id === batchId) {
        console.log('[WS-Batch] Batch status:', data)
        setBatch(prev => prev ? { ...prev, status: data.status, ...data } : null)
      }
    })

    return () => {
      socket.disconnect()
      socketRef.current = null
    }
  }, [batchId])

  return {
    batch,
    loading,
    error,
    connected,
    refresh: fetchBatch,
    startPolling: () => console.log('[WS-Batch] WebSocket active (legacy startPolling called)'),
    stopPolling: () => console.log('[WS-Batch] WebSocket active (legacy stopPolling called)'),
    isPolling: false
  }
}

export default useWebSocket
