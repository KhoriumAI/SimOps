import { useState, useEffect, useCallback, useRef } from 'react'
import { io } from 'socket.io-client'

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

    const wsUrl = import.meta.env.VITE_WS_URL || window.location.origin

    try {
      socketRef.current = io(wsUrl, {
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
 * useBatchPolling Hook
 * 
 * Fallback polling for batch status when WebSocket is unavailable.
 */
const API_BASE = import.meta.env.VITE_API_URL || '/api'

export function useBatchPolling(batchId, authFetch, interval = 2000) {
  const [batch, setBatch] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const timerRef = useRef(null)

  const fetchBatch = useCallback(async () => {
    if (!batchId || !authFetch) return

    try {
      console.log('[useBatchPolling] Fetching batch:', batchId)
      const response = await authFetch(`${API_BASE}/batch/${batchId}?include_files=true&include_jobs=true`)
      console.log('[useBatchPolling] Response status:', response.status)
      if (response.ok) {
        const data = await response.json()
        console.log('[useBatchPolling] Batch data:', data)
        setBatch(data)
        setError(null)
      } else {
        const err = await response.json()
        console.error('[useBatchPolling] Error response:', err)
        setError(err.error || 'Failed to fetch batch')
      }
    } catch (err) {
      console.error('[useBatchPolling] Fetch error:', err)
      setError(err.message)
    }
  }, [batchId, authFetch])

  const startPolling = useCallback(() => {
    if (timerRef.current) return

    fetchBatch()
    timerRef.current = setInterval(fetchBatch, interval)
  }, [fetchBatch, interval])

  const stopPolling = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current)
      timerRef.current = null
    }
  }, [])

  // Fetch immediately when batchId changes
  useEffect(() => {
    if (batchId && authFetch) {
      console.log('[useBatchPolling] batchId changed, fetching:', batchId)
      setLoading(true)
      setBatch(null) // Clear old batch while loading
      
      const doFetch = async () => {
        try {
          const response = await authFetch(`${API_BASE}/batch/${batchId}?include_files=true&include_jobs=true`)
          console.log('[useBatchPolling] Initial fetch response:', response.status)
          if (response.ok) {
            const data = await response.json()
            console.log('[useBatchPolling] Initial fetch data:', data)
            setBatch(data)
            setError(null)
          } else {
            const err = await response.json()
            setError(err.error || 'Failed to fetch batch')
          }
        } catch (err) {
          console.error('[useBatchPolling] Initial fetch error:', err)
          setError(err.message)
        } finally {
          setLoading(false)
        }
      }
      
      doFetch()
    } else if (!batchId) {
      setBatch(null)
      setLoading(false)
    }
  }, [batchId, authFetch])

  useEffect(() => {
    return () => stopPolling()
  }, [stopPolling])

  return {
    batch,
    loading,
    error,
    refresh: fetchBatch,
    startPolling,
    stopPolling
  }
}

export default useWebSocket
