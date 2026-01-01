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
 * useBatchPolling Hook
 */
export function useBatchPolling(batchId, authFetch, interval = 2000) {
  const [batch, setBatch] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const timerRef = useRef(null)
  const isPollingRef = useRef(false)

  // Check if batch is in a final state
  const isFinalState = (status) => {
    return ['completed', 'failed', 'cancelled'].includes(status)
  }

  const fetchBatch = useCallback(async () => {
    if (!batchId || !authFetch) return

    try {
      const response = await authFetch(`${API_BASE}/batch/${batchId}?include_files=true&include_jobs=true`)
      if (response.ok) {
        const data = await response.json()
        setBatch(data)
        setError(null)

        // Auto-stop polling when batch reaches final state
        if (isFinalState(data.status) && timerRef.current) {
          console.log('[Polling] Batch complete, stopping polling')
          clearInterval(timerRef.current)
          timerRef.current = null
          isPollingRef.current = false
        }
      } else {
        const err = await response.json()
        setError(err.error || 'Failed to fetch batch')
      }
    } catch (err) {
      // Only log actual errors, not routine fetches
      console.error('[Polling] Error:', err.message)
      setError(err.message)
    }
  }, [batchId, authFetch])

  const startPolling = useCallback(() => {
    if (timerRef.current || isPollingRef.current) return

    // Don't start polling if already in final state
    if (batch && isFinalState(batch.status)) return

    console.log('[Polling] Started')
    isPollingRef.current = true
    fetchBatch()
    timerRef.current = setInterval(fetchBatch, interval)
  }, [fetchBatch, interval, batch])

  const stopPolling = useCallback(() => {
    if (timerRef.current) {
      console.log('[Polling] Stopped')
      clearInterval(timerRef.current)
      timerRef.current = null
      isPollingRef.current = false
    }
  }, [])

  // Fetch immediately when batchId changes
  useEffect(() => {
    if (batchId && authFetch) {
      setLoading(true)
      setBatch(null)

      const doFetch = async () => {
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
          setError(err.message)
        } finally {
          setLoading(false)
        }
      }

      doFetch()
    } else if (!batchId) {
      setBatch(null)
      setLoading(false)
      stopPolling()
    }
  }, [batchId, authFetch, stopPolling])

  // Cleanup on unmount
  useEffect(() => {
    return () => stopPolling()
  }, [stopPolling])

  return {
    batch,
    loading,
    error,
    refresh: fetchBatch,
    startPolling,
    stopPolling,
    isPolling: isPollingRef.current
  }
}

export default useWebSocket
