/**
 * useSessionRestore - State Rehydration Hook
 * 
 * Automatically restores the active mesh, camera position, and selection state
 * from the database and localStorage upon page reload.
 * 
 * Features:
 * - Fetches the user's most recent session from /api/session/latest
 * - Reconnects to in-progress jobs via WebSocket
 * - Restores camera position and UI toggles from localStorage
 */

import { useState, useEffect, useCallback, useRef } from 'react'
import { API_BASE } from '../config'

// LocalStorage keys for persisted state
const STORAGE_KEYS = {
  CAMERA_POSITION: 'khorium_camera_position',
  CAMERA_TARGET: 'khorium_camera_target',
  VIEW_SETTINGS: 'khorium_view_settings',
  MESH_SETTINGS: 'khorium_mesh_settings',
  LAST_PROJECT_ID: 'khorium_last_project_id',
}

/**
 * Parse JSON safely from localStorage
 */
function safeParseJSON(value, fallback = null) {
  if (!value) return fallback
  try {
    return JSON.parse(value)
  } catch {
    return fallback
  }
}

/**
 * Hook to restore session state on page load
 */
export function useSessionRestore(authFetch) {
  const [isRestoring, setIsRestoring] = useState(true)
  const [restoredSession, setRestoredSession] = useState(null)
  const [error, setError] = useState(null)
  const hasRestored = useRef(false)

  const restoreSession = useCallback(async () => {
    // Prevent double restoration
    if (hasRestored.current) return
    hasRestored.current = true

    try {
      setIsRestoring(true)
      setError(null)

      // Fetch latest session from backend
      const response = await authFetch(`${API_BASE}/session/latest`)
      
      if (!response.ok) {
        if (response.status === 401) {
          // Not authenticated - expected for new users
          setIsRestoring(false)
          return null
        }
        throw new Error('Failed to fetch session')
      }

      const sessionData = await response.json()
      
      // If no project found, return empty session
      if (!sessionData.project) {
        console.log('[Session Restore] No previous session found')
        setIsRestoring(false)
        return null
      }

      // Load saved view settings from localStorage
      const viewSettings = safeParseJSON(
        localStorage.getItem(STORAGE_KEYS.VIEW_SETTINGS),
        {
          showAxes: true,
          colorMode: 'solid',
          qualityMetric: 'sicn',
          showHistogram: false,
          consoleOpen: true,
        }
      )

      // Load saved mesh settings from localStorage  
      const meshSettings = safeParseJSON(
        localStorage.getItem(STORAGE_KEYS.MESH_SETTINGS),
        {
          qualityPreset: 'Medium',
          maxElementSize: 10.0,
          minElementSize: 2.0,
          elementOrder: '1',
          meshStrategy: 'Tetrahedral (HXT)',
          curvatureAdaptive: false,
        }
      )

      // Load camera position if available
      const cameraPosition = safeParseJSON(
        localStorage.getItem(STORAGE_KEYS.CAMERA_POSITION),
        null
      )
      const cameraTarget = safeParseJSON(
        localStorage.getItem(STORAGE_KEYS.CAMERA_TARGET),
        null
      )

      const restoredData = {
        project: sessionData.project,
        hasActiveJob: sessionData.has_active_job,
        jobId: sessionData.job_id,
        internalJobId: sessionData.internal_job_id,
        status: sessionData.status,
        viewSettings,
        meshSettings,
        cameraPosition,
        cameraTarget,
      }

      console.log('[Session Restore] Restored session:', {
        projectId: sessionData.project?.id,
        status: sessionData.status,
        hasActiveJob: sessionData.has_active_job,
      })

      setRestoredSession(restoredData)
      setIsRestoring(false)
      return restoredData

    } catch (err) {
      console.error('[Session Restore] Error:', err)
      setError(err.message)
      setIsRestoring(false)
      return null
    }
  }, [authFetch])

  // Restore session on mount
  useEffect(() => {
    restoreSession()
  }, [restoreSession])

  return {
    isRestoring,
    restoredSession,
    error,
    restoreSession, // Allow manual re-trigger if needed
  }
}

/**
 * Hook to persist view and mesh settings to localStorage
 */
export function useLocalStatePersistence() {
  // Save view settings
  const saveViewSettings = useCallback((settings) => {
    localStorage.setItem(STORAGE_KEYS.VIEW_SETTINGS, JSON.stringify(settings))
  }, [])

  // Save mesh settings
  const saveMeshSettings = useCallback((settings) => {
    localStorage.setItem(STORAGE_KEYS.MESH_SETTINGS, JSON.stringify(settings))
  }, [])

  // Save camera position (call this from MeshViewer on orbit change)
  const saveCameraPosition = useCallback((position, target) => {
    if (position) {
      localStorage.setItem(STORAGE_KEYS.CAMERA_POSITION, JSON.stringify({
        x: position.x,
        y: position.y,
        z: position.z,
      }))
    }
    if (target) {
      localStorage.setItem(STORAGE_KEYS.CAMERA_TARGET, JSON.stringify({
        x: target.x,
        y: target.y,
        z: target.z,
      }))
    }
  }, [])

  // Save last project ID (for quick reference)
  const saveLastProjectId = useCallback((projectId) => {
    if (projectId) {
      localStorage.setItem(STORAGE_KEYS.LAST_PROJECT_ID, projectId)
    }
  }, [])

  // Load all persisted settings
  const loadPersistedState = useCallback(() => {
    return {
      viewSettings: safeParseJSON(localStorage.getItem(STORAGE_KEYS.VIEW_SETTINGS), null),
      meshSettings: safeParseJSON(localStorage.getItem(STORAGE_KEYS.MESH_SETTINGS), null),
      cameraPosition: safeParseJSON(localStorage.getItem(STORAGE_KEYS.CAMERA_POSITION), null),
      cameraTarget: safeParseJSON(localStorage.getItem(STORAGE_KEYS.CAMERA_TARGET), null),
      lastProjectId: localStorage.getItem(STORAGE_KEYS.LAST_PROJECT_ID),
    }
  }, [])

  // Clear persisted state (e.g., on logout)
  const clearPersistedState = useCallback(() => {
    Object.values(STORAGE_KEYS).forEach(key => {
      localStorage.removeItem(key)
    })
  }, [])

  return {
    saveViewSettings,
    saveMeshSettings,
    saveCameraPosition,
    saveLastProjectId,
    loadPersistedState,
    clearPersistedState,
  }
}

export default useSessionRestore

