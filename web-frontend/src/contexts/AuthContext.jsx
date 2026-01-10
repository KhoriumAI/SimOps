import { createContext, useContext, useState, useEffect, useRef } from 'react'
import { API_BASE } from '../config'

const AuthContext = createContext(null)
// API base URL - uses proxy in development, full URL in production


export function AuthProvider({ children }) {
  const [user, setUser] = useState(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState(null)

  // Refresh lock to prevent thundering herd problem
  // When multiple requests get 401 simultaneously, only the first one refreshes
  // Others wait for that refresh to complete
  const refreshPromiseRef = useRef(null)

  const clearTokens = () => {
    localStorage.removeItem('token')
    localStorage.removeItem('refresh_token')
    localStorage.removeItem('user')
  }

  // Centralized refresh logic with lock to prevent thundering herd
  // 
  // WHY: When multiple requests get 401 simultaneously, they all try to refresh.
  // With refresh token rotation, only the first succeeds - others fail because
  // the old refresh token is already invalidated, causing unwanted logout.
  //
  // HOW: Store the refresh promise in a ref. If a refresh is already in progress,
  // return the existing promise so all callers wait for the same result.
  //
  const refreshAccessToken = async () => {
    // If a refresh is already in progress, wait for it instead of starting another
    if (refreshPromiseRef.current) {
      return refreshPromiseRef.current
    }

    const refreshToken = localStorage.getItem('refresh_token')
    if (!refreshToken) {
      return { success: false, reason: 'no_refresh_token' }
    }

    // Create the refresh promise and store it
    const doRefresh = async () => {
      try {
        const refreshResponse = await fetch(`${API_BASE}/auth/refresh`, {
          method: 'POST',
          headers: { 'Authorization': `Bearer ${refreshToken}` }
        })

        if (refreshResponse.ok) {
          const data = await refreshResponse.json()
          localStorage.setItem('token', data.access_token)
          return { success: true, token: data.access_token }
        } else {
          // Refresh token is invalid/expired - user must re-login
          clearTokens()
          setUser(null)
          return { success: false, reason: 'refresh_token_invalid' }
        }
      } catch (err) {
        // Network error - don't clear tokens, might be temporary
        // User can retry, or will be logged out on next failed attempt
        console.error('Token refresh network error:', err.message)
        return { success: false, reason: 'network_error' }
      }
    }

    // Store promise, execute, then clear
    refreshPromiseRef.current = doRefresh()

    try {
      return await refreshPromiseRef.current
    } finally {
      // Clear lock immediately after completion
      // Any new 401s after this will trigger a fresh refresh attempt
      refreshPromiseRef.current = null
    }
  }

  // authFetch reads tokens directly from localStorage each time
  const authFetch = async (url, options = {}) => {
    const token = localStorage.getItem('token')
    const headers = {
      ...options.headers,
      'Authorization': `Bearer ${token}`,
    }

    // Don't set Content-Type for FormData if not explicitly set
    if (!headers['Content-Type'] && options.body && typeof options.body === 'object' && !(options.body instanceof FormData)) {
      headers['Content-Type'] = 'application/json'
      options.body = JSON.stringify(options.body)
    }

    let response;
    try {
      response = await fetch(url, { ...options, headers })
    } catch (err) {
      console.error('Fetch error:', err)
      throw err
    }

    if (response.status === 401) {
      const refreshResult = await refreshAccessToken()

      if (refreshResult.success) {
        // Retry with new token
        headers['Authorization'] = `Bearer ${refreshResult.token}`
        response = await fetch(url, { ...options, headers })
      }
      // If refresh failed, user is already logged out by refreshAccessToken()
    }

    return response
  }

  useEffect(() => {
    const checkAuth = async () => {
      const token = localStorage.getItem('token')
      if (!token) {
        setIsLoading(false)
        return
      }

      try {
        const response = await authFetch(`${API_BASE}/auth/me`)
        if (response.ok) {
          const data = await response.json()
          setUser(data.user)
        } else {
          clearTokens()
          setUser(null)
        }
      } catch (error) {
        clearTokens()
        setUser(null)
      }

      setIsLoading(false)
    }

    checkAuth()
  }, [])

  const register = async (email, password, name = '') => {
    setError(null)

    try {
      const response = await fetch(`${API_BASE}/auth/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password, name })
      })

      const data = await response.json()

      if (!response.ok) {
        setError(data.error || 'Registration failed')
        return false
      }

      // Store tokens
      localStorage.setItem('token', data.access_token)
      localStorage.setItem('refresh_token', data.refresh_token)
      localStorage.setItem('user', JSON.stringify(data.user))

      setUser(data.user)
      return true
    } catch (error) {
      setError('Network error')
      return false
    }
  }

  const login = async (email, password) => {
    setError(null)

    try {
      const response = await fetch(`${API_BASE}/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
      })

      const contentType = response.headers.get("content-type");
      if (response.ok && contentType && contentType.includes("application/json")) {
        const data = await response.json()
        localStorage.setItem('token', data.access_token)
        localStorage.setItem('refresh_token', data.refresh_token)
        localStorage.setItem('user', JSON.stringify(data.user))
        setUser(data.user)
        return true
      } else {
        let errorMessage = 'Login failed';
        if (contentType && contentType.includes("application/json")) {
          const errorData = await response.json();
          errorMessage = errorData.error || errorMessage;
        } else if (response.status === 405) {
          errorMessage = 'Method Not Allowed: Check if API endpoint is correct (S3 does not support POST).';
        }
        setError(errorMessage)
        return false
      }
    } catch (error) {
      console.error('Login error:', error)
      setError('Network error or server unreachable')
      return false
    }
  }

  const logout = async () => {
    try {
      // Send refresh token to backend so it can be revoked too
      const refreshToken = localStorage.getItem('refresh_token')
      await authFetch(`${API_BASE}/auth/logout`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ refresh_token: refreshToken })
      })
    } catch (error) {
      console.error('Logout error:', error)
    }
    clearTokens()
    setUser(null)
  }

  const requestPasswordReset = async (email) => {
    setError(null)
    try {
      const response = await fetch(`${API_BASE}/auth/forgot-password`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email })
      })
      const data = await response.json()
      if (!response.ok) {
        setError(data.error || 'Request failed')
        return { success: false, error: data.error }
      }
      return { success: true, message: data.message }
    } catch (error) {
      setError('Network error')
      return { success: false, error: 'Network error' }
    }
  }

  const resetPassword = async (token, password) => {
    setError(null)
    try {
      const response = await fetch(`${API_BASE}/auth/reset-password`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token, password })
      })
      const data = await response.json()
      if (!response.ok) {
        setError(data.error || 'Reset failed')
        return { success: false, error: data.error }
      }
      return { success: true, message: data.message }
    } catch (error) {
      setError('Network error')
      return { success: false, error: 'Network error' }
    }
  }

  return (
    <AuthContext.Provider value={{
      user,
      isLoading,
      isAuthenticated: !!user,
      error,
      login,
      logout,
      register,
      requestPasswordReset,
      resetPassword,
      authFetch,
      clearError: () => setError(null)
    }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}

export default AuthContext
