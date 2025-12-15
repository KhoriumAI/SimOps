import { createContext, useContext, useState, useEffect } from 'react'

const AuthContext = createContext(null)
// API base URL - uses proxy in development, full URL in production
const API_BASE = import.meta.env.VITE_API_URL || '/api'

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState(null)

  const clearTokens = () => {
    localStorage.removeItem('access_token')
    localStorage.removeItem('refresh_token')
  }

  // authFetch reads tokens directly from localStorage each time
  const authFetch = async (url, options = {}) => {
    const token = localStorage.getItem('access_token')
    const headers = { ...options.headers }
    
    console.log('authFetch called, token exists:', !!token)
    
    if (token) {
      headers['Authorization'] = `Bearer ${token}`
    }
    
    // Don't set Content-Type for FormData
    if (options.body && typeof options.body === 'object' && !(options.body instanceof FormData)) {
      headers['Content-Type'] = 'application/json'
      options.body = JSON.stringify(options.body)
    }

    let response = await fetch(url, { ...options, headers })

    // If unauthorized, try to refresh
    if (response.status === 401) {
      const refreshToken = localStorage.getItem('refresh_token')
      console.log('Got 401, refresh token exists:', !!refreshToken)
      
      if (refreshToken) {
        try {
          const refreshResponse = await fetch(`${API_BASE}/auth/refresh`, {
            method: 'POST',
            headers: { 'Authorization': `Bearer ${refreshToken}` }
          })

          if (refreshResponse.ok) {
            const data = await refreshResponse.json()
            localStorage.setItem('access_token', data.access_token)
            
            headers['Authorization'] = `Bearer ${data.access_token}`
            response = await fetch(url, { ...options, headers })
          } else {
            clearTokens()
            setUser(null)
          }
        } catch (err) {
          clearTokens()
          setUser(null)
        }
      } else {
        clearTokens()
        setUser(null)
      }
    }

    return response
  }

  useEffect(() => {
    const checkAuth = async () => {
      const token = localStorage.getItem('access_token')
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
      console.log('Register response:', data)

      if (!response.ok) {
        setError(data.error || 'Registration failed')
        return false
      }

      // Store tokens
      localStorage.setItem('access_token', data.access_token)
      localStorage.setItem('refresh_token', data.refresh_token)
      console.log('Tokens stored, access_token:', localStorage.getItem('access_token')?.substring(0, 20) + '...')
      
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

      const data = await response.json()
      console.log('Login response:', data)

      if (!response.ok) {
        setError(data.error || 'Login failed')
        return false
      }

      // Store tokens
      localStorage.setItem('access_token', data.access_token)
      localStorage.setItem('refresh_token', data.refresh_token)
      console.log('Tokens stored, access_token:', localStorage.getItem('access_token')?.substring(0, 20) + '...')
      
      setUser(data.user)
      return true
    } catch (error) {
      setError('Network error')
      return false
    }
  }

  const logout = async () => {
    try {
      await authFetch(`${API_BASE}/auth/logout`, { method: 'POST' })
    } catch (error) {
      console.error('Logout error:', error)
    }
    clearTokens()
    setUser(null)
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
