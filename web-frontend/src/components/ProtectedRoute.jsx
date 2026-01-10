import { useAuth } from '../contexts/AuthContext'
import LoginPage from '../pages/LoginPage'
import ResetPasswordPage from '../pages/ResetPasswordPage'
import { Routes, Route, Navigate, useLocation } from 'react-router-dom'

export default function ProtectedRoute({ children }) {
  const { isAuthenticated, isLoading } = useAuth()
  const location = useLocation()

  if (isLoading) {
    return (
      <div className="min-h-screen bg-white flex items-center justify-center">
        <div className="text-center">
          <div className="w-12 h-12 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-gray-500 font-medium">Initialising...</p>
        </div>
      </div>
    )
  }

  // If authenticated, we show the app (children)
  if (isAuthenticated) {
    // If they try to go to login or reset-password while logged in, redirect them
    if (location.pathname === '/login' || location.pathname === '/reset-password') {
      return <Navigate to="/" replace />
    }
    return children
  }

  // If not authenticated, we handle unauthenticated routes
  return (
    <Routes>
      <Route path="/reset-password" element={<ResetPasswordPage />} />
      <Route path="/login" element={<LoginPage />} />
      <Route path="*" element={<Navigate to="/login" replace />} />
    </Routes>
  )
}
