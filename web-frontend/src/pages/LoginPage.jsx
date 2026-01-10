import { useState } from 'react'
import { useAuth } from '../contexts/AuthContext'

export default function LoginPage() {
  const { login, register, requestPasswordReset, error, clearError, isLoading } = useAuth()
  const [isRegister, setIsRegister] = useState(false)
  const [isForgotPassword, setIsForgotPassword] = useState(false)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [resetSent, setResetSent] = useState(false)

  const [formData, setFormData] = useState({
    email: '',
    password: '',
    name: '',
    confirmPassword: ''
  })
  const [formError, setFormError] = useState('')

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value })
    setFormError('')
    clearError()
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setFormError('')

    if (isForgotPassword) {
      if (!formData.email) {
        setFormError('Email is required')
        return
      }
      setIsSubmitting(true)
      const result = await requestPasswordReset(formData.email)
      setIsSubmitting(false)
      if (result.success) {
        setResetSent(true)
      }
      return
    }

    if (!formData.email || !formData.password) {
      setFormError('Email and password are required')
      return
    }

    if (isRegister) {
      if (formData.password.length < 8) {
        setFormError('Password must be at least 8 characters')
        return
      }
      if (formData.password !== formData.confirmPassword) {
        setFormError('Passwords do not match')
        return
      }
    }

    setIsSubmitting(true)

    const success = isRegister
      ? await register(formData.email, formData.password, formData.name)
      : await login(formData.email, formData.password)

    setIsSubmitting(false)
  }

  const toggleMode = () => {
    setIsRegister(!isRegister)
    setIsForgotPassword(false)
    setResetSent(false)
    setFormError('')
    clearError()
    setFormData({ email: '', password: '', name: '', confirmPassword: '' })
  }

  const toggleForgotPassword = () => {
    setIsForgotPassword(!isForgotPassword)
    setIsRegister(false)
    setResetSent(false)
    setFormError('')
    clearError()
  }

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 flex flex-col items-center justify-center p-4">
      {/* Logo and Title */}
      <div className="mb-8 text-center">
        <div className="flex items-center justify-center gap-3 mb-2">
          <img src="/img/Khorium_logo.jpg" alt="Khorium" className="h-14 w-auto" />
          <h1 className="text-4xl font-bold text-gray-800">
            MeshGen
          </h1>
        </div>
        <p className="text-gray-500">Advanced 3D Mesh Generation • Quality-Driven • Web-Optimized</p>
      </div>

      {/* Login Card */}
      <div className="w-full max-w-md bg-white rounded-xl shadow-lg border border-gray-200 p-8">
        <h2 className="text-2xl font-bold text-gray-800 mb-6 text-center">
          {isForgotPassword ? 'Reset Password' : (isRegister ? 'Create Account' : 'Welcome Back')}
        </h2>

        {(error || formError) && (
          <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg text-red-600 text-sm">
            {error || formError}
          </div>
        )}

        {resetSent ? (
          <div className="text-center py-4">
            <div className="w-12 h-12 bg-green-100 text-green-600 rounded-full flex items-center justify-center mx-auto mb-4">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
            </div>
            <p className="text-gray-700 mb-6">Check your email for a password reset link.</p>
            <button
              onClick={toggleForgotPassword}
              className="text-blue-600 hover:text-blue-700 font-medium"
            >
              Back to Sign In
            </button>
          </div>
        ) : (
          <>
            <form onSubmit={handleSubmit} className="space-y-4">
              {isRegister && (
                <div>
                  <label className="block text-sm font-medium text-gray-600 mb-1">Name (optional)</label>
                  <input
                    type="text"
                    name="name"
                    value={formData.name}
                    onChange={handleChange}
                    placeholder="Your name"
                    className="w-full bg-white border border-gray-300 rounded-lg px-4 py-3 text-gray-800 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>
              )}

              <div>
                <label className="block text-sm font-medium text-gray-600 mb-1">Email</label>
                <input
                  type="email"
                  name="email"
                  value={formData.email}
                  onChange={handleChange}
                  placeholder="you@example.com"
                  required
                  className="w-full bg-white border border-gray-300 rounded-lg px-4 py-3 text-gray-800 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
              </div>

              {!isForgotPassword && (
                <div>
                  <div className="flex justify-between items-center mb-1">
                    <label className="block text-sm font-medium text-gray-600">Password</label>
                    {!isRegister && (
                      <button
                        type="button"
                        onClick={toggleForgotPassword}
                        className="text-xs text-blue-600 hover:text-blue-700 hover:underline"
                      >
                        Forgot password?
                      </button>
                    )}
                  </div>
                  <input
                    type="password"
                    name="password"
                    value={formData.password}
                    onChange={handleChange}
                    placeholder={isRegister ? 'Min 8 characters' : '••••••••'}
                    required
                    className="w-full bg-white border border-gray-300 rounded-lg px-4 py-3 text-gray-800 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>
              )}

              {isRegister && (
                <div>
                  <label className="block text-sm font-medium text-gray-600 mb-1">Confirm Password</label>
                  <input
                    type="password"
                    name="confirmPassword"
                    value={formData.confirmPassword}
                    onChange={handleChange}
                    placeholder="••••••••"
                    required
                    className="w-full bg-white border border-gray-300 rounded-lg px-4 py-3 text-gray-800 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>
              )}

              <button
                type="submit"
                disabled={isSubmitting}
                className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 disabled:cursor-not-allowed text-white font-semibold py-3 px-4 rounded-lg transition-colors shadow-sm"
              >
                {isSubmitting ? 'Please wait...' : (isForgotPassword ? 'Send Reset Link' : (isRegister ? 'Create Account' : 'Sign In'))}
              </button>
            </form>

            <div className="mt-6 text-center">
              <span className="text-gray-500">
                {isForgotPassword ? 'Remember your password?' : (isRegister ? 'Already have an account?' : "Don't have an account?")}
              </span>
              <button
                onClick={isForgotPassword ? toggleForgotPassword : toggleMode}
                className="ml-2 text-blue-600 hover:text-blue-700 font-medium"
              >
                {isForgotPassword ? 'Sign In' : (isRegister ? 'Sign In' : 'Create One')}
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  )
}
