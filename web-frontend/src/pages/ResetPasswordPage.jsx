import { useState, useEffect } from 'react'
import { useSearchParams, useNavigate } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'

export default function ResetPasswordPage() {
    const [searchParams] = useSearchParams()
    const navigate = useNavigate()
    const { resetPassword, error, clearError } = useAuth()

    const token = searchParams.get('token')

    const [password, setPassword] = useState('')
    const [confirmPassword, setConfirmPassword] = useState('')
    const [isSubmitting, setIsSubmitting] = useState(false)
    const [isSuccess, setIsSuccess] = useState(false)
    const [formError, setFormError] = useState('')

    useEffect(() => {
        if (!token) {
            setFormError('Invalid or missing reset token.')
        }
    }, [token])

    const handleSubmit = async (e) => {
        e.preventDefault()
        setFormError('')
        clearError()

        if (!token) {
            setFormError('Invalid or missing reset token.')
            return
        }

        if (password.length < 8) {
            setFormError('Password must be at least 8 characters')
            return
        }

        if (password !== confirmPassword) {
            setFormError('Passwords do not match')
            return
        }

        setIsSubmitting(true)
        const result = await resetPassword(token, password)
        setIsSubmitting(false)

        if (result.success) {
            setIsSuccess(true)
            setTimeout(() => {
                navigate('/login')
            }, 3000)
        }
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
            </div>

            {/* Reset Card */}
            <div className="w-full max-w-md bg-white rounded-xl shadow-lg border border-gray-200 p-8">
                <h2 className="text-2xl font-bold text-gray-800 mb-6 text-center">
                    Set New Password
                </h2>

                {(error || formError) && (
                    <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg text-red-600 text-sm">
                        {error || formError}
                    </div>
                )}

                {isSuccess ? (
                    <div className="text-center py-4">
                        <div className="w-12 h-12 bg-green-100 text-green-600 rounded-full flex items-center justify-center mx-auto mb-4">
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                            </svg>
                        </div>
                        <p className="text-gray-700 mb-2 font-medium">Password Reset Successful!</p>
                        <p className="text-gray-500 text-sm">Redirecting to login page...</p>
                    </div>
                ) : (
                    <form onSubmit={handleSubmit} className="space-y-4">
                        <div>
                            <label className="block text-sm font-medium text-gray-600 mb-1">New Password</label>
                            <input
                                type="password"
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                                placeholder="Min 8 characters"
                                required
                                disabled={!token}
                                className="w-full bg-white border border-gray-300 rounded-lg px-4 py-3 text-gray-800 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-50"
                            />
                        </div>

                        <div>
                            <label className="block text-sm font-medium text-gray-600 mb-1">Confirm New Password</label>
                            <input
                                type="password"
                                value={confirmPassword}
                                onChange={(e) => setConfirmPassword(e.target.value)}
                                placeholder="••••••••"
                                required
                                disabled={!token}
                                className="w-full bg-white border border-gray-300 rounded-lg px-4 py-3 text-gray-800 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-50"
                            />
                        </div>

                        <button
                            type="submit"
                            disabled={isSubmitting || !token}
                            className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 disabled:cursor-not-allowed text-white font-semibold py-3 px-4 rounded-lg transition-colors shadow-sm"
                        >
                            {isSubmitting ? 'Updating...' : 'Reset Password'}
                        </button>

                        <div className="text-center mt-4">
                            <button
                                type="button"
                                onClick={() => navigate('/login')}
                                className="text-sm text-gray-500 hover:text-gray-700"
                            >
                                Cancel and return to Sign In
                            </button>
                        </div>
                    </form>
                )}
            </div>
        </div>
    )
}
