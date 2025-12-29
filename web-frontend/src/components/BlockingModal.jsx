import { Loader2 } from 'lucide-react'

/**
 * Blocking Modal Component
 * Prevents user interaction during critical operations like file uploads and mesh generation.
 * Displays a prominent spinner and message to keep users informed.
 */
export default function BlockingModal({ isOpen, title, message, subMessage }) {
    if (!isOpen) return null

    return (
        <div className="fixed inset-0 bg-black bg-opacity-80 z-50 flex flex-col items-center justify-center">
            {/* Animated Spinner */}
            <div className="relative mb-6">
                {/* Outer ring */}
                <div className="w-24 h-24 border-4 border-blue-500/30 rounded-full animate-ping absolute"></div>

                {/* Main spinner */}
                <div className="w-24 h-24 border-t-4 border-blue-500 rounded-full animate-spin"></div>

                {/* Inner pulse */}
                <div className="absolute inset-0 flex items-center justify-center">
                    <div className="w-12 h-12 bg-blue-500/20 rounded-full animate-pulse"></div>
                </div>
            </div>

            {/* Title */}
            <h2 className="text-white text-2xl font-semibold mb-3">
                {title || 'Processing...'}
            </h2>

            {/* Main Message */}
            <p className="text-gray-300 text-lg mb-2">
                {message || 'Please wait...'}
            </p>

            {/* Sub Message / Warning */}
            {subMessage && (
                <p className="text-gray-400 text-sm max-w-md text-center px-4">
                    {subMessage}
                </p>
            )}

            {/* Visual indicator bar */}
            <div className="mt-8 w-64 h-1 bg-gray-700 rounded-full overflow-hidden">
                <div className="h-full bg-gradient-to-r from-blue-500 to-purple-500 animate-pulse"></div>
            </div>
        </div>
    )
}
