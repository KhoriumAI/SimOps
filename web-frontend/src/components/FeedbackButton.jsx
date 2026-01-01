import { useState } from 'react'
import { MessageCircle, X, Send, Bug, Lightbulb, AlertCircle } from 'lucide-react'
import { API_BASE } from '../config'

export default function FeedbackButton({ userEmail, jobId }) {
    const [isOpen, setIsOpen] = useState(false)
    const [feedbackType, setFeedbackType] = useState('feedback')
    const [message, setMessage] = useState('')
    const [isSending, setIsSending] = useState(false)
    const [sent, setSent] = useState(false)

    const handleSubmit = async (e) => {
        e.preventDefault()
        if (!message.trim()) return

        setIsSending(true)
        try {
            const token = localStorage.getItem('token')
            const response = await fetch(`${API_BASE}/feedback`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify({
                    type: feedbackType,
                    message: message.trim(),
                    userEmail: userEmail,
                    url: window.location.href,
                    userAgent: navigator.userAgent,
                    timestamp: new Date().toISOString(),
                    jobId: jobId || null  // Include Job ID if available
                })
            })

            if (response.ok) {
                setSent(true)
                setTimeout(() => {
                    setIsOpen(false)
                    setSent(false)
                    setMessage('')
                    setFeedbackType('feedback')
                }, 2000)
            } else {
                alert('Failed to send feedback. Please try again.')
            }
        } catch (err) {
            console.error('Feedback error:', err)
            alert('Failed to send feedback. Please try again.')
        } finally {
            setIsSending(false)
        }
    }

    const typeOptions = [
        { id: 'feedback', label: 'Feedback', icon: MessageCircle, color: 'blue' },
        { id: 'bug', label: 'Bug Report', icon: Bug, color: 'red' },
        { id: 'feature', label: 'Feature Request', icon: Lightbulb, color: 'yellow' }
    ]

    return (
        <>
            {/* Floating Feedback Button */}
            <button
                onClick={() => setIsOpen(true)}
                className="fixed bottom-4 right-4 z-50 bg-blue-600 hover:bg-blue-700 text-white p-3 rounded-full shadow-lg transition-all hover:scale-110"
                title="Send Feedback"
            >
                <MessageCircle className="w-5 h-5" />
            </button>

            {/* Feedback Modal */}
            {isOpen && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
                    <div className="bg-white rounded-xl shadow-2xl w-full max-w-md mx-4 overflow-hidden">
                        {/* Header */}
                        <div className="bg-gradient-to-r from-blue-600 to-blue-700 px-5 py-4 flex items-center justify-between">
                            <h2 className="text-white font-semibold text-lg">Send Feedback</h2>
                            <button
                                onClick={() => setIsOpen(false)}
                                className="text-white/80 hover:text-white transition-colors"
                            >
                                <X className="w-5 h-5" />
                            </button>
                        </div>

                        {sent ? (
                            <div className="p-8 text-center">
                                <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
                                    <AlertCircle className="w-8 h-8 text-green-600" />
                                </div>
                                <h3 className="text-lg font-medium text-gray-900 mb-2">Thank you!</h3>
                                <p className="text-gray-500">Your feedback has been sent to our team.</p>
                            </div>
                        ) : (
                            <form onSubmit={handleSubmit} className="p-5 space-y-4">
                                {/* Type Selection */}
                                <div>
                                    <label className="text-sm font-medium text-gray-700 block mb-2">Type</label>
                                    <div className="flex gap-2">
                                        {typeOptions.map(({ id, label, icon: Icon, color }) => (
                                            <button
                                                key={id}
                                                type="button"
                                                onClick={() => setFeedbackType(id)}
                                                className={`flex-1 flex items-center justify-center gap-2 py-2 px-3 rounded-lg border-2 transition-all text-sm font-medium ${feedbackType === id
                                                    ? `border-${color}-500 bg-${color}-50 text-${color}-700`
                                                    : 'border-gray-200 text-gray-600 hover:border-gray-300'
                                                    }`}
                                                style={{
                                                    borderColor: feedbackType === id ? (color === 'blue' ? '#3b82f6' : color === 'red' ? '#ef4444' : '#eab308') : undefined,
                                                    backgroundColor: feedbackType === id ? (color === 'blue' ? '#eff6ff' : color === 'red' ? '#fef2f2' : '#fefce8') : undefined,
                                                    color: feedbackType === id ? (color === 'blue' ? '#1d4ed8' : color === 'red' ? '#b91c1c' : '#a16207') : undefined
                                                }}
                                            >
                                                <Icon className="w-4 h-4" />
                                                {label}
                                            </button>
                                        ))}
                                    </div>
                                </div>

                                {/* Message */}
                                <div>
                                    <label className="text-sm font-medium text-gray-700 block mb-2">Message</label>
                                    <textarea
                                        value={message}
                                        onChange={(e) => setMessage(e.target.value)}
                                        placeholder={
                                            feedbackType === 'bug'
                                                ? 'Describe the bug: What happened? What did you expect?'
                                                : feedbackType === 'feature'
                                                    ? 'Describe the feature you would like to see...'
                                                    : 'Share your thoughts, suggestions, or comments...'
                                        }
                                        className="w-full h-32 px-3 py-2 border border-gray-300 rounded-lg text-sm resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white text-gray-900"
                                        required
                                    />
                                </div>

                                {/* Submit */}
                                <button
                                    type="submit"
                                    disabled={isSending || !message.trim()}
                                    className="w-full flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white font-medium py-2.5 rounded-lg transition-colors"
                                >
                                    {isSending ? (
                                        <>Sending...</>
                                    ) : (
                                        <>
                                            <Send className="w-4 h-4" />
                                            Send Feedback
                                        </>
                                    )}
                                </button>

                                <p className="text-xs text-gray-400 text-center">
                                    {jobId && <span className="block mb-1 text-gray-500">Job ID: <code className="bg-gray-100 px-1 rounded">{jobId}</code></span>}
                                    Your feedback helps us improve. Thank you!
                                </p>
                            </form>
                        )}
                    </div>
                </div>
            )}
        </>
    )
}
