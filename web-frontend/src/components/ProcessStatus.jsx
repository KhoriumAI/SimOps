import { CheckCircle, Circle, Loader2, AlertCircle } from 'lucide-react'

const STATUS_STEPS = [
  { id: 'uploaded', label: 'File Uploaded' },
  { id: 'processing', label: 'Generating Mesh' },
  { id: 'completed', label: 'Mesh Complete' }
]

export default function ProcessStatus({ status }) {
  if (!status) {
    return (
      <div className="text-center text-gray-500 text-sm">
        No active project
      </div>
    )
  }

  const getStepIcon = (stepId) => {
    const currentStatus = status.status
    const stepIndex = STATUS_STEPS.findIndex(s => s.id === stepId)
    const currentIndex = STATUS_STEPS.findIndex(s => s.id === currentStatus)

    if (currentStatus === 'error') {
      if (stepIndex <= 0) return <CheckCircle className="w-5 h-5 text-green-500" />
      return <AlertCircle className="w-5 h-5 text-red-500" />
    }

    // Completed shows all checkmarks
    if (currentStatus === 'completed') {
      return <CheckCircle className="w-5 h-5 text-green-500" />
    }

    // Previous steps are complete
    if (stepIndex < currentIndex) {
      return <CheckCircle className="w-5 h-5 text-green-500" />
    }

    // Current step: 'uploaded' is complete state, 'processing' is active state
    if (stepIndex === currentIndex) {
      if (currentStatus === 'uploaded') {
        return <CheckCircle className="w-5 h-5 text-green-500" />
      }
      return <Loader2 className="w-5 h-5 animate-spin text-blue-500" />
    }

    return <Circle className="w-5 h-5 text-gray-600" />
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="font-semibold text-sm text-gray-700">Status</h3>
        <span className={`text-xs px-2 py-1 rounded font-medium ${
          status.status === 'completed'
            ? 'bg-green-100 text-green-700'
            : status.status === 'error'
            ? 'bg-red-100 text-red-700'
            : 'bg-blue-100 text-blue-700'
        }`}>
          {status.status.replace(/_/g, ' ').toUpperCase()}
        </span>
      </div>

      <div className="space-y-3">
        {STATUS_STEPS.map(step => (
          <div key={step.id} className="flex items-center gap-3">
            {getStepIcon(step.id)}
            <span className="text-sm text-gray-700">{step.label}</span>
          </div>
        ))}
      </div>

      {status.error_message && (
        <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-md">
          <p className="text-sm text-red-600">{status.error_message}</p>
        </div>
      )}

      {status.strategy && (
        <div className="mt-4 p-3 bg-gray-100 rounded-md text-xs">
          <div className="text-gray-500">Strategy</div>
          <div className="text-gray-800 font-mono">{status.strategy}</div>
        </div>
      )}
    </div>
  )
}
