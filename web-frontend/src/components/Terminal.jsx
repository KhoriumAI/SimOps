import { useEffect, useRef } from 'react'
import { Terminal as TerminalIcon } from 'lucide-react'

export default function Terminal({ logs }) {
  const scrollRef = useRef(null)

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [logs])

  const getLogClass = (log) => {
    if (log.includes('[ERROR]')) return 'text-red-400'
    if (log.includes('[SUCCESS]') || log.includes('✓')) return 'text-green-400'
    if (log.includes('[INFO]')) return 'text-blue-400'
    if (log.includes('⚠')) return 'text-yellow-400'
    return 'text-gray-300'
  }

  return (
    <div className="h-full flex flex-col bg-gray-950">
      <div className="flex items-center gap-2 px-4 py-2 border-b border-gray-700 bg-gray-900">
        <TerminalIcon className="w-4 h-4" />
        <h3 className="text-sm font-semibold">Console</h3>
        <span className="ml-auto text-xs text-gray-600">
          {logs.length} {logs.length === 1 ? 'message' : 'messages'}
        </span>
      </div>

      <div ref={scrollRef} className="flex-1 overflow-auto p-4">
        <div className="space-y-1 font-mono text-xs">
          {logs.length === 0 ? (
            <p className="text-gray-600">Waiting for mesh generation...</p>
          ) : (
            logs.map((log, i) => (
              <div key={i} className={getLogClass(log)}>
                {log}
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  )
}
