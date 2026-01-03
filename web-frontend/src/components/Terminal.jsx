import { useEffect, useRef } from 'react'
import { Terminal as TerminalIcon, Copy } from 'lucide-react'

export default function Terminal({ logs, compact = false, noHeader = false }) {
  const scrollRef = useRef(null)

  const isAutoScroll = useRef(true)

  useEffect(() => {
    if (scrollRef.current && isAutoScroll.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [logs])

  const handleScroll = () => {
    if (scrollRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = scrollRef.current
      const atBottom = scrollHeight - scrollTop - clientHeight < 50
      isAutoScroll.current = atBottom
    }
  }

  const getLogClass = (log) => {
    if (log.includes('[ERROR]')) return 'text-red-400'
    if (log.includes('[SUCCESS]') || log.includes('✓')) return 'text-green-400'
    if (log.includes('[INFO]')) return 'text-blue-400'
    if (log.includes('⚠')) return 'text-yellow-400'
    if (log.includes('📐') || log.includes('📏') || log.includes('📦')) return 'text-cyan-400'
    if (log.startsWith('   •')) return 'text-gray-300'
    return 'text-gray-400'
  }

  const copyToClipboard = () => {
    const text = logs.join('\n')
    navigator.clipboard.writeText(text)
  }

  if (compact) {
    return (
      <div className="h-full flex flex-col bg-gray-800">
        <div className="flex items-center gap-1 px-2 py-1 border-b border-gray-700 bg-gray-700">
          <TerminalIcon className="w-3 h-3 text-gray-400" />
          <span className="text-[10px] font-medium text-gray-400">Console</span>
          <button
            onClick={copyToClipboard}
            className="ml-auto p-0.5 hover:bg-gray-600 rounded text-gray-400"
            title="Copy"
          >
            <Copy className="w-3 h-3" />
          </button>
        </div>
        <div ref={scrollRef} onScroll={handleScroll} className="flex-1 overflow-auto p-2">
          <div className="space-y-0.5 font-mono text-[10px]">
            {logs.length === 0 ? (
              <p className="text-gray-500">Waiting...</p>
            ) : (
              logs.slice(-50).map((log, i) => (
                <div key={i} className={`${getLogClass(log)} truncate`} title={log}>
                  {log}
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="h-full flex flex-col bg-gray-900">
      {!noHeader && (
        <div className="flex items-center gap-2 px-4 py-2 border-b border-gray-700 bg-gray-800">
          <TerminalIcon className="w-4 h-4 text-gray-400" />
          <h3 className="text-sm font-semibold text-gray-300">Console</h3>
          <span className="ml-auto text-xs text-gray-500">
            {logs.length} {logs.length === 1 ? 'message' : 'messages'}
          </span>
          <button
            onClick={copyToClipboard}
            className="ml-2 px-2 py-1 text-xs bg-blue-600 hover:bg-blue-700 text-white rounded flex items-center gap-1"
          >
            <Copy className="w-3 h-3" />
            Copy Console
          </button>
        </div>
      )}

      <div ref={scrollRef} onScroll={handleScroll} className="flex-1 overflow-auto p-4">
        <div className="space-y-0.5 font-mono text-xs">
          {logs.length === 0 ? (
            <p className="text-gray-500">Waiting for mesh generation...</p>
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
