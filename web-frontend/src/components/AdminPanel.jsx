import { useState, useEffect, useCallback } from 'react'
import { useAuth } from '../contexts/AuthContext'
import { API_BASE } from '../config'
import { ArrowLeft, BarChart3, Users, CheckCircle, XCircle, Calendar, RefreshCw, Settings, Shield, Menu, X } from 'lucide-react'

export default function AdminPanel({ onClose }) {
  const { authFetch } = useAuth()
  const [usageData, setUsageData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [activeSection, setActiveSection] = useState('usage')
  const [sidebarOpen, setSidebarOpen] = useState(false)

  const fetchUsageStats = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await authFetch(`${API_BASE}/admin/usage`)
      if (response.ok) {
        const data = await response.json()
        setUsageData(data)
      } else {
        const errorData = await response.json()
        setError(errorData.error || 'Failed to fetch usage statistics')
      }
    } catch (err) {
      setError(err.message || 'Failed to fetch usage statistics')
    } finally {
      setLoading(false)
    }
  }, [authFetch])

  useEffect(() => {
    if (activeSection === 'usage') {
      fetchUsageStats()
    }
  }, [activeSection, fetchUsageStats])

  const menuItems = [
    { id: 'usage', label: 'Usage Statistics', icon: BarChart3 },
    // Future menu items can be added here
    // { id: 'users', label: 'User Management', icon: Users },
    // { id: 'settings', label: 'Settings', icon: Settings },
  ]

  return (
    <div className="flex-1 flex flex-col bg-gray-50 overflow-hidden">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 px-3 sm:px-6 py-3 sm:py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 sm:gap-3">
            <button
              onClick={onClose}
              className="p-1.5 sm:p-2 hover:bg-gray-100 rounded-lg transition-colors"
              title="Back to Main"
            >
              <ArrowLeft className="w-4 h-4 sm:w-5 sm:h-5 text-gray-600" />
            </button>
            {/* Mobile menu button */}
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="p-1.5 sm:p-2 hover:bg-gray-100 rounded-lg transition-colors md:hidden"
              title="Toggle Menu"
            >
              <Menu className="w-5 h-5 text-gray-600" />
            </button>
            <Shield className="w-5 h-5 sm:w-6 sm:h-6 text-blue-600" />
            <h1 className="text-lg sm:text-2xl font-bold text-gray-900">Admin Panel</h1>
          </div>
          {activeSection === 'usage' && (
            <button
              onClick={fetchUsageStats}
              disabled={loading}
              className="flex items-center gap-1.5 sm:gap-2 px-2 sm:px-4 py-1.5 sm:py-2 text-xs sm:text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <RefreshCw className={`w-3.5 h-3.5 sm:w-4 sm:h-4 ${loading ? 'animate-spin' : ''}`} />
              <span className="hidden sm:inline">Refresh</span>
            </button>
          )}
        </div>
      </div>

      {/* Mobile Sidebar Overlay */}
      {sidebarOpen && (
        <div 
          className="fixed inset-0 bg-black/50 z-40 md:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Main Layout: Sidebar + Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Sidebar */}
        <div className={`fixed md:static inset-y-0 left-0 z-50 md:z-auto w-64 bg-white border-r border-gray-200 flex flex-col transform transition-transform duration-300 ease-in-out ${
          sidebarOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'
        }`}>
          {/* Mobile sidebar header */}
          <div className="flex items-center justify-between p-4 border-b border-gray-200 md:hidden">
            <h2 className="text-lg font-semibold text-gray-900">Menu</h2>
            <button
              onClick={() => setSidebarOpen(false)}
              className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
            >
              <X className="w-5 h-5 text-gray-600" />
            </button>
          </div>
          <nav className="flex-1 p-4 space-y-1 overflow-y-auto">
            {menuItems.map((item) => {
              const Icon = item.icon
              const isActive = activeSection === item.id
              return (
                <button
                  key={item.id}
                  onClick={() => {
                    setActiveSection(item.id)
                    setSidebarOpen(false)
                  }}
                  className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-colors ${
                    isActive
                      ? 'bg-blue-50 text-blue-700 border border-blue-200'
                      : 'text-gray-700 hover:bg-gray-50'
                  }`}
                >
                  <Icon className={`w-5 h-5 ${isActive ? 'text-blue-600' : 'text-gray-500'}`} />
                  <span className="font-medium text-sm">{item.label}</span>
                </button>
              )
            })}
          </nav>
        </div>

        {/* Content Area */}
        <div className="flex-1 overflow-y-auto bg-gray-50">
          <div className="p-3 sm:p-4 md:p-6">
          {loading ? (
            <div className="flex flex-col items-center justify-center py-12">
              <RefreshCw className="w-8 h-8 text-blue-600 animate-spin mb-4" />
              <p className="text-gray-600 text-sm sm:text-base">Loading usage statistics...</p>
            </div>
          ) : error ? (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4">
              <div className="flex items-center gap-2 text-red-800">
                <XCircle className="w-5 h-5" />
                <p className="font-medium text-sm sm:text-base">Error</p>
              </div>
              <p className="text-red-600 mt-2 text-sm sm:text-base">{error}</p>
              <button
                onClick={fetchUsageStats}
                className="mt-4 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors text-sm"
              >
                Retry
              </button>
            </div>
          ) : usageData ? (
            <div className="space-y-4 sm:space-y-6">
              {/* Period Info */}
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 sm:p-4">
                <div className="flex items-center gap-2 text-blue-800">
                  <Calendar className="w-4 h-4 sm:w-5 sm:h-5" />
                  <span className="font-semibold text-sm sm:text-base">Reporting Period</span>
                </div>
                <p className="text-blue-600 mt-1 text-xs sm:text-sm">{usageData.period}</p>
              </div>

              {/* Summary Stats */}
              {usageData.top_users && usageData.top_users.length > 0 && (
                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3 sm:gap-4">
                  <div className="bg-gray-50 rounded-lg p-3 sm:p-4 border border-gray-200">
                    <p className="text-xs sm:text-sm text-gray-600 mb-1">Total Jobs</p>
                    <p className="text-xl sm:text-2xl font-bold text-gray-900">
                      {usageData.top_users.reduce((sum, u) => sum + u.job_count, 0).toLocaleString()}
                    </p>
                  </div>
                  <div className="bg-green-50 rounded-lg p-3 sm:p-4 border border-green-200">
                    <p className="text-xs sm:text-sm text-green-600 mb-1">Total Completed</p>
                    <p className="text-xl sm:text-2xl font-bold text-green-900">
                      {usageData.top_users.reduce((sum, u) => sum + u.completed, 0).toLocaleString()}
                    </p>
                  </div>
                  <div className="bg-red-50 rounded-lg p-3 sm:p-4 border border-red-200 sm:col-span-2 md:col-span-1">
                    <p className="text-xs sm:text-sm text-red-600 mb-1">Total Failed</p>
                    <p className="text-xl sm:text-2xl font-bold text-red-900">
                      {usageData.top_users.reduce((sum, u) => sum + u.failed, 0).toLocaleString()}
                    </p>
                  </div>
                </div>
              )}

              {/* Top Users Table */}
              <div>
                <div className="flex items-center gap-2 mb-3 sm:mb-4">
                  <Users className="w-4 h-4 sm:w-5 sm:h-5 text-gray-700" />
                  <h3 className="text-base sm:text-lg font-semibold text-gray-900">Top 5 Users by Job Count</h3>
                </div>

                {usageData.top_users && usageData.top_users.length > 0 ? (
                  <div className="overflow-x-auto -mx-3 sm:mx-0">
                    <div className="inline-block min-w-full align-middle">
                      <div className="overflow-hidden">
                        <table className="min-w-full border-collapse bg-white rounded-lg shadow-sm">
                          <thead>
                            <tr className="bg-gray-50 border-b border-gray-200">
                              <th className="px-2 sm:px-4 py-2 sm:py-3 text-left text-[10px] sm:text-xs font-semibold text-gray-700 uppercase tracking-wider whitespace-nowrap">
                                Rank
                              </th>
                              <th className="px-2 sm:px-4 py-2 sm:py-3 text-left text-[10px] sm:text-xs font-semibold text-gray-700 uppercase tracking-wider whitespace-nowrap">
                                User ID
                              </th>
                              <th className="px-2 sm:px-4 py-2 sm:py-3 text-left text-[10px] sm:text-xs font-semibold text-gray-700 uppercase tracking-wider whitespace-nowrap">
                                Email
                              </th>
                              <th className="px-2 sm:px-4 py-2 sm:py-3 text-right text-[10px] sm:text-xs font-semibold text-gray-700 uppercase tracking-wider whitespace-nowrap">
                                Jobs
                              </th>
                              <th className="px-2 sm:px-4 py-2 sm:py-3 text-right text-[10px] sm:text-xs font-semibold text-gray-700 uppercase tracking-wider whitespace-nowrap">
                                Done
                              </th>
                              <th className="px-2 sm:px-4 py-2 sm:py-3 text-right text-[10px] sm:text-xs font-semibold text-gray-700 uppercase tracking-wider whitespace-nowrap">
                                Failed
                              </th>
                              <th className="px-2 sm:px-4 py-2 sm:py-3 text-right text-[10px] sm:text-xs font-semibold text-gray-700 uppercase tracking-wider whitespace-nowrap">
                                Rate
                              </th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-gray-200">
                            {usageData.top_users.map((user, index) => {
                              const successRate = user.job_count > 0 
                                ? ((user.completed / user.job_count) * 100).toFixed(1)
                                : '0.0'
                              return (
                                <tr key={user.user_id} className="hover:bg-gray-50 transition-colors">
                                  <td className="px-2 sm:px-4 py-2 sm:py-3 text-xs sm:text-sm font-medium text-gray-900">
                                    <span className="inline-flex items-center justify-center w-6 h-6 sm:w-8 sm:h-8 rounded-full bg-blue-100 text-blue-800 font-semibold text-[10px] sm:text-xs">
                                      {index + 1}
                                    </span>
                                  </td>
                                  <td className="px-2 sm:px-4 py-2 sm:py-3 text-xs sm:text-sm text-gray-600 font-mono">
                                    #{user.user_id}
                                  </td>
                                  <td className="px-2 sm:px-4 py-2 sm:py-3 text-xs sm:text-sm text-gray-900 break-words max-w-[120px] sm:max-w-none">
                                    <span className="truncate block">{user.email}</span>
                                  </td>
                                  <td className="px-2 sm:px-4 py-2 sm:py-3 text-xs sm:text-sm text-gray-900 text-right font-semibold whitespace-nowrap">
                                    {user.job_count.toLocaleString()}
                                  </td>
                                  <td className="px-2 sm:px-4 py-2 sm:py-3 text-xs sm:text-sm text-green-600 text-right whitespace-nowrap">
                                    <div className="flex items-center justify-end gap-1">
                                      <CheckCircle className="w-3 h-3 sm:w-4 sm:h-4" />
                                      <span className="hidden sm:inline">{user.completed.toLocaleString()}</span>
                                      <span className="sm:hidden">{user.completed}</span>
                                    </div>
                                  </td>
                                  <td className="px-2 sm:px-4 py-2 sm:py-3 text-xs sm:text-sm text-red-600 text-right whitespace-nowrap">
                                    <div className="flex items-center justify-end gap-1">
                                      <XCircle className="w-3 h-3 sm:w-4 sm:h-4" />
                                      <span className="hidden sm:inline">{user.failed.toLocaleString()}</span>
                                      <span className="sm:hidden">{user.failed}</span>
                                    </div>
                                  </td>
                                  <td className="px-2 sm:px-4 py-2 sm:py-3 text-right whitespace-nowrap">
                                    <span className={`inline-flex items-center px-1.5 sm:px-2.5 py-0.5 rounded-full text-[10px] sm:text-xs font-medium ${
                                      parseFloat(successRate) >= 90 
                                        ? 'bg-green-100 text-green-800'
                                        : parseFloat(successRate) >= 70
                                        ? 'bg-yellow-100 text-yellow-800'
                                        : 'bg-red-100 text-red-800'
                                    }`}>
                                      {successRate}%
                                    </span>
                                  </td>
                                </tr>
                              )
                            })}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="bg-gray-50 border border-gray-200 rounded-lg p-6 sm:p-8 text-center">
                    <Users className="w-10 h-10 sm:w-12 sm:h-12 text-gray-400 mx-auto mb-3" />
                    <p className="text-gray-600 text-sm sm:text-base">No usage data available for this period.</p>
                  </div>
                )}
              </div>
            </div>
          ) : null}
          </div>
        </div>
      </div>
    </div>
  )
}

