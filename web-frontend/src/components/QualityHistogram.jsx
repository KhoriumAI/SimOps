import { useMemo } from 'react'
import { BarChart3, TrendingUp, TrendingDown, Minus } from 'lucide-react'

/**
 * Quality Histogram Component
 * Shows the distribution of element quality (SICN values)
 */
export default function QualityHistogram({ histogramData: serverHistogramData, qualityData, qualityMetrics, isVisible = true }) {
  // Use server-provided histogram data, or generate from raw/metrics
  const histogramData = useMemo(() => {
    // Prefer server-provided histogram data (real data)
    if (serverHistogramData && serverHistogramData.bins) {
      return {
        ...serverHistogramData,
        isApproximate: false,
      }
    }
    // If we have raw quality data, use it
    if (qualityData && qualityData.length > 0) {
      return generateFromRawData(qualityData)
    }
    // Otherwise generate from summary metrics (estimated)
    if (qualityMetrics) {
      return generateFromMetrics(qualityMetrics)
    }
    return null
  }, [serverHistogramData, qualityData, qualityMetrics])

  // Generate from raw quality values
  function generateFromRawData(data) {
    const bins = 10
    const histogram = new Array(bins).fill(0)
    let validCount = 0
    let invalidCount = 0
    
    data.forEach(value => {
      if (value < 0) {
        invalidCount++
      } else {
        validCount++
        const binIndex = Math.min(Math.floor(value * 10), bins - 1)
        histogram[binIndex]++
      }
    })

    const maxCount = Math.max(...histogram)
    
    return {
      bins: histogram.map((count, i) => ({
        rangeStart: i / 10,
        count,
        normalized: maxCount > 0 ? count / maxCount : 0,
      })),
      totalElements: data.length,
      validCount,
      invalidCount,
      maxCount,
      isApproximate: false,
    }
  }

  // Generate approximate histogram from summary metrics
  function generateFromMetrics(metrics) {
    if (!metrics) return null
    
    const { min_sicn, max_sicn, avg_sicn, element_count } = metrics
    
    const mean = avg_sicn ?? 0.5
    const minVal = min_sicn ?? Math.max(0, mean - 0.3)
    const maxVal = max_sicn ?? Math.min(1, mean + 0.3)
    
    const bins = 10
    const histogram = []
    const sigma = Math.max(0.1, (maxVal - minVal) / 3)
    
    // Generate Gaussian distribution
    for (let i = 0; i < bins; i++) {
      const binCenter = (i + 0.5) / 10
      let value = 0
      
      // Only generate values within the data range
      if (binCenter >= minVal - 0.1 && binCenter <= maxVal + 0.1) {
        value = Math.exp(-Math.pow(binCenter - mean, 2) / (2 * sigma * sigma))
      }
      
      histogram.push(value)
    }
    
    // Normalize
    const maxVal2 = Math.max(...histogram)
    
    return {
      bins: histogram.map((value, i) => ({
        rangeStart: i / 10,
        count: Math.round((value / maxVal2) * (element_count || 100)),
        normalized: maxVal2 > 0 ? value / maxVal2 : 0,
      })),
      totalElements: element_count || 0,
      validCount: element_count || 0,
      invalidCount: 0,
      maxCount: element_count || 100,
      isApproximate: true,
    }
  }

  // Quality assessment
  const qualityAssessment = useMemo(() => {
    if (!qualityMetrics?.avg_sicn) return null
    
    const avg = qualityMetrics.avg_sicn
    if (avg >= 0.7) return { label: 'Excellent', icon: TrendingUp, color: 'text-green-400' }
    if (avg >= 0.5) return { label: 'Good', icon: TrendingUp, color: 'text-blue-400' }
    if (avg >= 0.3) return { label: 'Fair', icon: Minus, color: 'text-yellow-400' }
    if (avg >= 0.1) return { label: 'Poor', icon: TrendingDown, color: 'text-orange-400' }
    return { label: 'Bad', icon: TrendingDown, color: 'text-red-400' }
  }, [qualityMetrics])

  if (!isVisible) return null
  
  // No data message
  if (!histogramData && !qualityMetrics) {
    return (
      <div className="bg-gray-900/95 backdrop-blur rounded-lg p-3 min-w-[240px] border border-gray-700 shadow-xl">
        <div className="flex items-center gap-2 mb-2">
          <BarChart3 className="w-4 h-4 text-blue-400" />
          <span className="font-medium text-white text-sm">Quality Distribution</span>
        </div>
        <div className="text-gray-400 text-xs text-center py-4">
          <div className="mb-2">📊 No quality data available</div>
          <div className="text-[10px]">
            Click <span className="text-blue-400 font-medium">"Generate Mesh"</span> to create<br/>
            a mesh and view quality metrics.
          </div>
        </div>
      </div>
    )
  }

  const Icon = qualityAssessment?.icon || BarChart3
  const barHeight = 80 // SVG height for bars
  const barWidth = 20  // Width of each bar
  const barGap = 4     // Gap between bars

  return (
    <div className="bg-gray-900/95 backdrop-blur rounded-lg p-3 text-xs text-gray-300 w-64 shadow-xl border border-gray-700">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <BarChart3 className="w-4 h-4 text-blue-400" />
          <span className="font-medium text-white text-sm">Quality Distribution</span>
        </div>
        {qualityAssessment && (
          <div className={`flex items-center gap-1 ${qualityAssessment.color}`}>
            <Icon className="w-3 h-3" />
            <span className="text-[10px] font-medium">{qualityAssessment.label}</span>
          </div>
        )}
      </div>

      {/* SVG Histogram */}
      <div className="bg-gray-800 rounded-lg p-2 mb-2">
        <svg width="100%" height={barHeight + 20} viewBox={`0 0 ${10 * (barWidth + barGap)} ${barHeight + 20}`}>
          {/* Grid lines */}
          {[0, 0.25, 0.5, 0.75, 1].map((y, i) => (
            <line
              key={i}
              x1="0"
              y1={barHeight * (1 - y)}
              x2={10 * (barWidth + barGap)}
              y2={barHeight * (1 - y)}
              stroke="#374151"
              strokeWidth="1"
              strokeDasharray={y > 0 && y < 1 ? "2,2" : "0"}
            />
          ))}
          
          {/* Bars */}
          {histogramData?.bins?.map((bin, i) => {
            const hasData = bin.normalized > 0
            const height = hasData ? Math.max(bin.normalized * barHeight, 4) : 2
            const x = i * (barWidth + barGap)
            const y = barHeight - height
            const color = hasData ? getBarColor(bin.rangeStart) : '#374151' // Consistent gray for empty
            
            return (
              <g key={i}>
                <rect
                  x={x}
                  y={y}
                  width={barWidth}
                  height={height}
                  fill={color}
                  rx="2"
                  opacity={hasData ? 1 : 0.5}
                  className="transition-all duration-200 hover:brightness-125"
                />
                {/* Label below bar */}
                <text
                  x={x + barWidth / 2}
                  y={barHeight + 12}
                  textAnchor="middle"
                  fill="#6B7280"
                  fontSize="8"
                >
                  {(bin.rangeStart).toFixed(1)}
                </text>
              </g>
            )
          })}
        </svg>
      </div>

      {/* Legend */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-1">
          <div className="w-12 h-2 rounded" style={{ background: 'linear-gradient(to right, #ef4444, #f97316, #facc15, #84cc16, #22c55e)' }} />
          <span className="text-[10px] text-gray-500">Bad → Good</span>
        </div>
        {histogramData?.isApproximate ? (
          <span className="text-[10px] text-yellow-500 bg-yellow-500/10 px-1.5 py-0.5 rounded">~Estimated</span>
        ) : histogramData && (
          <span className="text-[10px] text-green-500 bg-green-500/10 px-1.5 py-0.5 rounded">✓ Real Data</span>
        )}
      </div>

      {/* Quality Metrics Summary */}
      {qualityMetrics && (
        <div className="border-t border-gray-700 pt-2 mt-2">
          <div className="grid grid-cols-3 gap-2 text-center">
            <div>
              <div className="text-[10px] text-gray-500">Min</div>
              <div className={`font-mono font-medium ${getQualityColor(qualityMetrics.min_sicn)}`}>
                {qualityMetrics.min_sicn?.toFixed(3) || 'N/A'}
              </div>
            </div>
            <div>
              <div className="text-[10px] text-gray-500">Avg</div>
              <div className={`font-mono font-medium ${getQualityColor(qualityMetrics.avg_sicn)}`}>
                {qualityMetrics.avg_sicn?.toFixed(3) || 'N/A'}
              </div>
            </div>
            <div>
              <div className="text-[10px] text-gray-500">Max</div>
              <div className={`font-mono font-medium ${getQualityColor(qualityMetrics.max_sicn)}`}>
                {qualityMetrics.max_sicn?.toFixed(3) || 'N/A'}
              </div>
            </div>
          </div>
          
          {qualityMetrics.element_count && (
            <div className="text-center mt-2 text-[10px] text-gray-500">
              {qualityMetrics.element_count.toLocaleString()} elements
            </div>
          )}
        </div>
      )}

      {/* Quality Scale Guide */}
      <div className="border-t border-gray-700 pt-2 mt-2">
        <div className="text-[10px] text-gray-500 mb-1">SICN Quality Scale:</div>
        <div className="flex h-2 rounded overflow-hidden">
          <div className="flex-1 bg-red-500" title="0.0-0.2: Bad"></div>
          <div className="flex-1 bg-orange-500" title="0.2-0.4: Poor"></div>
          <div className="flex-1 bg-yellow-500" title="0.4-0.6: Fair"></div>
          <div className="flex-1 bg-lime-500" title="0.6-0.8: Good"></div>
          <div className="flex-1 bg-green-500" title="0.8-1.0: Great"></div>
        </div>
        <div className="flex justify-between text-[8px] text-gray-500 mt-0.5">
          <span>0</span>
          <span>0.2</span>
          <span>0.4</span>
          <span>0.6</span>
          <span>0.8</span>
          <span>1.0</span>
        </div>
      </div>
    </div>
  )
}

// Get bar color based on quality value
function getBarColor(value) {
  if (value < 0.2) return '#ef4444' // Red
  if (value < 0.4) return '#f97316' // Orange
  if (value < 0.6) return '#facc15' // Yellow
  if (value < 0.8) return '#84cc16' // Lime
  return '#22c55e' // Green
}

// Get text color based on quality value
function getQualityColor(value) {
  if (value === undefined || value === null) return 'text-gray-400'
  if (value < 0.2) return 'text-red-400'
  if (value < 0.4) return 'text-orange-400'
  if (value < 0.6) return 'text-yellow-400'
  return 'text-green-400'
}
