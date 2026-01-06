import { CheckCircle, AlertTriangle, XCircle, Zap } from 'lucide-react'

export default function QualityReport({ metrics }) {
  if (!metrics) return null

  const getQualityGrade = (sicn, gamma) => {
    if (sicn >= 0.7 && gamma >= 0.7) return { grade: 'Excellent', color: 'text-green-400' }
    if (sicn >= 0.5 && gamma >= 0.5) return { grade: 'Good', color: 'text-blue-400' }
    if (sicn >= 0.3 && gamma >= 0.3) return { grade: 'Fair', color: 'text-yellow-400' }
    return { grade: 'Poor', color: 'text-red-400' }
  }

  const sicnMin = metrics.sicn_min || metrics['SICN (Gmsh)']?.min
  const sicnMax = metrics.sicn_max || metrics['SICN (Gmsh)']?.max
  const sicnAvg = metrics.sicn_avg || metrics['SICN (Gmsh)']?.avg

  const gammaMin = metrics.gamma_min || metrics['Gamma (Gmsh)']?.min
  const gammaMax = metrics.gamma_max || metrics['Gamma (Gmsh)']?.max
  const gammaAvg = metrics.gamma_avg || metrics['Gamma (Gmsh)']?.avg

  const skewnessMax = metrics.max_skewness || metrics['Skewness (converted)']?.max
  const aspectRatioMax = metrics.max_aspect_ratio || metrics['Aspect Ratio (converted)']?.max

  const { grade, color } = getQualityGrade(sicnMin, gammaMin)

  // CFD metrics (from OpenFOAM checkMesh equivalent)
  const cfd = metrics.cfd
  const nonOrthoMax = cfd?.geometry_checks?.non_orthogonality?.max
  const nonOrthoOk = cfd?.geometry_checks?.non_orthogonality?.ok
  const cfdSkewnessMax = cfd?.geometry_checks?.skewness?.max
  const cfdSkewnessOk = cfd?.geometry_checks?.skewness?.ok
  const facePyramidsOk = cfd?.geometry_checks?.face_pyramids?.ok
  const cfdReady = cfd?.cfd_ready

  // Non-orthogonality status icon
  const getNonOrthoIcon = () => {
    if (nonOrthoMax === undefined) return null
    if (nonOrthoMax <= 65) return <CheckCircle className="w-3 h-3 text-green-400" />
    if (nonOrthoMax <= 70) return <AlertTriangle className="w-3 h-3 text-yellow-400" />
    return <XCircle className="w-3 h-3 text-red-400" />
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="font-semibold text-sm uppercase text-gray-400">Quality Report</h3>
        <span className={`text-xs font-bold ${color}`}>{grade}</span>
      </div>

      {/* FEA Metrics */}
      <div className="space-y-2 text-xs">
        {sicnMin !== undefined && (
          <div className="flex justify-between items-center">
            <span className="text-gray-400">SICN (min)</span>
            <div className="flex items-center gap-2">
              <span className="font-mono">{sicnMin.toFixed(4)}</span>
              {sicnMin >= 0.3 ? (
                <CheckCircle className="w-3 h-3 text-green-400" />
              ) : (
                <AlertTriangle className="w-3 h-3 text-yellow-400" />
              )}
            </div>
          </div>
        )}

        {gammaMin !== undefined && (
          <div className="flex justify-between items-center">
            <span className="text-gray-400">Gamma (min)</span>
            <div className="flex items-center gap-2">
              <span className="font-mono">{gammaMin.toFixed(4)}</span>
              {gammaMin >= 0.2 ? (
                <CheckCircle className="w-3 h-3 text-green-400" />
              ) : (
                <AlertTriangle className="w-3 h-3 text-yellow-400" />
              )}
            </div>
          </div>
        )}

        {skewnessMax !== undefined && (
          <div className="flex justify-between items-center">
            <span className="text-gray-400">Max Skewness</span>
            <div className="flex items-center gap-2">
              <span className="font-mono">{skewnessMax.toFixed(4)}</span>
              {skewnessMax <= 0.7 ? (
                <CheckCircle className="w-3 h-3 text-green-400" />
              ) : (
                <AlertTriangle className="w-3 h-3 text-yellow-400" />
              )}
            </div>
          </div>
        )}

        {aspectRatioMax !== undefined && (
          <div className="flex justify-between items-center">
            <span className="text-gray-400">Max Aspect Ratio</span>
            <div className="flex items-center gap-2">
              <span className="font-mono">{aspectRatioMax.toFixed(2)}</span>
              {aspectRatioMax <= 5.0 ? (
                <CheckCircle className="w-3 h-3 text-green-400" />
              ) : (
                <AlertTriangle className="w-3 h-3 text-yellow-400" />
              )}
            </div>
          </div>
        )}
      </div>

      {/* CFD Metrics Section */}
      {cfd && (
        <>
          <div className="border-t border-gray-700 pt-2 mt-2">
            <div className="flex items-center justify-between mb-2">
              <h4 className="font-semibold text-xs uppercase text-gray-400 flex items-center gap-1">
                <Zap className="w-3 h-3" />
                CFD Quality
              </h4>
              <span className={`text-xs font-bold ${cfdReady ? 'text-green-400' : 'text-red-400'}`}>
                {cfdReady ? 'Ready' : 'Issues'}
              </span>
            </div>

            <div className="space-y-2 text-xs">
              {nonOrthoMax !== undefined && (
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Non-Orthogonality</span>
                  <div className="flex items-center gap-2">
                    <span className="font-mono">{nonOrthoMax.toFixed(1)}°</span>
                    {getNonOrthoIcon()}
                  </div>
                </div>
              )}

              {cfdSkewnessMax !== undefined && (
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">CFD Skewness</span>
                  <div className="flex items-center gap-2">
                    <span className="font-mono">{cfdSkewnessMax.toFixed(3)}</span>
                    {cfdSkewnessOk ? (
                      <CheckCircle className="w-3 h-3 text-green-400" />
                    ) : (
                      <AlertTriangle className="w-3 h-3 text-yellow-400" />
                    )}
                  </div>
                </div>
              )}

              {facePyramidsOk !== undefined && (
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Face Pyramids</span>
                  <div className="flex items-center gap-2">
                    <span className="font-mono">{facePyramidsOk ? 'OK' : 'FAIL'}</span>
                    {facePyramidsOk ? (
                      <CheckCircle className="w-3 h-3 text-green-400" />
                    ) : (
                      <XCircle className="w-3 h-3 text-red-400" />
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* CFD Ready Banner */}
          {cfdReady && (
            <div className="flex items-center gap-2 text-xs text-green-400 bg-green-900/20 p-2 rounded">
              <Zap className="w-4 h-4" />
              CFD-ready mesh (non-ortho ≤70°)
            </div>
          )}

          {!cfdReady && cfd.errors?.length > 0 && (
            <div className="flex items-start gap-2 text-xs text-red-400 bg-red-900/20 p-2 rounded">
              <AlertTriangle className="w-4 h-4 flex-shrink-0 mt-0.5" />
              <div>{cfd.errors[0]}</div>
            </div>
          )}
        </>
      )}

      {/* FEA All-pass banner */}
      {(sicnMin >= 0.3 && gammaMin >= 0.2 && skewnessMax <= 0.7 && aspectRatioMax <= 5.0) && !cfd && (
        <div className="flex items-center gap-2 text-xs text-green-400 bg-green-900/20 p-2 rounded">
          <CheckCircle className="w-4 h-4" />
          Mesh quality meets all standards
        </div>
      )}
    </div>
  )
}
