import { Loader2, CheckCircle, AlertCircle, TrendingUp } from 'lucide-react'

interface Summary {
  total_detections: number
  unique_classes: string[]
  total_counts: Record<string, number>
  max_simultaneous: Record<string, number>
  frames_with_detections: number
  total_frames_analyzed: number
}

interface FrameDetection {
  frame: number
  timestamp: number
  detections: any[]
  counts: Record<string, number>
}

interface DetectionSummaryProps {
  summary: Summary | null
  currentFrame: FrameDetection | null
  status: string
}

export function DetectionSummary({ summary, currentFrame, status }: DetectionSummaryProps) {
  return (
    <div className="bg-gray-800 rounded-lg p-4 space-y-4">
      {/* Status */}
      <div className="flex items-center gap-2">
        {status === 'processing' && (
          <>
            <Loader2 className="w-5 h-5 text-blue-400 animate-spin" />
            <span className="text-blue-400">Processing...</span>
          </>
        )}
        {status === 'completed' && (
          <>
            <CheckCircle className="w-5 h-5 text-green-400" />
            <span className="text-green-400">Analysis Complete</span>
          </>
        )}
        {status === 'error' && (
          <>
            <AlertCircle className="w-5 h-5 text-red-400" />
            <span className="text-red-400">Error</span>
          </>
        )}
      </div>

      {/* Summary Stats */}
      {summary && (
        <>
          <div className="grid grid-cols-2 gap-3">
            <div className="bg-gray-700/50 rounded-lg p-3">
              <p className="text-2xl font-bold text-blue-400">
                {summary.total_detections}
              </p>
              <p className="text-xs text-gray-400">Total Detections</p>
            </div>
            <div className="bg-gray-700/50 rounded-lg p-3">
              <p className="text-2xl font-bold text-green-400">
                {summary.unique_classes.length}
              </p>
              <p className="text-xs text-gray-400">Object Types</p>
            </div>
            <div className="bg-gray-700/50 rounded-lg p-3">
              <p className="text-2xl font-bold text-purple-400">
                {summary.frames_with_detections}
              </p>
              <p className="text-xs text-gray-400">Active Frames</p>
            </div>
            <div className="bg-gray-700/50 rounded-lg p-3">
              <p className="text-2xl font-bold text-yellow-400">
                {summary.total_frames_analyzed}
              </p>
              <p className="text-xs text-gray-400">Frames Analyzed</p>
            </div>
          </div>

          {/* Object Counts */}
          <div>
            <h3 className="text-sm font-semibold text-gray-400 mb-2 flex items-center gap-1">
              <TrendingUp className="w-4 h-4" />
              Detection Counts
            </h3>
            <div className="space-y-2">
              {Object.entries(summary.total_counts)
                .sort((a, b) => b[1] - a[1])
                .map(([cls, count]) => (
                  <div key={cls}>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="capitalize">{cls}</span>
                      <span className="text-gray-400">
                        {count} (max: {summary.max_simultaneous[cls]})
                      </span>
                    </div>
                    <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-blue-500"
                        style={{ 
                          width: `${(count / Math.max(...Object.values(summary.total_counts))) * 100}%` 
                        }}
                      />
                    </div>
                  </div>
                ))}
            </div>
          </div>
        </>
      )}

      {/* Current Frame Info */}
      {!summary && currentFrame && (
        <div>
          <h3 className="text-sm font-semibold text-gray-400 mb-2">Current Frame</h3>
          <div className="bg-gray-700/50 rounded-lg p-3">
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div>
                <p className="text-gray-400">Frame</p>
                <p className="font-medium">{currentFrame.frame}</p>
              </div>
              <div>
                <p className="text-gray-400">Objects</p>
                <p className="font-medium">{currentFrame.detections.length}</p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
