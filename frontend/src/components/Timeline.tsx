interface FrameDetection {
  frame: number
  timestamp: number
  detections: any[]
  counts: Record<string, number>
}

interface TimelineProps {
  detections: FrameDetection[]
  duration: number
}

export function Timeline({ detections, duration }: TimelineProps) {
  if (!detections.length || !duration) return null

  // Create activity heatmap
  const segments = 100
  const segmentDuration = duration / segments
  const activity = new Array(segments).fill(0)
  
  detections.forEach(d => {
    const segmentIndex = Math.min(
      Math.floor(d.timestamp / segmentDuration),
      segments - 1
    )
    activity[segmentIndex] += d.detections.length
  })

  const maxActivity = Math.max(...activity, 1)

  return (
    <div className="bg-gray-800 rounded-lg p-4">
      <h3 className="text-sm font-semibold text-gray-400 mb-3">Detection Timeline</h3>
      
      {/* Activity heatmap */}
      <div className="flex gap-px h-8">
        {activity.map((count, i) => (
          <div
            key={i}
            className="flex-1 rounded-sm transition-colors"
            style={{
              backgroundColor: count > 0 
                ? `rgba(59, 130, 246, ${0.2 + (count / maxActivity) * 0.8})`
                : 'rgba(55, 65, 81, 0.5)'
            }}
            title={`${(i * segmentDuration).toFixed(1)}s - ${((i + 1) * segmentDuration).toFixed(1)}s: ${count} detections`}
          />
        ))}
      </div>

      {/* Time labels */}
      <div className="flex justify-between mt-2 text-xs text-gray-500">
        <span>0:00</span>
        <span>{Math.floor(duration / 2 / 60)}:{String(Math.floor((duration / 2) % 60)).padStart(2, '0')}</span>
        <span>{Math.floor(duration / 60)}:{String(Math.floor(duration % 60)).padStart(2, '0')}</span>
      </div>

      {/* Legend */}
      <div className="flex items-center gap-4 mt-3 text-xs text-gray-400">
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded-sm bg-gray-700" />
          <span>No detections</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded-sm bg-blue-500/30" />
          <span>Low activity</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded-sm bg-blue-500" />
          <span>High activity</span>
        </div>
      </div>
    </div>
  )
}
