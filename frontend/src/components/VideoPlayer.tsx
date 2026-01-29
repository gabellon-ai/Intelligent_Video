import { useRef, useEffect } from 'react'

interface Detection {
  class: string
  confidence: number
  bbox: number[]
}

interface FrameDetection {
  frame: number
  timestamp: number
  detections: Detection[]
  counts: Record<string, number>
}

interface VideoPlayerProps {
  jobId: string
  currentDetection: FrameDetection | null
  videoInfo: any
}

const COLORS: Record<string, string> = {
  'forklift': '#FF6B6B',
  'person': '#4ECDC4',
  'pallet': '#FFE66D',
  'cardboard box': '#95E1D3',
  'AGV automated guided vehicle': '#A78BFA',
  'conveyor belt': '#F9A826',
}

export function VideoPlayer({ jobId, currentDetection, videoInfo }: VideoPlayerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    if (!currentDetection || !canvasRef.current) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Clear previous drawings
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Draw bounding boxes
    currentDetection.detections.forEach((det) => {
      const [x1, y1, x2, y2] = det.bbox
      const color = COLORS[det.class] || '#FFFFFF'
      
      // Scale coordinates to canvas size
      const scaleX = canvas.width / (videoInfo?.width || 1920)
      const scaleY = canvas.height / (videoInfo?.height || 1080)
      
      const sx1 = x1 * scaleX
      const sy1 = y1 * scaleY
      const sx2 = x2 * scaleX
      const sy2 = y2 * scaleY
      
      // Draw box
      ctx.strokeStyle = color
      ctx.lineWidth = 2
      ctx.strokeRect(sx1, sy1, sx2 - sx1, sy2 - sy1)
      
      // Draw label background
      const label = `${det.class} ${(det.confidence * 100).toFixed(0)}%`
      ctx.font = '12px Inter, sans-serif'
      const textWidth = ctx.measureText(label).width
      
      ctx.fillStyle = color
      ctx.fillRect(sx1, sy1 - 20, textWidth + 8, 20)
      
      // Draw label text
      ctx.fillStyle = '#000000'
      ctx.fillText(label, sx1 + 4, sy1 - 6)
    })
  }, [currentDetection, videoInfo])

  return (
    <div className="relative aspect-video bg-gray-900">
      {/* Placeholder for video - in production, this would be actual video */}
      <div className="absolute inset-0 flex items-center justify-center text-gray-500">
        {videoInfo ? (
          <div className="text-center">
            <p className="text-lg font-medium">Video Preview</p>
            <p className="text-sm">{videoInfo.width}x{videoInfo.height} • {videoInfo.fps?.toFixed(1)} FPS</p>
            <p className="text-sm">{videoInfo.duration?.toFixed(1)}s duration</p>
          </div>
        ) : (
          <p>Connecting...</p>
        )}
      </div>
      
      {/* Detection overlay canvas */}
      <canvas
        ref={canvasRef}
        width={960}
        height={540}
        className="absolute inset-0 w-full h-full pointer-events-none"
      />

      {/* Current detection count overlay */}
      {currentDetection && Object.keys(currentDetection.counts).length > 0 && (
        <div className="absolute top-4 left-4 bg-black/70 rounded-lg p-3">
          <div className="space-y-1">
            {Object.entries(currentDetection.counts).map(([cls, count]) => (
              <div key={cls} className="flex items-center gap-2 text-sm">
                <span 
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: COLORS[cls] || '#FFFFFF' }}
                />
                <span className="text-gray-300">{cls}:</span>
                <span className="font-medium">{count}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Timestamp */}
      {currentDetection && (
        <div className="absolute bottom-4 right-4 bg-black/70 rounded px-2 py-1 text-sm font-mono">
          {currentDetection.timestamp.toFixed(2)}s
        </div>
      )}
    </div>
  )
}
