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
  'forklift': '#E07B00',
  'person': '#0084D5',
  'pallet': '#10B981',
  'cardboard box': '#6B46C1',
  'AGV automated guided vehicle': '#A78BFA',
  'conveyor belt': '#F59E0B',
}

export function VideoPlayer({ jobId, currentDetection, videoInfo }: VideoPlayerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const lastSeekAt = useRef(0)

  useEffect(() => {
    const video = videoRef.current
    if (!video || !currentDetection) return
    // Throttle seeks to ~2/s — scrubbing on every detection (5/s) floods
    // Range requests and makes the video appear to never finish loading.
    const now = performance.now()
    if (now - lastSeekAt.current < 500) return
    const target = currentDetection.timestamp
    if (Math.abs(video.currentTime - target) > 0.5) {
      try {
        if (typeof (video as any).fastSeek === 'function') (video as any).fastSeek(target)
        else video.currentTime = target
        lastSeekAt.current = now
      } catch { /* seek before metadata loaded */ }
    }
  }, [currentDetection])

  useEffect(() => {
    if (!currentDetection || !canvasRef.current) return
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    ctx.clearRect(0, 0, canvas.width, canvas.height)

    currentDetection.detections.forEach((det) => {
      const [x1, y1, x2, y2] = det.bbox
      const color = COLORS[det.class] || '#0084D5'
      const scaleX = canvas.width / (videoInfo?.width || 1920)
      const scaleY = canvas.height / (videoInfo?.height || 1080)
      const sx1 = x1 * scaleX, sy1 = y1 * scaleY
      const sx2 = x2 * scaleX, sy2 = y2 * scaleY

      ctx.strokeStyle = color
      ctx.lineWidth = 2
      ctx.strokeRect(sx1, sy1, sx2 - sx1, sy2 - sy1)

      const label = `${det.class} ${(det.confidence * 100).toFixed(0)}%`
      ctx.font = '600 10px Arial'
      const textWidth = ctx.measureText(label).width
      ctx.fillStyle = color
      ctx.fillRect(sx1 - 1, sy1 - 20, textWidth + 10, 20)
      ctx.fillStyle = '#fff'
      ctx.fillText(label, sx1 + 4, sy1 - 6)
    })
  }, [currentDetection, videoInfo])

  return (
    <div style={{
      position: 'relative', aspectRatio: '16/9', background: 'var(--kv-n-900)',
      backgroundImage: 'repeating-linear-gradient(135deg, rgba(255,255,255,0.035) 0 24px, rgba(255,255,255,0.07) 24px 48px)',
      overflow: 'hidden',
    }}>
      {/* Placeholder */}
      {!currentDetection && (
        <div style={{
          position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column',
          alignItems: 'center', justifyContent: 'center', color: 'rgba(255,255,255,0.55)',
        }}>
          <div style={{
            fontFamily: 'var(--font-display)', fontWeight: 900, fontSize: 14,
            letterSpacing: '0.08em', color: 'rgba(255,255,255,0.8)',
          }}>WAREHOUSE FLOOR · VIDEO SOURCE</div>
          <div className="mono" style={{ fontSize: 10, marginTop: 6, letterSpacing: '0.06em' }}>
            {videoInfo
              ? `${videoInfo.width}×${videoInfo.height} · ${videoInfo.fps?.toFixed(1)} FPS · ${videoInfo.duration?.toFixed(1)}s`
              : 'Connecting…'}
          </div>
        </div>
      )}

      {/* Video source (seeks to the current analyzed frame) */}
      <video
        ref={videoRef}
        src={jobId ? `/api/videos/${jobId}/video` : undefined}
        muted
        playsInline
        preload="metadata"
        crossOrigin="anonymous"
        style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'contain', background: '#000' }}
      />

      {/* Canvas overlay */}
      <canvas ref={canvasRef} width={960} height={540}
        style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }}
      />

      {/* Counts overlay */}
      {currentDetection && Object.keys(currentDetection.counts).length > 0 && (
        <div style={{
          position: 'absolute', top: 12, left: 12,
          background: 'rgba(0,0,0,0.62)', border: '1px solid rgba(255,255,255,0.12)',
          borderRadius: 4, padding: '8px 10px',
          display: 'flex', flexDirection: 'column', gap: 4,
        }}>
          {Object.entries(currentDetection.counts).map(([cls, count]) => (
            <div key={cls} style={{
              display: 'flex', alignItems: 'center', gap: 8,
              fontFamily: 'var(--font-mono)', fontSize: 11, color: '#fff',
            }}>
              <span style={{
                width: 8, height: 8, borderRadius: 2,
                background: COLORS[cls] || '#0084D5',
              }} />
              <span style={{ color: 'rgba(255,255,255,0.7)' }}>{cls}</span>
              <span style={{ fontWeight: 700 }}>{count}</span>
            </div>
          ))}
        </div>
      )}

      {/* Analyzed badge */}
      {currentDetection && (
        <div style={{
          position: 'absolute', top: 12, right: 12,
          display: 'inline-flex', alignItems: 'center', gap: 6,
          background: 'rgba(16,185,129,0.18)', border: '1px solid rgba(16,185,129,0.5)',
          padding: '4px 10px', borderRadius: 4,
          fontFamily: 'var(--font-mono)', fontSize: 10, color: '#86efac',
          letterSpacing: '0.12em', fontWeight: 600,
        }}>
          ● ANALYZED
        </div>
      )}

      {/* Timestamp */}
      {currentDetection && (
        <div style={{
          position: 'absolute', bottom: 12, right: 12,
          background: 'rgba(0,0,0,0.62)', border: '1px solid rgba(255,255,255,0.12)',
          borderRadius: 4, padding: '4px 8px',
          fontFamily: 'var(--font-mono)', fontSize: 11, color: '#fff', letterSpacing: '0.04em',
        }}>
          {currentDetection.timestamp.toFixed(2)}s
        </div>
      )}
    </div>
  )
}
