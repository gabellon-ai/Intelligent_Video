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

  const segments = 100
  const segmentDuration = duration / segments
  const activity = new Array(segments).fill(0)

  detections.forEach(d => {
    const segmentIndex = Math.min(Math.floor(d.timestamp / segmentDuration), segments - 1)
    activity[segmentIndex] += d.detections.length
  })

  const maxActivity = Math.max(...activity, 1)

  return (
    <div style={{
      background: 'var(--kv-white)', border: '1px solid var(--kv-n-200)',
      borderRadius: 8, overflow: 'hidden',
    }}>
      <div style={{
        padding: '14px 16px', borderBottom: '1px solid var(--kv-n-200)',
        fontFamily: 'var(--font-display)', fontWeight: 900, fontSize: 12,
        letterSpacing: '0.08em', textTransform: 'uppercase' as const, color: 'var(--kv-black)',
      }}>
        Detection Timeline<span style={{ fontFamily: 'var(--font-serif)', color: 'var(--kv-blue)' }}>.</span>
      </div>
      <div style={{ padding: '12px 16px 16px' }}>
        {/* Heatmap */}
        <div style={{ display: 'flex', gap: 1, height: 32 }}>
          {activity.map((count, i) => (
            <div
              key={i}
              style={{
                flex: 1, borderRadius: 1,
                backgroundColor: count > 0
                  ? `rgba(0, 132, 213, ${0.15 + (count / maxActivity) * 0.85})`
                  : 'var(--kv-n-100)',
              }}
              title={`${(i * segmentDuration).toFixed(1)}s – ${((i + 1) * segmentDuration).toFixed(1)}s: ${count} detections`}
            />
          ))}
        </div>

        {/* Labels */}
        <div style={{
          display: 'flex', justifyContent: 'space-between', marginTop: 8,
          fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--kv-n-500)', letterSpacing: '0.04em',
        }}>
          <span>0:00</span>
          <span>{Math.floor(duration / 2 / 60)}:{String(Math.floor((duration / 2) % 60)).padStart(2, '0')}</span>
          <span>{Math.floor(duration / 60)}:{String(Math.floor(duration % 60)).padStart(2, '0')}</span>
        </div>

        {/* Legend */}
        <div style={{ display: 'flex', gap: 16, marginTop: 14, fontSize: 11, color: 'var(--kv-n-600)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <div style={{ width: 10, height: 10, borderRadius: 2, background: 'var(--kv-n-100)', border: '1px solid var(--kv-n-200)' }} />
            <span>No detections</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <div style={{ width: 10, height: 10, borderRadius: 2, background: 'rgba(0,132,213,0.3)' }} />
            <span>Low</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <div style={{ width: 10, height: 10, borderRadius: 2, background: 'rgba(0,132,213,1)' }} />
            <span>High</span>
          </div>
        </div>
      </div>
    </div>
  )
}
