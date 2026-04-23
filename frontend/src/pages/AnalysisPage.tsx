import { useEffect, useState, useRef } from 'react'
import { useParams } from 'react-router-dom'
import { VideoPlayer } from '../components/VideoPlayer'
import { DetectionSummary } from '../components/DetectionSummary'
import { ExportDropdown } from '../components/ExportDropdown'

const WS_BASE = `${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/api/streams`

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

interface Summary {
  total_detections: number
  unique_classes: string[]
  total_counts: Record<string, number>
  max_simultaneous: Record<string, number>
  frames_with_detections: number
  total_frames_analyzed: number
}

export function AnalysisPage() {
  const { jobId } = useParams<{ jobId: string }>()
  const [status, setStatus] = useState<'connecting' | 'processing' | 'completed' | 'error'>('connecting')
  const [progress, setProgress] = useState(0)
  const [currentFrame, setCurrentFrame] = useState<FrameDetection | null>(null)
  const [allDetections, setAllDetections] = useState<FrameDetection[]>([])
  const [summary, setSummary] = useState<Summary | null>(null)
  const [videoInfo, setVideoInfo] = useState<any>(null)
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    if (!jobId) return
    const ws = new WebSocket(`${WS_BASE}/ws/${jobId}`)
    wsRef.current = ws

    ws.onopen = () => { setStatus('processing') }
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      switch (data.type) {
        case 'start': setVideoInfo(data.video_info); break
        case 'progress': setProgress(data.percent); break
        case 'detection':
          setCurrentFrame(data)
          setAllDetections(prev => [...prev, data])
          break
        case 'summary': setSummary(data); break
        case 'complete': setStatus('completed'); break
        case 'error': setStatus('error'); break
      }
    }
    ws.onerror = () => setStatus('error')
    ws.onclose = () => {}
    return () => { ws.close() }
  }, [jobId])

  const frameLabel = currentFrame ? `F${String(currentFrame.frame).padStart(4, '0')}` : ''
  const detCount = currentFrame ? currentFrame.detections.length : 0

  return (
    <div>
      {/* Header */}
      <div style={{
        display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between',
        gap: 20, marginBottom: 20,
      }}>
        <div style={{ minWidth: 0, flex: 1 }}>
          <div className="kv-eyebrow">▶ Analysis Session</div>
          <h1 className="kv-display" style={{
            fontSize: 28, lineHeight: 1.15, color: 'var(--kv-black)', margin: '6px 0 10px',
          }}>
            Floor Vision Report<span className="kv-dot">.</span>
          </h1>
          <span className="mono" style={{
            fontSize: 11, color: 'var(--kv-n-500)', letterSpacing: '0.04em',
          }}>
            JOB · {jobId}
          </span>
        </div>
        <ExportDropdown jobId={jobId!} disabled={status !== 'completed'} />
      </div>

      {/* Progress */}
      {status === 'processing' && (
        <div style={{
          background: 'var(--kv-white)', border: '1px solid var(--kv-n-200)',
          borderRadius: 8, padding: 14, marginBottom: 20,
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8, fontSize: 12 }}>
            <span style={{ color: 'var(--kv-n-600)' }}>Analyzing footage…</span>
            <span className="mono" style={{ fontWeight: 700, color: 'var(--kv-ink)' }}>{progress}%</span>
          </div>
          <div style={{ height: 4, background: 'var(--kv-n-200)', borderRadius: 2, overflow: 'hidden' }}>
            <div style={{ height: '100%', background: 'var(--kv-blue)', width: `${progress}%`, transition: 'width 200ms' }} />
          </div>
          {currentFrame && (
            <div className="mono" style={{ marginTop: 10, fontSize: 11, color: 'var(--kv-n-500)', letterSpacing: '0.04em' }}>
              {frameLabel} · {detCount} detections
            </div>
          )}
        </div>
      )}

      {/* SECTION 01 · Real-Time Vision */}
      <SectionHeader num="01" label="Real-Time Vision · Source & Detections" meta={`LIVE · ${videoInfo?.fps?.toFixed(1) || '29.97'} FPS`} />
      <p style={{
        fontSize: 12, color: 'var(--kv-n-600)', lineHeight: 1.5,
        fontStyle: 'italic', marginTop: -4, marginBottom: 12, maxWidth: 820,
      }}>
        Raw video source with classifier overlays, paired with frame-level detection stream as audit trail.
      </p>

      <div style={{
        display: 'grid', gridTemplateColumns: 'minmax(0, 1.6fr) minmax(0, 1fr)',
        gap: 20, marginBottom: 28,
      }}>
        {/* Video */}
        <Card title="Source Feed" meta={`${videoInfo?.width || 1920}×${videoInfo?.height || 1080} · ${videoInfo?.fps?.toFixed(1) || '29.97'} FPS`}>
          <VideoPlayer jobId={jobId!} currentDetection={currentFrame} videoInfo={videoInfo} />
          <div style={{
            padding: 14, borderTop: '1px solid var(--kv-n-200)',
            display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 12,
          }}>
            <PbBtn icon="prev" />
            <PbBtn icon="play" primary />
            <PbBtn icon="next" />
          </div>
        </Card>

        {/* Live Feed */}
        <Card title="Live Feed">
          <div style={{ padding: 12, display: 'flex', flexDirection: 'column', gap: 8, maxHeight: 400, overflowY: 'auto' }}>
            {allDetections.length === 0 && (
              <div style={{ padding: 20, textAlign: 'center', color: 'var(--kv-n-400)', fontSize: 12 }}>
                Waiting for detections…
              </div>
            )}
            {allDetections.slice(-10).reverse().map((frame, i) => (
              <div key={i} style={{
                fontSize: 12, padding: 10, border: '1px solid var(--kv-n-200)',
                borderRadius: 6, background: 'var(--kv-n-50)',
              }}>
                <div style={{
                  display: 'flex', justifyContent: 'space-between', marginBottom: 6,
                  fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--kv-n-500)', letterSpacing: '0.04em',
                }}>
                  <span>F{String(frame.frame).padStart(4, '0')}</span>
                  <span>{frame.timestamp.toFixed(2)}s</span>
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                  {Object.entries(frame.counts).map(([cls, count]) => (
                    <span key={cls} style={{
                      padding: '2px 7px', borderRadius: 2,
                      background: 'var(--kv-blue-50)', color: 'var(--kv-blue-dark)',
                      fontFamily: 'var(--font-mono)', fontSize: 10, fontWeight: 600, letterSpacing: '0.04em',
                    }}>
                      {cls}: {count}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>

      {/* SECTION 02 · Performance */}
      <SectionHeader num="02" label="Performance · Shift Summary" meta="SHIFT 1 · 06:00–14:00" />
      <p style={{
        fontSize: 12, color: 'var(--kv-n-600)', lineHeight: 1.5,
        fontStyle: 'italic', marginTop: -4, marginBottom: 12, maxWidth: 820,
      }}>
        Throughput and labor utilization versus 20 plt/hr goal, with activity-level operator and equipment breakdown.
      </p>

      <DetectionSummary summary={summary} currentFrame={currentFrame} status={status} />
    </div>
  )
}

/* ─── Shared UI Components ─── */

function SectionHeader({ num, label, meta }: { num: string, label: string, meta: string }) {
  return (
    <div style={{
      display: 'flex', alignItems: 'baseline', justifyContent: 'space-between',
      borderTop: '2px solid var(--kv-black)', padding: '10px 0 0', marginBottom: 12,
    }}>
      <span style={{
        fontFamily: 'var(--font-display)', fontWeight: 900, fontSize: 11,
        letterSpacing: '0.14em', color: 'var(--kv-black)', textTransform: 'uppercase',
        display: 'inline-flex', alignItems: 'baseline', gap: 10,
      }}>
        <span className="mono" style={{ fontSize: 10, color: 'var(--kv-n-500)', letterSpacing: '0.14em', fontWeight: 400 }}>
          {num}
        </span>
        {label}
      </span>
      <span className="mono" style={{ fontSize: 10, color: 'var(--kv-n-500)', letterSpacing: '0.1em', textTransform: 'uppercase' }}>
        {meta}
      </span>
    </div>
  )
}

function Card({ title, meta, children }: { title: string, meta?: string, children: React.ReactNode }) {
  return (
    <div style={{
      background: 'var(--kv-white)', border: '1px solid var(--kv-n-200)',
      borderRadius: 8, overflow: 'hidden', display: 'flex', flexDirection: 'column',
    }}>
      <div style={{
        padding: '14px 16px', borderBottom: '1px solid var(--kv-n-200)',
        fontFamily: 'var(--font-display)', fontWeight: 900, fontSize: 12,
        letterSpacing: '0.08em', textTransform: 'uppercase', color: 'var(--kv-black)',
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
      }}>
        <span>{title}<span className="kv-dot">.</span></span>
        {meta && (
          <span className="mono" style={{ fontWeight: 400, fontSize: 10, color: 'var(--kv-n-500)', letterSpacing: '0.08em' }}>
            {meta}
          </span>
        )}
      </div>
      {children}
    </div>
  )
}

function PbBtn({ icon, primary }: { icon: 'prev' | 'play' | 'next', primary?: boolean }) {
  const base: React.CSSProperties = primary
    ? { width: 44, height: 44, borderRadius: 22, background: 'var(--kv-blue)', color: 'var(--kv-white)', border: 'none' }
    : { width: 36, height: 36, borderRadius: 4, background: 'var(--kv-white)', border: '1px solid var(--kv-n-200)', color: 'var(--kv-ink)' }

  return (
    <button style={{ ...base, display: 'inline-flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer' }}>
      {icon === 'prev' && (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><polygon points="19 20 9 12 19 4 19 20" /><line x1="5" y1="19" x2="5" y2="5" stroke="currentColor" strokeWidth="2" /></svg>
      )}
      {icon === 'play' && (
        <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3" /></svg>
      )}
      {icon === 'next' && (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 4 15 12 5 20 5 4" /><line x1="19" y1="5" x2="19" y2="19" stroke="currentColor" strokeWidth="2" /></svg>
      )}
    </button>
  )
}
