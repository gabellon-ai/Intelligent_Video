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
  // Derive values from summary or use placeholders
  const totalPallets = summary?.total_counts?.['pallet'] || 0
  const totalDetections = summary?.total_detections || 0
  const framesAnalyzed = summary?.total_frames_analyzed || 0
  const uniqueClasses = summary?.unique_classes?.length || 0

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* Rollup KPI strip */}
      <div style={{
        background: 'var(--kv-white)', border: '1px solid var(--kv-n-200)',
        borderRadius: 8, overflow: 'hidden',
      }}>
        <div style={{
          padding: '14px 16px', borderBottom: '1px solid var(--kv-n-200)',
          fontFamily: 'var(--font-display)', fontWeight: 900, fontSize: 12,
          letterSpacing: '0.08em', textTransform: 'uppercase' as const, color: 'var(--kv-black)',
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        }}>
          <span>Saw Outbound · Shift Rollup<span style={{ fontFamily: 'var(--font-serif)', color: 'var(--kv-blue)' }}>.</span></span>
          <span style={{
            fontFamily: 'var(--font-mono)', fontWeight: 400, fontSize: 10, letterSpacing: '0.08em',
            color: status === 'processing' ? 'var(--kv-blue)' : status === 'completed' ? 'var(--sem-success)' : 'var(--kv-n-500)',
            display: 'inline-flex', alignItems: 'center', gap: 6,
          }}>
            <span style={{
              width: 6, height: 6, borderRadius: '50%',
              background: status === 'processing' ? 'var(--kv-blue)' : status === 'completed' ? 'var(--sem-success)' : 'var(--kv-n-400)',
            }} />
            {status === 'processing' ? 'PROCESSING' : status === 'completed' ? 'COMPLETE' : 'WAITING'}
          </span>
        </div>
        <div style={{ padding: 14 }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 8 }}>
            <StatTile value={totalPallets || '—'} label="Total Pallets Moved" color="var(--sem-success)" />
            <StatTile value={framesAnalyzed ? (totalPallets / (framesAnalyzed / 150)).toFixed(1) : '—'} label="Pallets Per Hour" color="var(--kv-blue)" />
            <StatTile value={uniqueClasses || '—'} label="Object Types" color="#6B46C1" />
            <StatTile value={totalDetections || '—'} label="Total Detections" color="#E07B00" />
          </div>
        </div>
      </div>

      {/* Activity cards + Chart side-by-side */}
      <div style={{
        display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1.45fr)',
        gap: 20, alignItems: 'stretch',
      }}>
        {/* Left: two activity cards stacked */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <ActivityCard
            color="#E07B00"
            title="Activity 01 · Forktruck — Outbound Pallet to Floor"
            meta="1 OPERATOR · AVG CYCLE 2:24"
            palletsMoved={summary?.total_counts?.['forklift'] || 0}
            palletsPerHour={7.7}
            movePercent={82}
            utilPercent={76}
            entriesPerHour={18}
          />
          <ActivityCard
            color="#0084D5"
            title="Activity 02 · Walkie Stacker — Pallet to Work Cell"
            meta="2 OPERATORS · AVG CYCLE 3:12"
            palletsMoved={summary?.total_counts?.['pallet'] || 0}
            palletsPerHour={5.8}
            movePercent={64}
            utilPercent={67}
            entriesPerHour={12}
          />
        </div>

        {/* Right: Throughput chart */}
        <div style={{
          background: 'var(--kv-white)', border: '1px solid var(--kv-n-200)',
          borderRadius: 8, overflow: 'hidden', display: 'flex', flexDirection: 'column',
        }}>
          <div style={{
            padding: '14px 16px', borderBottom: '1px solid var(--kv-n-200)',
            fontFamily: 'var(--font-display)', fontWeight: 900, fontSize: 12,
            letterSpacing: '0.08em', textTransform: 'uppercase' as const, color: 'var(--kv-black)',
            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          }}>
            <span>Pallet Throughput · Staffing Efficiency<span style={{ fontFamily: 'var(--font-serif)', color: 'var(--kv-blue)' }}>.</span></span>
            <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 400, fontSize: 10, color: 'var(--kv-n-500)', letterSpacing: '0.08em' }}>
              HOURLY · SHIFT 1
            </span>
          </div>
          <div style={{ padding: '20px 20px 16px', flex: 1, display: 'flex', flexDirection: 'column' }}>
            {/* Legend */}
            <div style={{ display: 'flex', gap: 20, marginBottom: 14, fontSize: 11, color: 'var(--kv-n-600)', flexWrap: 'wrap' }}>
              <LegendItem color="#0084D5" label="Forktruck pallets/hr" type="bar" />
              <LegendItem color="#E07B00" label="Walkie stacker pallets/hr" type="bar" />
              <LegendItem color="#6B46C1" label="Labor utilization %" type="line" />
              <LegendItem color="#DC2626" label="Goal · 20/hr" type="dashed" />
            </div>

            {/* SVG Chart */}
            <svg viewBox="0 0 700 260" width="100%" style={{ display: 'block', overflow: 'visible', flex: 1 }}>
              {/* Grid */}
              <g fontFamily="var(--font-mono)" fontSize="10" fill="var(--kv-n-500)">
                {[20, 70, 120, 170].map(y => (
                  <line key={y} x1="48" y1={y} x2="652" y2={y} stroke="var(--kv-n-200)" strokeDasharray="2 3" />
                ))}
                <line x1="48" y1="220" x2="652" y2="220" stroke="var(--kv-n-300)" />
                {/* Left axis */}
                {[['20', 24], ['15', 74], ['10', 124], ['5', 174], ['0', 224]].map(([l, y]) => (
                  <text key={l} x="42" y={Number(y)} textAnchor="end">{l}</text>
                ))}
                {/* Right axis */}
                {[['100%', 24], ['75%', 74], ['50%', 124], ['25%', 174], ['0%', 224]].map(([l, y]) => (
                  <text key={l} x="658" y={Number(y)} textAnchor="start">{l}</text>
                ))}
              </g>
              <text x="10" y="12" fontFamily="var(--font-display)" fontSize="9" fontWeight="900" fill="var(--kv-ink)" letterSpacing="1.2">PALLETS/HR</text>
              <text x="690" y="12" fontFamily="var(--font-display)" fontSize="9" fontWeight="900" fill="var(--kv-ink)" letterSpacing="1.2" textAnchor="end">UTILIZATION</text>

              {/* Bars */}
              <g>
                {[
                  { x: 66, h1: 80, h2: 50 }, { x: 141, h1: 110, h2: 70 },
                  { x: 216, h1: 90, h2: 40 }, { x: 291, h1: 120, h2: 80 },
                  { x: 366, h1: 140, h2: 90 }, { x: 441, h1: 130, h2: 70 },
                  { x: 516, h1: 100, h2: 60 }, { x: 591, h1: 70, h2: 40 },
                ].map((b, i) => (
                  <g key={i}>
                    <rect x={b.x} y={220 - b.h1} width="20" height={b.h1} fill="#0084D5" />
                    <rect x={b.x + 22} y={220 - b.h2} width="20" height={b.h2} fill="#E07B00" />
                  </g>
                ))}
              </g>

              {/* Utilization line */}
              <polyline fill="none" stroke="#6B46C1" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"
                points="87,96 162,78 237,92 312,64 387,50 462,58 537,74 612,104" />
              <g fill="#6B46C1">
                {[[87, 96], [162, 78], [237, 92], [312, 64], [387, 50], [462, 58], [537, 74], [612, 104]].map(([cx, cy]) => (
                  <circle key={cx} cx={cx} cy={cy} r="4" stroke="#fff" strokeWidth="2" />
                ))}
              </g>

              {/* Goal line */}
              <line x1="48" y1="20" x2="652" y2="20" stroke="#DC2626" strokeWidth="1.5" strokeDasharray="5 4" />
              <rect x="54" y="10" width="78" height="14" fill="#DC2626" rx="2" />
              <text x="93" y="20" fontFamily="var(--font-display)" fontSize="9" fontWeight="900" fill="#fff" textAnchor="middle" letterSpacing="1">GOAL 20/HR</text>

              {/* X-axis labels */}
              <g fontFamily="var(--font-mono)" fontSize="10" fill="var(--kv-n-500)" textAnchor="middle" letterSpacing="0.04em">
                {['06:00', '07:00', '08:00', '09:00', '10:00', '11:00', '12:00', '13:00'].map((t, i) => (
                  <text key={t} x={87 + i * 75} y="240">{t}</text>
                ))}
              </g>
            </svg>

            {/* Insight strip */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2,1fr)', gap: 8, marginTop: 18 }}>
              <InsightTile color="#0084D5" label="FORKTRUCK · AVG" value="10.5" unit="plts/hr" pctOfGoal={52} />
              <InsightTile color="#E07B00" label="WALKIE STACKER · AVG" value="6.2" unit="plts/hr" pctOfGoal={31} />
            </div>
          </div>
        </div>
      </div>

      {/* Detection counts by class */}
      {summary && (
        <div style={{
          background: 'var(--kv-white)', border: '1px solid var(--kv-n-200)',
          borderRadius: 8, overflow: 'hidden',
        }}>
          <div style={{
            padding: '14px 16px', borderBottom: '1px solid var(--kv-n-200)',
            fontFamily: 'var(--font-display)', fontWeight: 900, fontSize: 12,
            letterSpacing: '0.08em', textTransform: 'uppercase' as const, color: 'var(--kv-black)',
          }}>
            Counts by class<span style={{ fontFamily: 'var(--font-serif)', color: 'var(--kv-blue)' }}>.</span>
          </div>
          <div style={{ padding: 16 }}>
            {Object.entries(summary.total_counts)
              .sort((a, b) => b[1] - a[1])
              .map(([cls, count]) => {
                const max = Math.max(...Object.values(summary.total_counts))
                return (
                  <div key={cls} style={{ marginBottom: 10 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 4 }}>
                      <span style={{ color: 'var(--kv-ink)', textTransform: 'capitalize' as const }}>{cls}</span>
                      <span className="mono" style={{ color: 'var(--kv-n-500)' }}>
                        {count} <span style={{ opacity: 0.6 }}>· max {summary.max_simultaneous[cls]}</span>
                      </span>
                    </div>
                    <div style={{ height: 4, background: 'var(--kv-n-200)', borderRadius: 2, overflow: 'hidden' }}>
                      <div style={{ height: '100%', background: 'var(--kv-blue)', width: `${(count / max) * 100}%` }} />
                    </div>
                  </div>
                )
              })}
          </div>
        </div>
      )}

      {/* Current frame info while processing */}
      {!summary && currentFrame && (
        <div style={{
          background: 'var(--kv-white)', border: '1px solid var(--kv-n-200)',
          borderRadius: 8, padding: 16,
        }}>
          <div style={{ fontFamily: 'var(--font-display)', fontWeight: 900, fontSize: 12, letterSpacing: '0.08em', textTransform: 'uppercase' as const, color: 'var(--kv-black)', marginBottom: 12 }}>
            Current Frame<span style={{ fontFamily: 'var(--font-serif)', color: 'var(--kv-blue)' }}>.</span>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 8 }}>
            <StatTile value={currentFrame.frame} label="Frame" color="var(--kv-blue)" />
            <StatTile value={currentFrame.detections.length} label="Objects" color="var(--sem-success)" />
          </div>
        </div>
      )}
    </div>
  )
}

/* ─── Sub-components ─── */

function StatTile({ value, label, color }: { value: string | number, label: string, color: string }) {
  return (
    <div style={{
      background: 'var(--kv-n-50)', border: '1px solid var(--kv-n-200)',
      borderRadius: 6, padding: '10px 12px', position: 'relative', overflow: 'hidden',
    }}>
      <span style={{ position: 'absolute', top: 0, left: 0, bottom: 0, width: 3, background: color }} />
      <div className="kv-display" style={{ fontSize: 22, color: 'var(--kv-black)', lineHeight: 1, marginLeft: 6 }}>
        {value}
      </div>
      <div style={{
        fontSize: 10, fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase' as const,
        color: 'var(--kv-n-500)', marginTop: 7, marginLeft: 6,
      }}>{label}</div>
    </div>
  )
}

function ActivityCard({ color, title, meta, palletsMoved, palletsPerHour, movePercent, utilPercent, entriesPerHour }: {
  color: string, title: string, meta: string, palletsMoved: number, palletsPerHour: number,
  movePercent: number, utilPercent: number, entriesPerHour: number
}) {
  return (
    <div style={{
      background: 'var(--kv-white)', border: '1px solid var(--kv-n-200)',
      borderRadius: 8, overflow: 'hidden', flex: 1, display: 'flex', flexDirection: 'column',
    }}>
      <div style={{
        padding: '14px 16px', borderBottom: '1px solid var(--kv-n-200)',
        fontFamily: 'var(--font-display)', fontWeight: 900, fontSize: 12,
        letterSpacing: '0.08em', textTransform: 'uppercase' as const, color: 'var(--kv-black)',
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
      }}>
        <span>
          <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: 2, background: color, marginRight: 6, verticalAlign: 'middle' }} />
          {title}<span style={{ fontFamily: 'var(--font-serif)', color: 'var(--kv-blue)' }}>.</span>
        </span>
        <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 400, fontSize: 10, color: 'var(--kv-n-500)', letterSpacing: '0.08em' }}>
          {meta}
        </span>
      </div>
      <div style={{ padding: 14 }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2,1fr)', gap: 8 }}>
          <div style={{
            background: 'var(--kv-n-50)', border: '1px solid var(--kv-n-200)',
            borderRadius: 6, padding: 10, position: 'relative', overflow: 'hidden',
          }}>
            <span style={{ position: 'absolute', top: 0, left: 0, bottom: 0, width: 3, background: color }} />
            <div className="kv-display" style={{ fontSize: 20, marginLeft: 6 }}>{palletsMoved}</div>
            <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase' as const, color: 'var(--kv-n-500)', marginTop: 4, marginLeft: 6 }}>Pallets Moved</div>
          </div>
          <div style={{
            background: 'var(--kv-n-50)', border: '1px solid var(--kv-n-200)',
            borderRadius: 6, padding: 10, position: 'relative', overflow: 'hidden',
          }}>
            <span style={{ position: 'absolute', top: 0, left: 0, bottom: 0, width: 3, background: color }} />
            <div className="kv-display" style={{ fontSize: 20, marginLeft: 6 }}>{palletsPerHour}</div>
            <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase' as const, color: 'var(--kv-n-500)', marginTop: 4, marginLeft: 6 }}>Pallets / Hour</div>
          </div>
        </div>

        {/* Bars */}
        <BarSplit label="Moving vs Idle" percent={movePercent} color={color} style={{ marginTop: 10 }} />
        <BarSplit label="Operator Utilization" percent={utilPercent} color="var(--kv-blue)" />

        <div style={{
          fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--kv-n-500)',
          letterSpacing: '0.04em', marginTop: 8,
        }}>
          {entriesPerHour} ENTRIES/HR · GOAL 20/HR
        </div>
      </div>
    </div>
  )
}

function BarSplit({ label, percent, color, style }: { label: string, percent: number, color: string, style?: React.CSSProperties }) {
  return (
    <div style={{ marginBottom: 8, ...style }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 4 }}>
        <span style={{ color: 'var(--kv-ink)' }}>{label}</span>
        <span className="mono" style={{ color: 'var(--kv-n-500)' }}>
          {percent}% <span style={{ opacity: 0.6 }}>· {100 - percent}% idle</span>
        </span>
      </div>
      <div style={{ display: 'flex', gap: 1, height: 4, borderRadius: 2, overflow: 'hidden' }}>
        <div style={{ width: `${percent}%`, background: color, height: '100%' }} />
        <div style={{ width: `${100 - percent}%`, background: 'var(--kv-n-300)', height: '100%' }} />
      </div>
    </div>
  )
}

function LegendItem({ color, label, type }: { color: string, label: string, type: 'bar' | 'line' | 'dashed' }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      {type === 'bar' && <span style={{ display: 'inline-block', width: 14, height: 10, background: color, borderRadius: 1 }} />}
      {type === 'line' && <span style={{ display: 'inline-block', width: 18, height: 2, background: color }} />}
      {type === 'dashed' && <span style={{ display: 'inline-block', width: 18, height: 0, borderTop: `2px dashed ${color}` }} />}
      <span>{label}</span>
    </div>
  )
}

function InsightTile({ color, label, value, unit, pctOfGoal }: { color: string, label: string, value: string, unit: string, pctOfGoal: number }) {
  return (
    <div style={{
      padding: '10px 12px', background: 'var(--kv-n-50)', border: '1px solid var(--kv-n-200)',
      borderRadius: 6, position: 'relative', overflow: 'hidden',
    }}>
      <span style={{ position: 'absolute', top: 0, left: 0, bottom: 0, width: 3, background: color }} />
      <div className="mono" style={{ fontSize: 10, color: 'var(--kv-n-500)', letterSpacing: '0.08em', marginLeft: 6 }}>{label}</div>
      <div style={{ marginLeft: 6, marginTop: 6, display: 'flex', alignItems: 'baseline', gap: 8 }}>
        <span className="kv-display" style={{ fontSize: 22, color: 'var(--kv-black)' }}>{value}</span>
        <span className="mono" style={{ fontSize: 10, color: 'var(--kv-n-500)' }}>{unit}</span>
        <span className="mono" style={{ marginLeft: 'auto', fontSize: 11, fontWeight: 700, color: '#DC2626', paddingRight: 4 }}>
          {pctOfGoal}% of goal
        </span>
      </div>
    </div>
  )
}
