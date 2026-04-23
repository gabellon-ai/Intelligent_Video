import { useState, useRef, useEffect } from 'react'

const API_BASE = 'http://localhost:8000/api'

interface ExportDropdownProps {
  jobId: string
  disabled?: boolean
}

type ExportFormat = 'csv' | 'pdf' | 'json'
type DownloadStatus = 'idle' | 'downloading' | 'success' | 'error'

const exportOptions: { format: ExportFormat, label: string }[] = [
  { format: 'pdf', label: 'PDF Report' },
  { format: 'csv', label: 'CSV Data' },
  { format: 'json', label: 'JSON Export' },
]

export function ExportDropdown({ jobId, disabled }: ExportDropdownProps) {
  const [isOpen, setIsOpen] = useState(false)
  const [downloadStatus, setDownloadStatus] = useState<Record<ExportFormat, DownloadStatus>>({
    csv: 'idle', pdf: 'idle', json: 'idle',
  })
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setIsOpen(false)
    }
    document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [])

  const handleExport = async (format: ExportFormat) => {
    if (downloadStatus[format] === 'downloading') return
    setDownloadStatus(prev => ({ ...prev, [format]: 'downloading' }))
    try {
      const response = await fetch(`${API_BASE}/videos/${jobId}/export/${format}`)
      if (!response.ok) throw new Error('Export failed')
      const contentDisposition = response.headers.get('content-disposition')
      let filename = `analysis_${jobId.slice(0, 8)}.${format}`
      if (contentDisposition) {
        const match = contentDisposition.match(/filename="?([^";\n]+)"?/)
        if (match) filename = match[1]
      }
      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url; a.download = filename
      document.body.appendChild(a); a.click()
      window.URL.revokeObjectURL(url); document.body.removeChild(a)
      setDownloadStatus(prev => ({ ...prev, [format]: 'success' }))
      setTimeout(() => setDownloadStatus(prev => ({ ...prev, [format]: 'idle' })), 2000)
    } catch {
      setDownloadStatus(prev => ({ ...prev, [format]: 'error' }))
      setTimeout(() => setDownloadStatus(prev => ({ ...prev, [format]: 'idle' })), 3000)
    }
  }

  return (
    <div style={{ position: 'relative' }} ref={ref}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        disabled={disabled}
        style={{
          display: 'inline-flex', alignItems: 'center', gap: 8,
          padding: '9px 16px', borderRadius: 4, border: 'none',
          fontSize: 13, fontWeight: 700, letterSpacing: '0.02em',
          background: disabled ? 'var(--kv-n-300)' : 'var(--kv-blue)',
          color: 'var(--kv-white)', cursor: disabled ? 'not-allowed' : 'pointer',
        }}
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="7 10 12 15 17 10" /><line x1="12" y1="15" x2="12" y2="3" />
        </svg>
        Export
      </button>

      {isOpen && !disabled && (
        <div style={{
          position: 'absolute', right: 0, marginTop: 8, width: 220,
          background: 'var(--kv-white)', borderRadius: 8,
          boxShadow: '0 12px 32px rgba(10,10,10,0.10), 0 4px 8px rgba(10,10,10,0.04)',
          border: '1px solid var(--kv-n-200)', overflow: 'hidden', zIndex: 50,
        }}>
          <div style={{
            padding: '8px 12px', fontSize: 10, fontWeight: 700, letterSpacing: '0.1em',
            textTransform: 'uppercase' as const, color: 'var(--kv-n-500)',
            borderBottom: '1px solid var(--kv-n-200)',
          }}>
            Export Results
          </div>
          {exportOptions.map(({ format, label }) => (
            <button
              key={format}
              onClick={() => handleExport(format)}
              disabled={downloadStatus[format] === 'downloading'}
              style={{
                display: 'block', width: '100%', padding: '10px 12px',
                background: downloadStatus[format] === 'success' ? 'rgba(16,185,129,0.08)' : 'transparent',
                border: 'none', borderBottom: '1px solid var(--kv-n-100)',
                textAlign: 'left' as const, fontSize: 13, fontWeight: 600,
                color: downloadStatus[format] === 'success' ? 'var(--sem-success)' : 'var(--kv-ink)',
                cursor: 'pointer',
              }}
            >
              {downloadStatus[format] === 'downloading' ? `Generating ${label}…`
                : downloadStatus[format] === 'success' ? `✓ ${label}`
                : label}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
