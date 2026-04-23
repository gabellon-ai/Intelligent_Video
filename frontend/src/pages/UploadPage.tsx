import { useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'

const API_BASE = '/api'

const presets = [
  { name: 'Warehouse · General', desc: 'Forklifts, pallets, associates, boxes, conveyors', active: true },
  { name: 'Safety Focus', desc: 'Associates, hi-vis vests, zone violations', active: false },
  { name: 'AGV Tracking', desc: 'AGVs, AMRs, autonomous equipment', active: false },
  { name: 'Loading Dock', desc: 'Trucks, trailers, dock-door activity', active: false },
]

export function UploadPage() {
  const [isDragging, setIsDragging] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [activePreset, setActivePreset] = useState(0)
  const navigate = useNavigate()

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    const file = e.dataTransfer.files[0]
    if (file && file.type.startsWith('video/')) {
      setSelectedFile(file)
    }
  }, [])

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) setSelectedFile(file)
  }, [])

  const handleUpload = async () => {
    if (!selectedFile) return
    setUploading(true)
    setUploadProgress(0)

    const formData = new FormData()
    formData.append('file', selectedFile)

    try {
      const xhr = new XMLHttpRequest()
      xhr.upload.addEventListener('progress', (e) => {
        if (e.lengthComputable) setUploadProgress(Math.round((e.loaded / e.total) * 100))
      })
      xhr.onload = () => {
        if (xhr.status === 200) {
          const response = JSON.parse(xhr.responseText)
          navigate(`/analysis/${response.job_id}`)
        } else {
          alert('Upload failed')
          setUploading(false)
        }
      }
      xhr.onerror = () => { alert('Upload failed'); setUploading(false) }
      xhr.open('POST', `${API_BASE}/videos/upload`)
      xhr.send(formData)
    } catch {
      setUploading(false)
    }
  }

  return (
    <div style={{ maxWidth: 880, margin: '0 auto' }}>
      {/* Header */}
      <div>
        <div className="kv-eyebrow">▶ New Analysis · Ashley Furniture</div>
        <h1 className="kv-display" style={{
          fontSize: 44, lineHeight: 1.05, color: 'var(--kv-black)', margin: '10px 0',
        }}>
          Upload floor footage<span className="kv-dot">.</span>
        </h1>
        <p style={{
          fontSize: 16, color: 'var(--kv-n-600)', lineHeight: 1.55, maxWidth: 640, margin: 0,
        }}>
          Drop a warehouse video and we'll identify associates, forklifts, pallets and material-handling
          events — frame-by-frame — with bounding-box overlays and a timestamped activity feed.
        </p>
      </div>

      {/* Dropzone */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => document.getElementById('fileInput')?.click()}
        style={{
          marginTop: 28, border: `1.5px dashed ${isDragging ? 'var(--kv-blue)' : 'var(--kv-n-300)'}`,
          background: isDragging ? 'var(--kv-blue-50)' : 'var(--kv-white)',
          borderRadius: 12, padding: 56, textAlign: 'center' as const,
          cursor: 'pointer', transition: 'border-color 200ms, background 200ms',
        }}
      >
        <input id="fileInput" type="file" accept="video/*" onChange={handleFileSelect} style={{ display: 'none' }} />
        <svg style={{ color: 'var(--kv-n-400)' }} width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="17 8 12 3 7 8" /><line x1="12" y1="3" x2="12" y2="15" />
        </svg>
        <div style={{
          fontFamily: 'var(--font-display)', fontSize: 16, color: 'var(--kv-black)',
          letterSpacing: '0.01em', marginTop: 14,
        }}>
          {selectedFile ? selectedFile.name : 'Drop video file or click to browse'}
        </div>
        <div style={{
          fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--kv-n-500)',
          marginTop: 6, letterSpacing: '0.08em',
        }}>
          {selectedFile
            ? `${(selectedFile.size / (1024 * 1024)).toFixed(1)} MB · ready`
            : 'MP4 · MOV · AVI · MKV'}
        </div>
      </div>

      {/* Upload progress */}
      {uploading && (
        <div style={{
          marginTop: 16, background: 'var(--kv-white)', border: '1px solid var(--kv-n-200)',
          borderRadius: 8, padding: 14,
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8, fontSize: 12 }}>
            <span style={{ color: 'var(--kv-n-600)' }}>Uploading…</span>
            <span className="mono" style={{ fontWeight: 700, color: 'var(--kv-ink)' }}>{uploadProgress}%</span>
          </div>
          <div style={{ height: 4, background: 'var(--kv-n-200)', borderRadius: 2, overflow: 'hidden' }}>
            <div style={{ height: '100%', background: 'var(--kv-blue)', width: `${uploadProgress}%`, transition: 'width 200ms' }} />
          </div>
        </div>
      )}

      {/* Start button */}
      <button
        onClick={handleUpload}
        disabled={!selectedFile || uploading}
        style={{
          width: '100%', padding: '12px 20px', borderRadius: 4, border: 'none',
          fontFamily: 'var(--font-sans)', fontWeight: 700, fontSize: 14, letterSpacing: '0.02em',
          background: !selectedFile || uploading ? 'var(--kv-n-300)' : 'var(--kv-blue)',
          color: 'var(--kv-white)', cursor: !selectedFile || uploading ? 'not-allowed' : 'pointer',
          display: 'inline-flex', alignItems: 'center', justifyContent: 'center', gap: 10,
          marginTop: 16, transition: 'background 140ms',
        }}
      >
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="17 8 12 3 7 8" /><line x1="12" y1="3" x2="12" y2="15" />
        </svg>
        {uploading ? 'Uploading…' : 'Start Analysis'}
      </button>

      {/* Presets */}
      <div style={{ marginTop: 48 }}>
        <div className="kv-eyebrow" style={{ marginBottom: 14 }}>Detection presets</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12 }}>
          {presets.map((p, i) => (
            <button
              key={p.name}
              onClick={() => setActivePreset(i)}
              style={{
                textAlign: 'left' as const, padding: 16,
                background: 'var(--kv-white)', border: `1px solid ${i === activePreset ? 'var(--kv-blue)' : 'var(--kv-n-200)'}`,
                borderRadius: 10, cursor: 'pointer', position: 'relative' as const,
                transition: 'border-color 140ms',
              }}
            >
              {i === activePreset && (
                <span style={{
                  position: 'absolute' as const, top: 0, left: 0, bottom: 0, width: 3,
                  background: 'var(--kv-blue)', borderRadius: '10px 0 0 10px',
                }} />
              )}
              <div style={{
                fontFamily: 'var(--font-display)', fontSize: 13, color: 'var(--kv-black)',
                marginBottom: 4, letterSpacing: '0.01em',
              }}>{p.name}</div>
              <div style={{ fontSize: 12, color: 'var(--kv-n-600)', lineHeight: 1.5 }}>{p.desc}</div>
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
