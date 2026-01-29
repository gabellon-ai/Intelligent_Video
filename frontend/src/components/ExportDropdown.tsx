import { useState, useRef, useEffect } from 'react'
import { Download, FileText, FileSpreadsheet, FileJson, ChevronDown, Loader2, Check } from 'lucide-react'

const API_BASE = 'http://localhost:8000/api'

interface ExportDropdownProps {
  jobId: string
  disabled?: boolean
}

type ExportFormat = 'csv' | 'pdf' | 'json'
type DownloadStatus = 'idle' | 'downloading' | 'success' | 'error'

interface ExportOption {
  format: ExportFormat
  label: string
  description: string
  icon: typeof FileText
}

const exportOptions: ExportOption[] = [
  {
    format: 'pdf',
    label: 'PDF Report',
    description: 'Comprehensive report with charts and key frames',
    icon: FileText
  },
  {
    format: 'csv',
    label: 'CSV Data',
    description: 'Frame-by-frame detections for analysis',
    icon: FileSpreadsheet
  },
  {
    format: 'json',
    label: 'JSON Export',
    description: 'Full analysis data for integration',
    icon: FileJson
  }
]

export function ExportDropdown({ jobId, disabled }: ExportDropdownProps) {
  const [isOpen, setIsOpen] = useState(false)
  const [downloadStatus, setDownloadStatus] = useState<Record<ExportFormat, DownloadStatus>>({
    csv: 'idle',
    pdf: 'idle',
    json: 'idle'
  })
  const dropdownRef = useRef<HTMLDivElement>(null)

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  const handleExport = async (format: ExportFormat) => {
    if (downloadStatus[format] === 'downloading') return

    setDownloadStatus(prev => ({ ...prev, [format]: 'downloading' }))

    try {
      const response = await fetch(`${API_BASE}/videos/${jobId}/export/${format}`)
      
      if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Export failed' }))
        throw new Error(error.detail || 'Export failed')
      }

      // Get filename from content-disposition header
      const contentDisposition = response.headers.get('content-disposition')
      let filename = `analysis_${jobId.slice(0, 8)}.${format}`
      if (contentDisposition) {
        const match = contentDisposition.match(/filename="?([^";\n]+)"?/)
        if (match) filename = match[1]
      }

      // Create blob and download
      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = filename
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(url)
      document.body.removeChild(a)

      setDownloadStatus(prev => ({ ...prev, [format]: 'success' }))
      
      // Reset success status after 2 seconds
      setTimeout(() => {
        setDownloadStatus(prev => ({ ...prev, [format]: 'idle' }))
      }, 2000)

    } catch (error) {
      console.error(`Export ${format} failed:`, error)
      setDownloadStatus(prev => ({ ...prev, [format]: 'error' }))
      
      // Reset error status after 3 seconds
      setTimeout(() => {
        setDownloadStatus(prev => ({ ...prev, [format]: 'idle' }))
      }, 3000)
    }
  }

  const getStatusIcon = (format: ExportFormat, Icon: typeof FileText) => {
    const status = downloadStatus[format]
    
    if (status === 'downloading') {
      return <Loader2 className="w-5 h-5 animate-spin text-blue-400" />
    }
    if (status === 'success') {
      return <Check className="w-5 h-5 text-green-400" />
    }
    return <Icon className="w-5 h-5 text-gray-400 group-hover:text-blue-400" />
  }

  return (
    <div className="relative" ref={dropdownRef}>
      {/* Trigger Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        disabled={disabled}
        className={`
          flex items-center gap-2 px-4 py-2 rounded-lg transition
          ${disabled 
            ? 'bg-gray-700 text-gray-500 cursor-not-allowed' 
            : 'bg-blue-600 hover:bg-blue-500 text-white'
          }
        `}
      >
        <Download className="w-4 h-4" />
        <span>Export</span>
        <ChevronDown className={`w-4 h-4 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
      </button>

      {/* Dropdown Menu */}
      {isOpen && !disabled && (
        <div className="absolute right-0 mt-2 w-72 bg-gray-800 rounded-lg shadow-xl border border-gray-700 overflow-hidden z-50">
          <div className="p-2">
            <p className="text-xs text-gray-500 px-3 py-2 uppercase font-medium">
              Export Analysis Results
            </p>
            
            {exportOptions.map(({ format, label, description, icon: Icon }) => (
              <button
                key={format}
                onClick={() => handleExport(format)}
                disabled={downloadStatus[format] === 'downloading'}
                className={`
                  group w-full flex items-start gap-3 px-3 py-3 rounded-lg transition
                  ${downloadStatus[format] === 'downloading' 
                    ? 'bg-gray-700/50 cursor-wait' 
                    : downloadStatus[format] === 'success'
                    ? 'bg-green-900/30'
                    : downloadStatus[format] === 'error'
                    ? 'bg-red-900/30'
                    : 'hover:bg-gray-700/50'
                  }
                `}
              >
                <div className="flex-shrink-0 mt-0.5">
                  {getStatusIcon(format, Icon)}
                </div>
                <div className="flex-1 text-left">
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-white">{label}</span>
                    {downloadStatus[format] === 'success' && (
                      <span className="text-xs text-green-400">Downloaded!</span>
                    )}
                    {downloadStatus[format] === 'error' && (
                      <span className="text-xs text-red-400">Failed</span>
                    )}
                  </div>
                  <p className="text-xs text-gray-400 mt-0.5">{description}</p>
                </div>
              </button>
            ))}
          </div>
          
          {/* Progress indicator for any active download */}
          {Object.values(downloadStatus).includes('downloading') && (
            <div className="border-t border-gray-700 px-4 py-2 bg-gray-900/50">
              <div className="flex items-center gap-2 text-xs text-gray-400">
                <Loader2 className="w-3 h-3 animate-spin" />
                <span>Generating export...</span>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
