import { useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { Upload, FileVideo, Loader2, CheckCircle } from 'lucide-react'

const API_BASE = 'http://localhost:8000/api'

export function UploadPage() {
  const [isDragging, setIsDragging] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
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
    if (file) {
      setSelectedFile(file)
    }
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
        if (e.lengthComputable) {
          setUploadProgress(Math.round((e.loaded / e.total) * 100))
        }
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

      xhr.onerror = () => {
        alert('Upload failed')
        setUploading(false)
      }

      xhr.open('POST', `${API_BASE}/videos/upload`)
      xhr.send(formData)
    } catch (error) {
      console.error('Upload error:', error)
      setUploading(false)
    }
  }

  return (
    <div className="max-w-2xl mx-auto">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold mb-2">Upload Video for Analysis</h1>
        <p className="text-gray-400">
          Drag & drop a warehouse video or click to browse
        </p>
      </div>

      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`
          border-2 border-dashed rounded-xl p-12 text-center transition-all cursor-pointer
          ${isDragging 
            ? 'border-blue-400 bg-blue-400/10' 
            : 'border-gray-600 hover:border-gray-500 bg-gray-800/50'
          }
        `}
        onClick={() => document.getElementById('fileInput')?.click()}
      >
        <input
          id="fileInput"
          type="file"
          accept="video/*"
          onChange={handleFileSelect}
          className="hidden"
        />

        {selectedFile ? (
          <div className="space-y-4">
            <FileVideo className="w-16 h-16 mx-auto text-blue-400" />
            <div>
              <p className="font-medium text-lg">{selectedFile.name}</p>
              <p className="text-gray-400 text-sm">
                {(selectedFile.size / (1024 * 1024)).toFixed(1)} MB
              </p>
            </div>
            <CheckCircle className="w-6 h-6 mx-auto text-green-400" />
          </div>
        ) : (
          <div className="space-y-4">
            <Upload className="w-16 h-16 mx-auto text-gray-500" />
            <div>
              <p className="font-medium text-lg">Drop video here</p>
              <p className="text-gray-400 text-sm">or click to browse</p>
            </div>
          </div>
        )}
      </div>

      {selectedFile && (
        <div className="mt-6 space-y-4">
          {uploading && (
            <div className="bg-gray-800 rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-400">Uploading...</span>
                <span className="text-sm font-medium">{uploadProgress}%</span>
              </div>
              <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-blue-500 transition-all duration-300"
                  style={{ width: `${uploadProgress}%` }}
                />
              </div>
            </div>
          )}

          <button
            onClick={handleUpload}
            disabled={uploading}
            className={`
              w-full py-3 px-6 rounded-lg font-medium transition-all
              flex items-center justify-center gap-2
              ${uploading 
                ? 'bg-gray-700 text-gray-400 cursor-not-allowed' 
                : 'bg-blue-600 hover:bg-blue-500 text-white'
              }
            `}
          >
            {uploading ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                Uploading...
              </>
            ) : (
              <>
                <Upload className="w-5 h-5" />
                Start Analysis
              </>
            )}
          </button>
        </div>
      )}

      {/* Detection presets */}
      <div className="mt-12">
        <h2 className="text-lg font-semibold mb-4">Detection Presets</h2>
        <div className="grid grid-cols-2 gap-4">
          {[
            { name: 'Warehouse - General', desc: 'Forklifts, pallets, people, boxes' },
            { name: 'Safety Focus', desc: 'People, vests, zone violations' },
            { name: 'AGV Tracking', desc: 'Robots, AMRs, autonomous vehicles' },
            { name: 'Loading Dock', desc: 'Trucks, trailers, dock activity' },
          ].map((preset) => (
            <div
              key={preset.name}
              className="p-4 bg-gray-800 rounded-lg border border-gray-700 hover:border-gray-600 cursor-pointer transition"
            >
              <p className="font-medium">{preset.name}</p>
              <p className="text-sm text-gray-400">{preset.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
