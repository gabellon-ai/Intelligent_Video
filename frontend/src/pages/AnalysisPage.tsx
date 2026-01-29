import { useEffect, useState, useRef } from 'react'
import { useParams } from 'react-router-dom'
import { Play, SkipBack, SkipForward } from 'lucide-react'
import { VideoPlayer } from '../components/VideoPlayer'
import { DetectionSummary } from '../components/DetectionSummary'
import { Timeline } from '../components/Timeline'
import { ExportDropdown } from '../components/ExportDropdown'

const API_BASE = 'http://localhost:8000/api'
const WS_BASE = 'ws://localhost:8000/api/streams'

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

    // Connect to WebSocket for real-time updates
    const ws = new WebSocket(`${WS_BASE}/ws/${jobId}`)
    wsRef.current = ws

    ws.onopen = () => {
      console.log('WebSocket connected')
      setStatus('processing')
    }

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      
      switch (data.type) {
        case 'start':
          setVideoInfo(data.video_info)
          break
        case 'progress':
          setProgress(data.percent)
          break
        case 'detection':
          setCurrentFrame(data)
          setAllDetections(prev => [...prev, data])
          break
        case 'summary':
          setSummary(data)
          break
        case 'complete':
          setStatus('completed')
          break
        case 'error':
          setStatus('error')
          console.error('Analysis error:', data.message)
          break
      }
    }

    ws.onerror = (error) => {
      console.error('WebSocket error:', error)
      setStatus('error')
    }

    ws.onclose = () => {
      console.log('WebSocket closed')
    }

    return () => {
      ws.close()
    }
  }, [jobId])

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Video Analysis</h1>
          <p className="text-gray-400">Job ID: {jobId}</p>
        </div>
        <ExportDropdown 
          jobId={jobId!} 
          disabled={status !== 'completed'} 
        />
      </div>

      {/* Progress Bar */}
      {status === 'processing' && (
        <div className="bg-gray-800 rounded-lg p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-400">Analyzing video...</span>
            <span className="text-sm font-medium">{progress}%</span>
          </div>
          <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
            <div 
              className="h-full bg-blue-500 transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
          {currentFrame && (
            <p className="text-xs text-gray-500 mt-2">
              Frame {currentFrame.frame} • {currentFrame.detections.length} detections
            </p>
          )}
        </div>
      )}

      <div className="grid grid-cols-3 gap-6">
        {/* Video Player with Overlays */}
        <div className="col-span-2">
          <div className="bg-gray-800 rounded-lg overflow-hidden">
            <VideoPlayer 
              jobId={jobId!}
              currentDetection={currentFrame}
              videoInfo={videoInfo}
            />
            
            {/* Playback Controls */}
            <div className="p-4 border-t border-gray-700">
              <div className="flex items-center justify-center gap-4">
                <button className="p-2 hover:bg-gray-700 rounded-lg transition">
                  <SkipBack className="w-5 h-5" />
                </button>
                <button className="p-3 bg-blue-600 hover:bg-blue-500 rounded-full transition">
                  <Play className="w-6 h-6" />
                </button>
                <button className="p-2 hover:bg-gray-700 rounded-lg transition">
                  <SkipForward className="w-5 h-5" />
                </button>
              </div>
            </div>
          </div>

          {/* Timeline */}
          {allDetections.length > 0 && (
            <div className="mt-4">
              <Timeline 
                detections={allDetections}
                duration={videoInfo?.duration || 0}
              />
            </div>
          )}
        </div>

        {/* Summary Panel */}
        <div className="space-y-4">
          <DetectionSummary 
            summary={summary}
            currentFrame={currentFrame}
            status={status}
          />

          {/* Live Detections Feed */}
          <div className="bg-gray-800 rounded-lg p-4">
            <h3 className="font-semibold mb-3">Live Detections</h3>
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {allDetections.slice(-10).reverse().map((frame, i) => (
                <div key={i} className="text-sm p-2 bg-gray-700/50 rounded">
                  <div className="flex justify-between text-xs text-gray-400 mb-1">
                    <span>Frame {frame.frame}</span>
                    <span>{frame.timestamp.toFixed(2)}s</span>
                  </div>
                  <div className="flex flex-wrap gap-1">
                    {Object.entries(frame.counts).map(([cls, count]) => (
                      <span 
                        key={cls}
                        className="px-2 py-0.5 bg-blue-600/30 rounded text-xs"
                      >
                        {cls}: {count}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
