import { ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { Video, BarChart3 } from 'lucide-react'

interface LayoutProps {
  children: ReactNode
}

export function Layout({ children }: LayoutProps) {
  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <nav className="bg-gray-800 border-b border-gray-700">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <Link to="/" className="flex items-center gap-2 text-xl font-bold">
              <Video className="w-6 h-6 text-blue-400" />
              <span>Intelligent Video</span>
            </Link>
            <div className="flex items-center gap-6">
              <Link to="/" className="hover:text-blue-400 transition">
                Upload
              </Link>
              <a href="#" className="hover:text-blue-400 transition flex items-center gap-1">
                <BarChart3 className="w-4 h-4" />
                Dashboard
              </a>
            </div>
          </div>
        </div>
      </nav>
      <main className="max-w-7xl mx-auto px-4 py-8">
        {children}
      </main>
    </div>
  )
}
