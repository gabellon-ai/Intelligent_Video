import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { UploadPage } from './pages/UploadPage'
import { AnalysisPage } from './pages/AnalysisPage'
import { Layout } from './components/Layout'
import './App.css'

function App() {
  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<UploadPage />} />
          <Route path="/analysis/:jobId" element={<AnalysisPage />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  )
}

export default App
