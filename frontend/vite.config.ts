import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      // `ws: true` is required so the real-time detection WebSocket
      // (/api/streams/ws/:jobId) is proxied to the backend in dev.
      '/api': {
        target: 'http://localhost:8000',
        ws: true,
      },
    },
  },
})
