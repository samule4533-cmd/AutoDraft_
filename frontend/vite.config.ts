import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5174,
    proxy: {
      '/chat': 'http://localhost:8001',
      '/health': 'http://localhost:8001',
      '/summaries': 'http://localhost:8001',
    },
  },
})
