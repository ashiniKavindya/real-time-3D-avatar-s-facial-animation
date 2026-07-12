import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // The real backend is the standalone Express server (server/index.ts),
    // run separately via `npm run dev:server` and proxied here in dev only.
    proxy: {
      '/api': 'http://localhost:3001',
    },
  },
})
