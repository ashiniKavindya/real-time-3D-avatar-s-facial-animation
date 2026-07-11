import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import { chatPlugin } from './server/chatPlugin.js'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  // loadEnv reads web/.env without exposing it to import.meta.env/the client bundle.
  const env = loadEnv(mode, process.cwd(), '')
  process.env.GEMINI_API_KEY = env.GEMINI_API_KEY

  return {
    plugins: [react(), chatPlugin()],
  }
})
