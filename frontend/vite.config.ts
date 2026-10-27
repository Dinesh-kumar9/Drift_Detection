import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // File-change events don't cross Docker bind mounts on Windows/macOS hosts;
    // docker-compose sets VITE_USE_POLLING so hot reload still works in the container.
    watch: process.env.VITE_USE_POLLING === 'true' ? { usePolling: true, interval: 300 } : undefined,
  },
})
