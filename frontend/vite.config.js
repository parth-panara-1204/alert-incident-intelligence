import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Split heavy deps (react, recharts, axios) into separate chunks to keep bundles small.
export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('node_modules/react')) return 'react'
          if (id.includes('node_modules/recharts')) return 'recharts'
          if (id.includes('node_modules/axios')) return 'axios'
          return undefined
        },
      },
    },
    chunkSizeWarningLimit: 900,
  },
})
