import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Tryb deweloperski: `npm run dev` serwuje interfejs na :5173, a wywołania /api idą do silnika na :8090.
// Produkcja: `npm run build` → ui/dist, które FastAPI serwuje pod "/" (patrz lexrag/web.py).
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { '/api': { target: process.env.LEXRAG_API ?? 'http://localhost:8090', changeOrigin: true } },
  },
  build: { outDir: 'dist', sourcemap: false },
})
