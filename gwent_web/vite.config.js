import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';

export default defineConfig({
  plugins: [vue()],
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
  server: {
    fs: {
      // Allow importing shared JSON data from ../shared/
      allow: ['..'],
    },
  },
});
