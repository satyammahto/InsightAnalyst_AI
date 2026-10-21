import express from 'express';
import cors from 'cors';
import path from 'path';
import { fileURLToPath } from 'url';
import { GoogleGenAI } from '@google/genai';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function startServer() {
  const app = express();
  const PORT = process.env.PORT ? parseInt(process.env.PORT, 10) : 3000;
  const isProd = process.env.NODE_ENV === 'production';

  app.use(cors());
  app.use(express.json({ limit: '50mb' }));

  // API Route: Chat with Gemini
  app.post('/api/chat', async (req, res) => {
    try {
      const { message, context, apiKey } = req.body;
      const effectiveKey = apiKey || process.env.GEMINI_API_KEY;

      if (!effectiveKey) {
        return res.status(400).json({
          error: 'No GEMINI_API_KEY configured',
          useFallback: true,
        });
      }

      const ai = new GoogleGenAI({ apiKey: effectiveKey });
      const promptText = `You are InsightAnalyst AI, an enterprise data analyst and statistical diagnostics engine.
Context regarding the user's active dataset and RAG knowledge vectors:
${context || 'No specific dataset context provided.'}

User's Question:
"${message}"

Instructions:
1. Provide a crisp, structured, plain-English executive explanation grounded in the data.
2. If citing numbers or trends, explain both What happened and Why it matters.
3. Suggest 1-2 actionable next steps or SQL queries if relevant.`;

      const response = await ai.models.generateContent({
        model: 'gemini-2.0-flash',
        contents: promptText,
      });

      res.json({
        text: response.text || 'No response generated.',
        success: true,
      });
    } catch (err: any) {
      console.error('Gemini API Error:', err);
      res.status(500).json({
        error: err.message || 'Failed to generate response',
        useFallback: true,
      });
    }
  });

  // Health check
  app.get('/api/health', (req, res) => {
    res.json({ status: 'ok', service: 'InsightAnalyst AI', timestamp: new Date().toISOString() });
  });

  if (!isProd) {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true, host: '0.0.0.0', port: PORT },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.resolve(__dirname, 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.resolve(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`⚡ InsightAnalyst AI running at http://0.0.0.0:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error('Failed to start server:', err);
  process.exit(1);
});
