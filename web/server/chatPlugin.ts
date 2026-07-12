import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import type { IncomingMessage } from 'node:http';
import type { Plugin } from 'vite';
import { ChatGoogleGenerativeAI } from '@langchain/google-genai';
import { AIMessage, HumanMessage, SystemMessage, type BaseMessage } from '@langchain/core/messages';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SYSTEM_PROMPT_PATH = path.resolve(__dirname, '../prompts/system.md');
const MODEL = 'gemini-flash-latest';

interface ChatMessage {
  role: 'user' | 'model';
  content: string;
}

interface ChatRequestBody {
  messages: ChatMessage[];
}

async function readRequestBody(req: IncomingMessage): Promise<string> {
  const chunks: Buffer[] = [];
  for await (const chunk of req) {
    chunks.push(chunk as Buffer);
  }
  return Buffer.concat(chunks).toString('utf-8');
}

export function chatPlugin(): Plugin {
  return {
    name: 'chat-api',
    configureServer(server) {
      const apiKey = process.env.GEMINI_API_KEY;
      const model = apiKey ? new ChatGoogleGenerativeAI({ model: MODEL, apiKey }) : null;

      server.middlewares.use('/api/chat', async (req, res) => {
        if (req.method !== 'POST') {
          res.statusCode = 405;
          res.end('Method not allowed');
          return;
        }

        if (!model) {
          res.statusCode = 500;
          res.setHeader('Content-Type', 'application/json');
          res.end(JSON.stringify({ error: 'GEMINI_API_KEY is not set in web/.env' }));
          return;
        }

        try {
          const raw = await readRequestBody(req);
          const body = JSON.parse(raw) as ChatRequestBody;

          const systemPrompt = fs.readFileSync(SYSTEM_PROMPT_PATH, 'utf-8');
          const messages: BaseMessage[] = [
            new SystemMessage(systemPrompt),
            ...body.messages.map((m) =>
              m.role === 'user' ? new HumanMessage(m.content) : new AIMessage(m.content),
            ),
          ];

          const response = await model.invoke(messages);

          res.setHeader('Content-Type', 'application/json');
          res.end(JSON.stringify({ reply: response.content }));
        } catch (err) {
          res.statusCode = 500;
          res.setHeader('Content-Type', 'application/json');
          res.end(JSON.stringify({ error: err instanceof Error ? err.message : 'Unknown error' }));
        }
      });
    },
  };
}
