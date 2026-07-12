import 'dotenv/config';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import express, { type NextFunction, type Request, type Response } from 'express';
import cookieSession from 'cookie-session';
import { OAuth2Client } from 'google-auth-library';
import { ChatGoogleGenerativeAI } from '@langchain/google-genai';
import { AIMessage, HumanMessage, SystemMessage, ToolMessage, type BaseMessage } from '@langchain/core/messages';
import { appendMemory, formatMemoriesForPrompt, readMemories } from './memoryStore.js';
import { chatTools } from './tools.js';
import { retrieveRelevantNotes } from './rag.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SYSTEM_PROMPT_PATH = path.resolve(__dirname, '../prompts/system.md');
const MODEL = 'gemini-flash-latest';
const PORT = Number(process.env.PORT ?? 3001);
const GOOGLE_CLIENT_ID = process.env.GOOGLE_CLIENT_ID ?? '';
const COOKIE_SECRET = process.env.COOKIE_SECRET ?? 'dev-only-insecure-secret-change-me';

const SUMMARY_INSTRUCTION =
  'Summarize this conversation in 2-4 sentences, capturing durable facts, feelings, or context ' +
  'about the user that would help a friend remember them next time. Skip small talk and filler. ' +
  'Write only the summary, no preamble.';

const MOOD_LABELS = ['happy', 'sad', 'angry', 'neutral'] as const;
type MoodLabel = (typeof MOOD_LABELS)[number];

function isMoodLabel(value: unknown): value is MoodLabel {
  return typeof value === 'string' && (MOOD_LABELS as readonly string[]).includes(value);
}

const MOOD_PROMPTS: Record<Exclude<MoodLabel, 'neutral'>, string> = {
  sad: "They seem a little down. Gently ask what's going on or why they seem sad, the way a close friend would - caring and direct, not clinical.",
  angry: "They seem frustrated or upset. Gently ask what's bothering them or what happened, the way a close friend would - caring and direct, not clinical.",
  happy: "They seem in a good mood. Warmly ask what's making them happy or what's got them smiling, the way a close friend would.",
};

function greetingInstruction(mood: MoodLabel | null): string {
  if (mood === null || mood === 'neutral') {
    return 'Write a short (1-2 sentence), warm opening greeting to start a new conversation. Do not mention mood or emotion detection at all.';
  }
  return (
    `Write a short (1-2 sentence) opening greeting to start a new conversation. ${MOOD_PROMPTS[mood]} ` +
    `Never say things like "I detected" or "my analysis shows" or name it as data from a system - just notice ` +
    `it the way an attentive friend naturally would.`
  );
}

interface ChatMessage {
  role: 'user' | 'model';
  content: string;
}

const apiKey = process.env.GEMINI_API_KEY;
const model = apiKey ? new ChatGoogleGenerativeAI({ model: MODEL, apiKey }) : null;
const modelWithTools = model ? model.bindTools(chatTools) : null;
const googleClient = new OAuth2Client(GOOGLE_CLIENT_ID);

const app = express();
app.use(express.json());
app.use(
  cookieSession({
    name: 'session',
    keys: [COOKIE_SECRET],
    maxAge: 30 * 24 * 60 * 60 * 1000,
    httpOnly: true,
    sameSite: 'lax',
  }),
);

function requireAuth(req: Request, res: Response, next: NextFunction): void {
  if (!req.session?.userId) {
    res.status(401).json({ error: 'Not signed in' });
    return;
  }
  next();
}

app.post('/api/auth/google', async (req, res) => {
  const { credential } = req.body as { credential?: string };
  if (!credential) {
    res.status(400).json({ error: 'Missing credential' });
    return;
  }

  try {
    const ticket = await googleClient.verifyIdToken({ idToken: credential, audience: GOOGLE_CLIENT_ID });
    const payload = ticket.getPayload();
    if (!payload?.sub) {
      res.status(401).json({ error: 'Invalid Google token' });
      return;
    }

    req.session = { userId: payload.sub, email: payload.email, name: payload.name };
    res.json({ email: payload.email, name: payload.name });
  } catch (err) {
    res.status(401).json({ error: err instanceof Error ? err.message : 'Google sign-in failed' });
  }
});

app.post('/api/auth/logout', (req, res) => {
  req.session = null;
  res.json({ ok: true });
});

app.get('/api/auth/me', (req, res) => {
  if (!req.session?.userId) {
    res.status(401).json({ error: 'Not signed in' });
    return;
  }
  res.json({ email: req.session.email, name: req.session.name });
});

app.post('/api/chat', requireAuth, async (req, res) => {
  if (!modelWithTools) {
    res.status(500).json({ error: 'GEMINI_API_KEY is not set in web/.env' });
    return;
  }

  try {
    const userId = req.session!.userId as string;
    const { messages } = req.body as { messages: ChatMessage[] };
    const latestUserMessage = [...messages].reverse().find((m) => m.role === 'user')?.content ?? '';

    const relevantNotes = apiKey ? await retrieveRelevantNotes(apiKey, latestUserMessage) : '';
    const systemPrompt =
      fs.readFileSync(SYSTEM_PROMPT_PATH, 'utf-8') + formatMemoriesForPrompt(readMemories(userId)) + relevantNotes;
    const chatMessages: BaseMessage[] = [
      new SystemMessage(systemPrompt),
      ...messages.map((m) => (m.role === 'user' ? new HumanMessage(m.content) : new AIMessage(m.content))),
    ];

    let response = await modelWithTools.invoke(chatMessages);

    // If the model asked to call a tool, run it and feed the result back in,
    // repeating until it's satisfied and produces a final natural-language reply.
    while (response.tool_calls && response.tool_calls.length > 0) {
      chatMessages.push(response);
      for (const call of response.tool_calls) {
        const matchedTool = chatTools.find((t) => t.name === call.name);
        const result = matchedTool
          ? await matchedTool.invoke(call.args as Record<string, never>)
          : `Unknown tool: ${call.name}`;
        chatMessages.push(new ToolMessage(String(result), call.id ?? ''));
      }
      response = await modelWithTools.invoke(chatMessages);
    }

    res.json({ reply: response.content });
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : 'Unknown error' });
  }
});

app.post('/api/greeting', requireAuth, async (req, res) => {
  if (!model) {
    res.status(500).json({ error: 'GEMINI_API_KEY is not set in web/.env' });
    return;
  }

  try {
    const userId = req.session!.userId as string;
    const { mood } = req.body as { mood?: unknown };
    const moodLabel = isMoodLabel(mood) ? mood : null;

    const systemPrompt =
      fs.readFileSync(SYSTEM_PROMPT_PATH, 'utf-8') + formatMemoriesForPrompt(readMemories(userId));
    const response = await model.invoke([
      new SystemMessage(systemPrompt),
      new HumanMessage(greetingInstruction(moodLabel)),
    ]);

    res.json({ greeting: String(response.content) });
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : 'Unknown error' });
  }
});

app.post('/api/end-session', requireAuth, async (req, res) => {
  if (!model) {
    res.status(500).json({ error: 'GEMINI_API_KEY is not set in web/.env' });
    return;
  }

  try {
    const userId = req.session!.userId as string;
    const { messages } = req.body as { messages: ChatMessage[] };

    if (messages.length === 0) {
      res.json({ summary: '' });
      return;
    }

    const transcript = messages.map((m) => `${m.role}: ${m.content}`).join('\n');
    const response = await model.invoke([new SystemMessage(SUMMARY_INSTRUCTION), new HumanMessage(transcript)]);
    const summary = String(response.content);
    appendMemory(userId, summary);

    res.json({ summary });
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : 'Unknown error' });
  }
});

// In production there's no Vite dev server, so this process also serves the built frontend.
if (process.env.NODE_ENV === 'production') {
  const distPath = path.resolve(__dirname, '../dist');
  app.use(express.static(distPath));
  app.get('*', (_req, res) => res.sendFile(path.join(distPath, 'index.html')));
}

app.listen(PORT, () => {
  console.log(`API server listening on http://localhost:${PORT}`);
});
