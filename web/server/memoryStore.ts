import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DATA_DIR = path.resolve(__dirname, '../data');

interface MemoryEntry {
  timestamp: string;
  summary: string;
}

// Google's `sub` claim is a numeric string, so this is already filesystem-safe,
// but we strip anything unexpected defensively before it touches a path.
function memoryFilePath(userId: string): string {
  const safeId = userId.replace(/[^a-zA-Z0-9_-]/g, '');
  return path.join(DATA_DIR, `memory-${safeId}.json`);
}

function ensureDataDir(): void {
  if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });
}

export function readMemories(userId: string): MemoryEntry[] {
  const file = memoryFilePath(userId);
  if (!fs.existsSync(file)) return [];
  try {
    return JSON.parse(fs.readFileSync(file, 'utf-8')) as MemoryEntry[];
  } catch {
    return [];
  }
}

export function appendMemory(userId: string, summary: string): void {
  ensureDataDir();
  const memories = readMemories(userId);
  memories.push({ timestamp: new Date().toISOString(), summary });
  fs.writeFileSync(memoryFilePath(userId), JSON.stringify(memories, null, 2), 'utf-8');
}

// Only the most recent entries are injected into the prompt so it doesn't
// grow unbounded as sessions accumulate over weeks/months.
export function formatMemoriesForPrompt(memories: MemoryEntry[], limit = 10): string {
  if (memories.length === 0) return '';
  const bullets = memories
    .slice(-limit)
    .map((m) => `- (${m.timestamp.slice(0, 10)}) ${m.summary}`)
    .join('\n');
  return `\n\n## What you remember about this user from past conversations:\n${bullets}`;
}
