import { db } from './db.js';

interface MemoryEntry {
  timestamp: string;
  summary: string;
}

export function readMemories(userId: string): MemoryEntry[] {
  const rows = db
    .prepare('SELECT timestamp, summary FROM memories WHERE user_id = ? ORDER BY id ASC')
    .all(userId);
  return rows.map((row) => ({ timestamp: String(row.timestamp), summary: String(row.summary) }));
}

export function appendMemory(userId: string, summary: string): void {
  db.prepare('INSERT INTO memories (user_id, timestamp, summary) VALUES (?, ?, ?)').run(
    userId,
    new Date().toISOString(),
    summary,
  );
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
