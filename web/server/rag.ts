import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { db } from './db.js';

const EMBEDDING_MODEL = 'gemini-embedding-001';
const TOP_K = 3;
// Below this similarity, a note is treated as irrelevant rather than force-fit into the prompt.
const MIN_SIMILARITY = 0.7;

let embeddings: GoogleGenerativeAIEmbeddings | null = null;

function getEmbeddings(apiKey: string): GoogleGenerativeAIEmbeddings {
  if (!embeddings) {
    embeddings = new GoogleGenerativeAIEmbeddings({ apiKey, model: EMBEDDING_MODEL });
  }
  return embeddings;
}

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

export interface NoteRecord {
  id: number;
  content: string;
  createdAt: string;
}

export async function addNote(apiKey: string, userId: string, content: string): Promise<void> {
  const vector = await getEmbeddings(apiKey).embedQuery(content);
  db.prepare('INSERT INTO notes (user_id, content, embedding, created_at) VALUES (?, ?, ?, ?)').run(
    userId,
    content,
    JSON.stringify(vector),
    new Date().toISOString(),
  );
}

export function listNotes(userId: string): NoteRecord[] {
  const rows = db
    .prepare('SELECT id, content, created_at FROM notes WHERE user_id = ? ORDER BY id DESC')
    .all(userId);
  return rows.map((row) => ({ id: Number(row.id), content: String(row.content), createdAt: String(row.created_at) }));
}

export function deleteNote(userId: string, noteId: number): void {
  db.prepare('DELETE FROM notes WHERE id = ? AND user_id = ?').run(noteId, userId);
}

export async function retrieveRelevantNotes(apiKey: string, userId: string, query: string): Promise<string> {
  const rows = db.prepare('SELECT content, embedding FROM notes WHERE user_id = ?').all(userId);
  if (rows.length === 0 || !query.trim()) return '';

  const chunks = rows.map((row) => ({
    text: String(row.content),
    embedding: JSON.parse(String(row.embedding)) as number[],
  }));

  const queryEmbedding = await getEmbeddings(apiKey).embedQuery(query);
  const scored = chunks
    .map((chunk) => ({ ...chunk, score: cosineSimilarity(queryEmbedding, chunk.embedding) }))
    .sort((a, b) => b.score - a.score)
    .filter((chunk) => chunk.score >= MIN_SIMILARITY)
    .slice(0, TOP_K);

  if (scored.length === 0) return '';

  const bullets = scored.map((c) => `- ${c.text.replace(/\n/g, ' ')}`).join('\n');
  return `\n\n## Relevant notes about this user:\n${bullets}`;
}
