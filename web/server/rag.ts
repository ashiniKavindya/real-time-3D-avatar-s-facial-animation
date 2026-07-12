import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const NOTES_DIR = path.resolve(__dirname, '../notes');
const EMBEDDING_MODEL = 'gemini-embedding-001';
const TOP_K = 3;
// Below this similarity, a note is treated as irrelevant rather than force-fit into the prompt.
const MIN_SIMILARITY = 0.7;

interface NoteChunk {
  text: string;
  embedding: number[];
}

let embeddings: GoogleGenerativeAIEmbeddings | null = null;
let chunksPromise: Promise<NoteChunk[]> | null = null;

function splitIntoChunks(text: string): string[] {
  return text
    .split(/\n\s*\n/)
    .map((chunk) => chunk.trim())
    .filter((chunk) => chunk.length > 0);
}

function loadRawChunks(): string[] {
  if (!fs.existsSync(NOTES_DIR)) return [];
  const files = fs.readdirSync(NOTES_DIR).filter((f) => f.endsWith('.md') || f.endsWith('.txt'));
  return files.flatMap((file) => splitIntoChunks(fs.readFileSync(path.join(NOTES_DIR, file), 'utf-8')));
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

// Chunks are embedded once and cached in memory for the life of the process -
// re-embedding on every chat message would be slow and unnecessary since the
// notes on disk don't change while the server is running.
async function getChunks(apiKey: string): Promise<NoteChunk[]> {
  if (!chunksPromise) {
    embeddings = new GoogleGenerativeAIEmbeddings({ apiKey, model: EMBEDDING_MODEL });
    const rawChunks = loadRawChunks();
    chunksPromise =
      rawChunks.length === 0
        ? Promise.resolve([])
        : embeddings
            .embedDocuments(rawChunks)
            .then((vectors) => rawChunks.map((text, i) => ({ text, embedding: vectors[i] })));
  }
  return chunksPromise;
}

export async function retrieveRelevantNotes(apiKey: string, query: string): Promise<string> {
  const chunks = await getChunks(apiKey);
  if (chunks.length === 0 || !embeddings) return '';

  const queryEmbedding = await embeddings.embedQuery(query);
  const scored = chunks
    .map((chunk) => ({ ...chunk, score: cosineSimilarity(queryEmbedding, chunk.embedding) }))
    .sort((a, b) => b.score - a.score)
    .filter((chunk) => chunk.score >= MIN_SIMILARITY)
    .slice(0, TOP_K);

  if (scored.length === 0) return '';

  const bullets = scored.map((c) => `- ${c.text.replace(/\n/g, ' ')}`).join('\n');
  return `\n\n## Relevant notes about this user:\n${bullets}`;
}
