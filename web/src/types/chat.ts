export interface ChatMessage {
  role: 'user' | 'model';
  content: string;
  // Client-side only, for display - not meaningful to the server.
  timestamp?: number;
}
