import { useCallback, useRef, useState } from 'react';
import { sendChatMessage } from '../lib/chatClient';
import type { ChatMessage } from '../types/chat';

export function ChatUI() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const listEndRef = useRef<HTMLDivElement>(null);

  const handleSend = useCallback(async () => {
    const text = input.trim();
    if (!text || isSending) return;

    const nextMessages: ChatMessage[] = [...messages, { role: 'user', content: text }];
    setMessages(nextMessages);
    setInput('');
    setError(null);
    setIsSending(true);

    try {
      const reply = await sendChatMessage(nextMessages);
      setMessages([...nextMessages, { role: 'model', content: reply }]);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong.');
    } finally {
      setIsSending(false);
      listEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [input, isSending, messages]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLInputElement>) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        void handleSend();
      }
    },
    [handleSend],
  );

  return (
    <div className="chat-ui">
      <div className="chat-messages">
        {messages.length === 0 && <p className="chat-empty">Say hello to start chatting.</p>}
        {messages.map((m, i) => (
          <div key={i} className={`chat-message chat-message-${m.role}`}>
            {m.content}
          </div>
        ))}
        {isSending && <div className="chat-message chat-message-model chat-typing">...</div>}
        <div ref={listEndRef} />
      </div>

      {error && <p className="chat-error">{error}</p>}

      <div className="chat-input-row">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type a message..."
          disabled={isSending}
          className="chat-input"
        />
        <button onClick={() => void handleSend()} disabled={isSending || !input.trim()} className="chat-send-button">
          Send
        </button>
      </div>
    </div>
  );
}
