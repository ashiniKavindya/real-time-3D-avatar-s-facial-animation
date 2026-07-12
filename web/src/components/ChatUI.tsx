import { useCallback, useEffect, useRef, useState } from 'react';
import { endChatSession, fetchGreeting, sendChatMessage } from '../lib/chatClient';
import type { ChatMessage } from '../types/chat';

interface ChatUIProps {
  // undefined = still waiting to know the mood (don't greet yet); null = known, no mood/camera off.
  moodHint?: 'happy' | 'sad' | 'angry' | 'neutral' | null;
}

export function ChatUI({ moodHint }: ChatUIProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [savedNotice, setSavedNotice] = useState<string | null>(null);
  const listEndRef = useRef<HTMLDivElement>(null);
  const hasGreetedRef = useRef(false);

  useEffect(() => {
    if (moodHint === undefined || hasGreetedRef.current) return;
    hasGreetedRef.current = true;
    fetchGreeting(moodHint)
      .then((greeting) => {
        if (greeting) setMessages((prev) => (prev.length === 0 ? [{ role: 'model', content: greeting }] : prev));
      })
      .catch((err) => setError(err instanceof Error ? err.message : 'Could not load greeting.'));
  }, [moodHint]);

  const handleSend = useCallback(async () => {
    const text = input.trim();
    if (!text || isSending) return;

    const nextMessages: ChatMessage[] = [...messages, { role: 'user', content: text }];
    setMessages(nextMessages);
    setInput('');
    setError(null);
    setSavedNotice(null);
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

  const handleEndSession = useCallback(async () => {
    if (messages.length === 0 || isSending) return;
    setIsSending(true);
    setError(null);
    try {
      await endChatSession(messages);
      setMessages([]);
      setSavedNotice('Saved to memory. Starting a fresh conversation.');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not save session.');
    } finally {
      setIsSending(false);
    }
  }, [messages, isSending]);

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
      {savedNotice && <p className="chat-saved-notice">{savedNotice}</p>}

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

      <button
        onClick={() => void handleEndSession()}
        disabled={isSending || messages.length === 0}
        className="chat-end-session-button"
      >
        End & remember this conversation
      </button>
    </div>
  );
}
