import { useCallback, useEffect, useRef, useState } from 'react';
import { endChatSession, fetchGreeting, sendChatMessage } from '../lib/chatClient';
import { fetchBotName, updateBotName } from '../lib/settingsClient';
import { Avatar, mapMoodToExpression } from './Avatar';
import type { ChatMessage } from '../types/chat';

type MoodLabel = 'happy' | 'sad' | 'angry' | 'neutral';

interface ChatUIProps {
  // undefined = still waiting to know the mood (don't greet yet); null = known, no mood/camera off.
  moodHint?: MoodLabel | null;
  // Continuously updated live reading, sent with every message so the whole
  // conversation stays mood-aware, not just the opening greeting.
  liveMood: MoodLabel | null;
}

function formatTime(timestamp: number | undefined): string {
  if (!timestamp) return '';
  return new Date(timestamp).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
}

export function ChatUI({ moodHint, liveMood }: ChatUIProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [savedNotice, setSavedNotice] = useState<string | null>(null);
  const [botName, setBotName] = useState('Friend');
  const [isEditingName, setIsEditingName] = useState(false);
  const [nameDraft, setNameDraft] = useState('');
  const listEndRef = useRef<HTMLDivElement>(null);
  const hasGreetedRef = useRef(false);

  useEffect(() => {
    fetchBotName().then(setBotName);
  }, []);

  useEffect(() => {
    if (moodHint === undefined || hasGreetedRef.current) return;
    hasGreetedRef.current = true;
    fetchGreeting(moodHint)
      .then((greeting) => {
        if (greeting) {
          setMessages((prev) =>
            prev.length === 0 ? [{ role: 'model', content: greeting, timestamp: Date.now() }] : prev,
          );
        }
      })
      .catch((err) => setError(err instanceof Error ? err.message : 'Could not load greeting.'));
  }, [moodHint]);

  const handleSend = useCallback(async () => {
    const text = input.trim();
    if (!text || isSending) return;

    const nextMessages: ChatMessage[] = [...messages, { role: 'user', content: text, timestamp: Date.now() }];
    setMessages(nextMessages);
    setInput('');
    setError(null);
    setSavedNotice(null);
    setIsSending(true);

    try {
      const reply = await sendChatMessage(nextMessages, liveMood);
      setMessages([...nextMessages, { role: 'model', content: reply, timestamp: Date.now() }]);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong.');
    } finally {
      setIsSending(false);
      listEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [input, isSending, messages, liveMood]);

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

  const startEditingName = useCallback(() => {
    setNameDraft(botName);
    setIsEditingName(true);
  }, [botName]);

  const saveName = useCallback(async () => {
    const trimmed = nameDraft.trim();
    setIsEditingName(false);
    if (!trimmed || trimmed === botName) return;
    try {
      setBotName(await updateBotName(trimmed));
    } catch {
      // Keep the previous name displayed if the save fails silently in the background.
    }
  }, [nameDraft, botName]);

  const expression = isSending ? 'thinking' : mapMoodToExpression(liveMood);

  return (
    <div className="chat-ui">
      <div className="chat-header">
        <Avatar expression={expression} compact />
        <div className="chat-header-info">
          {isEditingName ? (
            <input
              autoFocus
              value={nameDraft}
              onChange={(e) => setNameDraft(e.target.value)}
              onBlur={() => void saveName()}
              onKeyDown={(e) => {
                if (e.key === 'Enter') void saveName();
              }}
              className="chat-header-name-input"
              maxLength={40}
            />
          ) : (
            <button onClick={startEditingName} className="chat-header-name-button" title="Click to rename">
              {botName} <span className="chat-header-edit-icon">✎</span>
            </button>
          )}
          <span className="chat-header-status">{isSending ? 'typing…' : 'active now'}</span>
        </div>
      </div>

      <div className="chat-messages">
        {messages.length === 0 && <p className="chat-empty">Say hello to start chatting.</p>}
        {messages.map((m, i) => (
          <div key={i} className={`chat-message chat-message-${m.role}`}>
            <span className="chat-message-text">{m.content}</span>
            <span className="chat-message-time">{formatTime(m.timestamp)}</span>
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
