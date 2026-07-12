import { useCallback, useEffect, useState } from 'react';
import { ConsentModal } from './components/ConsentModal';
import { WebcamView } from './components/WebcamView';
import { DebugPanel, isDebugEnabled } from './components/DebugPanel';
import { ChatUI } from './components/ChatUI';
import { SignIn } from './components/SignIn';
import { fetchCurrentUser, logout, type AuthUser } from './lib/authClient';
import type { EmotionDebugInfo } from './lib/emotionPipeline';
import './App.css';

type ConsentState = 'pending' | 'granted' | 'declined';

function App() {
  const [user, setUser] = useState<AuthUser | null | 'loading'>('loading');
  const [consent, setConsent] = useState<ConsentState>('pending');
  const [emotionInfo, setEmotionInfo] = useState<EmotionDebugInfo | null>(null);
  const [debugToggle, setDebugToggle] = useState(false);

  useEffect(() => {
    fetchCurrentUser().then(setUser);
  }, []);

  const handleAccept = useCallback(() => setConsent('granted'), []);
  const handleDecline = useCallback(() => setConsent('declined'), []);
  const handleDisable = useCallback(() => {
    setConsent('declined');
    setEmotionInfo(null);
  }, []);
  const handleEmotionUpdate = useCallback((info: EmotionDebugInfo) => setEmotionInfo(info), []);
  const handleSignOut = useCallback(async () => {
    await logout();
    setUser(null);
  }, []);

  if (user === 'loading') {
    return (
      <div className="app">
        <h1>Emotion-Aware Chatbot</h1>
        <p>Loading...</p>
      </div>
    );
  }

  if (!user) {
    return (
      <div className="app">
        <h1>Emotion-Aware Chatbot</h1>
        <SignIn onSignedIn={setUser} />
      </div>
    );
  }

  return (
    <div className="app">
      <h1>Emotion-Aware Chatbot</h1>

      <div className="account-bar">
        <span>Signed in as {user.name ?? user.email}</span>
        <button onClick={() => void handleSignOut()} className="sign-out-button">
          Sign out
        </button>
      </div>

      <button
        onClick={() => setDebugToggle((v) => !v)}
        className="debug-toggle-button"
      >
        {debugToggle ? 'Hide debug panel' : 'Show debug panel'}
      </button>

      {consent === 'pending' && (
        <ConsentModal onAccept={handleAccept} onDecline={handleDecline} />
      )}
      {consent === 'granted' && (
        <WebcamView onEmotionUpdate={handleEmotionUpdate} onDisable={handleDisable} />
      )}
      {consent === 'declined' && (
        <p className="camera-off-notice">Camera is off. Chat works normally without it.</p>
      )}

      {(isDebugEnabled() || debugToggle) && <DebugPanel info={emotionInfo} />}

      <ChatUI />
    </div>
  );
}

export default App;
