import { useCallback, useEffect, useRef, useState } from 'react';
import { ConsentModal } from './components/ConsentModal';
import { WebcamView } from './components/WebcamView';
import { DebugPanel, isDebugEnabled } from './components/DebugPanel';
import { ChatUI } from './components/ChatUI';
import { NotesPanel } from './components/NotesPanel';
import { SignIn } from './components/SignIn';
import { fetchCurrentUser, logout, type AuthUser } from './lib/authClient';
import { EmotionPipeline, type EmotionDebugInfo } from './lib/emotionPipeline';
import './App.css';

type ConsentState = 'pending' | 'granted' | 'declined';
type Mood = EmotionDebugInfo['stableState'];

function App() {
  const [user, setUser] = useState<AuthUser | null | 'loading'>('loading');
  const [consent, setConsent] = useState<ConsentState>('pending');
  const [emotionInfo, setEmotionInfo] = useState<EmotionDebugInfo | null>(null);
  const [debugToggle, setDebugToggle] = useState(false);
  const [moodHint, setMoodHint] = useState<Mood | undefined>(undefined);
  // One pipeline instance for the app's lifetime so its EmotionStateMachine
  // warm-up timer/buffer persist across renders and across a decline->accept toggle.
  const pipelineRef = useRef(new EmotionPipeline());

  useEffect(() => {
    fetchCurrentUser().then(setUser);
  }, []);

  useEffect(() => {
    if (consent === 'granted') {
      pipelineRef.current.stateMachine.sessionSnapshot().then(setMoodHint);
    } else if (consent === 'declined') {
      setMoodHint(null);
    }
  }, [consent]);

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
      <div className="app-layout">
        <div className="chat-column">
          <ChatUI moodHint={moodHint} liveMood={consent === 'granted' ? (emotionInfo?.stableState ?? null) : null} />
        </div>

        <div className="side-column">
          {consent === 'pending' && (
            <ConsentModal onAccept={handleAccept} onDecline={handleDecline} />
          )}
          {consent === 'granted' && (
            <WebcamView pipeline={pipelineRef.current} onEmotionUpdate={handleEmotionUpdate} onDisable={handleDisable} />
          )}
          {consent === 'declined' && (
            <p className="camera-off-notice">Camera is off. Chat works normally without it.</p>
          )}

          <button onClick={() => setDebugToggle((v) => !v)} className="debug-toggle-button">
            {debugToggle ? 'Hide debug panel' : 'Show debug panel'}
          </button>
          {(isDebugEnabled() || debugToggle) && <DebugPanel info={emotionInfo} />}
        </div>
      </div>

      <NotesPanel />

      <div className="account-bar">
        <span>Signed in as {user.name ?? user.email}</span>
        <button onClick={() => void handleSignOut()} className="sign-out-button">
          Sign out
        </button>
      </div>
    </div>
  );
}

export default App;
