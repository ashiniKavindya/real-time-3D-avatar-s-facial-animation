import { useCallback, useState } from 'react';
import { ConsentModal } from './components/ConsentModal';
import { WebcamView } from './components/WebcamView';
import { DebugPanel, isDebugEnabled } from './components/DebugPanel';
import { ChatUI } from './components/ChatUI';
import type { EmotionDebugInfo } from './lib/emotionPipeline';
import './App.css';

type ConsentState = 'pending' | 'granted' | 'declined';

function App() {
  const [consent, setConsent] = useState<ConsentState>('pending');
  const [emotionInfo, setEmotionInfo] = useState<EmotionDebugInfo | null>(null);
  const [debugToggle, setDebugToggle] = useState(false);

  const handleAccept = useCallback(() => setConsent('granted'), []);
  const handleDecline = useCallback(() => setConsent('declined'), []);
  const handleDisable = useCallback(() => {
    setConsent('declined');
    setEmotionInfo(null);
  }, []);
  const handleEmotionUpdate = useCallback((info: EmotionDebugInfo) => setEmotionInfo(info), []);

  return (
    <div className="app">
      <h1>Emotion-Aware Chatbot</h1>

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
