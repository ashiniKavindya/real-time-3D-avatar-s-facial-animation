interface ConsentModalProps {
  onAccept: () => void;
  onDecline: () => void;
}

export function ConsentModal({ onAccept, onDecline }: ConsentModalProps) {
  return (
    <div className="consent-overlay">
      <div className="consent-modal">
        <h2>Camera access</h2>
        <p>
          This app watches your facial expression through your webcam to make the
          chatbot mood-aware. Video never leaves your device — only a short emotion
          label derived from it is used locally.
        </p>
        <p>You can turn the camera off at any time.</p>
        <div className="consent-actions">
          <button onClick={onDecline}>Continue without camera</button>
          <button onClick={onAccept} className="primary">
            Allow camera
          </button>
        </div>
      </div>
    </div>
  );
}
