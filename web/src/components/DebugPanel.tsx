import type { EmotionDebugInfo } from '../lib/emotionPipeline';

interface DebugPanelProps {
  info: EmotionDebugInfo | null;
}

export function isDebugEnabled(): boolean {
  return new URLSearchParams(window.location.search).get('debug') === '1';
}

function formatReading(reading: EmotionDebugInfo['rawReading']): string {
  if (!reading) return 'none';
  return `${reading.label} (${reading.confidence.toFixed(2)})`;
}

export function DebugPanel({ info }: DebugPanelProps) {
  const categories = info
    ? Object.entries(info.blendshapes).sort(([, a], [, b]) => b - a)
    : [];

  return (
    <div className="debug-panel">
      <h3>Debug: emotion pipeline</h3>

      {!info && <p className="debug-empty">No data yet</p>}

      {info && (
        <div className="debug-summary">
          <div>
            <strong>Face detected:</strong> {info.faceDetected ? 'yes' : 'no'}
          </div>
          <div>
            <strong>Pose:</strong> yaw {info.yaw.toFixed(1)}°, pitch {info.pitch.toFixed(1)}° —{' '}
            {info.poseUsable ? 'usable' : 'GATED (looking away)'}
          </div>
          <div>
            <strong>Raw reading (this frame):</strong> {formatReading(info.rawReading)}
          </div>
          <div>
            <strong>Gated reading (fed to state machine):</strong> {formatReading(info.gatedReading)}
          </div>
          <div>
            <strong>Stable state:</strong> {info.stableState ?? 'none yet'}
          </div>
          <div>
            <strong>Buffer ({info.buffer.length}/15):</strong>{' '}
            {info.buffer.map((r) => (r ? r.label[0].toUpperCase() : '-')).join(' ')}
          </div>
        </div>
      )}

      <h4>Blendshape bars</h4>
      {categories.length === 0 && <p className="debug-empty">No face detected</p>}
      <div className="debug-bars">
        {categories.map(([name, score]) => (
          <div key={name} className="debug-bar-row">
            <span className="debug-bar-label">{name}</span>
            <div className="debug-bar-track">
              <div
                className="debug-bar-fill"
                style={{ width: `${Math.min(score, 1) * 100}%` }}
              />
            </div>
            <span className="debug-bar-value">{score.toFixed(2)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
