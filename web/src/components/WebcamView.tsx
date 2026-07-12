import { useEffect, useRef, useState } from 'react';
import { detectFrame } from '../lib/faceLandmarker';
import type { EmotionPipeline, EmotionDebugInfo } from '../lib/emotionPipeline';

const TARGET_FPS = 8;
const FRAME_INTERVAL_MS = 1000 / TARGET_FPS;

interface WebcamViewProps {
  pipeline: EmotionPipeline;
  onEmotionUpdate: (info: EmotionDebugInfo) => void;
  onDisable: () => void;
}

export function WebcamView({ pipeline, onEmotionUpdate, onDisable }: WebcamViewProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let stream: MediaStream | null = null;
    let rafId: number;
    let lastFrameTime = 0;
    let cancelled = false;

    async function start() {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
      } catch {
        setError('Camera permission was denied or no camera is available.');
        return;
      }
      if (cancelled || !videoRef.current) return;

      const video = videoRef.current;
      video.srcObject = stream;
      await video.play();

      const loop = async (now: number) => {
        if (cancelled) return;
        if (now - lastFrameTime >= FRAME_INTERVAL_MS) {
          lastFrameTime = now;
          const result = await detectFrame(video, performance.now());
          if (!cancelled) {
            const info = pipeline.processFrame(result);
            onEmotionUpdate(info);
          }
        }
        rafId = requestAnimationFrame(loop);
      };
      rafId = requestAnimationFrame(loop);
    }

    start();

    return () => {
      cancelled = true;
      cancelAnimationFrame(rafId);
      stream?.getTracks().forEach((track) => track.stop());
    };
  }, [pipeline, onEmotionUpdate]);

  return (
    <div className="webcam-view">
      <video ref={videoRef} muted playsInline className="webcam-video" />
      {error && <p className="webcam-error">{error}</p>}
      <button onClick={onDisable} className="disable-camera-toggle">
        Turn off camera
      </button>
    </div>
  );
}
