export interface EmotionReading {
  label: 'happy' | 'sad' | 'angry' | 'neutral';
  confidence: number;
}

export interface EmotionDetector {
  detect(frame: {
    blendshapes: Record<string, number>;
    yaw: number;
    pitch: number;
  }): EmotionReading | null;
}
