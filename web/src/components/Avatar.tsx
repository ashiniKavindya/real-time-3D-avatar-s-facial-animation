export type AvatarExpression = 'joyful' | 'caring' | 'calm' | 'neutral' | 'thinking';
type Mood = 'happy' | 'sad' | 'angry' | 'neutral' | null;

const EXPRESSION_EMOJI: Record<AvatarExpression, string> = {
  joyful: '😄',
  caring: '🥺',
  calm: '😌',
  neutral: '🙂',
  thinking: '🤔',
};

const EXPRESSION_LABEL: Record<AvatarExpression, string> = {
  joyful: 'feeling glad for you',
  caring: "here for you",
  calm: 'staying calm with you',
  neutral: 'listening',
  thinking: 'thinking...',
};

// The avatar reacts to your mood the way an empathetic friend would, rather
// than mirroring your expression back at you (e.g. it doesn't scowl when you're angry).
export function mapMoodToExpression(mood: Mood): AvatarExpression {
  switch (mood) {
    case 'happy':
      return 'joyful';
    case 'sad':
      return 'caring';
    case 'angry':
      return 'calm';
    default:
      return 'neutral';
  }
}

interface AvatarProps {
  expression: AvatarExpression;
  compact?: boolean;
}

export function Avatar({ expression, compact }: AvatarProps) {
  return (
    <div className={compact ? 'avatar avatar-compact' : 'avatar'}>
      <div className="avatar-face" key={expression}>
        {EXPRESSION_EMOJI[expression]}
      </div>
      <div className="avatar-label">{EXPRESSION_LABEL[expression]}</div>
    </div>
  );
}
