# Build Spec: Emotion-Aware Chatbot (Chatbot-First Phase)

Hand this file to Claude Code. Build order is chatbot-first: the full product works end-to-end with a rule-based emotion detector before any ML training happens. The trained model is a later drop-in replacement behind a fixed interface.

## 1. What we're building

A browser web app: the user opens a chat page, the webcam (with consent) watches their expression, and the chatbot greets them mood-aware ("Hey, you seem in a good mood today!") when — and only when — it's confident. Mid-conversation, detected mood silently modulates the bot's tone. Powered by MediaPipe blendshapes + rules now; a trained ONNX model later.

## 2. Architecture

```
Webcam ─▶ MediaPipe FaceLandmarker (JS, VIDEO mode, blendshapes ON)
              │  52 blendshape scores + face transform matrix, sampled ~8 fps
              ▼
        EmotionDetector (interface)  ◀── RuleBasedDetector now / OnnxDetector later
              │  {label, confidence} or null
              ▼
        PoseGate: suppress when |yaw| or |pitch| > 30° (from transform matrix)
              ▼
        EmotionStateMachine: confidence ≥ 0.8, hysteresis ≈ 15 samples,
              emits: stableState, transition events, sessionStartSnapshot
              ▼
        Chat layer (Anthropic API): greeting flow + tone modulation
```

## 3. Tech stack

- Vite + TypeScript + React (plain CSS or Tailwind, keep it light)
- `@mediapipe/tasks-vision` — FaceLandmarker with `outputFaceBlendshapes: true` and `outputFacialTransformationMatrixes: true`
- Anthropic API (`claude-sonnet-4-6`) for chat — server-side proxy route or direct depending on deployment; never expose a raw API key in shipped client code
- No ONNX yet — but define the `EmotionDetector` interface now

## 4. Module specs

### 4.1 `EmotionDetector` interface (the seam for the future model)
```ts
interface EmotionReading { label: 'happy'|'sad'|'angry'|'neutral'; confidence: number; }
interface EmotionDetector {
  detect(frame: { blendshapes: Record<string, number>; yaw: number; pitch: number; }): EmotionReading | null;
}
```
`null` = no face / gated / unusable frame. 4 classes only in this phase (research showed Fear/Disgust are unreliable; the chatbot never needs them).

### 4.2 `RuleBasedDetector` (initial implementation)
Score from blendshapes (all values are 0..1 from MediaPipe):
- happy: mean(mouthSmileLeft, mouthSmileRight), boosted by cheekSquint values
- sad: mean(mouthFrownLeft, mouthFrownRight, browInnerUp)
- angry: mean(browDownLeft, browDownRight) with low smile
- neutral: 1 − max(others)
Label = argmax; confidence = winning score with margin over runner-up folded in (e.g. `score − 0.5·runnerUp`). Tune thresholds live with a debug panel (see 4.6). Rules are deliberately conservative: it's fine to be unsure often; it's not fine to be confidently wrong.

### 4.3 `PoseGate`
Extract yaw/pitch from the facial transformation matrix. If |yaw|>30° or |pitch|>30°, return null upstream of the state machine.

### 4.4 `EmotionStateMachine`
- Ring buffer of last N=15 readings (~2 s at 8 fps)
- `stableState`: a label becomes stable when it holds majority of the buffer AND mean confidence ≥ 0.8
- Emits `transition` events (old → new stable state) — never fires on identical consecutive states
- `sessionSnapshot()`: after a 4 s warm-up from webcam start, returns the stable state (or null) exactly once — consumed by the greeting flow

### 4.5 Chat layer
- **Greeting flow:** on chat open + webcam consent granted → warm-up → `sessionSnapshot()`:
  - happy → LLM asked for a short warm greeting that lightly acknowledges the good mood
  - sad/angry → LLM asked for a gentle greeting that does NOT name the emotion ("hope everything's going okay")
  - neutral or null → plain friendly greeting, no mood reference
- **Mid-conversation:** on each user message, attach emotion context ONLY if a transition occurred since the last message, as a system-side note: `[context: user's expression shifted from neutral to sad ~20s ago; adapt tone, do not mention it]`
- **System prompt rules (web/prompts/system.md):** never state confidence numbers or mention detection/cameras; never announce negative emotions directly; never announce anything mid-conversation; positive mood may be acknowledged once, at greeting only; below-threshold = behave as if no signal exists.

### 4.6 Debug panel (dev-only, toggle with `?debug=1`)
Live view of: raw blendshape bars, current reading, buffer contents, stable state, gate status, last prompt sent to the LLM. This panel is how rule thresholds get tuned — build it early, it pays for itself immediately.

### 4.7 Consent & privacy (launch requirement, not polish)
- Explicit opt-in modal before camera starts; visible always-on toggle to disable
- Frames and blendshapes are never sent to any server — only the derived emotion context string reaches the LLM
- A camera-off mode where the chatbot works as a normal chatbot

## 5. Build order (each step = one Claude Code session, each ends runnable)

1. **Scaffold + webcam + MediaPipe**: Vite app, consent modal, FaceLandmarker running, raw blendshapes rendered in the debug panel. *Done when: smiling visibly moves mouthSmile bars.*
2. **Detector + gate + state machine**: RuleBasedDetector, PoseGate, EmotionStateMachine, all visible in debug panel. *Done when: sustained smile → stable 'happy'; head turn → gated; rapid expression flicker → no state flapping.*
3. **Chat UI + Anthropic API**: plain chat working with the system prompt, no emotion yet. *Done when: normal conversation works.*
4. **Greeting flow**: warm-up → snapshot → mood-aware or plain greeting. *Done when: smile at load → warm mood greeting; neutral → plain; camera covered → plain; NEVER a wrong announcement in 10 manual trials.*
5. **Transition modulation**: mid-conversation context injection on transitions only. *Done when: shifting to a sad expression makes the next reply noticeably gentler without mentioning it.*
6. **Polish**: camera-off mode, error states (no camera/permission denied), threshold tuning session using the debug panel.

## 6. Definition of done for this phase

- Demo video: three runs (happy / neutral / covered camera) showing correct greeting behavior in all three
- Zero wrong mood announcements across 20 manual trials (missed moods are acceptable; wrong ones are the failure mode)
- `EmotionDetector` interface untouched by any chat-layer code — proving the ONNX model can drop in later without changes

## 7. What comes after (unchanged from the full proposal)

The research phases (feature engineering, GRU training on the HF dataset + CREMA-D, experiments E1–E10) proceed next and produce an `OnnxDetector implements EmotionDetector`. The user study (E10) runs against this very app. Full details: emotion-chatbot-proposal-final.md.
