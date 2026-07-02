# Emotion-Synced 3D Avatar / VTuber System

**Real-Time Facial Landmark Retargeting and Emotion-Triggered Animation**
Dataset: [PSewmuthu/Emotion_Video_Facial_Landmarks](https://huggingface.co/datasets/PSewmuthu/Emotion_Video_Facial_Landmarks) (Hugging Face)

---

## 1. Overview

This project drives a real-time 3D avatar's facial animation and expression state using facial landmark data, trained and validated on the `Emotion_Video_Facial_Landmarks` dataset. The dataset provides pre-extracted, normalized 3D facial landmark features — 478 MediaPipe landmarks × x/y/z, across six emotion classes, ~556,000 frames.

The pipeline has two layers that can be built and shipped independently:

- **Layer A — Direct mesh-driving:** Stream normalized, head-pose-decoupled landmark deltas onto a 3D face mesh for linear blendshape mapping or live ARKit retargeting.
- **Layer B — Emotion-triggered animation:** Classify the landmark sequence into one of six emotion classes using a temporal sequence model running in a background thread to trigger stylized avatar reactions (blend shape presets, particle effects, idle-pose switches).

Both layers run live from a webcam via MediaPipe Face Landmarker at inference time, matching the data topology of the training set.

## 2. Dataset Fit

| Property | Detail | Relevance |
|---|---|---|
| Rows | ~556,000 frames | Enough for a sequence/temporal model, not just per-frame |
| Features | `x_0..x_477`, `y_0..y_477`, `z_0..z_477` (1,434 columns) | Maps 1:1 onto MediaPipe's live topology. Requires local transformation to decouple head translation/scale. |
| Labels | Angry, Disgust, Fear, Happy, Neutral, Sad | Clip-level, not per-frame — treated as weak/sequence labels |
| Grouping | `video_filename`, `frame_num` | Reconstructs temporal windows per clip for sequence models (LSTM/GRU) |
| Normalization | Coordinates in [0, 1] | **Critical:** Represents screen space. Must be transformed using a rigid local coordinate anchor (e.g., nose bridge) to ensure distance-invariant model performance. |
| Format | CSV (16.24 GB) or Parquet (4.62 GB, `Optimized_Video_Facial_Landmarks`) | Use Parquet for faster loading during development |

> **Caveat:** Labels are clip-level, so individual frames (especially near clip boundaries or neutral moments within an "Angry" clip) will be noisy. This drives several sequence-windowing and smoothing choices below.

## 3. System Architecture

To avoid blocking the main thread during real-time 3D rendering, the runtime architecture splits extraction/rendering from inference using Web Workers.

              ┌────────────────────── MAIN THREAD ──────────────────────┐
              │                                                         │
Webcam ───────────┼─▶ MediaPipe Face Landmarker ──▶ 478×(x,y,z) Landmarks   │
│                                    │            │       │
│         ┌──────────────────────────┘            │       │
│         ▼                                       ▼       │
│   Layer A: Retargeting                    [Post Message]│
│   (Raw-to-Blendshape Matrix)                    │       │
│         │                                       │       │
│         ▼                                       ▼       │
│  3D Avatar Mesh ◀───────── [Post Message] ──────┼───────┼──┐
│  (Three.js Scene)           Emotion Trigger     │       │  │
│         │                                       │       │  │
│         ▼                                       │       │  │
│  Rendered Output                                │       │  │
│                                                 │       │  │
└─────────────────────────────────────────────────┼───────┘  │
│          │
┌──────────────────── WEB WORKER ─────────────────┼──────────┘
│                                                 ▼          │
│                                       Local Normalization  │
│                                       (De-pose & Scale)    │
│                                                 │          │
│                                                 ▼          │
│                                       Layer B: ONNX Model  │
│                                       (Sliding Window GRU) │
└────────────────────────────────────────────────────────────┘


## 4. Requirements

### Hardware
- GPU recommended for training (RTX 3060+).
- Webcam (720p+ / 30fps+).
- ~5–17 GB free storage (Parquet vs CSV) plus space for checkpoints/assets.

### Software
**Python (Data + Training):** Python 3.10+, `pandas`, `pyarrow`, `numpy`, `torch` (or `tensorflow`), `scikit-learn`, `onnx`

**Web/Avatar (Three.js Route - Client Deployment):** Node.js 18+, `three`, `@mediapipe/tasks-vision`, `onnxruntime-web` (for multi-threaded Web Worker inference), a rigged GLTF/VRM avatar with ARKit blendshapes.

### Assets
- A rigged 3D avatar with blendshapes/morph targets (ARKit-style 52 blendshapes standard).
- FX assets for Layer B triggers (tears, sparkle, vignette).

---

## 5. Anticipated Difficulties & Mitigation Solutions

1. **Screen Space Coordinate Traps** — Coordinates normalized in $[0, 1]$ carry structural distance, scaling, and tilt bias. Leaning back or shifting side-to-side will trigger erratic model predictions. *Mitigation:* Normalize face coordinates relative to stable, rigid face structural regions (e.g., distance vectors from the nose bridge base).
2. **Web Main-Thread Bottleneck** — Running canvas rendering, MediaPipe face tracking, and sequence classifications sequentially breaks the 16.6ms frame budget (60 FPS). *Mitigation:* Offload the Layer B sequence model entirely to a Web Worker via `onnxruntime-web`, keeping the main thread clear for immediate Layer A tracking and WebGL updates.
3. **Clip-level labels, not frame-level** — Naive per-frame training learns to associate neutral-looking faces with whatever label the clip carries. *Mitigation:* Train using short overlapping sliding windows (e.g., 30 frames with a stride of 5) matching the runtime rolling inference loop.
4. **Offline Dataset Feature Disconnect** — Live MediaPipe runtimes can generate automated blendshape values on the fly, but the offline dataset consists *only* of raw coordinate sets. *Mitigation:* Layer A will map raw keypoint deltas directly to the avatar morph coefficients via linear interpolation or distance-ratio metrics rather than bypassing the raw point logic entirely.
5. **Jitter in raw landmark tracking** — Frame-to-frame noise makes the avatar twitch. *Mitigation:* Implement a localized One Euro Filter or Exponential Moving Average (EMA) tuning wrapper on the input array stream.

---

## 6. Workflow

### Phase 0 — Sanity Check & Coordinate Engineering (1–2 days)
Prove the core data and normalization pipelines before writing real model code.
- Pull a small sample of the dataset; inspect layout and column format.
- Run MediaPipe locally on a webcam feed. Map and test a coordinate normalization algorithm to prove that moving your body relative to the camera frame yields steady, normalized landmark arrays.

### Phase 1 — Minimal End-to-End Mesh Setup (3–5 days)
Get immediate feedback on avatar structures.
- Set up a barebones Three.js scene containing your free pre-rigged VRM avatar.
- Wire `@mediapipe/tasks-vision` to drive the basic mouth and eye structural channels using direct distance vectors calculated from the raw stream (e.g., distance between upper/lower lip points driving the `mouthOpen` shape).

### Phase 2 — Data Preprocessing & Sequence Engineering (3–4 days)
- Load the Parquet dataset; group by `video_filename` and sort by `frame_num`.
- Normalize the entire dataset to remove positional, orientation, and camera scale biases.
- Partition continuous sequences into fixed-size overlapping sliding windows.

### Phase 3 — Baseline & ONNX Verification (4–5 days)
- Train a simple multi-layer perceptron (MLP) as a structural baseline.
- Export the baseline model to `.onnx` format.
- Integrate the ONNX runtime into your web environment to confirm the web-assembly runtime compiles and parses inference inputs cleanly without throwing dependency issues.

### Phase 4 — Temporal Model Deployment (1 week)
- Move to a windowed GRU/LSTM architecture trained over your windowed dataset matrices.
- Apply cross-entropy weight profiles to manage target class distribution imbalances (e.g., Disgust/Fear underrepresentation).
- Re-export the sequence pipeline to ONNX.

### Phase 5 — Multi-Thread Worker Integration (3–5 days)
- Move the ONNX runtime pipeline into a dedicated browser Web Worker.
- Set up the `postMessage` data channel passing the frame buffer coordinates from the Main Thread over to the Web Worker.
- Integrate a One Euro Filter onto the incoming stream data to strip high-frequency twitching and jitter.

### Phase 6 — Emotion-Triggered State Machine (4–5 days)
- Build a robust animation state machine: Emotion + Confidence Threshold + Hysteresis Debouncing -> Animation/FX trigger.
- Add timers to prevent rapid oscillation or flickering between emotion responses (e.g., requiring an emotion state to remain dominant for at least 15 frames before transitioning states).
- Hook up placeholder custom shaders, environmental filters, or particle responses.

### Phase 7 — Calibration & System Profiling (3–5 days)
- Benchmark system latency profile ensuring end-to-end performance runs under 100ms.
- Collect a small, targeted calibration folder of user-specific frames to tweak structural scaling values if any specific blendshape limits are hitting clipping distortions.

### Phase 8 — Polish & Demo (3–5 days)
- Optimize assets, background styling, and asset materials.
- Output your system window onto an OBS virtual camera target canvas.
- Record system performance test results.

---

## 7. Stretch Goals

- **Multi-Avatar Rig Swapping:** Define standard translation coefficients enabling clean structural profiles on any standard loaded GLTF model without code recompilation.
- **Continuous Intensity Head (Regression):** Add a parallel output head onto the sequence model backbone measuring emotion intensity value on an open scalar interval $[0, 1]$.

## 8. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Coordinate translation frame shifts | Transform coordinates into local distance-independent space centered at a facial structural bone root anchor. |
| Single-thread blocking lag | Run Layer B sequence logic strictly within a dedicated background browser Web Worker using `onnxruntime-web`. |
| Clip-level training data noise | Apply rolling inference windows, confidence score bounds, and state debouncing logic. |