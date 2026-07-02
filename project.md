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