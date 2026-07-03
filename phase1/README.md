# Phase 1 Starter

This folder contains a minimal browser demo for the Phase 1 goal in `project.md`.

What it does:
- opens the webcam in a browser
- runs `@mediapipe/tasks-vision` Face Landmarker on each frame
- uses a tiny Three.js avatar to visualize mouth and eye channels
- draws the detected landmarks on an overlay for quick inspection

## Run it

Serve the repo from the project root so the browser can load `models/face_landmarker.task`:

```bash
python -m http.server 8000
```

Then open:

```text
http://localhost:8000/phase1/
```

## Notes

- The avatar is intentionally simple. It is a placeholder for a VRM or GLTF rig.
- Mouth and eye channels are based on direct landmark distances so you can verify the retargeting path before moving to full blendshape mapping.
- Open the page from `http://localhost:8000/phase1/` or another secure local server. `file://` will usually break camera access and model loading.
- If the CDN is blocked in your environment, install the dependencies locally and bundle the app with your preferred toolchain.
