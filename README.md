# real-time-3D-avatar-s-facial-animation

## Phase 0 Starter

The workspace now includes Colab-friendly Phase 0 notebooks for two checks:

- Dataset sampling and layout validation with [notebooks/phase0_dataset_sanity_check_colab.ipynb](notebooks/phase0_dataset_sanity_check_colab.ipynb)
- Webcam landmark normalization probing with [notebooks/phase0_webcam_probe_colab.ipynb](notebooks/phase0_webcam_probe_colab.ipynb)

Install the Python dependencies from [requirements.txt](requirements.txt) before running either notebook.

## Phase 1 Starter

The repo also includes a browser-based Phase 1 demo:

- Avatar and webcam starter at [phase1/index.html](phase1/index.html)
- Run notes at [phase1/README.md](phase1/README.md)

This demo loads the existing [models/face_landmarker.task](models/face_landmarker.task) and drives simple mouth/eye channels on a placeholder Three.js head.

## Phase 2 Starter

The repo now includes a chunked sequence preprocessing pipeline:

- Preprocessor CLI at [phase2/preprocess_sequences.py](phase2/preprocess_sequences.py)
- Run notes at [phase2/README.md](phase2/README.md)

For the full dataset run:

```bash
.\.venv\Scripts\python.exe -m phase2.preprocess_sequences
```

For a quick capped verification run:

```bash
.\.venv\Scripts\python.exe -m phase2.preprocess_sequences --output-dir phase2\output_sample --max-rows 5000
```
