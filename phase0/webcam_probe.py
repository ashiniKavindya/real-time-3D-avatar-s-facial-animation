from __future__ import annotations

import argparse

import cv2
import mediapipe as mp

from phase0.landmark_normalization import normalize_landmarks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 0 webcam probe")
    parser.add_argument("--model-path", required=True, help="Path to a MediaPipe Face Landmarker task file")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam device index")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    base_options = mp.tasks.BaseOptions(model_asset_path=args.model_path)
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
    )

    capture = cv2.VideoCapture(args.camera_index)
    if not capture.isOpened():
        raise SystemExit(f"Unable to open camera {args.camera_index}")

    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            success, frame = capture.read()
            if not success:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection = landmarker.detect(mp_image)

            if detection.face_landmarks:
                landmarks = detection.face_landmarks[0]
                point_array = [[point.x, point.y, point.z] for point in landmarks]
                normalized, stats = normalize_landmarks(point_array)
                cv2.putText(
                    frame,
                    f"scale={stats.scale:.4f} norm0={normalized.mean():.4f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("Phase 0 Webcam Probe", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    capture.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
