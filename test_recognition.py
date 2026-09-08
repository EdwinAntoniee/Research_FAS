import os
import sys
import time
import pickle
import argparse
from typing import Dict, List, Tuple

import cv2
import numpy as np
import tensorflow as tf
from scipy.spatial.distance import cosine
import insightface
from insightface.app import FaceAnalysis

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class FaceAntiSpoofingGate:
    def __init__(self, model_path: str, liveness_threshold: float = 0.5):
        self.threshold = liveness_threshold
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"FAS model checkpoint not found: {model_path}")
        self.model = tf.keras.models.load_model(model_path)

    def predict_liveness(self, frame: np.ndarray, bbox: np.ndarray) -> Tuple[bool, float]:
        h, w, _ = frame.shape
        x1, y1, x2, y2 = bbox.astype(int)

        pad_x = int((x2 - x1) * 0.1)
        pad_y = int((y2 - y1) * 0.1)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        face_roi = frame[y1:y2, x1:x2]
        if face_roi.size == 0:
            return False, 0.0

        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (224, 224))
        input_tensor = np.expand_dims(face_resized.astype(np.float32) / 255.0, axis=0)

        pred = self.model.predict(input_tensor, verbose=0)
        liveness_score = float(pred[0][0])
        is_live = liveness_score >= self.threshold

        return is_live, liveness_score


class FaceRecognizer:
    def __init__(self, db_path: str, cosine_threshold: float = 0.65):
        self.threshold = cosine_threshold
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Face database not found: {db_path}")

        with open(db_path, "rb") as f:
            data = pickle.load(f)
        self.known_encodings: np.ndarray = data["encodings"]
        self.known_names: List[str] = data["names"]

    def identify(self, embedding: np.ndarray) -> Tuple[str, float]:
        best_name = "Unknown"
        min_distance = 1.0

        for idx, known_enc in enumerate(self.known_encodings):
            dist = float(cosine(embedding, known_enc))
            if dist < min_distance:
                min_distance = dist
                if min_distance < self.threshold:
                    best_name = self.known_names[idx]

        return best_name, min_distance


class BiometricVerificationPipeline:
    def __init__(
        self,
        fas_model_path: str,
        db_path: str,
        liveness_threshold: float = 0.50,
        cosine_threshold: float = 0.65,
        det_size: Tuple[int, int] = (320, 320),
        frame_skip: int = 3,
        ctx_id: int = -1
    ):
        self.app = FaceAnalysis(allowed_modules=['detection', 'recognition'])
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

        self.fas_gate = FaceAntiSpoofingGate(fas_model_path, liveness_threshold)
        self.recognizer = FaceRecognizer(db_path, cosine_threshold)

        self.frame_skip = frame_skip
        self.frame_count = 0
        self.cached_detections: List[Dict] = []

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        self.frame_count += 1

        if self.frame_count % self.frame_skip == 0:
            faces = self.app.get(frame)
            current_results = []

            for face in faces:
                bbox = face.bbox.astype(int)
                is_live, liveness_score = self.fas_gate.predict_liveness(frame, bbox)

                if is_live:
                    identity, distance = self.recognizer.identify(face.embedding)
                    status = "AUTHENTICATED" if identity != "Unknown" else "UNREGISTERED"
                else:
                    identity = "ACCESS DENIED"
                    distance = 1.0
                    status = "SPOOF_ATTACK"

                current_results.append({
                    "bbox": bbox,
                    "is_live": is_live,
                    "liveness": liveness_score,
                    "identity": identity,
                    "distance": distance,
                    "status": status
                })

            self.cached_detections = current_results

        display_frame = frame.copy()
        self._render_hud(display_frame)

        for det in self.cached_detections:
            self._render_detection_box(display_frame, det)

        return display_frame

    def _render_detection_box(self, frame: np.ndarray, det: Dict) -> None:
        bbox = det["bbox"]
        status = det["status"]
        liveness = det["liveness"]
        identity = det["identity"]
        dist = det["distance"]

        if status == "SPOOF_ATTACK":
            box_color = (0, 0, 255)
            tag_title = f"[SPOOF DETECTED] ({liveness*100:.1f}%)"
            tag_sub = "Presentation Attack Blocked"
        elif status == "AUTHENTICATED":
            box_color = (0, 255, 0)
            tag_title = f"[REAL] {identity.upper()}"
            tag_sub = f"Live: {liveness*100:.0f}% | Dist: {dist:.2f}"
        else:
            box_color = (0, 200, 255)
            tag_title = f"[REAL] Unknown Guest"
            tag_sub = f"Live: {liveness*100:.0f}% | No Match"

        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), box_color, 2)

        header_h = 42
        cv2.rectangle(
            frame,
            (bbox[0], max(0, bbox[1] - header_h)),
            (bbox[2], bbox[1]),
            box_color,
            cv2.FILLED
        )

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            frame,
            tag_title,
            (bbox[0] + 6, max(18, bbox[1] - 22)),
            font,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            frame,
            tag_sub,
            (bbox[0] + 6, max(36, bbox[1] - 6)),
            font,
            0.42,
            (240, 240, 240),
            1,
            cv2.LINE_AA
        )

    def _render_hud(self, frame: np.ndarray) -> None:
        h, w, _ = frame.shape
        cv2.rectangle(frame, (0, 0), (w, 35), (25, 25, 25), cv2.FILLED)
        cv2.putText(
            frame,
            "Biometric Verification | Stage 1: MobileNetV2 FAS | Stage 2: ArcFace",
            (15, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (220, 220, 220),
            1,
            cv2.LINE_AA
        )


def open_capture_stream(source: str) -> cv2.VideoCapture:
    if source.isdigit():
        idx = int(source)
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(idx)
        return cap
    return cv2.VideoCapture(source)


def main():
    parser = argparse.ArgumentParser(description="Real-Time Face Anti-Spoofing and Recognition Pipeline")
    parser.add_argument("--source", type=str, default="0", help="Camera index or video file path")
    parser.add_argument(
        "--fas-model",
        type=str,
        default=os.path.join(SCRIPT_DIR, "model_fas_mobilenet.h5"),
        help="Path to FAS model"
    )
    parser.add_argument(
        "--db",
        type=str,
        default=os.path.join(SCRIPT_DIR, "model_recognition_facenet.pkl"),
        help="Path to facial embeddings database"
    )
    parser.add_argument("--liveness-thresh", type=float, default=0.50, help="Liveness threshold")
    parser.add_argument("--cosine-thresh", type=float, default=0.65, help="Cosine distance threshold")
    parser.add_argument("--skip", type=int, default=3, help="Frames to skip between AI inferences")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for face detection")
    args = parser.parse_args()

    ctx = 0 if args.gpu else -1
    pipeline = BiometricVerificationPipeline(
        fas_model_path=args.fas_model,
        db_path=args.db,
        liveness_threshold=args.liveness_thresh,
        cosine_threshold=args.cosine_thresh,
        frame_skip=args.skip,
        ctx_id=ctx
    )

    cap = open_capture_stream(args.source)
    if not cap.isOpened():
        print(f"Error: Unable to open video source '{args.source}'")
        sys.exit(1)

    print("Pipeline active. Press 'q' to exit.")
    fps_history = []

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        processed = pipeline.process_frame(frame)

        fps = 1.0 / max(1e-5, (time.time() - t0))
        fps_history.append(fps)
        if len(fps_history) > 30:
            fps_history.pop(0)
        avg_fps = sum(fps_history) / len(fps_history)

        cv2.putText(
            processed,
            f"FPS: {avg_fps:.1f}",
            (processed.shape[1] - 110, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 200),
            2,
            cv2.LINE_AA
        )

        cv2.imshow("Biometric Verification System", processed)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()