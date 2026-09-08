import os
import pickle
import argparse
from typing import Dict, List, Tuple
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class FaceDatabaseEncoder:
    def __init__(self, det_size: Tuple[int, int] = (640, 640), ctx_id: int = -1):
        self.app = FaceAnalysis(allowed_modules=['detection', 'recognition'])
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

    def encode_directory(self, dataset_path: str) -> Dict[str, np.ndarray]:
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

        known_face_encodings: List[np.ndarray] = []
        known_face_names: List[str] = []
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

        subdirs = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
        if not subdirs:
            raise ValueError(f"No identity directories found inside {dataset_path}")

        for person_name in subdirs:
            person_dir = os.path.join(dataset_path, person_name)
            image_files = [f for f in os.listdir(person_dir) if f.lower().endswith(valid_exts)]
            enrolled = 0

            for filename in image_files:
                img_path = os.path.join(person_dir, filename)
                img = cv2.imread(img_path)
                if img is None:
                    continue

                faces = self.app.get(img)
                if len(faces) == 1:
                    known_face_encodings.append(faces[0].embedding)
                    known_face_names.append(person_name)
                    enrolled += 1

            print(f"[{person_name}] Enrolled {enrolled}/{len(image_files)} faces.")

        if not known_face_encodings:
            raise RuntimeError("No valid facial embeddings could be extracted from dataset.")

        return {
            "encodings": np.array(known_face_encodings),
            "names": known_face_names
        }

    @staticmethod
    def save_database(data: Dict[str, np.ndarray], output_file: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        with open(output_file, "wb") as f:
            pickle.dump(data, f)
        print(f"Database saved: {output_file} ({len(data['names'])} embeddings)")


def main():
    parser = argparse.ArgumentParser(description="Extract ArcFace embeddings from identity image directories")
    parser.add_argument(
        "--dataset",
        type=str,
        default=os.path.join(SCRIPT_DIR, "dataset_wajah"),
        help="Path to folder containing identity subfolders"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(SCRIPT_DIR, "model_recognition_facenet.pkl"),
        help="Path to output pickle file"
    )
    parser.add_argument("--det-size", type=int, default=640, help="Detector image resolution")
    parser.add_argument("--gpu", action="store_true", help="Use GPU context (ctx_id=0)")
    args = parser.parse_args()

    ctx = 0 if args.gpu else -1
    encoder = FaceDatabaseEncoder(det_size=(args.det_size, args.det_size), ctx_id=ctx)
    db = encoder.encode_directory(args.dataset)
    encoder.save_database(db, args.output)


if __name__ == "__main__":
    main()