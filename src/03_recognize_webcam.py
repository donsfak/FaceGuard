"""
Étape 3 — Reconnaissance faciale temps réel (webcam + fenêtre OpenCV).

Pipeline : webcam -> détection SCRFD -> embedding ArcFace -> recherche FAISS
-> vote k-NN + seuil -> lissage temporel (+ anti-spoofing optionnel) -> affichage.

Affiche pour chaque visage : un cadre, le nom AU-DESSUS du visage (ou « Inconnu »),
le score de similarité, et en haut de l'image les FPS et le nombre de personnes.

Touches : q = quitter · s = capture d'écran dans docs/ · l = activer/désactiver la vivacité

Usage :
    python src/03_recognize_webcam.py
    python src/03_recognize_webcam.py --threshold 0.5 --det-size 320 --no-liveness
"""

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from core import config  # noqa: E402
from core.drawing import draw_hud, draw_result  # noqa: E402
from core.engine import FaceEngine  # noqa: E402
from core.tracking import FaceTracker  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Reconnaissance faciale temps réel")
    p.add_argument("--encodings", type=Path, default=config.ENCODINGS_PATH)
    p.add_argument("--threshold", type=float, default=config.THRESHOLD,
                   help="Similarité cosinus minimale (plus haut = plus strict)")
    p.add_argument("--knn-k", type=int, default=config.KNN_K)
    p.add_argument("--det-size", type=int, default=config.DET_SIZE, help="320 = rapide, 640 = précis")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--process-every-n", type=int, default=1,
                   help="Analyser 1 image sur N (2 ou 3 si le PC est lent)")
    p.add_argument("--smoothing", type=int, default=config.SMOOTHING)
    p.add_argument("--no-liveness", action="store_true", help="Désactive l'anti-spoofing géométrique")
    p.add_argument("--gpu", action="store_true")
    return p.parse_args()


def log_event(path: Path, name: str, sim: float, liveness: str):
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "user_name", "liveness_status", "confidence_score"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), name, liveness, f"{sim:.4f}"])


def main():
    args = parse_args()
    engine = FaceEngine(args.encodings, threshold=args.threshold, knn_k=args.knn_k,
                        det_size=args.det_size, gpu=args.gpu)
    if not engine.names:
        sys.exit("[ERREUR] Base vide : lancez d'abord src/02_encode_faces.py")
    tracker = FaceTracker(smoothing=args.smoothing, liveness=not args.no_liveness)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit("[ERREUR] Webcam inaccessible (déjà utilisée par une autre application ?)")

    print("[INFO] q = quitter · s = capture · l = vivacité on/off")
    results, frame_idx = [], 0
    fps, last = 0.0, time.perf_counter()
    last_logged = {}

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[ERREUR] Lecture webcam impossible.")
            break

        if frame_idx % args.process_every_n == 0:
            results = tracker.update(engine.analyze(frame))
            now = time.time()
            for r in results:
                if r["confirmed"] and now - last_logged.get(r["identity"], 0) > 10:
                    last_logged[r["identity"]] = now
                    log_event(config.RECOGNITION_LOG, r["identity"], r["similarity"], r["liveness"])
        frame_idx += 1

        for r in results:
            draw_result(frame, r, show_liveness=tracker.liveness)
        known = sum(r["identity"] not in (config.UNKNOWN_LABEL, config.SPOOF_LABEL) for r in results)

        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))  # moyenne glissante, plus lisible
        last = now
        draw_hud(frame, fps, len(results), known)

        cv2.imshow("Reconnaissance faciale - ArcFace", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            out = config.DOCS_DIR / f"capture_{datetime.now():%Y%m%d_%H%M%S}.jpg"
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), frame)
            print(f"[INFO] Capture enregistrée : {out}")
        if key == ord("l"):
            tracker.liveness = not tracker.liveness
            print(f"[INFO] Vivacité {'activée' if tracker.liveness else 'désactivée'}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
