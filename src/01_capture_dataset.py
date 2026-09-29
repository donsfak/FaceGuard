"""
Étape 1 — Constitution du dataset via la webcam.

Enregistre des photos dans dataset/<nom>/ (ou dataset_test_externe/<nom>/).
Une photo n'est prise automatiquement que si le détecteur SCRFD — le même que
celui du système de reconnaissance — voit EXACTEMENT UN visage net et assez grand.

Pourquoi ? L'analyse du dataset a montré des photos où la personne était de dos
ou cachée, et d'autres où une personne en arrière-plan était visible : ces images
dégradent la reconnaissance (embeddings mal étiquetés -> faux positifs).

⚠ N'enregistrez que des personnes ayant donné leur accord (consigne du projet).

Usage :
    python src/01_capture_dataset.py --name cedric --count 20
    python src/01_capture_dataset.py --name cedric --count 6 --dataset_dir dataset_test_externe

Photos de TEST par condition (voir docs/protocole_tests.md) — toutes les images sont
gardées, même difficiles, sinon le test serait biaisé :
    python src/01_capture_dataset.py --name cedric --count 5 --dataset_dir conditions/tourne --raw
    python src/01_capture_dataset.py --name "cedric+marie" --count 5 --dataset_dir conditions/plusieurs --group
Touches : ESPACE = capture forcée (si un visage est visible) · q = quitter
"""

import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from core import config  # noqa: E402
from core.engine import FaceEngine, largest_face, normalize_name  # noqa: E402

MIN_FACE_PX = 80
MIN_DET_SCORE = 0.6


def parse_args():
    p = argparse.ArgumentParser(description="Capture d'un dataset de visages via webcam")
    p.add_argument("--name", required=True, help="Nom de la personne (= nom du dossier)")
    p.add_argument("--count", type=int, default=20)
    p.add_argument("--dataset_dir", type=Path, default=config.DATASET_DIR)
    p.add_argument("--interval", type=float, default=0.8, help="Secondes entre deux captures automatiques")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--raw", action="store_true",
                   help="Tests par condition : capture à intervalle régulier SANS contrôle qualité")
    p.add_argument("--group", action="store_true",
                   help="Photos de groupe : --name 'marie+jean' (ou 'marie+inconnu'), fichiers à plat dans le dossier")
    return p.parse_args()


def next_index(person_dir: Path, name: str) -> int:
    """Premier numéro libre (avant : len(fichiers) pouvait écraser une photo existante)."""
    nums = [int(m.group(1)) for f in person_dir.glob(f"{name}_*.jpg")
            if (m := re.match(rf"{re.escape(name)}_(\d+)\.jpg$", f.name))]
    return max(nums, default=-1) + 1


def main():
    args = parse_args()
    if args.group:
        args.raw = True
        name = "+".join("inconnu" if n.strip().lower() == "inconnu" else normalize_name(n)
                        for n in args.name.split("+"))
    else:
        name = normalize_name(args.name)
    dataset_dir = args.dataset_dir if args.dataset_dir.is_absolute() else config.ROOT_DIR / args.dataset_dir
    person_dir = dataset_dir if args.group else dataset_dir / name
    person_dir.mkdir(parents=True, exist_ok=True)
    index = next_index(person_dir, name)

    engine = FaceEngine(encodings_path=None, det_size=320)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit("[ERREUR] Webcam inaccessible (déjà utilisée par une autre application ?)")

    print(f"[INFO] Capture de {args.count} photos pour '{name}' -> {person_dir}")
    print("[INFO] Variez : face, 3/4 gauche, 3/4 droite, sourire/neutre, un peu plus près/loin.")

    captured, last = 0, 0.0
    while captured < args.count:
        ok, frame = cap.read()
        if not ok:
            print("[ERREUR] Lecture webcam échouée.")
            break

        faces = [f for f in engine.detect(frame) if f.det_score >= MIN_DET_SCORE]
        good = [f for f in faces if (f.bbox[2] - f.bbox[0]) >= MIN_FACE_PX]
        display = frame.copy()
        for f in faces:
            x1, y1, x2, y2 = f.bbox.astype(int)
            color = (0, 200, 0) if f in good else (0, 165, 255)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

        if args.raw:
            msg, color = f"MODE TEST : {len(faces)} visage(s) detecte(s)", (255, 200, 0)
        elif len(faces) > 1:
            msg, color = "Plusieurs visages : une seule personne dans le champ", (0, 0, 255)
        elif not good:
            msg, color = "Approchez-vous / regardez la camera", (0, 165, 255)
        else:
            msg, color = "OK", (0, 200, 0)
        cv2.putText(display, f"{name}: {captured}/{args.count}  {msg}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imshow("Capture dataset (q = quitter)", display)

        key = cv2.waitKey(1) & 0xFF
        now = time.time()
        if args.raw:  # tests : on garde tout, y compris les images où rien n'est détecté
            auto = now - last >= max(args.interval, 1.5)
            forced = key == ord(" ")
        else:
            auto = len(faces) == 1 and len(good) == 1 and now - last >= args.interval
            forced = key == ord(" ") and largest_face(faces) is not None
        if auto or forced:
            path = person_dir / f"{name}_{index:03d}.jpg"
            cv2.imwrite(str(path), frame)  # image brute : l'encodage refait la détection
            index, captured, last = index + 1, captured + 1, now
            print(f"[CAPTURE] {path}")
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    nxt = "python src/04_evaluate.py" if args.raw else "python src/02_encode_faces.py"
    print(f"[TERMINÉ] {captured} photos enregistrées. Lancez ensuite : {nxt}")


if __name__ == "__main__":
    main()
