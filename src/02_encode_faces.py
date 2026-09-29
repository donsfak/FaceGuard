"""
Étape 2 — Encodage du dataset avec InsightFace (SCRFD + ArcFace).

Parcourt dataset/<personne>/*.jpg, extrait un embedding 512-D par photo et
sauvegarde la base dans models/encodings_arcface.npz (format sûr, sans pickle).

Qualité des données : si une photo contient plusieurs visages, on garde celui
qui ressemble aux autres photos de la personne (et pas forcément le plus grand).
Les photos inexploitables sont listées à la fin : à revoir ou à reprendre.

Usage (depuis n'importe quel dossier) :
    python src/02_encode_faces.py
    python src/02_encode_faces.py --det-size 320 --gpu
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import config  # noqa: E402
from core.engine import FaceEngine, encode_dataset  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Encodage du dataset (InsightFace / ArcFace)")
    p.add_argument("--dataset", type=Path, default=config.DATASET_DIR)
    p.add_argument("--encodings", type=Path, default=config.ENCODINGS_PATH)
    p.add_argument("--det-size", type=int, default=config.DET_SIZE)
    p.add_argument("--gpu", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.dataset.is_dir():
        sys.exit(f"[ERREUR] Dossier dataset introuvable : {args.dataset}")

    engine = FaceEngine(encodings_path=None, det_size=args.det_size, gpu=args.gpu)
    engine.encodings_path = args.encodings

    t0 = time.perf_counter()
    encodings, names, report = encode_dataset(engine, args.dataset)
    if len(names) == 0:
        sys.exit("[ERREUR] Aucun visage encodé : base non modifiée.")

    if args.encodings.exists():
        backup = args.encodings.with_suffix(".npz.bak")
        args.encodings.replace(backup)
        print(f"[INFO] Ancienne base sauvegardée : {backup}")
    engine.set_gallery(encodings, names)
    engine.save()

    print("\n[RÉSUMÉ]")
    for person, count in engine.people().items():
        flag = "  <- moins de 5 photos exploitables !" if count < 5 else ""
        print(f"  - {person:<15} {count:>3} embeddings{flag}")
    print(f"  Visages encodés   : {report['processed']}")
    print(f"  Photos ignorées   : {report['skipped']}")
    print(f"  Photos multi-visages : {report['multi_face_images']}")
    print(f"  Durée             : {time.perf_counter() - t0:.1f} s")
    print(f"  Base              : {args.encodings}")


if __name__ == "__main__":
    main()
