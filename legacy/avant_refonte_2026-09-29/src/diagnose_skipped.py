"""
Diagnostic — Identifie et analyse les images du dataset sur lesquelles
aucun visage n'a été détecté (celles marquées [SKIP] par 02_encode_faces.py).

Pour chaque image ignorée, on calcule :
  - la luminosité moyenne (image trop sombre/trop claire ?)
  - un indice de netteté (variance du Laplacien -> image floue ?)
  - la taille de l'image

On génère aussi une planche-contact (montage) des images ignorées pour
inspection visuelle rapide.

Usage :
    python diagnose_skipped.py --detection-method hog
"""

import argparse
import os

import cv2
import face_recognition
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnostique les images sans visage détecté")
    parser.add_argument("--dataset_dir", default="../dataset")
    parser.add_argument("--detection-method", choices=["hog", "cnn"], default="hog")
    parser.add_argument("--also-try-cnn", action="store_true",
                         help="Pour chaque image ignorée en hog, retente avec cnn (plus lent, plus précis)")
    parser.add_argument("--output", default="../docs/skipped_montage.jpg")
    return parser.parse_args()


def sharpness_score(gray_image):
    """Variance du Laplacien : plus c'est bas, plus l'image est floue."""
    return cv2.Laplacian(gray_image, cv2.CV_64F).var()


def main():
    args = parse_args()

    skipped = []  # (name, path, image_bgr)

    for person_name in sorted(os.listdir(args.dataset_dir)):
        person_dir = os.path.join(args.dataset_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
        for filename in sorted(os.listdir(person_dir)):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            path = os.path.join(person_dir, filename)

            image_rgb = face_recognition.load_image_file(path)
            boxes = face_recognition.face_locations(image_rgb, model=args.detection_method)

            if len(boxes) == 0:
                image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                skipped.append((person_name, path, image_bgr))

    print(f"[INFO] {len(skipped)} images sans visage détecté (méthode={args.detection_method})\n")

    print(f"{'Personne':<12}{'Fichier':<28}{'Taille':<12}{'Luminosité':<12}{'Netteté':<10}")
    per_person_count = {}
    for name, path, image_bgr in skipped:
        per_person_count[name] = per_person_count.get(name, 0) + 1

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        sharpness = sharpness_score(gray)
        h, w = image_bgr.shape[:2]

        filename = os.path.basename(path)
        print(f"{name:<12}{filename:<28}{f'{w}x{h}':<12}{brightness:<12.1f}{sharpness:<10.1f}")

    print("\n[RÉPARTITION PAR PERSONNE]")
    for name, count in sorted(per_person_count.items(), key=lambda x: -x[1]):
        print(f"   - {name}: {count} images ignorées")

    print("\n[REPÈRES] Luminosité : 0=noir, 255=blanc. En dessous de ~50 ou au-dessus de ~200, souvent problématique.")
    print("[REPÈRES] Netteté (variance Laplacien) : en dessous de ~50-100, l'image est souvent trop floue.")

    if args.also_try_cnn and args.detection_method != "cnn":
        print("\n[INFO] Nouvel essai avec le détecteur 'cnn' (plus lent, patience...)")
        still_skipped = 0
        for name, path, _ in skipped:
            image_rgb = face_recognition.load_image_file(path)
            boxes = face_recognition.face_locations(image_rgb, model="cnn")
            status = "détecté avec cnn !" if len(boxes) > 0 else "toujours rien"
            if len(boxes) == 0:
                still_skipped += 1
            print(f"   - {os.path.basename(path)}: {status}")
        print(f"\n[RÉSUMÉ CNN] {len(skipped) - still_skipped}/{len(skipped)} récupérées avec le détecteur cnn")

    # --- Planche-contact des images ignorées ---
    if skipped:
        thumb_size = 150
        cols = 6
        rows = (len(skipped) + cols - 1) // cols
        montage = np.ones((rows * thumb_size, cols * thumb_size, 3), dtype=np.uint8) * 255

        for i, (name, path, image_bgr) in enumerate(skipped):
            thumb = cv2.resize(image_bgr, (thumb_size, thumb_size))
            cv2.putText(thumb, name[:10], (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            r, c = divmod(i, cols)
            montage[r * thumb_size:(r + 1) * thumb_size, c * thumb_size:(c + 1) * thumb_size] = thumb

        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        cv2.imwrite(args.output, montage)
        print(f"\n[INFO] Planche-contact sauvegardée : {args.output}")
        print("[INFO] Ouvrez ce fichier pour inspecter visuellement les images ignorées.")


if __name__ == "__main__":
    main()

