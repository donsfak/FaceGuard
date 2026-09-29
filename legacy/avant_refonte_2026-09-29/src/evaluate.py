"""
Étape 5 — Évaluation quantitative du système de reconnaissance.

Principe : split train/test par personne (80/20 par défaut).
  - Les images "train" servent à construire la base d'embeddings de référence.
  - Les images "test" (jamais vues à la construction) servent à mesurer si le
    pipeline (détection + embedding + seuil) reconnaît correctement des photos
    nouvelles de la même personne.

Produit :
  - Une accuracy globale
  - Une matrice de confusion (qui est confondu avec qui)
  - Une courbe accuracy en fonction du seuil, pour justifier le choix du seuil

Usage :
    python evaluate.py --detection-method hog --test-ratio 0.2
"""

import argparse
import os
import random
from collections import defaultdict

import face_recognition
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Évalue le système de reconnaissance faciale")
    parser.add_argument("--dataset_dir", default="../dataset", help="Dossier racine du dataset")
    parser.add_argument("--detection-method", choices=["hog", "cnn"], default="hog")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Proportion d'images réservées au test")
    parser.add_argument("--threshold", type=float, default=0.6, help="Seuil utilisé pour l'accuracy détaillée")
    parser.add_argument("--seed", type=int, default=42, help="Graine aléatoire pour un split reproductible")
    return parser.parse_args()


def load_split(dataset_dir, test_ratio, seed):
    """Sépare les images de chaque personne en train/test."""
    rng = random.Random(seed)
    train_paths, test_paths = [], []

    for person_name in sorted(os.listdir(dataset_dir)):
        person_dir = os.path.join(dataset_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
        images = [f for f in os.listdir(person_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        rng.shuffle(images)

        n_test = max(1, int(len(images) * test_ratio))
        test_imgs = images[:n_test]
        train_imgs = images[n_test:]

        for f in train_imgs:
            train_paths.append((person_name, os.path.join(person_dir, f)))
        for f in test_imgs:
            test_paths.append((person_name, os.path.join(person_dir, f)))

    return train_paths, test_paths


def encode_paths(paths, detection_method, label):
    encodings, names = [], []
    skipped = 0
    for i, (name, path) in enumerate(paths):
        image = face_recognition.load_image_file(path)
        boxes = face_recognition.face_locations(image, model=detection_method)
        if len(boxes) == 0:
            skipped += 1
            continue
        enc = face_recognition.face_encodings(image, boxes[:1])[0]
        encodings.append(enc)
        names.append(name)
    print(f"[INFO] {label}: {len(encodings)} visages encodés, {skipped} ignorés (aucun visage détecté)")
    return np.array(encodings), names


def predict(encoding, known_encodings, known_names, threshold):
    distances = face_recognition.face_distance(known_encodings, encoding)
    best_index = np.argmin(distances)
    best_distance = distances[best_index]
    if best_distance < threshold:
        return known_names[best_index], best_distance
    return "Inconnu", best_distance


def main():
    args = parse_args()

    print(f"[INFO] Split train/test (test_ratio={args.test_ratio}, seed={args.seed})")
    train_paths, test_paths = load_split(args.dataset_dir, args.test_ratio, args.seed)
    print(f"[INFO] {len(train_paths)} images train, {len(test_paths)} images test")

    print("\n[ÉTAPE 1] Encodage du set d'entraînement (base de référence)...")
    train_encodings, train_names = encode_paths(train_paths, args.detection_method, "Train")

    print("\n[ÉTAPE 2] Encodage du set de test...")
    test_encodings, test_names = encode_paths(test_paths, args.detection_method, "Test")

    # --- Évaluation au seuil choisi ---
    print(f"\n[ÉTAPE 3] Évaluation au seuil = {args.threshold}")
    correct = 0
    confusion = defaultdict(lambda: defaultdict(int))  # confusion[vrai_nom][nom_prédit] = count

    for true_name, encoding in zip(test_names, test_encodings):
        predicted_name, distance = predict(encoding, train_encodings, train_names, args.threshold)
        confusion[true_name][predicted_name] += 1
        if predicted_name == true_name:
            correct += 1

    accuracy = correct / len(test_names) if test_names else 0.0
    print(f"\n[RÉSULTAT] Accuracy globale : {accuracy * 100:.1f}% ({correct}/{len(test_names)})")

    print("\n[MATRICE DE CONFUSION] (lignes = vraie identité, colonnes = prédiction)")
    all_predicted_labels = sorted(set(p for row in confusion.values() for p in row.keys()))
    header = "Vrai \\ Prédit".ljust(15) + "".join(l[:10].ljust(12) for l in all_predicted_labels)
    print(header)
    for true_name in sorted(confusion.keys()):
        row = confusion[true_name]
        line = true_name[:14].ljust(15) + "".join(str(row.get(l, 0)).ljust(12) for l in all_predicted_labels)
        print(line)

    # --- Impact du seuil sur l'accuracy (pour justifier le choix du seuil) ---
    print("\n[ÉTAPE 4] Accuracy en fonction du seuil (pour justifier le choix retenu)")
    print(f"{'Seuil':<10}{'Accuracy':<12}{'% Inconnu':<12}")
    for t in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        correct_t = 0
        unknown_t = 0
        for true_name, encoding in zip(test_names, test_encodings):
            predicted_name, _ = predict(encoding, train_encodings, train_names, t)
            if predicted_name == true_name:
                correct_t += 1
            if predicted_name == "Inconnu":
                unknown_t += 1
        acc_t = correct_t / len(test_names) if test_names else 0.0
        pct_unknown = unknown_t / len(test_names) if test_names else 0.0
        print(f"{t:<10.2f}{acc_t * 100:<11.1f}%{pct_unknown * 100:<11.1f}%")

    print("\n[INFO] Utilisez ce tableau dans votre rapport pour justifier le seuil choisi :")
    print("       un seuil trop bas augmente le taux d'Inconnu (faux négatifs),")
    print("       un seuil trop haut réduit l'accuracy en confondant des personnes (faux positifs).")


if __name__ == "__main__":
    main()

