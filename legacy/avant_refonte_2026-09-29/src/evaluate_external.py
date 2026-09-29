"""
Évaluation externe — Mesure la généralisation du système sur des photos
prises à un autre moment que la construction du dataset (autre séance,
idéalement autre éclairage/jour), pour donner une accuracy plus honnête
que celle obtenue avec un simple split train/test sur les mêmes photos.

Produit :
  - Accuracy globale + matrice de confusion (comme evaluate.py)
  - Un graphique accuracy vs seuil (docs/accuracy_vs_threshold.png)
  - Un graphique de distribution des distances genuine vs impostor
    (docs/distance_distribution.png) -- LE graphique standard en
    reconnaissance faciale pour justifier le choix du seuil.

Usage :
    python evaluate_external.py --detection-method cnn
"""

import argparse
import os
from collections import defaultdict

import face_recognition
import matplotlib
matplotlib.use("Agg")  # pas d'affichage interactif nécessaire, on sauvegarde direct en fichier
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Évalue le système sur un dataset externe")
    parser.add_argument("--train_dir", default="../dataset", help="Dataset principal (base de référence)")
    parser.add_argument("--test_dir", default="../dataset_test_externe", help="Dataset de test externe")
    parser.add_argument("--detection-method", choices=["hog", "cnn"], default="cnn")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--output_dir", default="../docs")
    return parser.parse_args()


def list_images(dataset_dir):
    paths = []
    for person_name in sorted(os.listdir(dataset_dir)):
        person_dir = os.path.join(dataset_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
        for filename in sorted(os.listdir(person_dir)):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append((person_name, os.path.join(person_dir, filename)))
    return paths


def encode_paths(paths, detection_method, label):
    encodings, names = [], []
    skipped = 0
    for name, path in paths:
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


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.isdir(args.test_dir):
        raise FileNotFoundError(
            f"{args.test_dir} introuvable. Capturez d'abord des photos de test externes avec "
            f"01_capture_dataset.py --dataset_dir {args.test_dir}"
        )

    print("[ÉTAPE 1] Encodage du dataset d'entraînement complet...")
    train_paths = list_images(args.train_dir)
    train_encodings, train_names = encode_paths(train_paths, args.detection_method, "Train")

    print("\n[ÉTAPE 2] Encodage du dataset de test externe...")
    test_paths = list_images(args.test_dir)
    test_encodings, test_names = encode_paths(test_paths, args.detection_method, "Test externe")

    if len(test_encodings) == 0:
        print("[ERREUR] Aucune image de test exploitable.")
        return

    # --- Accuracy au seuil choisi + matrice de confusion ---
    print(f"\n[ÉTAPE 3] Évaluation au seuil = {args.threshold}")
    correct = 0
    confusion = defaultdict(lambda: defaultdict(int))
    genuine_distances = []   # distance quand la comparaison est avec la BONNE personne
    impostor_distances = []  # distance quand la comparaison est avec une AUTRE personne

    train_names_array = np.array(train_names)

    for true_name, test_encoding in zip(test_names, test_encodings):
        distances = face_recognition.face_distance(train_encodings, test_encoding)

        best_index = np.argmin(distances)
        best_distance = distances[best_index]
        predicted_name = train_names_array[best_index] if best_distance < args.threshold else "Inconnu"

        confusion[true_name][predicted_name] += 1
        if predicted_name == true_name:
            correct += 1

        # Pour le graphique de distribution : on regarde TOUTES les distances,
        # pas seulement la meilleure, en séparant "même personne" et "autre personne".
        for name, dist in zip(train_names_array, distances):
            if name == true_name:
                genuine_distances.append(dist)
            else:
                impostor_distances.append(dist)

    accuracy = correct / len(test_names)
    print(f"\n[RÉSULTAT] Accuracy (test externe) : {accuracy * 100:.1f}% ({correct}/{len(test_names)})")

    print("\n[MATRICE DE CONFUSION]")
    all_predicted_labels = sorted(set(p for row in confusion.values() for p in row.keys()))
    header = "Vrai \\ Prédit".ljust(15) + "".join(l[:10].ljust(12) for l in all_predicted_labels)
    print(header)
    for true_name in sorted(confusion.keys()):
        row = confusion[true_name]
        line = true_name[:14].ljust(15) + "".join(str(row.get(l, 0)).ljust(12) for l in all_predicted_labels)
        print(line)

    # --- Graphique 1 : accuracy vs seuil ---
    thresholds = np.arange(0.30, 0.85, 0.02)
    accuracies = []
    for t in thresholds:
        correct_t = 0
        for true_name, test_encoding in zip(test_names, test_encodings):
            distances = face_recognition.face_distance(train_encodings, test_encoding)
            best_index = np.argmin(distances)
            predicted_name = train_names_array[best_index] if distances[best_index] < t else "Inconnu"
            if predicted_name == true_name:
                correct_t += 1
        accuracies.append(correct_t / len(test_names) * 100)

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, accuracies, marker="o", markersize=3, color="#2563eb")
    plt.axvline(x=args.threshold, color="red", linestyle="--", label=f"Seuil retenu = {args.threshold}")
    plt.xlabel("Seuil de distance")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy en fonction du seuil (dataset de test externe)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path1 = os.path.join(args.output_dir, "accuracy_vs_threshold.png")
    plt.savefig(path1, dpi=150)
    print(f"\n[GRAPHIQUE] Sauvegardé : {path1}")

    # --- Graphique 2 : distribution des distances genuine vs impostor ---
    plt.figure(figsize=(8, 5))
    plt.hist(genuine_distances, bins=30, alpha=0.6, label="Même personne (genuine)", color="#22c55e")
    plt.hist(impostor_distances, bins=30, alpha=0.6, label="Personne différente (impostor)", color="#ef4444")
    plt.axvline(x=args.threshold, color="black", linestyle="--", label=f"Seuil retenu = {args.threshold}")
    plt.xlabel("Distance euclidienne")
    plt.ylabel("Nombre de comparaisons")
    plt.title("Distribution des distances : bonnes vs mauvaises correspondances")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path2 = os.path.join(args.output_dir, "distance_distribution.png")
    plt.savefig(path2, dpi=150)
    print(f"[GRAPHIQUE] Sauvegardé : {path2}")

    print("\n[INFO] Le graphique de distribution est LE visuel à mettre en avant dans le rapport :")
    print("       il montre concrètement pourquoi un seuil autour de 0.6 sépare bien les deux cas,")
    print("       et où se trouve la zone de recouvrement (source des erreurs possibles).")


if __name__ == "__main__":
    main()
