"""
Benchmark offline — Compare 3 détecteurs de visages sur le dataset :
    - hog   (dlib, classique, rapide)
    - cnn   (dlib, deep learning, plus lent mais plus robuste)
    - mtcnn (Multi-task Cascaded CNN, deep learning, autre architecture)

Pour chaque détecteur, on mesure :
    - le taux de détection (% d'images où au moins un visage est trouvé)
    - le temps moyen de traitement par image

Ce script ne modifie PAS encodings.pickle ni le pipeline temps réel : il sert
uniquement à produire un tableau comparatif pour le rapport.

Usage :
    python benchmark_detectors.py
"""

import os
import time

import cv2
import face_recognition
import numpy as np

try:
    from mtcnn import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False


DATASET_DIR = "../dataset"


def list_images(dataset_dir):
    paths = []
    for person_name in sorted(os.listdir(dataset_dir)):
        person_dir = os.path.join(dataset_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
        for filename in sorted(os.listdir(person_dir)):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(person_dir, filename))
    return paths


def benchmark_dlib(image_paths, method):
    detected = 0
    total_time = 0.0
    for path in image_paths:
        image = face_recognition.load_image_file(path)
        start = time.time()
        boxes = face_recognition.face_locations(image, model=method)
        total_time += time.time() - start
        if len(boxes) > 0:
            detected += 1
    return detected, total_time


def benchmark_mtcnn(image_paths):
    detector = MTCNN()
    detected = 0
    total_time = 0.0
    for path in image_paths:
        image_bgr = cv2.imread(path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        start = time.time()
        results = detector.detect_faces(image_rgb)
        total_time += time.time() - start
        if len(results) > 0:
            detected += 1
    return detected, total_time


def main():
    image_paths = list_images(DATASET_DIR)
    n = len(image_paths)
    print(f"[INFO] {n} images dans le dataset\n")

    results = {}

    print("[1/3] Benchmark hog...")
    detected, elapsed = benchmark_dlib(image_paths, "hog")
    results["hog"] = (detected, elapsed)

    print("[2/3] Benchmark cnn (patience, plus lent)...")
    detected, elapsed = benchmark_dlib(image_paths, "cnn")
    results["cnn"] = (detected, elapsed)

    if MTCNN_AVAILABLE:
        print("[3/3] Benchmark mtcnn...")
        detected, elapsed = benchmark_mtcnn(image_paths)
        results["mtcnn"] = (detected, elapsed)
    else:
        print("[3/3] mtcnn non installé (pip install mtcnn tensorflow) -> ignoré")

    print(f"\n{'Détecteur':<12}{'Taux détection':<18}{'Temps moyen/image':<20}")
    for name, (detected, elapsed) in results.items():
        rate = detected / n * 100 if n else 0
        avg_time = elapsed / n * 1000 if n else 0  # en ms
        print(f"{name:<12}{f'{rate:.1f}% ({detected}/{n})':<18}{f'{avg_time:.1f} ms':<20}")

    print("\n[INFO] Ce tableau est directement exploitable dans le rapport pour justifier")
    print("       le choix de hog (temps réel) + fallback cnn (robustesse ponctuelle).")


if __name__ == "__main__":
    main()
