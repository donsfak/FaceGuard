"""
Étape 4 (Refactorisée) — Reconnaissance faciale temps réel avec InsightFace (ArcFace).

Modifications de niveau Production :
  - Remplacement de dlib/face_recognition par InsightFace (modèles buffalo_l).
  - Détection SCRFD : surpasse de loin HOG/CNN en vitesse et robustesse.
  - Embeddings 512-D normalisés avec perte ArcFace.
  - Calcul de similarité Cosinus au lieu de la distance Euclidienne.

Usage :
    python 03_recognize_webcam_insightface.py --threshold 0.45 --gpu
"""

import argparse
import csv
import os
import pickle
import time
from collections import deque, Counter
from datetime import datetime

import cv2
import numpy as np
from insightface.app import FaceAnalysis

def parse_args():
    parser = argparse.ArgumentParser(description="Reconnaissance faciale temps réel avec InsightFace")
    parser.add_argument("--encodings", default="../models/encodings_arcface.pickle", help="Fichier d'encodages (InsightFace 512-D)")
    # Attention: Avec la similarité cosinus, un seuil HAUT signifie PLUS STRICT.
    parser.add_argument("--threshold", type=float, default=0.45,
                         help="Seuil de similarité cosinus. (Plus haut = plus strict. Défaut 0.45)")
    parser.add_argument("--knn-k", type=int, default=3, help="Nombre de voisins pour le vote k-NN.")
    parser.add_argument("--det-size", type=int, default=640,
                         help="Taille d'entrée du détecteur. 320 = très rapide, 640 = très précis.")
    parser.add_argument("--gpu", action="store_true", help="Activer l'accélération GPU (CUDA)")
    parser.add_argument("--smoothing", type=int, default=8, help="Frames pour vote de stabilisation")
    parser.add_argument("--process-every-n", type=int, default=3, help="Saut de frames pour perf CPU")
    return parser.parse_args()


def knn_predict_cosine(target_encoding, known_encodings, known_names, threshold, k=3):
    """
    Identification par k-NN basée sur la similarité cosinus.
    Les embeddings InsightFace (normed_embedding) sont déjà normalisés L2.
    Le produit scalaire (dot product) équivaut donc directement à la similarité cosinus.
    """
    # Calcul vectorisé de la similarité avec toute la base
    similarities = np.dot(known_encodings, target_encoding)
    
    # On trie de manière décroissante (les plus hautes similarités d'abord)
    nearest_indices = np.argsort(similarities)[::-1][:k]
    
    # On filtre par notre seuil (seuls ceux au-dessus du seuil sont valides)
    candidates = [(known_names[i], similarities[i]) for i in nearest_indices if similarities[i] >= threshold]

    if not candidates:
        # On renvoie la meilleure similarité même si elle a échoué, pour debug
        overall_best = similarities[nearest_indices[0]] if len(similarities) > 0 else 0.0
        return "Inconnu", overall_best

    vote_counts = Counter(name for name, _ in candidates)
    max_votes = max(vote_counts.values())
    tied_names = [name for name, count in vote_counts.items() if count == max_votes]

    if len(tied_names) == 1:
        winner = tied_names[0]
    else:
        # En cas d'égalité, on tranche par la similarité maximale
        best_per_name = {}
        for name, sim in candidates:
            if name not in best_per_name or sim > best_per_name[name]:
                best_per_name[name] = sim
        winner = max(tied_names, key=lambda n: best_per_name[n])

    representative_sim = max([sim for name, sim in candidates if name == winner])
    return winner, representative_sim


def main():
    args = parse_args()

    # 1. Initialisation d'InsightFace (Modèle 'buffalo_l' inclut détection SCRFD et Reco ArcFace ResNet50)
    print("[INFO] Initialisation d'InsightFace...")
    ctx_id = 0 if args.gpu else -1
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'] if args.gpu else ['CPUExecutionProvider'])
    # det_size gère le redimensionnement interne en gardant l'aspect ratio
    app.prepare(ctx_id=ctx_id, det_size=(args.det_size, args.det_size))

    # 2. Chargement de la base d'encodages (ATTENTION: Il faut regénérer les encodages avec InsightFace)
    print(f"[INFO] Chargement de {args.encodings}...")
    if not os.path.exists(args.encodings):
        raise FileNotFoundError(f"Fichier d'encodages introuvable. Vous devez re-lancer l'encodage avec InsightFace.")
    
    with open(args.encodings, "rb") as f:
        data = pickle.load(f)
    
    known_encodings = np.array(data["encodings"])
    known_names = data["names"]
    print(f"[INFO] {len(known_names)} embeddings (512-D) chargés pour {len(set(known_names))} personnes")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la webcam.")

    print("[INFO] Moteur de reconnaissance démarré. Appuyez sur 'q' pour quitter.")

    tracks = [] 
    MAX_CENTROID_DIST = 80 
    frame_index = 0
    last_display_data = [] 

    prev_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % args.process_every_n == 0:
            # InsightFace gère la détection, l'alignement et l'embedding en une seule passe.
            # Il prend directement la frame BGR d'OpenCV.
            faces = app.get(frame)

            raw_results = []
            for face in faces:
                # bounding box sous format [left, top, right, bottom]
                x1, y1, x2, y2 = face.bbox.astype(int)
                centroid = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                
                # ArcFace utilise des embeddings normalisés de 512 dimensions
                encoding = face.normed_embedding

                name, best_sim = knn_predict_cosine(encoding, known_encodings, known_names, args.threshold, k=args.knn_k)
                raw_results.append((centroid, (x1, y1, x2, y2), name, best_sim, face.kps))

            # --- Association temporelle (Tracking basique) ---
            matched_track_ids = set()
            display_data = []

            for centroid, box, name, sim, kps in raw_results:
                best_track_idx = None
                best_dist = MAX_CENTROID_DIST
                for idx, track in enumerate(tracks):
                    if idx in matched_track_ids:
                        continue
                    d = np.hypot(centroid[0] - track["centroid"][0], centroid[1] - track["centroid"][1])
                    if d < best_dist:
                        best_dist = d
                        best_track_idx = idx

                if best_track_idx is None:
                    new_track = {"centroid": centroid, "history": deque(maxlen=max(args.smoothing, 1))}
                    tracks.append(new_track)
                    best_track_idx = len(tracks) - 1

                track = tracks[best_track_idx]
                track["centroid"] = centroid
                track["history"].append(name)
                matched_track_ids.add(best_track_idx)

                smoothed_name = Counter(track["history"]).most_common(1)[0][0] if args.smoothing > 0 else name
                display_data.append((box, smoothed_name, sim, kps))

            tracks = [t for i, t in enumerate(tracks) if i in matched_track_ids]
            last_display_data = display_data 

        frame_index += 1

        # --- Affichage OpenCV ---
        for (x1, y1, x2, y2), name, sim, kps in last_display_data:
            color = (0, 255, 0) if name != "Inconnu" else (0, 0, 255)
            
            # Boîte englobante
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label et Score de similarité
            label = f"{name} ({sim:.2f})"
            cv2.rectangle(frame, (x1, y2 - 25), (x2, y2), color, cv2.FILLED)
            cv2.putText(frame, label, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Affichage des 5 points faciaux clefs (Alignement ArcFace)
            if kps is not None:
                for pt in kps.astype(int):
                    cv2.circle(frame, tuple(pt), 2, (255, 255, 0), -1)

        # Calcul FPS
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time) if current_time != prev_time else fps
        prev_time = current_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

        cv2.imshow("Reconnaissance Faciale - ArcFace (Production)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
