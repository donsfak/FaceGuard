"""
Étape 4 (Production) — Reconnaissance faciale temps réel avec InsightFace + FAISS.

Modifications de niveau Production :
  - Intégration de FAISS (IndexFlatIP) pour une recherche vectorielle O(1).
  - Conversion stricte en float32 (requis par l'architecture C++ de FAISS).
"""

import argparse
import pickle
import time
from collections import deque, Counter
import cv2
import numpy as np
import faiss  # <-- Nouvel import indispensable
from insightface.app import FaceAnalysis

def parse_args():
    parser = argparse.ArgumentParser(description="Reconnaissance temps réel (InsightFace + FAISS)")
    parser.add_argument("--encodings", default="../models/encodings_arcface.pickle")
    parser.add_argument("--threshold", type=float, default=0.45, help="Seuil de similarité cosinus (plus haut = plus strict)")
    parser.add_argument("--knn-k", type=int, default=3, help="Nombre de voisins pour le vote k-NN.")
    parser.add_argument("--det-size", type=int, default=640)
    parser.add_argument("--gpu", action="store_true", help="Activer l'accélération GPU")
    parser.add_argument("--smoothing", type=int, default=8)
    parser.add_argument("--process-every-n", type=int, default=3)
    return parser.parse_args()


def faiss_knn_predict(target_encoding, index, known_names, threshold, k=3):
    """
    Identification par k-NN propulsée par FAISS.
    """
    # FAISS exige des matrices 2D en float32. On reshape notre vecteur (512,) en (1, 512)
    query_vector = np.array([target_encoding], dtype=np.float32)
    
    # Recherche instantanée dans l'index
    # similarities : matrice des scores de similarité
    # indices : matrice des positions dans la liste known_names
    similarities, indices = index.search(query_vector, k)
    
    candidates = []
    for i in range(k):
        sim = similarities[0][i]
        idx = indices[0][i]
        # FAISS renvoie -1 si l'index est vide ou s'il n'y a pas assez de voisins
        if idx != -1 and sim >= threshold:
            candidates.append((known_names[idx], sim))

    if not candidates:
        overall_best = similarities[0][0] if indices[0][0] != -1 else 0.0
        return "Inconnu", overall_best

    # Logique de vote (identique à la version précédente)
    vote_counts = Counter(name for name, _ in candidates)
    max_votes = max(vote_counts.values())
    tied_names = [name for name, count in vote_counts.items() if count == max_votes]

    if len(tied_names) == 1:
        winner = tied_names[0]
    else:
        best_per_name = {}
        for name, sim in candidates:
            if name not in best_per_name or sim > best_per_name[name]:
                best_per_name[name] = sim
        winner = max(tied_names, key=lambda n: best_per_name[n])

    representative_sim = max([sim for name, sim in candidates if name == winner])
    return winner, representative_sim


def main():
    args = parse_args()

    print("[INFO] Initialisation d'InsightFace...")
    ctx_id = 0 if args.gpu else -1
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'] if args.gpu else ['CPUExecutionProvider'])
    app.prepare(ctx_id=ctx_id, det_size=(args.det_size, args.det_size))

    print(f"[INFO] Chargement de {args.encodings}...")
    with open(args.encodings, "rb") as f:
        data = pickle.load(f)
    
    # ---------------------------------------------------------
    # NOUVEAU MOTEUR FAISS
    # 1. FAISS ne tolère QUE le type float32. C'est une erreur classique de l'oublier.
    known_encodings = np.array(data["encodings"], dtype=np.float32)
    known_names = data["names"]
    
    dimension = known_encodings.shape[1] # Devrait être 512 avec ArcFace
    
    # 2. Création de l'index Produit Interne (Cosine Similarity sur vecteurs normalisés)
    index = faiss.IndexFlatIP(dimension)
    
    # 3. Injection des données dans la base vectorielle C++
    index.add(known_encodings)
    print(f"[INFO] FAISS : {index.ntotal} embeddings (dim {dimension}) indexés avec succès.")
    # ---------------------------------------------------------

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la webcam.")

    print("[INFO] Moteur FAISS démarré. Appuyez sur 'q' pour quitter.")

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
            faces = app.get(frame)
            raw_results = []
            
            for face in faces:
                x1, y1, x2, y2 = face.bbox.astype(int)
                centroid = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                
                # Appel de notre nouvelle fonction FAISS
                name, best_sim = faiss_knn_predict(
                    face.normed_embedding, 
                    index, 
                    known_names, 
                    args.threshold, 
                    k=args.knn_k
                )
                
                raw_results.append((centroid, (x1, y1, x2, y2), name, best_sim, face.kps))

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

        for (x1, y1, x2, y2), name, sim, kps in last_display_data:
            color = (0, 255, 0) if name != "Inconnu" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{name} ({sim:.2f})"
            cv2.rectangle(frame, (x1, y2 - 25), (x2, y2), color, cv2.FILLED)
            cv2.putText(frame, label, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            if kps is not None:
                for pt in kps.astype(int):
                    cv2.circle(frame, tuple(pt), 2, (255, 255, 0), -1)

        current_time = time.time()
        fps = 1.0 / (current_time - prev_time) if current_time != prev_time else fps
        prev_time = current_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

        cv2.imshow("Reconnaissance Faciale - FAISS", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
