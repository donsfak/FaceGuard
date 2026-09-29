"""
Étape 5 (Sécurité) — Reconnaissance temps réel avec Anti-Spoofing Géométrique.

Modifications de niveau Production :
  - Intégration d'un contrôle de Liveness (Vivacité) par analyse de variance spatiale.
  - Différencie un vrai visage 3D (micro-mouvements) d'une photo 2D rigide (écran/papier).
  - Chargement sécurisé depuis une archive .npz pour bloquer les injections de code.
"""

import argparse
import time
from collections import deque, Counter
import cv2
import numpy as np
import faiss
from insightface.app import FaceAnalysis

def parse_args():
    parser = argparse.ArgumentParser(description="Reconnaissance FAISS + Liveness Detection")
    parser.add_argument("--encodings", default="../models/encodings_arcface.npz")
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--knn-k", type=int, default=3)
    parser.add_argument("--gpu", action="store_true")
    # On augmente légèrement le lissage pour avoir le temps de capter la variance
    parser.add_argument("--smoothing", type=int, default=15) 
    return parser.parse_args()


def faiss_knn_predict(target_encoding, index, known_names, threshold, k=3):
    query_vector = np.array([target_encoding], dtype=np.float32)
    similarities, indices = index.search(query_vector, k)
    
    candidates = []
    for i in range(k):
        sim = similarities[0][i]
        idx = indices[0][i]
        if idx != -1 and sim >= threshold:
            candidates.append((known_names[idx], sim))

    if not candidates:
        return "Inconnu", similarities[0][0] if indices[0][0] != -1 else 0.0

    vote_counts = Counter(name for name, _ in candidates)
    winner = max(vote_counts.keys(), key=lambda n: vote_counts[n])
    representative_sim = max([sim for name, sim in candidates if name == winner])
    return winner, representative_sim


def calculate_3d_ratio(kps):
    """
    Calcule le ratio de distance entre le nez et le centre des yeux, 
    divisé par la distance entre les deux yeux.
    kps : tableau de 5 points [Oeil G, Oeil D, Nez, Bouche G, Bouche D]
    """
    left_eye = kps[0]
    right_eye = kps[1]
    nose = kps[2]
    
    # Distance entre les deux yeux
    eye_distance = np.linalg.norm(left_eye - right_eye)
    
    # Point central entre les deux yeux
    eye_center = (left_eye + right_eye) / 2.0
    
    # Distance entre le centre des yeux et le nez
    nose_distance = np.linalg.norm(nose - eye_center)
    
    # Si la distance des yeux est nulle (erreur de détection), on retourne 0
    if eye_distance == 0:
        return 0.0
        
    return nose_distance / eye_distance


def main():
    # 1. On restaure la lecture des arguments
    args = parse_args()
    
    # 2. On restaure l'initialisation du réseau de neurones InsightFace
    print("[INFO] Initialisation du système complet (InsightFace + FAISS + Anti-Spoofing)...")
    ctx_id = 0 if args.gpu else -1
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'] if args.gpu else ['CPUExecutionProvider'])
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))

    print(f"[INFO] Chargement sécurisé de {args.encodings}...")
    
    # 3. Chargement propre et sécurisé depuis l'archive NPZ
    data = np.load(args.encodings)
    
    # On force le float32 pour FAISS et on convertit le tableau de noms en liste Python
    known_encodings = data["encodings"].astype(np.float32)
    known_names = data["names"].tolist()
    
    # 4. Initialisation de l'index FAISS
    index = faiss.IndexFlatIP(known_encodings.shape[1])
    index.add(known_encodings)

    cap = cv2.VideoCapture(0)
    
    tracks = []
    MAX_CENTROID_DIST = 80
    frame_index = 0
    last_display_data = []

    print("[INFO] Moteur sécurisé démarré. Montrez une photo sur votre téléphone pour tester l'anti-spoofing !")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Analyse à chaque frame pour ne rater aucune donnée de micro-mouvement
        faces = app.get(frame)
        raw_results = []
        
        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            centroid = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            name, best_sim = faiss_knn_predict(face.normed_embedding, index, known_names, args.threshold, k=args.knn_k)
            
            # Calcul du ratio géométrique pour cette frame
            ratio = calculate_3d_ratio(face.kps)
            raw_results.append((centroid, (x1, y1, x2, y2), name, best_sim, face.kps, ratio))

        matched_track_ids = set()
        display_data = []

        for centroid, box, name, sim, kps, ratio in raw_results:
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
                # Ajout de l'historique des ratios géométriques
                new_track = {
                    "centroid": centroid, 
                    "history": deque(maxlen=args.smoothing),
                    "ratio_history": deque(maxlen=args.smoothing)
                }
                tracks.append(new_track)
                best_track_idx = len(tracks) - 1

            track = tracks[best_track_idx]
            track["centroid"] = centroid
            track["history"].append(name)
            track["ratio_history"].append(ratio)
            matched_track_ids.add(best_track_idx)

            # --- LOGIQUE DE LIVENESS (ANTI-SPOOFING) ---
            smoothed_name = Counter(track["history"]).most_common(1)[0][0]
            is_real = False
            liveness_status = "Analyse..."
            
            # On attend d'avoir assez de frames pour calculer une variance significative
            if len(track["ratio_history"]) == args.smoothing:
                # Calcul de la variance du ratio sur les X dernières frames
                ratio_variance = np.var(track["ratio_history"])
                
                # Un visage 3D (vivant) aura toujours des micro-mouvements
                # Une photo ou un écran 2D gardera des proportions parfaitement fixes
                if ratio_variance < 0.00007:
                    liveness_status = f"SPOOF (Photo) var:{ratio_variance:.5f}"
                    is_real = False
                else:
                    liveness_status = f"VIVANT var:{ratio_variance:.5f}"
                    is_real = True

            # Si c'est un faux, on écrase l'identité
            if not is_real and len(track["ratio_history"]) == args.smoothing:
                smoothed_name = "FRAUDE DETECTEE"

            display_data.append((box, smoothed_name, sim, kps, liveness_status, is_real))

        tracks = [t for i, t in enumerate(tracks) if i in matched_track_ids]
        
        # Affichage
        for (x1, y1, x2, y2), name, sim, kps, status, is_real in display_data:
            # Code couleur : Vert si vivant, Orange si analyse en cours, Rouge si fraude ou inconnu
            if name == "FRAUDE DETECTEE":
                color = (0, 0, 255) # Rouge
            elif name == "Inconnu":
                color = (0, 165, 255) # Orange
            elif not is_real:
                color = (0, 255, 255) # Jaune (Analyse)
            else:
                color = (0, 255, 0) # Vert (OK)
                
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Label identité
            label = f"{name} ({sim:.2f})"
            cv2.rectangle(frame, (x1, y2 - 25), (x2, y2), color, cv2.FILLED)
            cv2.putText(frame, label, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Label sécurité (au-dessus de la tête)
            cv2.putText(frame, status, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            for pt in kps.astype(int):
                cv2.circle(frame, tuple(pt), 2, (255, 255, 0), -1)

        cv2.imshow("Reconnaissance Faciale - Securite", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
