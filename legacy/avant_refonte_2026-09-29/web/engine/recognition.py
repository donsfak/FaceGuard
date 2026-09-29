import numpy as np
import faiss
import cv2
from collections import deque, Counter
from insightface.app import FaceAnalysis

class FaceRecognitionEngine:
    def __init__(self, encodings_path="../models/encodings_arcface.npz", threshold=0.45, knn_k=3, smoothing=15, gpu=False):
        print("[IA] Initialisation du moteur de reconnaissance...")
        self.threshold = threshold
        self.knn_k = knn_k
        self.smoothing = smoothing
        self.MAX_CENTROID_DIST = 80
        
        ctx_id = 0 if gpu else -1
        self.app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'] if gpu else ['CPUExecutionProvider'])
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        
        print(f"[IA] Chargement de la base {encodings_path}...")
        data = np.load(encodings_path)
        self.known_encodings = data["encodings"].astype(np.float32)
        self.known_names = data["names"].tolist()
        
        self.index = faiss.IndexFlatIP(self.known_encodings.shape[1])
        self.index.add(self.known_encodings)
        
        self.tracks = []
        print("[IA] Moteur prêt pour la production.")

    def reload_encodings(self, encodings_path="../models/encodings_arcface.npz"):
        """Recharge les encodings et met à jour l'index FAISS à chaud."""
        data = np.load(encodings_path)
        self.known_encodings = data["encodings"].astype(np.float32)
        self.known_names = data["names"].tolist()
        
        self.index = faiss.IndexFlatIP(self.known_encodings.shape[1])
        self.index.add(self.known_encodings)
        print("[IA] Base d'encodages rechargée avec succès.")

    def _faiss_knn_predict(self, target_encoding):
        query_vector = np.array([target_encoding], dtype=np.float32)
        similarities, indices = self.index.search(query_vector, self.knn_k)
        
        candidates = []
        for i in range(self.knn_k):
            # Conversion explicite en types natifs Python pour éviter les erreurs de sérialisation JSON
            sim = float(similarities[0][i])
            idx = int(indices[0][i])
            if idx != -1 and sim >= self.threshold:
                candidates.append((self.known_names[idx], sim))

        if not candidates:
            fallback_sim = float(similarities[0][0]) if indices[0][0] != -1 else 0.0
            return "Inconnu", fallback_sim

        vote_counts = Counter(name for name, _ in candidates)
        winner = max(vote_counts.keys(), key=lambda n: vote_counts[n])
        representative_sim = float(max([sim for name, sim in candidates if name == winner]))
        return winner, representative_sim

    def _calculate_3d_ratio(self, kps):
        left_eye, right_eye, nose = kps[0], kps[1], kps[2]
        eye_distance = float(np.linalg.norm(left_eye - right_eye))
        eye_center = (left_eye + right_eye) / 2.0
        nose_distance = float(np.linalg.norm(nose - eye_center))
        return nose_distance / eye_distance if eye_distance != 0 else 0.0

    def process_frame(self, frame):
        faces = self.app.get(frame)
        raw_results = []
        
        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            centroid = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            name, best_sim = self._faiss_knn_predict(face.normed_embedding)
            ratio = self._calculate_3d_ratio(face.kps)
            # face.kps : 5 points [Œil G, Œil D, Nez, Bouche G, Bouche D], en coordonnées
            # de la frame originale (même repère que face.bbox, pas besoin de mise à l'échelle).
            raw_results.append((centroid, (int(x1), int(y1), int(x2), int(y2)), name, best_sim, ratio, face.kps))

        matched_track_ids = set()
        final_results = []

        for centroid, box, name, sim, ratio, kps in raw_results:
            best_track_idx = None
            best_dist = self.MAX_CENTROID_DIST
            
            for idx, track in enumerate(self.tracks):
                if idx in matched_track_ids:
                    continue
                d = np.hypot(centroid[0] - track["centroid"][0], centroid[1] - track["centroid"][1])
                if d < best_dist:
                    best_dist = d
                    best_track_idx = idx

            if best_track_idx is None:
                new_track = {
                    "centroid": centroid, 
                    "history": deque(maxlen=self.smoothing),
                    "ratio_history": deque(maxlen=self.smoothing)
                }
                self.tracks.append(new_track)
                best_track_idx = len(self.tracks) - 1

            track = self.tracks[best_track_idx]
            track["centroid"] = centroid
            track["history"].append(name)
            track["ratio_history"].append(ratio)
            matched_track_ids.add(best_track_idx)

            smoothed_name = Counter(track["history"]).most_common(1)[0][0]
            is_real = False
            liveness_status = "Analyse..."
            
            if len(track["ratio_history"]) == self.smoothing:
                ratio_variance = float(np.var(track["ratio_history"]))
                if ratio_variance < 0.00007:
                    liveness_status = "SPOOF (Photo)"
                    is_real = False
                else:
                    liveness_status = "VIVANT"
                    is_real = True

            if not is_real and len(track["ratio_history"]) == self.smoothing:
                smoothed_name = "FRAUDE DETECTEE"

            final_results.append({
                "status": "success",
                "identity": str(smoothed_name),
                "similarity": float(sim),
                "liveness": str(liveness_status),
                "is_real": bool(is_real),
                "box": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                # .tolist() convertit déjà les float32 numpy en float Python natifs -> JSON-safe
                "kps": kps.tolist(),
            })

        self.tracks = [t for i, t in enumerate(self.tracks) if i in matched_track_ids]
        return final_results
