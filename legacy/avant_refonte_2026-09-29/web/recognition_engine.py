"""
Moteur de reconnaissance faciale, partagé par toute la plateforme web.

Encapsule InsightFace (détection SCRFD + embedding ArcFace) et un index
FAISS pour la recherche des plus proches voisins. Conçu pour être
rechargé à chaud (méthode `reload`) après l'ajout ou la suppression
d'une personne, sans redémarrer le serveur Flask.
"""

import os
import threading
from collections import Counter

import cv2
import numpy as np
import faiss
from insightface.app import FaceAnalysis


class RecognitionEngine:
    def __init__(self, encodings_path, dataset_dir, det_size=640, gpu=False, threshold=0.45, knn_k=3):
        self.encodings_path = encodings_path
        self.dataset_dir = dataset_dir
        self.threshold = threshold
        self.knn_k = knn_k
        self._lock = threading.Lock()

        print("[ENGINE] Initialisation d'InsightFace (SCRFD + ArcFace)...")
        ctx_id = 0 if gpu else -1
        self.app = FaceAnalysis(
            name="buffalo_l", providers=["CUDAExecutionProvider"] if gpu else ["CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))

        self.index = None
        self.known_names = []
        self.reload()

    def reload(self):
        """Recharge la base d'embeddings depuis le fichier .npz (après ajout/suppression d'une personne)."""
        with self._lock:
            if not os.path.exists(self.encodings_path):
                print(f"[ENGINE] Aucune base trouvée ({self.encodings_path}) -> base vide")
                self.index = None
                self.known_names = []
                return

            data = np.load(self.encodings_path)
            known_encodings = data["encodings"].astype(np.float32)
            self.known_names = data["names"].tolist()

            if len(self.known_names) == 0:
                self.index = None
                return

            dimension = known_encodings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(known_encodings)
            print(f"[ENGINE] Base rechargée : {self.index.ntotal} embeddings, {len(set(self.known_names))} personnes")

    def get_enrolled_people(self):
        """Liste les personnes enregistrées avec leur nombre de photos, depuis le dossier dataset/."""
        people = []
        if not os.path.isdir(self.dataset_dir):
            return people
        for name in sorted(os.listdir(self.dataset_dir)):
            person_dir = os.path.join(self.dataset_dir, name)
            if not os.path.isdir(person_dir):
                continue
            photo_count = len([f for f in os.listdir(person_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
            people.append({"name": name, "photo_count": photo_count})
        return people

    def detect_and_identify(self, frame):
        """
        Détecte tous les visages d'une frame et les identifie.
        Retourne une liste de dicts : box, name, similarity, kps.
        """
        faces = self.app.get(frame)
        results = []

        with self._lock:
            has_index = self.index is not None and self.index.ntotal > 0

            for face in faces:
                x1, y1, x2, y2 = face.bbox.astype(int)

                if has_index:
                    name, sim = self._knn_predict(face.normed_embedding)
                else:
                    name, sim = "Inconnu", 0.0

                results.append({
                    "box": (x1, y1, x2, y2),
                    "name": name,
                    "similarity": float(sim),
                    "kps": face.kps,
                })
        return results

    def _knn_predict(self, target_encoding):
        query_vector = np.array([target_encoding], dtype=np.float32)
        similarities, indices = self.index.search(query_vector, min(self.knn_k, self.index.ntotal))

        candidates = [
            (self.known_names[idx], sim)
            for sim, idx in zip(similarities[0], indices[0])
            if idx != -1 and sim >= self.threshold
        ]

        if not candidates:
            overall_best = similarities[0][0] if indices[0][0] != -1 else 0.0
            return "Inconnu", overall_best

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

        representative_sim = max(sim for name, sim in candidates if name == winner)
        return winner, representative_sim


def encode_dataset(dataset_dir, encodings_path, det_size=640, gpu=False, app=None):
    """
    (Ré)encode tout le dataset en embeddings ArcFace, sauvegarde en .npz.
    Réutilise une instance FaceAnalysis existante si fournie (`app`), pour
    éviter de recharger le modèle inutilement depuis la plateforme web.
    """
    if app is None:
        ctx_id = 0 if gpu else -1
        app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider"] if gpu else ["CPUExecutionProvider"])
        app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))

    known_encodings = []
    known_names = []
    total_processed, total_skipped = 0, 0

    person_folders = sorted(f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f)))

    for person_name in person_folders:
        person_dir = os.path.join(dataset_dir, person_name)
        images = [f for f in os.listdir(person_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        for filename in images:
            image = cv2.imread(os.path.join(person_dir, filename))
            if image is None:
                total_skipped += 1
                continue

            faces = app.get(image)
            if len(faces) == 0:
                total_skipped += 1
                continue

            best_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            known_encodings.append(best_face.normed_embedding)
            known_names.append(person_name)
            total_processed += 1

    os.makedirs(os.path.dirname(encodings_path), exist_ok=True)
    if known_encodings:
        np.savez_compressed(encodings_path, encodings=np.array(known_encodings), names=np.array(known_names))
    else:
        # Base vide (aucune personne enregistrée) -> on supprime l'ancien fichier s'il existe
        if os.path.exists(encodings_path):
            os.remove(encodings_path)

    return {"processed": total_processed, "skipped": total_skipped, "people": len(set(known_names))}
