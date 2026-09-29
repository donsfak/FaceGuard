"""
Moteur de reconnaissance faciale partagé (scripts CLI + plateforme web).

Pipeline pour chaque image :
    image BGR
      -> SCRFD      : détection des visages + 5 points clés (yeux, nez, bouche)
      -> alignement : le visage est redressé à partir des 5 points (112x112)
      -> ArcFace    : embedding de 512 dimensions, normalisé (norme L2 = 1)
      -> FAISS      : recherche des k plus proches voisins par produit scalaire
                      (= similarité cosinus, puisque les vecteurs sont normalisés)
      -> vote k-NN + seuil : identité ou "Inconnu"

La base de référence (les "embeddings connus") est un fichier .npz contenant
deux tableaux : `encodings` (N x 512, float32) et `names` (N chaînes).
Le format .npz est lu avec allow_pickle=False : contrairement à un .pickle,
il ne peut pas exécuter de code à l'ouverture.
"""

from __future__ import annotations

import os
import re
import tempfile
import threading
import unicodedata
from collections import Counter
from pathlib import Path

import cv2
import faiss
import numpy as np
from insightface.app import FaceAnalysis

from core import config


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

_NAME_ALLOWED = re.compile(r"^[\w\- '.]+$", re.UNICODE)


def normalize_name(raw_name: str) -> str:
    """Nettoie un nom saisi par l'utilisateur et vérifie qu'il est utilisable
    comme nom de dossier (pas de '/', pas de '..', longueur raisonnable).

    Lève ValueError avec un message lisible si le nom est invalide.
    """
    name = unicodedata.normalize("NFC", (raw_name or "")).strip()
    name = re.sub(r"\s+", " ", name)
    if not name:
        raise ValueError("Le nom est vide.")
    if len(name) > 50:
        raise ValueError("Le nom ne doit pas dépasser 50 caractères.")
    if not _NAME_ALLOWED.match(name) or name.startswith(".") or ".." in name:
        raise ValueError("Le nom ne peut contenir que des lettres, chiffres, espaces, tirets, apostrophes et points.")
    if name.lower() in (config.UNKNOWN_LABEL.lower(), config.SPOOF_LABEL.lower()):
        raise ValueError("Ce nom est réservé par le système.")
    return name


def largest_face(faces):
    """Visage à la plus grande boîte englobante (évite d'encoder quelqu'un en arrière-plan)."""
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def choose_consistent_faces(faces_per_image, min_sim: float = 0.30):
    """Pour une MÊME personne, choisit dans chaque image le visage qui lui correspond.

    Problème réel rencontré dans le dataset : sur certaines photos, une autre
    personne apparaît en arrière-plan et son visage est parfois le plus grand
    (ex. oyetola_000.jpg). "Garder le plus grand visage" encodait alors la
    mauvaise personne sous le mauvais nom -> faux positifs.

    Méthode :
      1. on prend le plus grand visage de chaque image (la personne photographiée,
         dans la grande majorité des cas) ;
      2. le "médoïde" = celui qui ressemble le plus aux autres : c'est notre
         référence. Un intrus présent sur quelques photos seulement ne peut pas
         l'être (ex. dataset kadi : une personne en arrière-plan sur toutes les
         photos, mais toujours plus petite que la personne photographiée) ;
      3. centroïde = moyenne des plus grands visages proches du médoïde ;
      4. pour chaque image, on garde le visage le plus proche du centroïde,
         ou rien si sa similarité est < min_sim (image aberrante, signalée).

    Retourne (embeddings, similarités_au_centroïde) — None pour une image rejetée.
    """
    embs = [[f.normed_embedding.astype(np.float32) for f in faces] for faces in faces_per_image]
    largest = [largest_face(f).normed_embedding.astype(np.float32) for f in faces_per_image if f]
    if not largest:
        return [None] * len(embs), [0.0] * len(embs)

    ref = np.stack(largest)
    if len(ref) > 1:
        pair = ref @ ref.T
        np.fill_diagonal(pair, np.nan)
        medoid = ref[int(np.nanargmax(np.nanmedian(pair, axis=1)))]
    else:
        medoid = ref[0]
    inliers = ref[ref @ medoid >= min_sim]
    centroid = inliers.mean(axis=0) if len(inliers) else medoid
    centroid = centroid / np.linalg.norm(centroid)

    chosen, sims = [], []
    for candidates in embs:
        if not candidates:
            chosen.append(None)
            sims.append(0.0)
            continue
        scores = [float(c @ centroid) for c in candidates]
        best = int(np.argmax(scores))
        ok = scores[best] >= min_sim
        chosen.append(candidates[best] if ok else None)
        sims.append(scores[best])

    # 4. Garde-fou par paires : chaque visage retenu doit ressembler à la
    #    MAJORITÉ des autres (médiane). Un centroïde seul ne suffit pas : avec
    #    3 personnes différentes, chacune est à ~0.55 de leur moyenne.
    kept = [i for i, c in enumerate(chosen) if c is not None]
    if len(kept) >= 2:
        mat = np.stack([chosen[i] for i in kept])
        pair = mat @ mat.T
        np.fill_diagonal(pair, np.nan)
        medians = np.nanmedian(pair, axis=1)
        for i, med in zip(kept, medians):
            if med < min_sim:
                chosen[i] = None
                sims[i] = float(med)
    return chosen, sims


def list_dataset_images(dataset_dir: Path):
    """Retourne [(nom_personne, chemin_image), ...] pour un dossier dataset/<personne>/*.jpg."""
    dataset_dir = Path(dataset_dir)
    items = []
    if not dataset_dir.is_dir():
        return items
    for person_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        for img in sorted(person_dir.iterdir()):
            if img.suffix.lower() in config.IMAGE_EXTENSIONS:
                items.append((person_dir.name, img))
    return items


# ---------------------------------------------------------------------------
# Moteur
# ---------------------------------------------------------------------------

class FaceEngine:
    """Détection + embedding (InsightFace) et identification (FAISS + vote k-NN)."""

    def __init__(
        self,
        encodings_path: Path | str | None = config.ENCODINGS_PATH,
        threshold: float = config.THRESHOLD,
        knn_k: int = config.KNN_K,
        det_size: int = config.DET_SIZE,
        gpu: bool = config.USE_GPU,
        model_name: str = config.MODEL_NAME,
    ):
        self.encodings_path = Path(encodings_path) if encodings_path else None
        self.threshold = threshold
        self.knn_k = knn_k

        print(f"[IA] Chargement d'InsightFace '{model_name}' (det_size={det_size}, gpu={gpu})...")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if gpu else ["CPUExecutionProvider"]
        # buffalo_l contient 5 réseaux (détection, reconnaissance, âge/genre,
        # landmarks 2D-106 et 3D-68). On ne charge QUE les deux utiles :
        # les 3 autres étaient exécutés pour chaque visage sans jamais être lus
        # -> gain de vitesse important sur CPU.
        self.app = FaceAnalysis(name=model_name, allowed_modules=["detection", "recognition"], providers=providers)
        self.app.prepare(ctx_id=0 if gpu else -1, det_size=(det_size, det_size))

        self._infer_lock = threading.Lock()   # sérialise l'inférence ONNX
        self._db_lock = threading.RLock()     # protège index FAISS + noms

        self.encodings = np.zeros((0, 512), dtype=np.float32)
        self.names: list[str] = []
        self.index = None
        if self.encodings_path is not None:
            self.reload()

    # ------------------------------------------------------------------ base
    def reload(self):
        """(Re)charge la base .npz depuis le disque et reconstruit l'index FAISS."""
        with self._db_lock:
            if self.encodings_path is None or not self.encodings_path.exists():
                print(f"[IA] Aucune base trouvée ({self.encodings_path}) -> base vide.")
                self.set_gallery(np.zeros((0, 512), dtype=np.float32), [])
                return
            with np.load(self.encodings_path, allow_pickle=False) as data:
                encodings = data["encodings"].astype(np.float32)
                names = [str(n) for n in data["names"].tolist()]
            self.set_gallery(encodings, names)
            print(f"[IA] Base chargée : {len(names)} embeddings, {len(set(names))} personnes.")

    def set_gallery(self, encodings: np.ndarray, names: list[str]):
        """Remplace la base de référence en mémoire (utilisé aussi par l'évaluation)."""
        encodings = np.ascontiguousarray(encodings, dtype=np.float32).reshape(-1, 512)
        if len(encodings) != len(names):
            raise ValueError("encodings et names n'ont pas la même longueur")
        with self._db_lock:
            self.encodings = encodings
            self.names = list(names)
            if len(names) == 0:
                self.index = None
            else:
                index = faiss.IndexFlatIP(encodings.shape[1])
                index.add(encodings)
                self.index = index

    def save(self):
        """Écriture atomique du .npz : on écrit dans un fichier temporaire puis on
        le renomme. Un crash pendant l'écriture ne peut donc plus corrompre la base."""
        if self.encodings_path is None:
            raise RuntimeError("Aucun chemin de base configuré")
        with self._db_lock:
            self.encodings_path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self.encodings_path.parent, suffix=".npz.tmp")
            try:
                with os.fdopen(fd, "wb") as f:
                    np.savez_compressed(f, encodings=self.encodings, names=np.array(self.names, dtype=str))
                os.replace(tmp, self.encodings_path)
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)

    def people(self) -> dict[str, int]:
        """{nom: nombre d'embeddings} pour chaque personne connue."""
        with self._db_lock:
            return dict(sorted(Counter(self.names).items(), key=lambda kv: kv[0].lower()))

    def canonical_name(self, name: str) -> str:
        """Réutilise l'orthographe existante si le nom est déjà connu à la casse près
        (évite d'avoir 'falibeta' et 'Falibeta' comme deux personnes différentes)."""
        with self._db_lock:
            for existing in set(self.names):
                if existing.lower() == name.lower():
                    return existing
        return name

    def add_person(self, name: str, embeddings: list[np.ndarray], save: bool = True) -> str:
        """Ajoute les embeddings d'une personne (tous, pas seulement la moyenne :
        le vote k-NN a besoin de plusieurs vecteurs par personne)."""
        name = self.canonical_name(normalize_name(name))
        new = np.asarray(embeddings, dtype=np.float32).reshape(-1, 512)
        new /= np.linalg.norm(new, axis=1, keepdims=True)
        with self._db_lock:
            self.set_gallery(np.vstack([self.encodings, new]), self.names + [name] * len(new))
            if save:
                self.save()
        return name

    def similarity_to_person(self, embedding: np.ndarray, name: str) -> float:
        """Similarité cosinus entre un embedding et le profil moyen d'une personne
        (0.0 si la personne n'existe pas). Sert à vérifier qu'une photo ajoutée
        à un profil existant montre bien la même personne."""
        with self._db_lock:
            idx = [i for i, n in enumerate(self.names) if n == name]
            if not idx:
                return 0.0
            centroid = self.encodings[idx].mean(axis=0)
        centroid /= np.linalg.norm(centroid)
        query = np.asarray(embedding, dtype=np.float32).reshape(-1)
        return float(query @ centroid / np.linalg.norm(query))

    def remove_person(self, name: str, save: bool = True) -> int:
        with self._db_lock:
            keep = [i for i, n in enumerate(self.names) if n != name]
            removed = len(self.names) - len(keep)
            if removed:
                self.set_gallery(self.encodings[keep], [self.names[i] for i in keep])
                if save:
                    self.save()
            return removed

    # ------------------------------------------------------------- inférence
    def detect(self, frame_bgr: np.ndarray):
        """Détection + embedding de tous les visages d'une image (objets insightface Face)."""
        with self._infer_lock:
            return self.app.get(frame_bgr)

    def identify(self, embedding: np.ndarray) -> tuple[str, float]:
        """Vote k-NN parmi les voisins dont la similarité dépasse le seuil.

        Retourne (identité, similarité). Si aucun voisin ne dépasse le seuil,
        retourne ("Inconnu", meilleure similarité trouvée) : ce score reste utile
        pour comprendre pourquoi la personne a été rejetée.
        """
        with self._db_lock:
            if self.index is None or self.index.ntotal == 0:
                return config.UNKNOWN_LABEL, 0.0
            k = min(self.knn_k, self.index.ntotal)
            query = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
            sims, idxs = self.index.search(query, k)
            names = self.names

        pairs = [(names[int(i)], float(s)) for s, i in zip(sims[0], idxs[0]) if i != -1]
        if not pairs:
            return config.UNKNOWN_LABEL, 0.0
        candidates = [(n, s) for n, s in pairs if s >= self.threshold]
        if not candidates:
            return config.UNKNOWN_LABEL, pairs[0][1]

        votes = Counter(n for n, _ in candidates)
        best_sim = {}
        for n, s in candidates:
            best_sim[n] = max(best_sim.get(n, -1.0), s)
        # Majorité des votes ; en cas d'égalité, la meilleure similarité l'emporte.
        winner = max(votes, key=lambda n: (votes[n], best_sim[n]))
        return winner, best_sim[winner]

    def analyze(self, frame_bgr: np.ndarray) -> list[dict]:
        """Détecte et identifie tous les visages. Types Python natifs -> sérialisable en JSON."""
        results = []
        for face in self.detect(frame_bgr):
            name, sim = self.identify(face.normed_embedding)
            x1, y1, x2, y2 = (int(v) for v in face.bbox)
            results.append({
                "box": [x1, y1, x2, y2],
                "kps": face.kps.astype(float).tolist(),
                "identity": name,
                "similarity": round(float(sim), 4),
                "det_score": round(float(face.det_score), 4),
            })
        return results

    def embed_largest_face(self, image_bgr: np.ndarray):
        """Embedding du visage principal d'une photo, ou None si aucun visage."""
        face = largest_face(self.detect(image_bgr))
        return None if face is None else face.normed_embedding.astype(np.float32)


def encode_dataset(engine: FaceEngine, dataset_dir: Path = config.DATASET_DIR, verbose: bool = True):
    """Encode toutes les images de dataset/<personne>/ -> (encodings, names, rapport).

    Le rapport liste les images ignorées (illisibles, sans visage, ou dont aucun
    visage ne ressemble au reste des photos de la personne) et celles qui
    contenaient plusieurs visages : c'est la partie "qualité du dataset".
    """
    by_person: dict[str, list] = {}
    for person, path in list_dataset_images(dataset_dir):
        by_person.setdefault(person, []).append(path)

    encodings, names = [], []
    skipped, multi_face = [], []
    for person, paths in by_person.items():
        faces_per_image, kept_paths = [], []
        for path in paths:
            image = cv2.imread(str(path))
            if image is None:
                skipped.append((str(path), "illisible"))
                continue
            faces = engine.detect(image)
            if not faces:
                skipped.append((str(path), "aucun visage détecté"))
                continue
            if len(faces) > 1:
                multi_face.append(str(path))
            faces_per_image.append(faces)
            kept_paths.append(path)

        chosen, sims = choose_consistent_faces(faces_per_image)
        for path, emb, sim in zip(kept_paths, chosen, sims):
            if emb is None:
                skipped.append((str(path), f"visage incohérent avec les autres photos (sim={sim:.2f})"))
                continue
            encodings.append(emb)
            names.append(person)

    if verbose:
        for path in multi_face:
            print(f"  [INFO] plusieurs visages : {path} (visage de la personne sélectionné)")
        for path, why in skipped:
            print(f"  [SKIP] {path} : {why}")
    report = {"processed": len(names), "skipped": len(skipped), "people": len(set(names)),
              "multi_face_images": len(multi_face), "skipped_details": skipped}
    arr = np.array(encodings, dtype=np.float32).reshape(-1, 512)
    return arr, names, report
