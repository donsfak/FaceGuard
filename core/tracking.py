"""
Suivi temporel des visages d'une frame à l'autre + anti-spoofing géométrique.

Pourquoi un suivi ?
  Sur une seule image, la reconnaissance peut hésiter (flou, visage tourné).
  On associe chaque visage à une "piste" (track) grâce à la position de son
  centre, puis on fait un vote majoritaire sur les N dernières prédictions :
  l'identité affichée est beaucoup plus stable.

IMPORTANT : un FaceTracker contient l'état d'UNE caméra. Dans la webapp,
chaque connexion WebSocket a son propre tracker (avant, un tracker global
mélangeait les visages de deux navigateurs ouverts en même temps).

Anti-spoofing (prototype pédagogique) :
  On mesure le ratio  distance(nez, milieu des yeux) / distance(yeux).
  Sur un vrai visage, les micro-mouvements 3D font varier ce ratio ;
  sur une photo plate tenue immobile il reste quasi constant.
  Limites connues : une photo qu'on incline/bouge peut passer, une personne
  parfaitement immobile peut être suspectée. Ce n'est PAS une vraie
  détection de vivacité (qui demanderait un modèle dédié, ex. Silent-Face).
"""

from __future__ import annotations

import itertools
from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np

from core import config


def geometric_ratio(kps) -> float:
    kps = np.asarray(kps, dtype=np.float32)
    left_eye, right_eye, nose = kps[0], kps[1], kps[2]
    eye_dist = float(np.linalg.norm(left_eye - right_eye))
    if eye_dist == 0:
        return 0.0
    return float(np.linalg.norm(nose - (left_eye + right_eye) / 2.0)) / eye_dist


_track_ids = itertools.count(1)


@dataclass
class Track:
    centroid: tuple
    size: float
    history: deque
    ratios: deque
    id: int = field(default_factory=lambda: next(_track_ids))
    missed: int = 0
    live_confirmed: bool = False


class FaceTracker:
    def __init__(
        self,
        smoothing: int = config.SMOOTHING,
        liveness: bool = config.LIVENESS_ENABLED,
        liveness_var_threshold: float = config.LIVENESS_VAR_THRESHOLD,
        max_missed: int = 5,
        min_votes: int = 3,
    ):
        self.smoothing = max(1, smoothing)
        self.liveness = liveness
        self.var_threshold = liveness_var_threshold
        self.max_missed = max_missed
        self.min_votes = min_votes
        self.tracks: list[Track] = []

    def _match(self, detections):
        """Association visage <-> piste la plus proche (glouton sur toutes les paires).
        La distance maximale est relative à la taille du visage (et non 80 px fixes),
        donc indépendante de la résolution de la caméra."""
        pairs = []
        for di, det in enumerate(detections):
            x1, y1, x2, y2 = det["box"]
            c = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            size = max(x2 - x1, y2 - y1, 1)
            for ti, tr in enumerate(self.tracks):
                d = float(np.hypot(c[0] - tr.centroid[0], c[1] - tr.centroid[1]))
                if d < 0.6 * max(size, tr.size):
                    pairs.append((d, di, ti))
        pairs.sort()
        det_to_track, used_d, used_t = {}, set(), set()
        for _, di, ti in pairs:
            if di in used_d or ti in used_t:
                continue
            det_to_track[di] = ti
            used_d.add(di)
            used_t.add(ti)
        return det_to_track

    def update(self, detections: list[dict]) -> list[dict]:
        """detections : sortie de FaceEngine.analyze(). Retourne les mêmes dicts
        enrichis de l'identité lissée et du statut de vivacité."""
        det_to_track = self._match(detections)
        seen = set()
        results = []

        for di, det in enumerate(detections):
            x1, y1, x2, y2 = det["box"]
            centroid = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            size = float(max(x2 - x1, y2 - y1, 1))
            if di in det_to_track:
                track = self.tracks[det_to_track[di]]
            else:
                track = Track(centroid, size, deque(maxlen=self.smoothing), deque(maxlen=self.smoothing))
                self.tracks.append(track)
            track.centroid, track.size, track.missed = centroid, size, 0
            track.history.append((det["identity"], det["similarity"]))
            track.ratios.append(geometric_ratio(det["kps"]))
            seen.add(track.id)

            votes = Counter(n for n, _ in track.history)
            identity, n_votes = votes.most_common(1)[0]
            sims = [s for n, s in track.history if n == identity]
            similarity = float(np.mean(sims))

            # --- Vivacité ---
            if not self.liveness:
                liveness, is_real = "Désactivé", True
            elif track.live_confirmed:
                liveness, is_real = "VIVANT", True
            elif len(track.ratios) < self.smoothing:
                liveness, is_real = "Analyse...", False
            elif float(np.var(track.ratios)) >= self.var_threshold:
                # Une fois le mouvement 3D observé, la piste reste "vivante" :
                # une personne réelle qui s'immobilise ne bascule plus en FRAUDE.
                track.live_confirmed = True
                liveness, is_real = "VIVANT", True
            else:
                liveness, is_real = "SPOOF (Photo)", False

            if liveness == "SPOOF (Photo)":
                identity = config.SPOOF_LABEL

            known = identity not in (config.UNKNOWN_LABEL, config.SPOOF_LABEL)
            results.append({
                **det,
                "raw_identity": det["identity"],
                "identity": identity,
                "similarity": round(similarity, 4),
                "liveness": liveness,
                "is_real": is_real,
                "track_id": track.id,
                # 0 -> 1 pendant la collecte des images nécessaires à l'anti-spoofing
                "liveness_progress": 1.0 if (is_real or liveness == "SPOOF (Photo)")
                else round(len(track.ratios) / self.smoothing, 2),
                # "confirmed" = on peut enregistrer un pointage : identité connue,
                # stable sur plusieurs frames ET vivacité validée.
                "confirmed": bool(known and is_real and n_votes >= self.min_votes),
            })

        # Les pistes non vues sont gardées quelques frames (une détection ratée
        # ne remet plus l'historique à zéro), puis supprimées.
        for tr in self.tracks:
            if tr.id not in seen:
                tr.missed += 1
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        return results
