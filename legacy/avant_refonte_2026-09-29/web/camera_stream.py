"""
Capture webcam en arrière-plan, partagée par tous les clients connectés.

Un seul thread lit la webcam du serveur en continu, exécute la
reconnaissance (détection + identification + lissage + journal), et
maintient la dernière frame annotée en mémoire (JPEG encodé). Les
requêtes /video_feed de chaque client se contentent de lire cette
frame partagée -> pas besoin d'ouvrir la webcam plusieurs fois.
"""

import csv
import os
import threading
import time
from collections import deque, Counter
from datetime import datetime

import cv2
import numpy as np


class CameraStream:
    def __init__(self, engine, process_every_n=3, smoothing=8, event_cooldown=8.0,
                 log_csv="../docs/recognition_log.csv", unknown_dir="../docs/unknown_faces"):
        self.engine = engine
        self.process_every_n = process_every_n
        self.smoothing = smoothing
        self.event_cooldown = event_cooldown
        self.log_csv = log_csv
        self.unknown_dir = unknown_dir

        self._lock = threading.Lock()
        self._latest_jpeg = None
        self._latest_status = []  # noms actuellement visibles à l'écran
        self._running = False
        self._thread = None
        self._tracks = []
        self._frame_index = 0
        self._last_display_data = []

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def get_latest_jpeg(self):
        with self._lock:
            return self._latest_jpeg

    def get_latest_status(self):
        with self._lock:
            return list(self._latest_status)

    def _run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[CAMERA] Impossible d'ouvrir la webcam du serveur.")
            self._running = False
            return

        MAX_CENTROID_DIST = 80

        while self._running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            if self._frame_index % self.process_every_n == 0:
                results = self.engine.detect_and_identify(frame)
                self._last_display_data = self._update_tracks(results, MAX_CENTROID_DIST)

            self._frame_index += 1

            annotated = self._draw_annotations(frame, self._last_display_data)

            ok, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                with self._lock:
                    self._latest_jpeg = buffer.tobytes()
                    self._latest_status = [d["name"] for d in self._last_display_data]

        cap.release()

    def _update_tracks(self, results, max_centroid_dist):
        matched_track_ids = set()
        display_data = []

        for r in results:
            x1, y1, x2, y2 = r["box"]
            centroid = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

            best_track_idx, best_dist = None, max_centroid_dist
            for idx, track in enumerate(self._tracks):
                if idx in matched_track_ids:
                    continue
                d = np.hypot(centroid[0] - track["centroid"][0], centroid[1] - track["centroid"][1])
                if d < best_dist:
                    best_dist, best_track_idx = d, idx

            if best_track_idx is None:
                self._tracks.append({
                    "centroid": centroid,
                    "history": deque(maxlen=max(self.smoothing, 1)),
                    "last_event_time": 0.0,
                    "last_event_name": None,
                })
                best_track_idx = len(self._tracks) - 1

            track = self._tracks[best_track_idx]
            track["centroid"] = centroid
            track["history"].append(r["name"])
            matched_track_ids.add(best_track_idx)

            smoothed_name = Counter(track["history"]).most_common(1)[0][0] if self.smoothing > 0 else r["name"]

            history_is_stable = (
                len(track["history"]) == track["history"].maxlen
                and all(h == smoothed_name for h in track["history"])
            )
            now = time.time()
            cooldown_elapsed = (now - track["last_event_time"]) >= self.event_cooldown
            name_changed = track["last_event_name"] != smoothed_name

            if history_is_stable and (cooldown_elapsed or name_changed):
                if smoothed_name != "Inconnu":
                    self._log_recognition(smoothed_name, r["similarity"])
                else:
                    self._save_unknown_face_placeholder(r)
                track["last_event_time"] = now
                track["last_event_name"] = smoothed_name

            display_data.append({
                "box": r["box"], "name": smoothed_name, "similarity": r["similarity"], "kps": r["kps"],
                "_capture_unknown": r.get("_capture_unknown", False),
            })

        self._tracks = [t for i, t in enumerate(self._tracks) if i in matched_track_ids]
        return display_data

    def _log_recognition(self, name, similarity):
        file_exists = os.path.isfile(self.log_csv)
        os.makedirs(os.path.dirname(self.log_csv), exist_ok=True)
        with open(self.log_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "nom", "similarite_cosinus"])
            writer.writerow([datetime.now().isoformat(timespec="seconds"), name, f"{similarity:.3f}"])

    def _save_unknown_face_placeholder(self, result):
        # Le recadrage réel se fait dans _draw_annotations car on n'a la frame
        # complète qu'à cet endroit ; ici on note juste l'intention via un flag.
        result["_capture_unknown"] = True

    def _draw_annotations(self, frame, display_data):
        annotated = frame.copy()
        for d in display_data:
            x1, y1, x2, y2 = d["box"]
            name, sim = d["name"], d["similarity"]
            color = (196, 212, 45) if name != "Inconnu" else (68, 68, 239)  # BGR : cyan-ish / rouge

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{name} ({sim:.2f})"
            cv2.rectangle(annotated, (x1, y2 - 24), (x2, y2), color, cv2.FILLED)
            cv2.putText(annotated, label, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            if d.get("kps") is not None:
                for pt in d["kps"].astype(int):
                    cv2.circle(annotated, tuple(pt), 2, (255, 255, 0), -1)

            if d.pop("_capture_unknown", False):
                os.makedirs(self.unknown_dir, exist_ok=True)
                margin = 20
                h, w = frame.shape[:2]
                cx1, cy1 = max(0, x1 - margin), max(0, y1 - margin)
                cx2, cy2 = min(w, x2 + margin), min(h, y2 + margin)
                crop = frame[cy1:cy2, cx1:cx2]
                if crop.size > 0:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    cv2.imwrite(os.path.join(self.unknown_dir, f"unknown_{timestamp}.jpg"), crop)

        return annotated
