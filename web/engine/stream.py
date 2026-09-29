"""
WebSocket temps réel : le navigateur envoie des images JPEG (data URL base64),
le serveur répond avec la liste des visages détectés et identifiés.

Corrections par rapport à la version précédente :
  - un FaceTracker PAR connexion (avant : état partagé entre tous les clients) ;
  - l'inférence tourne dans un thread (asyncio.to_thread) : elle ne bloque plus
    la boucle asyncio, donc le dashboard et les autres clients restent réactifs ;
  - un pointage n'est enregistré que si la vivacité est validée ET l'identité
    stable (avant, les visages en "Analyse..." étaient déjà pointés).
"""

import asyncio
import base64
import binascii
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from core.tracking import FaceTracker
from database.crud import log_attendance

COOLDOWN_SECONDS = 10.0
MAX_FRAME_BYTES = 2_000_000
_last_logged: dict[str, float] = {}
# L'écriture en base part dans un thread dédié : un Supabase lent ne ralentit pas la vidéo.
_db_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="db")


def decode_frame(data: str):
    """data URL 'data:image/jpeg;base64,...' -> image BGR (ou None si invalide)."""
    if "," not in data or len(data) > MAX_FRAME_BYTES * 4 // 3 + 100:
        return None
    try:
        raw = base64.b64decode(data.split(",", 1)[1], validate=True)
    except (binascii.Error, ValueError):
        return None
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


def _log_if_needed(result: dict) -> bool:
    """Enregistre un pointage si nécessaire. Retourne True si un pointage vient d'être créé
    (le scanner l'affiche alors dans son journal en direct)."""
    if not result["confirmed"]:
        return False
    name, now = result["identity"], time.time()
    if now - _last_logged.get(name, 0.0) < COOLDOWN_SECONDS:
        return False
    _last_logged[name] = now
    _db_pool.submit(_write_log, name, result["liveness"], result["similarity"])
    return True


def _write_log(name, liveness, similarity):
    try:
        log_attendance(user_name=name, liveness_status=liveness, confidence_score=similarity)
        print(f"[DB] Pointage enregistré : {name} ({similarity:.2f})")
    except Exception as exc:
        print(f"[DB] Supabase indisponible, pointage gardé en CSV local : {exc}")


async def websocket_endpoint(websocket: WebSocket, engine):
    await websocket.accept()
    tracker = FaceTracker()
    print("[WS] Client connecté.")

    def process(frame):
        t0 = time.perf_counter()
        results = tracker.update(engine.analyze(frame))
        for r in results:
            r["logged"] = _log_if_needed(r)
        return results, (time.perf_counter() - t0) * 1000

    try:
        while True:
            data = await websocket.receive_text()
            frame = decode_frame(data)
            if frame is None:
                await websocket.send_json({"faces": [], "error": "image invalide"})
                continue
            try:
                results, ms = await asyncio.to_thread(process, frame)
                await websocket.send_json({"faces": results, "processing_ms": round(ms, 1),
                                           "frame_size": [int(frame.shape[1]), int(frame.shape[0])]})
            except WebSocketDisconnect:
                raise
            except Exception as exc:
                # Une frame qui plante ne ferme pas la connexion : le flux continue.
                print(f"[WS] Erreur sur une frame (ignorée) : {exc}")
                await websocket.send_json({"faces": [], "error": "erreur de traitement"})
    except WebSocketDisconnect:
        print("[WS] Client déconnecté.")
