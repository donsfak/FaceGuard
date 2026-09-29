"""
Plateforme web FaceGuard (FastAPI).

Lancement (depuis le dossier web/) :
    uvicorn app:app --host 127.0.0.1 --port 8001

Routes :
    GET  /                      tableau de bord des pointages
    GET  /scanner               scanner temps réel (webcam du navigateur)
    GET  /register              enrôlement d'une nouvelle personne (5 captures)
    GET  /people                personnes connues du système
    POST /register-user         API d'enrôlement (multipart : name + files[])
    POST /people/{name}/delete  retire une personne de la base
    GET  /api/logs              pointages + statistiques en JSON (tableau de bord en direct)
    POST /api/check-face        contrôle qualité d'une image (enrôlement guidé)
    GET  /people/{name}/photo   miniature du visage d'une personne
    GET  /health                état du moteur
    WS   /ws/detect             flux de reconnaissance
"""

import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import List

WEB_DIR = Path(__file__).resolve().parent
ROOT_DIR = WEB_DIR.parent
sys.path.insert(0, str(ROOT_DIR))  # accès au module partagé core/

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket  # noqa: E402
from fastapi.responses import HTMLResponse, RedirectResponse, Response  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from fastapi.templating import Jinja2Templates  # noqa: E402

from core import config  # noqa: E402
from core.engine import FaceEngine, choose_consistent_faces, largest_face, normalize_name  # noqa: E402
from database.crud import get_recent_logs, get_recent_logs_local, supabase_status  # noqa: E402
from engine.stream import websocket_endpoint  # noqa: E402

MAX_UPLOAD_FILES = 10
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MIN_VALID_CAPTURES = 3
# Similarité minimale d'une capture au profil moyen des autres captures.
SAME_PERSON_MIN_SIM = 0.35
# Au-dessus, le nouveau visage est déjà connu sous un autre nom.
DUPLICATE_SIM = 0.55


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Le moteur (≈ 300 Mo de modèles) est chargé une seule fois au démarrage.
    app.state.engine = FaceEngine()
    yield


app = FastAPI(title="FaceGuard — Contrôle d'accès IA", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")
templates = Jinja2Templates(directory=WEB_DIR / "templates")


def _is_suspicious(status) -> bool:
    status = (status or "").upper()
    return "SPOOF" in status or "FRAUDE" in status


def _load_logs(limit: int):
    """Supabase si disponible, sinon CSV local : le tableau de bord ne bloque jamais."""
    try:
        return get_recent_logs(limit=limit), "Supabase"
    except Exception:
        return get_recent_logs_local(limit=limit), "CSV local"


def _dashboard_payload(limit: int = 100):
    logs, source = _load_logs(limit)
    today = datetime.now(timezone.utc).date().isoformat()
    today_logs = [l for l in logs if str(l.get("timestamp", "")).startswith(today)]
    present = sorted({l.get("user_name") for l in today_logs
                      if l.get("user_name") and not _is_suspicious(l.get("liveness_status"))})
    registered = sorted(app.state.engine.people())
    return {
        "source": source,
        "supabase_status": supabase_status(),
        "logs": logs,
        "stats": {
            "today": len(today_logs),
            "present_today": present,
            "absent_today": [p for p in registered if p not in present],
            "registered": len(registered),
            "alerts": sum(_is_suspicious(l.get("liveness_status")) for l in logs),
        },
    }


# ------------------------------------------------------------------- pages
@app.get("/", response_class=HTMLResponse)
def read_dashboard(request: Request):
    # Les données initiales sont intégrées à la page (affichage immédiat),
    # puis rafraîchies en direct par static/js/dashboard.js via /api/logs.
    return templates.TemplateResponse(request, "dashboard.html", {"payload": _dashboard_payload()})


@app.get("/scanner", response_class=HTMLResponse)
def read_scanner(request: Request):
    return templates.TemplateResponse(request, "scanner.html", {})


@app.get("/register", response_class=HTMLResponse)
def read_register(request: Request):
    return templates.TemplateResponse(request, "register.html", {})


@app.get("/people", response_class=HTMLResponse)
def read_people(request: Request):
    logs, _ = _load_logs(300)
    last_seen = {}
    for l in logs:  # logs triés du plus récent au plus ancien
        n = l.get("user_name")
        if n and n not in last_seen and not _is_suspicious(l.get("liveness_status")):
            last_seen[n] = l.get("timestamp")
    people = [{"name": n, "count": c, "last_seen": last_seen.get(n),
               "has_photo": (config.DATASET_DIR / n).is_dir()}
              for n, c in app.state.engine.people().items()]
    return templates.TemplateResponse(request, "people.html", {"people": people})


# --------------------------------------------------------------------- API
@app.post("/register-user")
def register_user(name: str = Form(...), files: List[UploadFile] = File(...)):
    """Enrôlement multi-captures.

    `def` (et non `async def`) : FastAPI exécute la fonction dans un thread,
    l'extraction des embeddings ne bloque donc pas le serveur.
    """
    engine: FaceEngine = app.state.engine
    try:
        clean_name = engine.canonical_name(normalize_name(name))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if len(files) > MAX_UPLOAD_FILES:
        raise HTTPException(status_code=400, detail=f"Maximum {MAX_UPLOAD_FILES} photos.")

    faces_per_image, images = [], []
    for upload in files:
        content = upload.file.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=400, detail="Une photo dépasse 5 Mo.")
        frame = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        faces = engine.detect(frame)
        if faces:
            faces_per_image.append(faces)
            images.append(frame)

    # Dans chaque capture on garde le visage cohérent avec les autres captures
    # (une personne passant derrière n'est pas enrôlée par erreur).
    chosen, _ = choose_consistent_faces(faces_per_image, min_sim=SAME_PERSON_MIN_SIM)
    kept = [(e, img) for e, img in zip(chosen, images) if e is not None]
    embeddings = [e for e, _ in kept]
    images = [img for _, img in kept]

    if len(embeddings) < MIN_VALID_CAPTURES:
        raise HTTPException(
            status_code=400,
            detail=f"Visage exploitable sur {len(embeddings)} capture(s) seulement (minimum {MIN_VALID_CAPTURES}). "
                   "Une seule personne, face à la caméra et bien éclairée, puis recommencez.",
        )
    mean = np.mean(embeddings, axis=0)
    mean /= np.linalg.norm(mean)

    # Contrôle 2 : ce visage n'est pas déjà enregistré sous un autre nom.
    existing, sim = engine.identify(mean)
    if existing != config.UNKNOWN_LABEL and existing != clean_name and sim >= DUPLICATE_SIM:
        raise HTTPException(status_code=409, detail=f"Ce visage est déjà enregistré sous le nom « {existing} » "
                                                    f"(similarité {sim:.2f}).")

    # Les photos sont aussi rangées dans dataset/<nom>/ : un ré-encodage complet
    # (src/02_encode_faces.py) conservera donc les personnes enrôlées via le web.
    person_dir = config.DATASET_DIR / clean_name
    person_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    for i, img in enumerate(images):
        cv2.imwrite(str(person_dir / f"{clean_name}_web_{stamp}_{i}.jpg"), img)

    engine.add_person(clean_name, embeddings)  # sauvegarde .npz + mise à jour FAISS à chaud
    _thumb_cache.pop(clean_name, None)
    return {
        "status": "success",
        "name": clean_name,
        "message": f"{clean_name} enrôlé(e) avec succès ({len(embeddings)}/{len(files)} captures validées).",
    }


@app.post("/people/{name}/delete")
def delete_person(name: str, request: Request):
    engine: FaceEngine = app.state.engine
    _thumb_cache.pop(name, None)
    if engine.remove_person(name) == 0:
        raise HTTPException(status_code=404, detail="Personne inconnue.")
    # On archive le dossier photo plutôt que de le supprimer.
    src = config.DATASET_DIR / name
    if src.is_dir() and src.resolve().parent == config.DATASET_DIR.resolve():
        archive = config.ROOT_DIR / "dataset_archive"
        archive.mkdir(exist_ok=True)
        src.rename(archive / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}")
    if "application/json" in request.headers.get("accept", ""):
        return {"status": "success", "message": f"{name} a été retiré(e) de la base."}
    return RedirectResponse("/people", status_code=303)


@app.get("/api/logs")
def api_logs(limit: int = 100):
    return _dashboard_payload(max(1, min(limit, 500)))


@app.post("/api/check-face")
def check_face(file: UploadFile = File(...)):
    """Contrôle qualité d'une image avant capture (utilisé par l'enrôlement guidé).

    Retourne un statut simple + un message à afficher à l'utilisateur :
    none | multiple | too_small | too_dark | blurry | ok
    """
    engine: FaceEngine = app.state.engine
    frame = cv2.imdecode(np.frombuffer(file.file.read(MAX_UPLOAD_BYTES), np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Image illisible.")
    h, w = frame.shape[:2]
    faces = [f for f in engine.detect(frame) if f.det_score >= 0.5]

    def reply(status, message, face=None, **extra):
        box = [int(v) for v in face.bbox] if face is not None else None
        return {"status": status, "message": message, "faces": len(faces), "box": box,
                "frame_size": [w, h], **extra}

    if not faces:
        return reply("none", "Aucun visage détecté : placez votre visage dans le cadre.")
    if len(faces) > 1:
        return reply("multiple", "Plusieurs visages détectés : une seule personne devant la caméra.",
                     largest_face(faces))
    face = faces[0]
    x1, y1, x2, y2 = (int(v) for v in face.bbox)
    if (x2 - x1) < 0.16 * w:
        return reply("too_small", "Approchez-vous de la caméra.", face)
    crop = cv2.cvtColor(frame[max(0, y1):y2, max(0, x1):x2], cv2.COLOR_BGR2GRAY)
    if crop.size and crop.mean() < 55:
        return reply("too_dark", "Visage trop sombre : ajoutez de la lumière face à vous.", face)
    if crop.size and cv2.Laplacian(crop, cv2.CV_64F).var() < 25:
        return reply("blurry", "Image floue : restez immobile un instant.", face)
    name, sim = engine.identify(face.normed_embedding)
    known = None if name == config.UNKNOWN_LABEL else name
    return reply("ok", "Parfait, ne bougez plus.", face, known_as=known, similarity=round(sim, 3))


_thumb_cache: dict[str, bytes] = {}


@app.get("/people/{name}/photo")
def person_photo(name: str):
    """Miniature du visage d'une personne (première photo de dataset/<nom>/, recadrée)."""
    if name not in app.state.engine.people():
        raise HTTPException(status_code=404)
    if name not in _thumb_cache:
        folder = config.DATASET_DIR / name
        if folder.resolve().parent != config.DATASET_DIR.resolve() or not folder.is_dir():
            raise HTTPException(status_code=404)
        photos = sorted(p for p in folder.iterdir() if p.suffix.lower() in config.IMAGE_EXTENSIONS)
        # On choisit le visage le plus proche du profil moyen de la personne dans la base :
        # "le plus grand visage" montrait parfois quelqu'un d'autre (personne en arrière-plan).
        engine: FaceEngine = app.state.engine
        with engine._db_lock:
            vecs = engine.encodings[[i for i, n in enumerate(engine.names) if n == name]]
        centroid = vecs.mean(axis=0) / np.linalg.norm(vecs.mean(axis=0))
        best, best_score = None, -1.0
        for path in photos[:10]:
            img = cv2.imread(str(path))
            if img is None:
                continue
            for face in engine.detect(img):
                score = float(face.normed_embedding @ centroid)
                if score > best_score:
                    best, best_score = (img, face), score
            if best_score > 0.75:  # assez ressemblant : inutile de continuer
                break
        if best is None or best_score < 0.3:
            raise HTTPException(status_code=404)
        img, face = best
        x1, y1, x2, y2 = (int(v) for v in face.bbox)
        m = int(0.35 * (x2 - x1))
        crop = img[max(0, y1 - m):y2 + m, max(0, x1 - m):x2 + m]
        side = min(crop.shape[:2])
        cy, cx = crop.shape[0] // 2, crop.shape[1] // 2
        crop = cv2.resize(crop[cy - side // 2:cy + side // 2, cx - side // 2:cx + side // 2], (160, 160))
        _thumb_cache[name] = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
    return Response(_thumb_cache[name], media_type="image/jpeg", headers={"Cache-Control": "max-age=300"})


@app.get("/health")
def health():
    engine: FaceEngine = app.state.engine
    people = engine.people()
    return {"status": "ok", "people": len(people), "embeddings": sum(people.values()),
            "threshold": engine.threshold, "knn_k": engine.knn_k, "supabase": supabase_status()}


@app.websocket("/ws/detect")
async def detect_faces(websocket: WebSocket):
    await websocket_endpoint(websocket, websocket.app.state.engine)
