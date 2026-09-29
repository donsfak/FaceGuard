"""
Configuration centrale du projet.

Tous les chemins sont calculés à partir de l'emplacement de ce fichier :
le projet fonctionne donc quel que soit le dossier depuis lequel on lance
les scripts (avant, `../models/...` ne marchait que depuis src/ ou web/).

Chaque paramètre peut être surchargé par une variable d'environnement
(pratique pour Docker ou pour tester un autre seuil sans toucher au code).
"""

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = Path(os.getenv("FACE_DATASET_DIR", ROOT_DIR / "dataset"))
TEST_DATASET_DIR = Path(os.getenv("FACE_TEST_DATASET_DIR", ROOT_DIR / "dataset_test_externe"))
MODELS_DIR = Path(os.getenv("FACE_MODELS_DIR", ROOT_DIR / "models"))
ENCODINGS_PATH = Path(os.getenv("FACE_ENCODINGS_PATH", MODELS_DIR / "encodings_arcface.npz"))
DOCS_DIR = Path(os.getenv("FACE_DOCS_DIR", ROOT_DIR / "docs"))
UNKNOWN_DIR = DOCS_DIR / "unknown_faces"
RECOGNITION_LOG = DOCS_DIR / "recognition_log.csv"

# --- Modèle ---------------------------------------------------------------
MODEL_NAME = os.getenv("FACE_MODEL_NAME", "buffalo_l")
# 640 = détecte les petits visages (loin de la caméra) ; 320 = ~3x plus rapide sur CPU.
DET_SIZE = int(os.getenv("FACE_DET_SIZE", "640"))
USE_GPU = os.getenv("FACE_USE_GPU", "0") == "1"

# --- Identification -------------------------------------------------------
# Similarité cosinus minimale pour accepter une identité (plus haut = plus strict).
# Valeur à justifier avec src/04_evaluate.py (courbe FAR/FRR).
THRESHOLD = float(os.getenv("FACE_THRESHOLD", "0.45"))
KNN_K = int(os.getenv("FACE_KNN_K", "3"))

# --- Suivi temporel / anti-spoofing ---------------------------------------
SMOOTHING = int(os.getenv("FACE_SMOOTHING", "15"))
LIVENESS_ENABLED = os.getenv("FACE_LIVENESS", "1") == "1"
LIVENESS_VAR_THRESHOLD = float(os.getenv("FACE_LIVENESS_VAR", "0.00007"))

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

UNKNOWN_LABEL = "Inconnu"
SPOOF_LABEL = "FRAUDE DETECTEE"
