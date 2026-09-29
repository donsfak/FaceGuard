#!/usr/bin/env bash
# Range l'ancien code (versions dlib / Flask / doublons) dans legacy/ au lieu de le supprimer.
# À lancer UNE fois depuis la racine du projet :   bash scripts/ranger_ancien_code.sh
set -euo pipefail
cd "$(dirname "$0")/.."

DEST="legacy/avant_refonte_2026-09-29"
mkdir -p "$DEST"

FILES=(
  main.py                               # doublon de web/app.py, ne pouvait pas démarrer depuis la racine
  web/app.py7                           # ancienne copie de app.py
  web/recognition_engine.py             # moteur de la version Flask (remplacé par core/engine.py)
  web/camera_stream.py                  # flux MJPEG de la version Flask
  web/engine/recognition.py             # remplacé par core/engine.py + core/tracking.py
  web/templates/add_person.html         # template Flask (url_for incompatible avec FastAPI)
  web/templates/touch.html              # ancien formulaire d'enrôlement à 1 photo
  src/02_encode_faces_insightface.py    # fusionné dans src/02_encode_faces.py
  src/03_recognize_webcam_insightface.py
  src/03_recognize_webcam_faiss.py
  src/03_recognize_webcam_liveness.py   # les trois fusionnés dans src/03_recognize_webcam.py
  src/evaluate.py                       # évaluation dlib -> remplacée par src/04_evaluate.py
  src/evaluate_external.py
  src/benchmark_detectors.py            # benchmark HOG/CNN (dlib), utile pour l'historique
  src/diagnose_skipped.py
  src/Diagnose                          # doublon de diagnose_skipped.py
  models/encodings.pickle               # base dlib 128-D
  models/encodings_arcface.pickle       # ancienne base ArcFace au format pickle (non sûr)
  docs/accuracy_vs_threshold.png        # graphiques de la version dlib
  docs/distance_distribution.png
)

in_git=false
git rev-parse --is-inside-work-tree >/dev/null 2>&1 && in_git=true

for f in "${FILES[@]}"; do
  [ -e "$f" ] || continue
  mkdir -p "$DEST/$(dirname "$f")"
  if $in_git && git ls-files --error-unmatch "$f" >/dev/null 2>&1; then
    git mv "$f" "$DEST/$f"
  else
    mv "$f" "$DEST/$f"
  fi
  echo "rangé : $f -> $DEST/$f"
done

# Caches Python (régénérés automatiquement)
find . -name "__pycache__" -type d -not -path "*/venv/*" -prune -exec rm -rf {} + 2>/dev/null || true
echo "Terminé. Anciennes versions dans $DEST/"
