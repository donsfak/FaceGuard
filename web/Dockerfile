# Image de la plateforme web. À construire depuis la RACINE du projet :
#   docker build -t faceguard -f web/Dockerfile .
#   docker run -p 8001:8001 --env-file web/.env -v "$PWD/models:/app/models" -v "$PWD/dataset:/app/dataset" faceguard
FROM python:3.11-slim

# build-essential : compilation de l'extension C++ d'insightface
# libgl1 / libglib2.0-0 : dépendances système d'OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY web/requirements.txt /app/web/requirements.txt
RUN pip install --no-cache-dir -r /app/web/requirements.txt

# Pré-téléchargement des modèles buffalo_l dans l'image (sinon ~280 Mo au 1er démarrage)
RUN python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l', allowed_modules=['detection','recognition'], providers=['CPUExecutionProvider'])"

COPY core /app/core
COPY web /app/web
COPY models /app/models

WORKDIR /app/web
EXPOSE 8001
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
