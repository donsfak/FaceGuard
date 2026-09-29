"""
Étape 2 (Refactorisée) — Encodage du dataset avec InsightFace (ArcFace).

Ce script parcourt le dossier dataset/, détecte les visages avec SCRFD,
extrait les embeddings 512-D normalisés avec ArcFace, et les sauvegarde
dans un fichier pickle.

Modifications de niveau Production :
  - Utilisation d'InsightFace (format BGR natif, pas besoin de conversion RGB).
  - Gestion des visages multiples : sélectionne le plus grand visage (surface de la BBox)
    pour éviter d'encoder quelqu'un en arrière-plan.
  - Sauvegarde sous un nouveau nom (encodings_arcface.pickle) pour éviter
    les conflits de dimensionnalité avec l'ancienne base dlib.

Usage :
    python 02_encode_faces_insightface.py --gpu
"""

import argparse
import os
import pickle
import cv2
import numpy as np
from insightface.app import FaceAnalysis

def parse_args():
    parser = argparse.ArgumentParser(description="Encodage du dataset de visages avec InsightFace")
    parser.add_argument("--dataset", default="../dataset", 
                        help="Dossier contenant les dossiers par personne")
    parser.add_argument("--encodings", default="../models/encodings_arcface.npz", 
                    help="Chemin du fichier de sortie pour les embeddings (Format sécurisé NPZ)")
    parser.add_argument("--det-size", type=int, default=640, 
                        help="Taille d'entrée du détecteur. 640 assure de trouver même les petits visages.")
    parser.add_argument("--gpu", action="store_true", 
                        help="Activer l'accélération GPU (CUDA)")
    return parser.parse_args()

def get_largest_face(faces):
    """
    Retourne le visage ayant la plus grande Bounding Box (surface).
    Crucial pour éviter d'encoder un visage en arrière-plan dans le dataset.
    """
    if not faces:
        return None
    
    largest_face = None
    max_area = 0
    
    for face in faces:
        x1, y1, x2, y2 = face.bbox
        area = (x2 - x1) * (y2 - y1)
        if area > max_area:
            max_area = area
            largest_face = face
            
    return largest_face

def main():
    args = parse_args()

    print("[INFO] Initialisation d'InsightFace (Modèle buffalo_l)...")
    ctx_id = 0 if args.gpu else -1
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'] if args.gpu else ['CPUExecutionProvider'])
    app.prepare(ctx_id=ctx_id, det_size=(args.det_size, args.det_size))

    known_encodings = []
    known_names = []
    
    total_processed = 0
    total_skipped = 0

    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Le dossier dataset '{args.dataset}' est introuvable.")

    # Parcours des dossiers (chaque sous-dossier = une personne)
    person_folders = [f for f in os.listdir(args.dataset) if os.path.isdir(os.path.join(args.dataset, f))]
    
    print(f"[INFO] Traitement de {len(person_folders)} identités trouvées dans {args.dataset}")

    for person_name in sorted(person_folders):
        person_dir = os.path.join(args.dataset, person_name)
        images = [f for f in os.listdir(person_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        
        if not images:
            continue
            
        print(f"[INFO] Extraction pour '{person_name}' ({len(images)} images)...")
        
        for filename in images:
            image_path = os.path.join(person_dir, filename)
            
            # InsightFace et OpenCV utilisent tous les deux BGR.
            # Plus besoin du cv2.cvtColor(img, cv2.COLOR_BGR2RGB) !
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"  [ATTENTION] Impossible de lire {filename}. Fichier corrompu ?")
                total_skipped += 1
                continue

            faces = app.get(image)
            
            if len(faces) == 0:
                print(f"  [SKIP] Aucun visage détecté dans {filename}.")
                total_skipped += 1
                continue
                
            if len(faces) > 1:
                print(f"  [WARNING] {len(faces)} visages détectés dans {filename}. Conservation du plus grand.")
            
            # Application de la règle métier : on prend le visage principal
            best_face = get_largest_face(faces)
            
            known_encodings.append(best_face.normed_embedding)
            known_names.append(person_name)
            total_processed += 1

    # Sauvegarde des données
    print("[INFO] Sérialisation des encodages...")
    os.makedirs(os.path.dirname(args.encodings), exist_ok=True)
    
    # Sauvegarde des données en format compressé sécurisé
    print("[INFO] Sérialisation sécurisée des encodages (NPZ)...")
    os.makedirs(os.path.dirname(args.encodings), exist_ok=True)
    
    np.savez_compressed(args.encodings, encodings=known_encodings, names=known_names)

    print("\n[RÉSUMÉ FINAL]")
    print(f" - Visages encodés avec succès : {total_processed}")
    print(f" - Images ignorées (sans visage) : {total_skipped}")
    print(f" - Fichier généré : {args.encodings}")
    print("[INFO] Vous pouvez maintenant lancer 03_recognize_webcam_insightface.py !")

if __name__ == "__main__":
    main()


