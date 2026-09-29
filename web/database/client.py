"""Client Supabase (optionnel).

Avant : une ValueError était levée à l'import si le .env manquait -> toute
l'application refusait de démarrer, même la reconnaissance faciale qui n'a
pas besoin de la base. Désormais Supabase est facultatif : sans identifiants,
l'application tourne et les pointages sont conservés dans docs/recognition_log.csv.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Le .env est cherché à côté de ce projet web, quel que soit le dossier de lancement.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = None
# Raison lisible si Supabase n'est pas utilisé (affichée dans le tableau de bord et /health)
STATUS = "ok"
if SUPABASE_URL and SUPABASE_KEY:
    try:
        from supabase import create_client
        from supabase.lib.client_options import SyncClientOptions

        # Timeouts courts : sans eux, le dashboard attend jusqu'à 120 s si Supabase ne répond pas.
        supabase = create_client(
            SUPABASE_URL,
            SUPABASE_KEY,
            options=SyncClientOptions(postgrest_client_timeout=5, storage_client_timeout=10),
        )
    except ImportError:
        STATUS = (f"paquet 'supabase' non installé pour ce Python ({sys.executable}). "
                  f"Installez-le : {sys.executable} -m pip install supabase")
    except Exception as exc:  # URL ou clé invalide...
        STATUS = f"connexion impossible : {exc}"
else:
    STATUS = "SUPABASE_URL / SUPABASE_KEY absents de web/.env"
if STATUS != "ok":
    print(f"[DB] Supabase désactivé : {STATUS}")
    print("[DB] Les pointages sont enregistrés dans docs/recognition_log.csv. "
          "Diagnostic complet : python scripts/diagnostic_supabase.py")
