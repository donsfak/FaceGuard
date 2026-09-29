"""
Diagnostic Supabase — explique pourquoi les pointages ne partent pas dans Supabase.

Usage (depuis la racine du projet, avec le MÊME Python que celui qui lance le serveur) :
    python scripts/diagnostic_supabase.py

Vérifie dans l'ordre : paquet installé, fichier .env, type de clé, connexion,
lecture de la table, écriture (une ligne de test, supprimée ensuite).
"""

import base64
import json
import os
import sys
from pathlib import Path

ENV_PATH = Path(__file__).resolve().parent.parent / "web" / ".env"
OK, KO, WARN = "  [OK] ", "  [ERREUR] ", "  [ATTENTION] "


def fail(msg, fix):
    print(KO + msg)
    print("        -> " + fix)
    sys.exit(1)


print(f"Python utilisé : {sys.executable} ({sys.version.split()[0]})\n")

# 1. Paquets
try:
    from dotenv import load_dotenv
    from supabase import create_client
    print(OK + "paquets 'supabase' et 'python-dotenv' installés")
except ImportError as exc:
    fail(f"paquet manquant ({exc.name}) dans CE Python",
         f"{sys.executable} -m pip install supabase python-dotenv\n"
         "           (le serveur doit être lancé avec ce même Python / ce même venv)")

# 2. .env
if not ENV_PATH.exists():
    fail(f"{ENV_PATH} introuvable", "créez web/.env avec SUPABASE_URL=... et SUPABASE_KEY=...")
load_dotenv(ENV_PATH)
url, key = os.getenv("SUPABASE_URL", "").strip(), os.getenv("SUPABASE_KEY", "").strip()
if not url or not key:
    fail("SUPABASE_URL ou SUPABASE_KEY vide dans web/.env", "copiez-les depuis Project Settings -> API")
print(OK + f"web/.env lu ({url})")

# 3. Type de clé
role = None
if key.startswith("sb_secret_"):
    role = "service_role"
elif key.startswith("sb_publishable_"):
    role = "anon"
else:
    try:
        part = key.split(".")[1]
        role = json.loads(base64.urlsafe_b64decode(part + "=" * (-len(part) % 4))).get("role")
    except Exception:
        pass
if role == "service_role":
    print(OK + "clé service_role (secrète) : fonctionne avec RLS activé")
elif role == "anon":
    print(WARN + "clé ANON (publique). Si RLS est activé sans policy, l'écriture sera refusée.\n"
                 "        -> recommandé : RLS activé + clé service_role dans web/.env (voir docs/supabase_setup.sql)")
else:
    print(WARN + "type de clé non reconnu")

client = create_client(url, key)

# 4. Lecture
try:
    rows = client.table("attendance_logs").select("*").order("timestamp", desc=True).limit(3).execute().data
    print(OK + f"lecture de attendance_logs ({len(rows)} ligne(s) récentes)")
    if rows:
        print("        colonnes :", ", ".join(rows[0].keys()))
except Exception as exc:
    msg = str(exc)
    if "timestamp" in msg and "does not exist" in msg:
        fail("la colonne 'timestamp' n'existe pas", "voir la ligne 'rename column created_at' dans docs/supabase_setup.sql")
    if "relation" in msg and "does not exist" in msg or "PGRST205" in msg:
        fail("la table attendance_logs n'existe pas", "exécutez docs/supabase_setup.sql dans le SQL Editor de Supabase")
    fail(f"lecture impossible : {msg}", "vérifiez l'URL, la clé et votre connexion Internet")

# 5. Écriture (ligne de test supprimée ensuite)
try:
    inserted = client.table("attendance_logs").insert(
        {"user_name": "__test_diagnostic__", "liveness_status": "TEST", "confidence_score": 0}).execute().data
    client.table("attendance_logs").delete().eq("user_name", "__test_diagnostic__").execute()
    print(OK + "écriture de test réussie (ligne supprimée)")
except Exception as exc:
    msg = str(exc)
    if "42501" in msg or "row-level security" in msg:
        fail("écriture refusée par RLS", "mettez la clé service_role dans web/.env (voir docs/supabase_setup.sql)")
    fail(f"écriture impossible : {msg}", "vérifiez les colonnes (docs/supabase_setup.sql)")

print("\nTout est bon : relancez le serveur, les pointages iront dans Supabase.")
