"""Accès aux pointages : Supabase si configuré, sinon CSV local (docs/recognition_log.csv)."""

import csv
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core import config  # noqa: E402

from database import client as db_client  # noqa: E402
from database.client import supabase  # noqa: E402

_csv_lock = threading.Lock()
CSV_HEADER = ["timestamp", "user_name", "liveness_status", "confidence_score"]


class DatabaseUnavailable(RuntimeError):
    pass


def _append_csv(row: dict):
    path = config.RECOGNITION_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    with _csv_lock:
        if path.exists() and path.stat().st_size > 0:
            with open(path, encoding="utf-8") as f:
                header = f.readline().strip().split(",")
            if header != CSV_HEADER:
                # Ancien format (ex : "timestamp,nom,distance" de la version dlib) :
                # on l'archive au lieu de mélanger deux formats dans le même fichier.
                path.rename(path.with_name(path.stem + "_ancien_format.csv"))
        new_file = not path.exists() or path.stat().st_size == 0
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADER, extrasaction="ignore")
            if new_file:
                writer.writeheader()
            writer.writerow(row)


def log_attendance(user_name: str, liveness_status: str, confidence_score: float):
    """Enregistre un pointage. Toujours écrit dans le CSV local (trace hors-ligne),
    et dans Supabase si disponible."""
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "user_name": user_name,
        "liveness_status": liveness_status,
        "confidence_score": round(float(confidence_score), 4),
    }
    _append_csv(row)
    if supabase is None:
        return row
    data = {k: row[k] for k in ("user_name", "liveness_status", "confidence_score")}
    # Supabase génère l'id et le timestamp.
    try:
        result = supabase.table("attendance_logs").insert(data).execute().data
    except Exception as exc:
        db_client.STATUS = _explain(exc)
        raise
    db_client.STATUS = "ok"
    return result


def _explain(exc) -> str:
    msg = str(exc)
    if "42501" in msg or "row-level security" in msg:
        return ("écriture refusée par RLS : utilisez la clé service_role dans web/.env "
                "(voir docs/supabase_setup.sql)")
    if "does not exist" in msg or "PGRST205" in msg:
        return "table ou colonne absente : exécutez docs/supabase_setup.sql dans Supabase"
    return f"erreur Supabase : {msg[:200]}"


def supabase_status() -> str:
    return db_client.STATUS


def get_recent_logs(limit: int = 10):
    """Derniers pointages (du plus récent au plus ancien)."""
    if supabase is None:
        raise DatabaseUnavailable(db_client.STATUS)
    try:
        response = supabase.table("attendance_logs").select("*").order("timestamp", desc=True).limit(limit).execute()
    except Exception as exc:
        db_client.STATUS = _explain(exc)
        raise
    db_client.STATUS = "ok"
    return response.data


def get_recent_logs_local(limit: int = 10):
    """Lecture du CSV local (mode hors-ligne). Ignore les anciennes lignes au format dlib."""
    path = config.RECOGNITION_LOG
    if not path.exists():
        return []
    with _csv_lock, open(path, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("user_name")]
    rows = rows[-limit:][::-1]
    for i, r in enumerate(rows):
        r["id"] = len(rows) - i
        try:
            r["confidence_score"] = float(r.get("confidence_score") or 0)
        except ValueError:
            r["confidence_score"] = 0.0
    return rows
