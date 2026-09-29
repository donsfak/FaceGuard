-- =====================================================================
-- FaceGuard — configuration sécurisée de la table des pointages
-- À exécuter dans Supabase : SQL Editor -> New query -> coller -> Run
-- =====================================================================

-- 1. Table (créée seulement si elle n'existe pas déjà)
create table if not exists public.attendance_logs (
    id               bigint generated always as identity primary key,
    "timestamp"      timestamptz not null default now(),
    user_name        text        not null,
    liveness_status  text,
    confidence_score real
);

-- Si ta table existait déjà avec une colonne "created_at" au lieu de "timestamp",
-- décommente la ligne suivante (le code FaceGuard trie par "timestamp") :
-- alter table public.attendance_logs rename column created_at to "timestamp";

create index if not exists attendance_logs_timestamp_idx
    on public.attendance_logs ("timestamp" desc);

-- 2. Row Level Security : corrige l'erreur "RLS Disabled in Public"
--    Sans RLS, n'importe qui possédant la clé publique "anon" (visible dans
--    n'importe quelle appli cliente) peut lire, modifier ou effacer les pointages.
alter table public.attendance_logs enable row level security;

-- 3. Aucune policy pour les rôles "anon" et "authenticated" :
--    la table n'est plus accessible depuis Internet avec la clé publique.
--    Le serveur FaceGuard (Python) utilise la clé "service_role", qui contourne
--    RLS. Elle doit rester SECRÈTE : uniquement dans web/.env, jamais dans le
--    navigateur ni sur GitHub.
--
--    Où la trouver : Project Settings -> API Keys -> "service_role"
--    (ou une "Secret key" sb_secret_... sur les projets récents),
--    puis dans web/.env :   SUPABASE_KEY=<clé service_role>

-- 4. Vérification : doit afficher "true" dans la colonne relrowsecurity
select relname, relrowsecurity from pg_class where relname = 'attendance_logs';
