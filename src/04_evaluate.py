"""
Étape 4 — Évaluation du système ArcFace (celui réellement utilisé en démo).

Les anciens scripts evaluate*.py mesuraient la version dlib (distance
euclidienne, seuil 0.6) : leurs chiffres ne décrivaient plus le système actuel.

Ce script mesure ce que demande le cahier des charges :
  1. Personnes connues   : bien reconnues / mal reconnues / non reconnues
                           - sur le dataset de test externe (autre séance photo)
                           - en "leave-one-out" sur le dataset principal
  2. Personnes inconnues : chaque personne est retirée de la base à tour de rôle
                           puis présentée au système -> doit répondre « Inconnu »
                           (mesure des faux positifs)
  3. Choix du seuil      : distributions genuine/impostor, courbes FAR/FRR, EER
  4. Performances        : temps de traitement par image et FPS estimés
  5. Conditions (option) : dossier conditions/<condition>/<personne>/*.jpg
                           ex. conditions/face, conditions/tourne, conditions/sombre,
                           conditions/inconnu/<personne hors base>/*.jpg ;
                           photos de groupe : conditions/plusieurs/delphine+kadi_001.jpg
                           (protocole détaillé : docs/protocole_tests.md)

Sorties : docs/evaluation_report.md, docs/evaluation_metrics.json,
          docs/similarity_distribution.png, docs/threshold_tradeoff.png

Usage :
    python src/04_evaluate.py
    python src/04_evaluate.py --threshold 0.5 --conditions_dir conditions
"""

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from core import config  # noqa: E402
from core.engine import FaceEngine, encode_dataset, largest_face, list_dataset_images  # noqa: E402

UNKNOWN = config.UNKNOWN_LABEL


def parse_args():
    p = argparse.ArgumentParser(description="Évaluation du système de reconnaissance (ArcFace)")
    p.add_argument("--dataset", type=Path, default=config.DATASET_DIR)
    p.add_argument("--test_dir", type=Path, default=config.TEST_DATASET_DIR)
    p.add_argument("--conditions_dir", type=Path, default=config.ROOT_DIR / "conditions")
    p.add_argument("--threshold", type=float, default=config.THRESHOLD)
    p.add_argument("--knn-k", type=int, default=config.KNN_K)
    p.add_argument("--det-size", type=int, default=config.DET_SIZE)
    p.add_argument("--output_dir", type=Path, default=config.DOCS_DIR)
    return p.parse_args()


# ----------------------------------------------------------------- helpers
def probe_images(engine, folder: Path):
    """Pour chaque photo de test : embedding du plus grand visage + temps de traitement.
    (En test, on ne connaît pas la « bonne » personne à l'avance : on fait comme le
    système en production, sans tricher avec l'étiquette.)"""
    probes, timings, no_face = [], [], []
    for person, path in list_dataset_images(folder):
        img = cv2.imread(str(path))
        if img is None:
            continue
        t0 = time.perf_counter()
        faces = engine.detect(img)
        timings.append((time.perf_counter() - t0) * 1000)
        face = largest_face(faces)
        if face is None:
            no_face.append(str(path))
            continue
        probes.append({"person": person, "path": str(path), "emb": face.normed_embedding.astype(np.float32),
                       "n_faces": len(faces)})
    return probes, timings, no_face


KNOWN_KEYS = ("correct", "mal_reconnu", "non_reconnu")
UNKNOWN_KEYS = ("inconnu_rejete", "faux_positif")


def zeros(*keys):
    return Counter({k: 0 for k in keys})


def classify(engine, probes, known_people):
    """Compte bien reconnus / mal reconnus / non reconnus (+ faux positifs pour les inconnus)."""
    out = zeros(*KNOWN_KEYS, *UNKNOWN_KEYS)
    confusion = defaultdict(Counter)
    errors = []
    for p in probes:
        pred, sim = engine.identify(p["emb"])
        truth = p["person"] if p["person"] in known_people else UNKNOWN
        confusion[truth][pred] += 1
        if truth == UNKNOWN:
            out["inconnu_rejete" if pred == UNKNOWN else "faux_positif"] += 1
        elif pred == truth:
            out["correct"] += 1
        elif pred == UNKNOWN:
            out["non_reconnu"] += 1
        else:
            out["mal_reconnu"] += 1
        if pred != truth:
            errors.append({"image": Path(p["path"]).name, "vrai": truth, "predit": pred, "similarite": round(sim, 3)})
    return out, confusion, errors


def pair_scores(emb, names):
    """Similarités cosinus de toutes les paires : même personne (genuine) vs différentes (impostor)."""
    sims = emb @ emb.T
    names = np.array(names)
    same = names[:, None] == names[None, :]
    iu = np.triu_indices(len(names), k=1)
    return sims[iu][same[iu]], sims[iu][~same[iu]]


def far_frr(genuine, impostor, thresholds):
    far = np.array([(impostor >= t).mean() for t in thresholds])
    frr = np.array([(genuine < t).mean() for t in thresholds])
    i = int(np.argmin(np.abs(far - frr)))
    return far, frr, float(thresholds[i]), float((far[i] + frr[i]) / 2)


def pct(a, b):
    return f"{100 * a / b:.1f} %" if b else "n/a"


# -------------------------------------------------------------------- main
def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    engine = FaceEngine(encodings_path=None, threshold=args.threshold, knn_k=args.knn_k, det_size=args.det_size)
    metrics = {"threshold": args.threshold, "knn_k": args.knn_k, "det_size": args.det_size}

    # ---- Base de référence (même traitement que 02_encode_faces.py)
    print("[1/5] Encodage du dataset de référence...")
    gal_emb, gal_names, rep = encode_dataset(engine, args.dataset, verbose=False)
    people = sorted(set(gal_names))
    engine.set_gallery(gal_emb, gal_names)
    metrics["dataset"] = {"personnes": len(people), "embeddings": len(gal_names),
                          "photos_ignorees": rep["skipped"], "photos_multi_visages": rep["multi_face_images"],
                          "par_personne": dict(Counter(gal_names))}
    print(f"      {len(gal_names)} embeddings, {len(people)} personnes, {rep['skipped']} photos ignorées")

    # ---- 1a. Test externe (connus)
    print("[2/5] Test externe (autre séance photo)...")
    ext_probes, timings, ext_noface = probe_images(engine, args.test_dir)
    ext, ext_conf, ext_err = classify(engine, ext_probes, set(people))
    n_ext = len(ext_probes)
    metrics["test_externe"] = {**ext, "total": n_ext, "sans_visage": len(ext_noface), "erreurs": ext_err,
                               "confusion": {k: dict(v) for k, v in ext_conf.items()}}

    # ---- 1b. Leave-one-out sur le dataset (connus)
    print("[3/5] Leave-one-out sur le dataset...")
    loo = zeros(*KNOWN_KEYS)
    loo_err = []
    for i in range(len(gal_names)):
        keep = np.arange(len(gal_names)) != i
        engine.set_gallery(gal_emb[keep], [n for j, n in enumerate(gal_names) if j != i])
        pred, sim = engine.identify(gal_emb[i])
        truth = gal_names[i]
        if pred == truth:
            loo["correct"] += 1
        elif pred == UNKNOWN:
            loo["non_reconnu"] += 1
        else:
            loo["mal_reconnu"] += 1
            loo_err.append({"vrai": truth, "predit": pred, "similarite": round(sim, 3)})
    metrics["leave_one_out"] = {**loo, "total": len(gal_names), "erreurs": loo_err}

    # ---- 2. Personnes inconnues : on retire chaque personne à tour de rôle
    print("[4/5] Personnes absentes de la base (open-set)...")
    unk = zeros(*UNKNOWN_KEYS)
    unk_err = []
    worst = []
    for person in people:
        keep = [j for j, n in enumerate(gal_names) if n != person]
        engine.set_gallery(gal_emb[keep], [gal_names[j] for j in keep])
        probes = [e for e, n in zip(gal_emb, gal_names) if n == person]
        probes += [p["emb"] for p in ext_probes if p["person"] == person]
        for e in probes:
            pred, sim = engine.identify(e)
            worst.append(sim)
            if pred == UNKNOWN:
                unk["inconnu_rejete"] += 1
            else:
                unk["faux_positif"] += 1
                unk_err.append({"personne_absente": person, "confondue_avec": pred, "similarite": round(sim, 3)})
    n_unk = sum(unk.values())
    metrics["inconnus"] = {**unk, "total": n_unk, "erreurs": unk_err,
                           "similarite_max_observee": round(float(max(worst, default=0)), 3)}
    engine.set_gallery(gal_emb, gal_names)

    # ---- Conditions optionnelles
    cond_rows, group_rows = [], []
    if args.conditions_dir.is_dir():
        for cond in sorted(p for p in args.conditions_dir.iterdir() if p.is_dir()):
            # a) Une personne par photo : conditions/<condition>/<personne>/*.jpg
            #    (une personne absente de la base doit être répondue « Inconnu »)
            probes, cond_times, noface = probe_images(engine, cond)
            if probes or noface:
                res, _, _ = classify(engine, probes, set(people))
                cond_rows.append((cond.name, res, len(probes) + len(noface), len(noface),
                                  float(np.mean(cond_times)) if cond_times else 0.0))
            # b) Plusieurs personnes par photo : conditions/<condition>/delphine+kadi_001.jpg
            #    (noms séparés par « + » ; une personne hors base s'écrit « inconnu »)
            groups = sorted(p for p in cond.iterdir() if p.is_file() and p.suffix.lower() in config.IMAGE_EXTENSIONS)
            if groups:
                ok_imgs, detail = 0, []
                for path in groups:
                    expected = sorted(n for n in path.stem.rsplit("_", 1)[0].split("+") if n.lower() != "inconnu")
                    n_expected_faces = len(path.stem.rsplit("_", 1)[0].split("+"))
                    img = cv2.imread(str(path))
                    faces = engine.detect(img) if img is not None else []
                    found = sorted(n for n, _ in (engine.identify(f.normed_embedding) for f in faces) if n != UNKNOWN)
                    good = found == expected and len(faces) >= n_expected_faces
                    ok_imgs += good
                    detail.append({"image": path.name, "attendu": expected, "reconnu": found,
                                   "visages": len(faces), "correct": good})
                group_rows.append((cond.name, ok_imgs, len(groups), detail))
        metrics["conditions"] = {c: {**r, "total": t, "sans_visage": nf, "ms_par_image": round(ms, 1)}
                                 for c, r, t, nf, ms in cond_rows}
        metrics["conditions_groupes"] = {c: {"images_correctes": ok, "total": t, "detail": d}
                                         for c, ok, t, d in group_rows}

    # ---- 3. Seuil : genuine / impostor
    print("[5/5] Courbes de seuil et performances...")
    gen, imp = pair_scores(gal_emb, gal_names)
    if ext_probes:
        pe = np.stack([p["emb"] for p in ext_probes])
        pn = np.array([p["person"] for p in ext_probes])
        cross = pe @ gal_emb.T
        same = pn[:, None] == np.array(gal_names)[None, :]
        gen_ext, imp_ext = cross[same], cross[~same]
    else:
        gen_ext = imp_ext = np.array([])
    thr = np.round(np.arange(0.10, 0.90, 0.01), 2)
    far, frr, eer_thr, eer = far_frr(gen, imp, thr)
    metrics["seuil"] = {"genuine_moyenne": round(float(gen.mean()), 3), "impostor_moyenne": round(float(imp.mean()), 3),
                        "impostor_max": round(float(imp.max()), 3), "genuine_min": round(float(gen.min()), 3),
                        "eer": round(eer, 4), "seuil_eer": eer_thr,
                        "far_au_seuil": round(float((imp >= args.threshold).mean()), 4),
                        "frr_au_seuil": round(float((gen < args.threshold).mean()), 4)}

    # Taux d'identification (vote k-NN) en fonction du seuil
    id_rate, fp_rate = [], []
    saved = engine.threshold
    for t in thr:
        engine.threshold = float(t)
        ok = sum(engine.identify(p["emb"])[0] == p["person"] for p in ext_probes)
        id_rate.append(ok / n_ext if n_ext else np.nan)
        fp = 0
        for person in people:
            keep = [j for j, n in enumerate(gal_names) if n != person]
            engine.set_gallery(gal_emb[keep], [gal_names[j] for j in keep])
            fp += sum(engine.identify(e)[0] != UNKNOWN for e, n in zip(gal_emb, gal_names) if n == person)
        fp_rate.append(fp / len(gal_names))
        engine.set_gallery(gal_emb, gal_names)
    engine.threshold = saved

    # ---- 4. Performances
    t_ms = np.array(timings) if timings else np.array([np.nan])
    search = []
    for p in ext_probes[:50] or [{"emb": gal_emb[0]}]:
        t0 = time.perf_counter()
        engine.identify(p["emb"])
        search.append((time.perf_counter() - t0) * 1000)
    metrics["performance"] = {"detection_embedding_ms_moyen": round(float(np.nanmean(t_ms)), 1),
                              "detection_embedding_ms_median": round(float(np.nanmedian(t_ms)), 1),
                              "recherche_faiss_ms": round(float(np.mean(search)), 3),
                              "fps_estime": round(1000 / float(np.nanmean(t_ms)), 1)}

    # ---- Graphiques
    plt.figure(figsize=(8, 4.5))
    bins = np.linspace(-0.3, 1.0, 66)
    plt.hist(imp, bins=bins, alpha=0.6, density=True, color="#ef4444", label="Personnes différentes (impostor)")
    plt.hist(gen, bins=bins, alpha=0.6, density=True, color="#22c55e", label="Même personne (genuine)")
    if len(gen_ext):
        plt.hist(gen_ext, bins=bins, histtype="step", lw=2, density=True, color="#15803d",
                 label="Même personne, test externe")
    plt.axvline(args.threshold, color="black", ls="--", label=f"Seuil = {args.threshold}")
    plt.xlabel("Similarité cosinus entre deux embeddings ArcFace")
    plt.ylabel("Densité")
    plt.title("Pourquoi un seuil ? Les deux distributions sont séparées")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(args.output_dir / "similarity_distribution.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(thr, far * 100, color="#ef4444", label="FAR : faux positifs (paires)")
    ax[0].plot(thr, frr * 100, color="#2563eb", label="FRR : faux négatifs (paires)")
    ax[0].axvline(args.threshold, color="black", ls="--", lw=1)
    ax[0].set_xlabel("Seuil")
    ax[0].set_ylabel("%")
    ax[0].set_title(f"Comparaison de paires (EER {eer * 100:.2f} % à {eer_thr})")
    ax[0].legend(fontsize=8)
    ax[1].plot(thr, np.array(id_rate) * 100, color="#16a34a", label="Connus bien reconnus (test externe)")
    ax[1].plot(thr, np.array(fp_rate) * 100, color="#ef4444", label="Inconnus acceptés à tort")
    ax[1].axvline(args.threshold, color="black", ls="--", lw=1)
    ax[1].set_xlabel("Seuil")
    ax[1].set_title("Identification complète (vote k-NN)")
    ax[1].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output_dir / "threshold_tradeoff.png", dpi=150)
    plt.close()

    # ---- Rapport
    lo = metrics["leave_one_out"]
    perf = metrics["performance"]
    s = metrics["seuil"]
    lines = [
        "# Rapport d'évaluation — reconnaissance faciale ArcFace",
        "",
        f"Seuil de similarité : **{args.threshold}** · vote k-NN k={args.knn_k} · det_size={args.det_size}",
        "",
        "## Dataset",
        f"- {len(people)} personnes, {len(gal_names)} embeddings retenus",
        f"- {rep['skipped']} photo(s) ignorée(s) (pas de visage, ou visage d'une autre personne)",
        f"- {rep['multi_face_images']} photo(s) contenant plusieurs visages",
        "",
        "| Personne | Embeddings |", "|---|---|",
        *[f"| {n} | {c} |" for n, c in sorted(Counter(gal_names).items())],
        "",
        "## 1. Personnes connues",
        "| Jeu de test | Bien reconnues | Mal reconnues | Non reconnues (« Inconnu ») | Total |",
        "|---|---|---|---|---|",
        f"| Test externe (autre séance) | {ext['correct']} ({pct(ext['correct'], n_ext)}) | {ext['mal_reconnu']} "
        f"| {ext['non_reconnu']} | {n_ext} |",
        f"| Leave-one-out (dataset) | {lo['correct']} ({pct(lo['correct'], lo['total'])}) | {lo['mal_reconnu']} "
        f"| {lo['non_reconnu']} | {lo['total']} |",
        "",
        "Le leave-one-out est optimiste : les photos d'une même séance se ressemblent beaucoup.",
        "Le test externe (photos prises un autre jour) est plus représentatif de la démo.",
        "",
        "## 2. Personnes absentes de la base",
        "Chaque personne est retirée de la base puis présentée au système.",
        f"- Correctement rejetées (« Inconnu ») : **{unk['inconnu_rejete']} / {n_unk}** ({pct(unk['inconnu_rejete'], n_unk)})",
        f"- Faux positifs (confondues avec quelqu'un) : **{unk['faux_positif']}**",
        f"- Similarité maximale observée pour un inconnu : {metrics['inconnus']['similarite_max_observee']}",
        "",
        "## 3. Choix du seuil",
        f"- Similarité moyenne même personne : {s['genuine_moyenne']} (min {s['genuine_min']})",
        f"- Similarité moyenne personnes différentes : {s['impostor_moyenne']} (max {s['impostor_max']})",
        f"- Au seuil {args.threshold} : FAR = {s['far_au_seuil'] * 100:.2f} % · FRR = {s['frr_au_seuil'] * 100:.2f} %",
        f"- Equal Error Rate : {s['eer'] * 100:.2f} % (seuil {s['seuil_eer']})",
        "",
        "![distribution](similarity_distribution.png)",
        "![seuil](threshold_tradeoff.png)",
        "",
        "## 4. Performances (CPU de la machine qui a lancé l'évaluation)",
        f"- Détection + embedding : {perf['detection_embedding_ms_moyen']} ms / image "
        f"(médiane {perf['detection_embedding_ms_median']} ms)",
        f"- Recherche FAISS + vote : {perf['recherche_faiss_ms']} ms",
        f"- FPS estimés (analyse de chaque image) : ~{perf['fps_estime']}",
    ]
    if cond_rows:
        lines += ["", "## 5. Conditions de test",
                  "| Condition | Photos | Connus bien reconnus | Mal reconnus | Non reconnus | Inconnus rejetés "
                  "| Faux positifs | Sans visage | ms/image |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for c, r, t, nf, ms in cond_rows:
            lines.append(f"| {c} | {t} | {r['correct']} | {r['mal_reconnu']} | {r['non_reconnu']} "
                         f"| {r['inconnu_rejete']} | {r['faux_positif']} | {nf} | {ms:.0f} |")
    if group_rows:
        lines += ["", "### Plusieurs personnes sur la même image",
                  "| Condition | Images où tout le monde est correctement identifié | Total |", "|---|---|---|"]
        lines += [f"| {c} | {ok} | {t} |" for c, ok, t, _ in group_rows]
        for c, _, _, detail in group_rows:
            lines += [f"- `{c}/{d['image']}` : attendu {d['attendu'] or '—'}, reconnu {d['reconnu'] or '—'} "
                      f"({d['visages']} visage(s)) {'✓' if d['correct'] else '✗'}" for d in detail]
    errs = ext_err + [{"image": "(leave-one-out)", **e} for e in loo_err]
    if errs or unk_err:
        lines += ["", "## Erreurs détaillées"]
        lines += [f"- {e}" for e in errs[:30]]
        lines += [f"- inconnu : {e}" for e in unk_err[:30]]
    if rep["skipped_details"]:
        lines += ["", "## Photos ignorées lors de l'encodage (à revoir)"]
        lines += [f"- `{Path(p).parent.name}/{Path(p).name}` : {why}" for p, why in rep["skipped_details"]]

    (args.output_dir / "evaluation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output_dir / "evaluation_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print("\n" + "\n".join(lines[:40]))
    print(f"\n[OK] Rapport : {args.output_dir / 'evaluation_report.md'}")


if __name__ == "__main__":
    main()
