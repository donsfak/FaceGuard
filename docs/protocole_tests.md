# Protocole de tests par condition

Le sujet (partie 9) demande de tester le système dans différentes conditions :
visage de face, légèrement tourné, éclairage différent, plusieurs personnes,
personne absente du dataset. Ce protocole produit ces chiffres en ~20 minutes.

## Règles pour que le test soit honnête

- Prendre les photos **un autre jour** que le dataset (sinon les images se ressemblent trop).
- **Garder toutes les photos**, même ratées : le mode `--raw` n'applique aucun filtre.
  Supprimer les photos difficiles fausserait le résultat.
- Ne pas mettre ces photos dans `dataset/` : elles servent uniquement à tester.
- Chaque personne photographiée doit avoir donné son accord.

## 1. Photos à prendre

Remplacer `marie`, `jean`, `paul` par les noms **exactement** tels qu'ils apparaissent
dans `dataset/`. Prévoir **5 photos par personne et par condition**, pour au moins
3 personnes enregistrées.

| Condition | Consigne | Commande (une par personne) |
|---|---|---|
| `face` | De face, bon éclairage, à ~1 m | `python src/01_capture_dataset.py --name marie --count 5 --dataset_dir conditions/face --raw` |
| `tourne` | Tête tournée de 30 à 45° (gauche puis droite) | `python src/01_capture_dataset.py --name marie --count 5 --dataset_dir conditions/tourne --raw` |
| `sombre` | Lumière de la pièce éteinte, seulement l'écran ou une fenêtre | `python src/01_capture_dataset.py --name marie --count 5 --dataset_dir conditions/sombre --raw` |
| `contre_jour` | Dos à une fenêtre lumineuse | `python src/01_capture_dataset.py --name marie --count 5 --dataset_dir conditions/contre_jour --raw` |
| `loin` | À 2-3 m de la caméra | `python src/01_capture_dataset.py --name marie --count 5 --dataset_dir conditions/loin --raw` |
| `inconnu` | 2 ou 3 personnes **absentes** du dataset, de face | `python src/01_capture_dataset.py --name paul --count 5 --dataset_dir conditions/inconnu --raw` |
| `plusieurs` | 2 ou 3 personnes ensemble devant la caméra | `python src/01_capture_dataset.py --name "marie+jean" --count 5 --dataset_dir conditions/plusieurs --group` |

Pour une photo de groupe avec quelqu'un qui n'est pas dans la base, écrire `inconnu`
à sa place : `--name "marie+inconnu"`. Le système doit reconnaître marie et répondre
« Inconnu » pour l'autre personne.

Arborescence obtenue :

```
conditions/
├── face/marie/marie_000.jpg …
├── tourne/marie/…
├── sombre/marie/…
├── inconnu/paul/…              (paul n'est PAS dans dataset/)
└── plusieurs/marie+jean_000.jpg …
```

## 2. Lancer l'évaluation

```bash
python src/04_evaluate.py
```

La section **5. Conditions de test** de `docs/evaluation_report.md` donne, pour chaque condition :
bien reconnus, mal reconnus (confondus avec quelqu'un d'autre), non reconnus (« Inconnu » à tort),
inconnus correctement rejetés, faux positifs, images sans visage détecté, et le temps par image.

## 3. Mesure du temps réel (FPS)

Le rapport mesure le temps de traitement d'une image. Pour les FPS en conditions réelles :

1. `python src/03_recognize_webcam.py` : les FPS s'affichent en haut à gauche.
   Noter la valeur avec 1, 2 puis 3 personnes devant la caméra.
2. Recommencer avec `--det-size 320` pour comparer vitesse et détection des visages éloignés.
3. Dans la plateforme web, le bas de la vidéo du scanner affiche `ms` et `img/s`.

| Configuration | 1 personne | 2 personnes | 3 personnes |
|---|---|---|---|
| det-size 640 | … FPS | … FPS | … FPS |
| det-size 320 | … FPS | … FPS | … FPS |

## 4. Tests de démonstration (à répéter avant l'oral)

Les 5 points obligatoires de la partie 7, à vérifier dans les conditions de la salle :

- [ ] Une personne connue est reconnue (nom au-dessus du visage)
- [ ] Une personne inconnue est affichée « Inconnu »
- [ ] Deux personnes simultanément devant la caméra
- [ ] Un cadre autour de chaque visage
- [ ] Le fonctionnement est fluide (noter les FPS)

Bonus anti-fraude : présenter la photo d'une personne connue sur un téléphone **immobile**.
Elle doit passer en « FRAUDE ». Expliquer ensuite la limite : une photo qu'on bouge peut passer.

Plan B si la salle est très sombre ou si l'anti-fraude gêne :
`python src/03_recognize_webcam.py --no-liveness`, ou `FACE_LIVENESS=0` pour la plateforme web.
