# Fiche de révision pour l'oral

Réponses courtes aux questions de la partie 8 du sujet, avec les chiffres de **notre**
système. Chaque réponse tient en 30 secondes à l'oral. Les chiffres viennent de
`docs/evaluation_report.md` : relancer `python src/04_evaluate.py` avant l'oral s'ils changent.

---

## Les 10 questions du sujet

### 1. Qu'est-ce que la Computer Vision ?
Le domaine de l'IA qui permet à une machine d'extraire de l'information d'images ou de
vidéos. Pour un ordinateur, une image n'est qu'un tableau de nombres (640 × 480 pixels × 3
couleurs pour notre webcam). La vision par ordinateur transforme ces nombres en informations
utiles : « il y a deux visages ici, celui-ci est delphine ».

### 2. Différence entre détection et reconnaissance ?
- **Détection** : *où* sont les visages ? → des cadres (SCRFD). Ne sait pas qui c'est.
- **Reconnaissance** : *qui* est-ce ? → on compare le visage détecté aux visages connus (ArcFace + comparaison).

La détection vient toujours en premier : on ne peut pas reconnaître un visage qu'on n'a pas trouvé.
Dans notre pipeline, c'est aussi la détection qui fournit 5 points clés (yeux, nez, coins de la
bouche) pour **aligner** le visage avant la reconnaissance.

### 3. Comment un modèle représente-t-il numériquement un visage ?
Par un **vecteur de nombres**. Le visage aligné (112 × 112 pixels) passe dans un réseau de
neurones convolutif (ResNet, modèle ArcFace) qui produit **512 nombres**. Le réseau a appris,
sur des millions de visages, à placer les photos d'une même personne proches les unes des
autres et celles de personnes différentes loin, quels que soient l'éclairage ou l'expression.

### 4. Qu'est-ce qu'un embedding facial ?
Ce vecteur de 512 nombres : une « empreinte » du visage dans un espace mathématique. On ne
stocke pas les photos dans la base de reconnaissance mais ces vecteurs
(`models/encodings_arcface.npz`). On le normalise pour que sa longueur vaille 1 : seule sa
**direction** compte.

### 5. Comment deux visages sont-ils comparés ?
On compare leurs embeddings par **similarité cosinus** : le produit scalaire de deux vecteurs
de longueur 1, c'est-à-dire le cosinus de l'angle entre eux. **FAISS** calcule cette similarité
entre le visage de la caméra et tous les embeddings de la base en une fraction de milliseconde
(0,02 ms mesurés). On garde les **3 plus proches voisins** et ils **votent** (k-NN, k = 3).

### 6. Distance ou similarité ?
Deux façons de mesurer la même chose :
- **Similarité cosinus** (notre choix) : 1 = identiques, 0 = sans rapport. **Plus c'est haut, plus c'est ressemblant.**
- **Distance euclidienne** (notre première version, avec dlib) : 0 = identiques. **Plus c'est bas, plus c'est ressemblant.**

Sur nos données : même personne → similarité moyenne **0,72** ; personnes différentes → **0,06**.

### 7. Pourquoi un seuil de reconnaissance ?
Parce que la base contient toujours *quelqu'un* de plus proche que les autres, même pour un
inconnu. Sans seuil, un inconnu serait toujours reconnu comme la personne la moins différente.
Le seuil (similarité ≥ **0,45**) répond à « est-ce assez ressemblant pour être la même personne ? ».

Pourquoi 0,45 ? Le graphique `similarity_distribution.png` le montre : deux personnes
différentes ne dépassent jamais **0,34** chez nous, alors que la même personne est presque
toujours au-dessus de 0,45. Le seuil se place dans cet écart.

### 8. Influence de la qualité du dataset ?
**C'est notre découverte principale.** 66 photos sur 100 contenaient plusieurs visages
(personnes en arrière-plan). Avec la règle « garder le plus grand visage », **11 embeddings
de la base appartenaient à la mauvaise personne**. L'un des vecteurs « kadi » était en réalité
le visage d'oyetola : si oyetola était absente de la base, elle était reconnue comme kadi
**20 fois sur 20**. Le modèle n'y était pour rien : le problème venait des données.

Corrections : l'encodage choisit dans chaque photo le visage cohérent avec les autres photos
de la personne ; la capture refuse les images avec plusieurs visages ; l'enrôlement vérifie la
qualité en direct et impose 5 poses différentes (face, gauche, droite, menton levé, sourire).

### 9. Faux positifs et faux négatifs ?
- **Faux positif** : un inconnu (ou quelqu'un d'autre) accepté sous l'identité d'une personne connue. **Le plus grave** pour un contrôle d'accès : quelqu'un entre à la place d'un autre.
- **Faux négatif** : une personne connue non reconnue (« Inconnu »). Gênant, mais elle peut réessayer.

Le seuil règle le compromis : plus haut, moins de faux positifs mais plus de faux négatifs.
Notre choix, au seuil 0,45 :
- **FAR 0 %** : faux positifs sur les paires de photos ;
- **FRR 1,34 %** : faux négatifs sur les paires ;
- **0 faux positif sur 109** tentatives de personnes absentes de la base.

### 10. Limites du système ?
- **Anti-spoofing basique** : détecte une photo immobile, pas une photo qu'on bouge ni une vidéo.
- **Petit dataset** : 5 personnes, une ou deux séances. Les 100 % sont encourageants mais pas statistiquement solides.
- **Conditions difficiles** : visage très tourné, contre-jour, très loin (< 40 px), masque.
- **Vitesse sur CPU** : quelques centaines de ms par image selon la machine.
- **Biais possibles** : un modèle pré-entraîné peut être moins précis sur des populations peu représentées dans ses données d'entraînement.
- **Données biométriques** : très sensibles (RGPD). Il faut le consentement, un stockage protégé et un droit de retrait.

---

## Questions probables du jury

**Pourquoi être passés de dlib (face_recognition) à InsightFace ?**
Nous avons mesuré dlib : le détecteur HOG était rapide mais ratait ~30 % des visages, le CNN
détectait ~97 % mais était très lent. InsightFace combine un détecteur plus robuste (SCRFD) et
des embeddings plus discriminants (512 dimensions entraînés avec la perte ArcFace, contre 128
pour dlib).

**Avez-vous entraîné un modèle ?**
Non, et c'est volontaire : on utilise un modèle **pré-entraîné** (apprentissage par transfert).
Entraîner un réseau de reconnaissance demande des millions de visages. Notre « apprentissage »
consiste à calculer les embeddings de nos personnes et à les stocker. Ajouter une personne ne
demande donc aucun ré-entraînement : 5 photos suffisent.

**Qu'est-ce que la perte ArcFace ?**
Une fonction de coût utilisée pendant l'entraînement du modèle. Elle impose une **marge
angulaire** entre les personnes : les embeddings d'une même personne sont rassemblés dans un
cône étroit, bien séparé des autres. D'où l'usage de la similarité cosinus (un angle).

**Pourquoi FAISS alors que vous n'avez que ~100 embeddings ?**
À cette échelle, numpy suffirait. FAISS montre comment passer à l'échelle : une entreprise
avec 10 000 employés et 20 photos chacun, soit 200 000 vecteurs, garde des recherches en
quelques millisecondes.

**À quoi sert le vote k-NN ?**
À ne pas dépendre d'une seule photo. Si un embedding de la base est mauvais (photo floue, mauvais
visage), il ne peut pas imposer seul son identité : il faut la majorité des 3 plus proches voisins
au-dessus du seuil.

**Et le lissage temporel ?**
Chaque visage est suivi d'une image à l'autre. Le nom affiché est le vote majoritaire des
15 dernières prédictions, ce qui évite qu'un nom clignote sur une image floue. Un pointage n'est
enregistré qu'une fois l'identité stable **et** la vivacité validée.

**Comment fonctionne votre anti-spoofing ?**
On mesure le rapport distance(nez, milieu des yeux) / distance entre les yeux. Sur un vrai visage,
les micro-mouvements 3D font varier ce rapport ; sur une photo plate tenue immobile, il reste
constant. C'est un prototype. Une vraie solution utiliserait un modèle dédié ou un défi actif
(« clignez des yeux »).

**Comment avez-vous évalué le système ?**
Quatre tests (`src/04_evaluate.py`) :
1. photos prises **un autre jour** : 12/12 reconnues ;
2. **leave-one-out** : chaque photo testée contre toutes les autres, 97/97 ;
3. chaque personne **retirée de la base** puis présentée : 109/109 rejetées ;
4. courbes **FAR/FRR** en fonction du seuil : EER 0,10 %.

Ces chiffres sont complétés par les tests par condition (`docs/protocole_tests.md`).

**Qu'est-ce que l'EER ?**
L'*Equal Error Rate* : le taux d'erreur au seuil où les faux positifs égalent les faux négatifs.
Un seul chiffre pour résumer la séparation des deux distributions : plus il est bas, mieux c'est.

**Pourquoi le leave-one-out est-il « optimiste » ?**
Les photos d'une même séance se ressemblent énormément : tester une photo contre sa quasi-jumelle
est trop facile. Le test externe, avec des photos prises un autre jour, est plus représentatif
d'une utilisation réelle.

**Que feriez-vous avec plus de temps ?**
Agrandir le dataset (plus de personnes, plusieurs séances, éclairages variés), utiliser un vrai
modèle anti-spoofing, un GPU pour la vitesse, une authentification sur la plateforme web, et un
test de biais par groupe de population.
