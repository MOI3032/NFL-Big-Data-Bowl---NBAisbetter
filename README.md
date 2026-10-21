# Pocket Integrity Score (PIS)
### Mesurer la protection de passe action par action, à partir du tracking NFL

Le **Pocket Integrity Score** note de 0 à 100 la propreté de la poche du quarterback entre le snap et le lancer, sur chaque dropback, à partir des positions des 22 joueurs (10 images par seconde). Il est accompagné du **Blitz Time Pressure** : le temps avant que le premier rusher arrive à moins de 2 yards du QB. Les deux métriques sont calculées sur la saison NFL 2021 (semaines 1 à 8, 8 057 dropbacks, données NFL Big Data Bowl) et sont consultables dans une application web avec un replay animé de chaque action.

---

## Ce qui rend le PIS original

### 1. Une note pour chaque dropback, pas seulement pour les jeux ratés

Les statistiques classiques de protection sont binaires et rares : un sack arrive sur **6,4 %** des dropbacks, une pression notée par PFF sur **36 %**. Les autres actions comptent toutes comme « propres », qu'aucun rusher n'ait approché ou qu'un défenseur soit passé à un bras du QB.

Le PIS donne une note continue à **toutes** les actions. Ses médianes séparent nettement les trois situations :

| Résultat selon PFF | PIS médian |
|---|---|
| Poche propre | **67** |
| Pression sans sack | **42** |
| Sack | **1** |

### 2. Il mesure le film de l'action, pas un seul instant

Un sack ou une pression est un événement ponctuel. Le PIS suit la poche **image par image, du snap au lancer**. Il combine trois questions qu'un coach de ligne se pose devant la vidéo :

| Ingrédient | Question | Poids |
|---|---|---|
| **Coussin** | À quelle distance le rusher le plus proche est-il resté du QB, en moyenne ? | 45 % |
| **Temps propre** | Pendant quelle part de l'action aucun rusher n'était à moins de 2 yards ? | 40 % |
| **Aire de la poche** | Quelle surface les bloqueurs ont-ils gardée autour du QB ? (enveloppe convexe des bloqueurs) | 15 % |

L'aire a volontairement un petit poids : une grande surface peut aussi traduire une ligne étirée plutôt que solide.

### 3. Il révèle les pressions que personne ne compte

Parmi les 5 130 dropbacks que PFF juge propres, **618 (12 %)** ont un PIS inférieur à 40 : un rusher a frôlé le QB, mais le ballon est parti à temps. Ces poches serrées n'apparaissent dans aucune statistique, alors qu'elles annoncent les sacks à venir.

### 4. Il ne récompense pas un QB qui lance vite

Un QB qui lâche le ballon en 1,8 s subit naturellement moins de pression qu'un QB qui le garde 3,5 s. Sans correction, une ligne paraît bonne simplement parce que son QB joue vite. Le PIS brut est corrélé au temps de lancer (r = −0,35, soit environ −8,5 points par seconde).

Le **PIS ajusté** retire cet effet : c'est l'écart à ce qu'on attend pour ce temps de lancer (50 = normal). Sa corrélation avec le temps de lancer tombe à **0**. Le classement des équipes change de façon visible : PHI passe du 15e au 10e rang (QB qui garde le ballon 3,4 s), NYG recule du 11e au 16e (lancers rapides).

### 5. Un temps avant pression qui ne triche pas avec les données manquantes

Quand le ballon part avant toute pression, on ne sait pas quand la pression serait arrivée. Supprimer ces actions ne garderait que les poches qui cèdent. Leur donner une valeur inventée fausserait le résultat. Elles sont donc traitées comme des observations **censurées**, avec des courbes de survie de Kaplan-Meier codées à la main. Résultat : le temps médian avant pression passe de **3,6 s avec 4 rushers à 3,0 s avec un blitz à 6 ou plus**. Le nombre de défenseurs dans la boîte, visible avant le snap, prédit beaucoup moins bien la pression que le nombre réel de rushers.

### 6. Objectif, validé de façon indépendante et robuste

- **Objectif** : le score vient uniquement du tracking, sans notation vidéo. Il est donc reproductible et calculable sur n'importe quel match tracké.
- **Validé contre PFF sans avoir été entraîné dessus** : il repère une pression PFF avec une **AUC de 0,80** et un sack avec une **AUC de 0,95** (0,5 = hasard, 1 = parfait). Le temps de lancer seul ne fait que 0,74.
- **Il isole la ligne du QB** : au niveau des équipes, le PIS ajusté suit le taux de bloqueurs « battus » selon PFF (ρ = −0,44) bien plus que le taux de pression global (ρ = −0,13), qui inclut aussi les pressions causées par le QB ou par le schéma.
- **Robuste au choix des poids** : avec 5 autres pondérations, le classement des équipes reste quasi identique (ρ ≥ 0,96).
- **Lisible** : 50 = moyenne de la ligue, environ 84 = un écart-type au-dessus, et chaque note se décompose en trois ingrédients compréhensibles.

---

## Comment il est calculé

```
Fenêtre    : du snap au lancer ou au sack (événements ball_snap → pass_forward / qb_sack / qb_strip_sack)
Z          = 0,45 · z(coussin) + 0,40 · z(temps propre) + 0,15 · z(aire de la poche)
PIS        = 100 · Φ(Z / σ_Z)            Φ = loi normale, z = écart à la moyenne de la ligue en écarts-types
PIS ajusté = 50 + (PIS − PIS attendu pour ce temps de lancer, par régression linéaire)
```

- Les coordonnées sont retournées pour que l'attaque aille toujours de gauche à droite.
- Le coussin est écrêté aux percentiles 1 et 99 pour limiter l'effet des valeurs extrêmes.
- **Pression (tracking)** : un rusher entre dans le cercle de 2 yards autour du QB. Un rusher déjà dans ce cercle au snap (par exemple un nose tackle face à un QB sous le centre) n'est compté qu'après en être sorti.
- **Sensibilité** : les résultats sont aussi calculés avec des rayons de 1,5 et 3 yards, avec 5 jeux de poids différents, et avec ou sans play-action et pocket-roll.

---

## Lancer le projet

### Prérequis

- Windows, macOS ou Linux, avec **Python 3.12**.
- Les données du Big Data Bowl dans `nfl-big-data-bowl-regional-event-data-main/data/` : `games.csv`, `plays.csv`, `players.csv`, `pffScoutingData.csv` et le dossier `tracking/`.

### 1. Installer (une seule fois)

Avec [uv](https://docs.astral.sh/uv/) (recommandé, il lit `pyproject.toml` et `uv.lock`) :

```powershell
uv sync
```

Ou avec pip :

```powershell
py -3.12 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

Sur macOS ou Linux, remplace `.venv\Scripts\` par `.venv/bin/` dans toutes les commandes.

### 2. Calculer les métriques (une seule fois, environ 2 à 3 minutes)

```powershell
.venv\Scripts\python.exe main.py
```

Le script lit le tracking, applique les filtres, calcule le PIS et le temps avant pression, valide contre PFF, puis écrit les CSV et les figures dans `outputs/`. Le tracking nettoyé est mis en cache dans `data/processed/` (parquet), ce qui rend les lancements suivants plus rapides.

| Option | Effet |
|---|---|
| `--weeks 1 2` | ne traite que certaines semaines |
| `--force` | reconstruit le cache depuis les CSV bruts |
| `--no-figures` | ne régénère que les CSV |

### 3. Ouvrir l'application web

```powershell
.venv\Scripts\python.exe app.py
```

Attends le message `Ouvre http://127.0.0.1:8050`, puis ouvre **http://127.0.0.1:8050** dans ton navigateur. Laisse le terminal ouvert pendant l'utilisation et fais Ctrl+C pour arrêter. Si le port est déjà pris : `app.py --port 8051`.

> Usage local uniquement : l'application n'écoute que sur 127.0.0.1 et n'a pas d'authentification.

### 4. (Optionnel) Le notebook

Ouvre `analysis.ipynb` dans VS Code, Kiro ou Jupyter, choisis le kernel `.venv` et lance « Run All ». Il refait toute l'analyse de haut en bas et affiche chaque figure.

---

## L'application en un coup d'œil

| Onglet | À quoi il sert |
|---|---|
| **Guide** | le PIS en 30 secondes, ses ingrédients, l'échelle de lecture, le glossaire et les données |
| **Actions & replays** | filtrer et trier les 8 057 actions (semaine, match, équipe en attaque ou en défense, résultat, blitz, pression). Pour l'action choisie : diagnostic en une phrase, 9 fiches chiffrées, replay animé (poche des bloqueurs, cercle de 2 yards qui devient rouge) et courbe de pression seconde par seconde |
| **Survie de la poche** | courbes de Kaplan-Meier selon les rushers, la boîte, la couverture, la formation, le play-action ou le down, pour la ligue ou une équipe, avec un rayon de 1,5, 2 ou 3 yards |
| **Classement équipes** | PIS ajusté par équipe, comparaison avec la pression notée par PFF, et un clic sur une équipe ouvre ses actions |
| **Validation PFF** | AUC, boxplots, effet du temps de lancer et sensibilité des poids |
| **Bloqueurs** | part des duels où le défenseur bloqué atteint le QB, avec au moins 100 blocs par joueur |

Chaque page commence par un encadré « Comment lire… », et chaque en-tête de tableau affiche sa définition au survol.

**À qui ça sert** : coach de ligne offensive et coordinateur offensif (évaluer ses protections et revoir les actions où la poche cède), coordinateur défensif (combien de rushers envoyer et contre quelle formation), recruteur (classement des bloqueurs).

---

## Contenu du dépôt

```
main.py              pipeline complet
app.py               application web Dash (6 onglets)
assets/style.css     design de l'application
analysis.ipynb       notebook exécutable de haut en bas
src/
  config.py          chemins, noms d'événements et de rôles, poids du PIS
  data_loading.py    lecture des CSV (avec cache)
  preprocessing.py   filtres, fenêtre snap → lancer, normalisation, cache parquet, contrôles
  metrics.py         distances, enveloppe convexe, PIS, temps avant pression, Kaplan-Meier, bloqueurs
  validation.py      AUC, point-bisériale, Spearman, sensibilité des poids
  viz.py             figures Plotly en anglais ou en français
  pipeline.py        enchaînement des calculs
outputs/             CSV (play_summary, team_ranking, blocker_ranking, validation…) et figures HTML / PNG
data/processed/      cache parquet (ignoré par git)
```

---

## Limites et hypothèses

- **Peu de données** : 8 semaines seulement, soit environ 250 dropbacks par équipe. Les classements bougeront avec une saison complète.
- **Temps de lancer** : le PIS brut en dépend fortement. Pour comparer des équipes, il faut toujours utiliser le PIS ajusté.
- **Rayon de 2 yards** : c'est une convention. Les conclusions tiennent à 1,5 et à 3 yards.
- **Bloqueurs** : c'est un indicateur de duel. Sur un stunt ou un échange de blocs, le défenseur peut atteindre le QB par la faute d'un autre bloqueur.
- **Données réelles** : saison **2021** (Big Data Bowl 2023), un fichier de tracking par match, rôles PFF écrits `Pass Rush` / `Pass Block`, pas d'événement `pass_shovel` dans ces données.
- **Filtres** : 449 scrambles et 1 action sans rusher sont exclus, ainsi que 50 actions sans snap ou sans fin exploitable. Les 39 actions dont la fin est un `autoevent_passforward` et les 16 dont le temps de lancer sort de 0,5-8 s sont gardées et signalées.
