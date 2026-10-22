# Mettre l'application en ligne (gratuit)

L'app est une app **Dash / Flask** (un serveur Python). Elle a besoin, au démarrage, des CSV de `outputs/`, des fichiers `games/plays/players/pffScoutingData.csv` et du cache `data/replay/frames.parquet` (29 Mo) qui alimente les replays. Le dossier `tracking/` brut (810 Mo) **n'est pas** nécessaire en ligne : il est remplacé par ce cache.

## Avant tout : construire le cache des replays

```powershell
.venv\Scripts\python.exe build_replay_cache.py
```

Ça crée `data/replay/frames.parquet` (~29 Mo). À refaire seulement si les données changent.

---

## Option recommandée : Render (depuis GitHub)

Gratuit, se branche sur ton dépôt GitHub, détecte le `Dockerfile` et le `render.yaml` tout seul.

1. Pousse le projet sur GitHub (voir « Préparer le dépôt » plus bas).
2. Va sur https://render.com, crée un compte (tu peux te connecter avec GitHub).
3. **New** → **Blueprint** → choisis ton dépôt `DataNFL`. Render lit `render.yaml` et crée le service web « pocket-integrity » en plan *Free*. (Sinon : **New → Web Service**, Render détecte le `Dockerfile`, rien à régler.)
4. Clique **Apply** / **Create**. Render construit l'image puis démarre `gunicorn`. L'URL publique (`https://pocket-integrity-xxxx.onrender.com`) s'affiche quand le build est fini (5-10 min la première fois).
5. À chaque `git push` suivant, Render redéploie automatiquement.

Limite du plan Free : l'app se met en veille après 15 min sans visite et met ~30 s à redémarrer à la visite suivante. La RAM est limitée à 512 Mo, d'où le réglage « 1 worker » dans le `Dockerfile`.

### Préparer le dépôt (une fois)

Le dépôt est déjà prêt : le tracking brut est exclu, le cache `data/replay/frames.parquet` et le gros CSV passent par Git LFS. Il reste à pousser :

```powershell
git push origin master
```

> Render sait lire les fichiers Git LFS au build, donc le parquet des replays arrive bien dans l'image.

---

## Option B : Hugging Face Spaces (Docker)

Gratuit, durable, sans mise en veille, et il gère les fichiers de données.

1. Crée un compte sur https://huggingface.co puis un **Space** : *New Space* → SDK **Docker** → *Blank*.
2. Active Git LFS pour les fichiers de données et pousse le projet :

```powershell
# dans le dossier DataNFL
git lfs install
git lfs track "*.parquet" "*.csv"
git add .gitattributes

# fichiers nécessaires à l'app en ligne (PAS le dossier tracking/)
git add Dockerfile requirements-app.txt .dockerignore app.py src assets README.md
git add outputs/play_summary.csv outputs/team_ranking.csv outputs/blocker_ranking.csv outputs/radius_sensitivity.csv outputs/validation_play_level.csv outputs/validation_team_level.csv outputs/pis_weight_sensitivity.csv outputs/subset_summary_PA_PR.csv outputs/exclusions.csv
git add data/replay/frames.parquet
git add "nfl-big-data-bowl-regional-event-data-main/data/games.csv" "nfl-big-data-bowl-regional-event-data-main/data/plays.csv" "nfl-big-data-bowl-regional-event-data-main/data/players.csv" "nfl-big-data-bowl-regional-event-data-main/data/pffScoutingData.csv"

git commit -m "Déploiement de l'app Pocket Integrity"
git remote add space https://huggingface.co/spaces/<ton-compte>/<ton-space>
git push space main
```

3. Le Space construit l'image à partir du `Dockerfile` et lance `gunicorn` sur le port 7860. L'URL publique s'affiche en haut du Space au bout de quelques minutes.

> Le `README.md` doit commencer par l'en-tête YAML que Spaces attend. Ajoute ces lignes tout en haut du README **au moment de pousser vers le Space** (elles sont inutiles en local) :
>
> ```
> ---
> title: Pocket Integrity Score
> emoji: 🏈
> colorFrom: blue
> colorTo: green
> sdk: docker
> app_port: 7860
> ---
> ```

---

---

## Tester l'image Docker en local (facultatif)

Si Docker Desktop est installé :

```powershell
docker build -t pocket-integrity .
docker run -p 7860:7860 pocket-integrity
# puis ouvrir http://localhost:7860
```

---

## Notes

- **Mémoire** : chaque worker gunicorn charge les données au démarrage (~400 Mo). Le free tier de Render (512 Mo) tient avec 1 seul worker ; sur Spaces, les 2 workers du Dockerfile passent largement. Pour réduire : baisser `--workers` à 1 dans le `Dockerfile`.
- **Licence des données** : la rediffusion publique du tracking du Big Data Bowl doit rester autorisée. L'app ne publie que les positions filtrées nécessaires aux replays, pas les fichiers bruts complets.
- **Sécurité** : l'app n'a pas d'authentification. Un Space public est ouvert à tous ; mets le Space en *Private* si tu veux restreindre l'accès.
