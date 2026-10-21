---
title: Pocket Integrity Score
emoji: 🏈
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

- **Pocket Integrity Score (PIS)** : note de 0 à 100 de la propreté de la poche du QB du snap au lancer, calculée image par image sur le tracking NFL 2021 (8 057 dropbacks).
- **Originalité** : il note chaque action au lieu de compter les seuls sacks, se corrige du temps de lancer et colle aux notes PFF sans être entraîné dessus (AUC 0,80 pour les pressions, 0,95 pour les sacks).
- **Lancer** : `uv sync`, puis `.venv\Scripts\python.exe main.py`, puis `.venv\Scripts\python.exe app.py` et ouvrir http://127.0.0.1:8050.
