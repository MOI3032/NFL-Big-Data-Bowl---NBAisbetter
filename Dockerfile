# Image de déploiement pour l'app Dash (Hugging Face Spaces / Render / tout hôte Docker).
FROM python:3.12-slim

WORKDIR /app

# Dépendances d'exécution de l'app web (pas les outils de notebook ni kaleido).
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

# Code + données nécessaires à l'exécution (voir .dockerignore pour ce qui est exclu).
COPY . .

# Hugging Face Spaces expose le port 7860.
ENV PORT=7860
EXPOSE 7860

# gunicorn sert l'objet WSGI `server` défini dans app.py.
# 1 worker + 8 threads : un seul chargement des données en mémoire (~400 Mo), ce qui tient
# dans les 512 Mo du free tier de Render. Les threads gèrent plusieurs visiteurs à la fois.
# $PORT est fourni par l'hébergeur (Render) ; 7860 par défaut (Hugging Face / local Docker).
CMD gunicorn --bind 0.0.0.0:${PORT:-7860} --workers 1 --threads 8 --timeout 120 app:server
