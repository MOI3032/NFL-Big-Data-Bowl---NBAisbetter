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
# 2 workers, 4 threads : assez pour un usage de démo ; chaque worker charge les données au boot.
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "2", "--threads", "4", "--timeout", "120", "app:server"]
