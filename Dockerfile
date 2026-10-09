FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY recommender/ recommender/
COPY api.py .
COPY static/ static/
# The trained artifact is created by `python train.py` BEFORE building the image.
COPY artifacts/recommender.joblib artifacts/recommender.joblib

ENV MODEL_PATH=artifacts/recommender.joblib
ENV PORT=8000
EXPOSE 8000
# Hosts such as Render/Railway/Fly inject $PORT; locally it defaults to 8000.
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]
