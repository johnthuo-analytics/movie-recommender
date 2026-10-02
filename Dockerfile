FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY recommender/ recommender/
COPY api.py .
# The trained artifact is created by `python train.py` BEFORE building the image.
COPY artifacts/recommender.joblib artifacts/recommender.joblib

ENV MODEL_PATH=artifacts/recommender.joblib
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
