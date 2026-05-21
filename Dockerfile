FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy dataset and training script, then train the model at build time
# so container startup is instant (no training delay on Cloud Run)
COPY mtcars.csv .
COPY scripts/ ./scripts/
RUN mkdir -p models && python scripts/train.py

# Copy application code
COPY app/ ./app/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=models/model.pkl \
    LOG_LEVEL=INFO

EXPOSE 8080

# python -m uvicorn is used instead of the uvicorn binary
# so it always works regardless of PATH
CMD ["python", "-m", "uvicorn", "app.main:app", \
     "--host", "0.0.0.0", "--port", "8080"]