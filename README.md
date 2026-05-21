# MTCars MPG Prediction API

A FastAPI application that predicts **miles per gallon (mpg)** using a linear regression model trained on the classic [mtcars dataset](https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/mtcars.html).

## Model

| | |
|---|---|
| **Type** | Linear Regression (scikit-learn) |
| **Response** | `mpg` — miles per gallon |
| **Predictors** | `wt` (weight), `hp` (horsepower), `cyl` (cylinders) |
| **R²** | 0.815 |

Predictors were chosen based on correlation with fuel efficiency. Heavier, more powerful, larger-engine cars use more fuel.

## Repo Structure

```
.
├── app/
│   └── main.py          # FastAPI application
├── models/
│   └── model.pkl        # trained model artifact
├── scripts/
│   └── train.py         # training script
├── tests/
│   └── test_api.py      # automated tests
├── mtcars.csv           # dataset
├── Dockerfile
├── .dockerignore
├── requirements.txt
└── README.md
```

## Local Setup (no Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model (generates models/model.pkl)
python scripts/train.py

# Start the API
uvicorn app.main:app --reload --port 8080
```

Swagger UI: http://localhost:8080/docs

## Run with Podman

```bash
# Build
podman build -t mtcars-fastapi .

# Run
podman run --rm -p 8080:8080 mtcars-fastapi
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Returns ok if the API is running |
| `GET` | `/ready` | Returns ok if the model is loaded (503 if not) |
| `POST` | `/predict` | Predict mpg |

### POST /predict

**Request body:**

```json
{
  "wt":  2.62,
  "hp":  110,
  "cyl": 4
}
```

| Field | Type | Constraint | Description |
|-------|------|-----------|-------------|
| `wt` | float | > 0 | Weight in 1000s of lbs |
| `hp` | float | > 0 | Gross horsepower |
| `cyl` | int | 2–16 | Number of cylinders |

**Response:**

```json
{
  "predicted_mpg": 24.56,
  "predictors_used": ["wt", "hp", "cyl"]
}
```

## Example curl

```bash
curl -X POST "http://localhost:8080/predict" \
  -H "Content-Type: application/json" \
  -d '{"wt": 2.62, "hp": 110, "cyl": 4}'
```

## Run Tests

```bash
pytest tests/ -v
```

## Deploy to Cloud Run

```bash
export PROJECT_ID=your-project-id
export REGION=us-central1
export IMAGE=${REGION}-docker.pkg.dev/${PROJECT_ID}/ml-apis/mtcars-fastapi

# Enable required APIs (once)
gcloud services enable run.googleapis.com \
                       cloudbuild.googleapis.com \
                       artifactregistry.googleapis.com

# Create Artifact Registry repo (once)
gcloud artifacts repositories create ml-apis \
  --repository-format=docker \
  --location=${REGION}

# Build and push with Google Cloud Build
# (avoids QEMU issues when building on Apple Silicon)
gcloud builds submit --tag ${IMAGE} .

# Deploy
gcloud run deploy mtcars-fastapi \
  --image ${IMAGE} \
  --platform managed \
  --region ${REGION} \
  --allow-unauthenticated \
  --port 8080 \
  --memory 512Mi
```

**Deployed API URL:** `[https://mtcars-fastapi-<hash>-uc.a.run.app/docs](https://mtcars-fastapi-494286024315.us-central1.run.app/docs)`

## Rebuild the Model

```bash
python scripts/train.py
```

Loads `mtcars.csv`, fits a linear regression on `wt`, `hp`, and `cyl`, prints evaluation metrics, and saves the model to `models/model.pkl`.
