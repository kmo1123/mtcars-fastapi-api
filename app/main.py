"""
app/main.py - MTCars MPG Prediction API

Predicts miles per gallon (mpg) using a linear regression model
trained on the mtcars dataset.

Predictors: wt (weight), hp (horsepower), cyl (cylinders)

Run locally:
    uvicorn app.main:app --reload --port 8080

Swagger UI:
    http://localhost:8080/docs
"""

import logging
import os
import pickle
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.pkl"))

# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------

model: object           = None
predictors: list[str]   = []
model_meta: dict        = {}
model_ready: bool       = False


def load_model() -> None:
    """Load the trained model from MODEL_PATH into module-level state."""
    global model, predictors, model_meta, model_ready

    if not MODEL_PATH.exists():
        logger.error("Model file not found at %s", MODEL_PATH)
        logger.error("Run 'python scripts/train.py' to generate it.")
        return

    with open(MODEL_PATH, "rb") as f:
        payload = pickle.load(f)

    model       = payload["model"]
    predictors  = payload["predictors"]
    model_meta  = {k: v for k, v in payload.items() if k != "model"}
    model_ready = True

    logger.info(
        "Model loaded | predictors=%s | R²=%.4f",
        predictors,
        payload["r2"],
    )


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield
    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="MTCars MPG Prediction API",
    description=(
        "Predicts **miles per gallon (mpg)** using a linear regression "
        "model trained on the mtcars dataset.\n\n"
        "**Predictors:** `wt` (weight in 1000 lbs), "
        "`hp` (horsepower), `cyl` (cylinders)"
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class PredictionRequest(BaseModel):
    """Input features for mpg prediction."""

    wt: float = Field(
        ...,
        gt=0,
        description="Vehicle weight in 1000s of lbs (e.g. 3.215 = 3215 lbs)",
        examples=[3.215],
    )
    hp: float = Field(
        ...,
        gt=0,
        description="Gross horsepower",
        examples=[110],
    )
    cyl: int = Field(
        ...,
        ge=2,
        le=16,
        description="Number of cylinders (typically 4, 6, or 8)",
        examples=[6],
    )


class PredictionResponse(BaseModel):
    predicted_mpg: float        = Field(description="Predicted miles per gallon")
    predictors_used: list[str]  = Field(description="Features used by the model")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Ops"], summary="Liveness check")
def health():
    """Returns ok if the API process is running. No auth required."""
    return {"status": "ok"}


@app.get("/ready", tags=["Ops"], summary="Readiness check")
def ready():
    """Returns ok if the model is loaded. Returns 503 if not ready."""
    if not model_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Model is not loaded. "
                "Run 'python scripts/train.py' to generate models/model.pkl."
            ),
        )
    return {
        "status":     "ok",
        "predictors": predictors,
        "r2":         model_meta.get("r2", 0.0),
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Predict mpg",
)
def predict(request: PredictionRequest):
    """
    Predict miles per gallon for a vehicle.

    - **wt**: weight in 1000s of lbs
    - **hp**: gross horsepower
    - **cyl**: number of cylinders
    """
    if not model_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded.",
        )

    X = pd.DataFrame([{"wt": request.wt, "hp": request.hp, "cyl": request.cyl}])
    predicted = float(model.predict(X)[0])

    logger.info(
        "predict | wt=%.3f hp=%.0f cyl=%d -> mpg=%.2f",
        request.wt, request.hp, request.cyl, predicted,
    )

    return {
        "predicted_mpg":   round(predicted, 2),
        "predictors_used": predictors,
    }