"""
scripts/train.py - Train and save the MPG prediction model.

Trains a linear regression model on the mtcars dataset using:
    wt  - vehicle weight (1000 lbs)
    hp  - gross horsepower
    cyl - number of cylinders

These predictors were chosen because they have the strongest
correlation with fuel efficiency: heavier, more powerful, and
larger-engine cars consume more fuel.

Run from the project root:
    python scripts/train.py

Output:
    models/model.pkl
"""

import logging
import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH  = Path(__file__).parent.parent / "mtcars.csv"
MODEL_PATH = Path(__file__).parent.parent / "models" / "model.pkl"

PREDICTORS = ["wt", "hp", "cyl"]
RESPONSE   = "mpg"


def train() -> None:
    # Load
    df = pd.read_csv(DATA_PATH)
    logger.info("Loaded %d rows from %s", len(df), DATA_PATH)

    X = df[PREDICTORS]
    y = df[RESPONSE]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Fit
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    r2  = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    logger.info("Predictors : %s", PREDICTORS)
    logger.info("R²         : %.4f", r2)
    logger.info("MSE        : %.4f", mse)
    logger.info("Intercept  : %.4f", model.intercept_)
    for name, coef in zip(PREDICTORS, model.coef_):
        logger.info("  %-4s coef: %+.4f", name, coef)

    # Save
    MODEL_PATH.parent.mkdir(exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({
            "model":      model,
            "predictors": PREDICTORS,
            "response":   RESPONSE,
            "r2":         round(float(r2),  4),
            "mse":        round(float(mse), 4),
        }, f)

    logger.info("Saved to %s", MODEL_PATH)


if __name__ == "__main__":
    train()