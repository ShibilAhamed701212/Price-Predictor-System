"""Predict a house price with the trained scikit-learn pipeline.

By default the latest ``sklearn_pipeline`` artifact produced by ``run_pipeline.py`` is
loaded from the active ZenML store. Pass ``--model-path`` (or set ``MODEL_PATH``) to load
a pickled pipeline file instead.
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd


def load_pipeline(model_path=None):
    if model_path:
        # Only load pickle files you created yourself: unpickling can run arbitrary code.
        with open(model_path, "rb") as f:
            return pickle.load(f)

    from zenml.client import Client

    return Client().get_artifact_version("sklearn_pipeline").load()


parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument(
    "--model-path",
    default=os.environ.get("MODEL_PATH"),
    help="Path to a pickled sklearn pipeline (defaults to the latest ZenML artifact).",
)
args = parser.parse_args()

pipeline = load_pipeline(args.model_path)

sample = pd.DataFrame([{
    "Order": 1, "PID": 1, "MS SubClass": 60, "Lot Frontage": 65.0, "Lot Area": 8450,
    "Overall Qual": 7, "Overall Cond": 5, "Year Built": 2003, "Year Remod/Add": 2003,
    "Mas Vnr Area": 196.0, "BsmtFin SF 1": 706, "BsmtFin SF 2": 0, "Bsmt Unf SF": 150,
    "Total Bsmt SF": 856, "1st Flr SF": 856, "2nd Flr SF": 854, "Low Qual Fin SF": 0,
    "Gr Liv Area": 1710, "Bsmt Full Bath": 1, "Bsmt Half Bath": 0, "Full Bath": 2,
    "Half Bath": 1, "Bedroom AbvGr": 3, "Kitchen AbvGr": 1, "TotRms AbvGrd": 8,
    "Fireplaces": 1, "Garage Yr Blt": 2003.0, "Garage Cars": 2, "Garage Area": 548,
    "Wood Deck SF": 0, "Open Porch SF": 61, "Enclosed Porch": 0, "3Ssn Porch": 0,
    "Screen Porch": 0, "Pool Area": 0, "Misc Val": 0, "Mo Sold": 2, "Yr Sold": 2008
}])

# Apply log transform to match training preprocessing
sample["Gr Liv Area"] = np.log1p(sample["Gr Liv Area"])

pred_log = pipeline.predict(sample)[0]
pred_price = np.expm1(pred_log)

print(f"Predicted SalePrice: ${pred_price:,.2f}")
