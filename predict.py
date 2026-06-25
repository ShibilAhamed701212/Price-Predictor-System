import pickle
import numpy as np
import pandas as pd

MODEL_PATH = r"C:\Users\sclip\AppData\Roaming\zenml\local_stores\f4dc7a44-1501-47f7-85d1-4c6eeb9f08f2\model_building_step\sklearn_pipeline\7abc11dc-2256-4c1f-87af-022aad2c508f\3af4d1ed\artifact.pkl"

with open(MODEL_PATH, "rb") as f:
    pipeline = pickle.load(f)

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
input("\nPress Enter to exit...")
