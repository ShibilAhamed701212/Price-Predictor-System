import os
import sys

# Make sure we can import from the project
sys.path.insert(0, os.path.dirname(__file__))

os.environ["ZENML_ENABLE_RICH_TRACEBACK"] = "FALSE"

print("=" * 70)
print("  HOME PRICE PREDICTOR - STEP BY STEP PIPELINE")
print("  Press Enter after each screenshot to continue")
print("=" * 70)
input("\nPress Enter to start...")

# ------------------------------------------------------------
# STEP 1: Data Ingestion
# ------------------------------------------------------------
print("\n" + "-" * 70)
print("STEP 1: DATA INGESTION")
print("Loading the house prices dataset from archive.zip")
print("-" * 70)
input("\nPress Enter to run this step...")

import zipfile

import pandas as pd

data_path = os.path.join(os.path.dirname(__file__), "data", "archive.zip")
with zipfile.ZipFile(data_path) as z:
    csv_file = [f for f in z.namelist() if f.endswith(".csv")][0]
    df = pd.read_csv(z.open(csv_file))

print("\nDataset loaded successfully!")
print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print("\nFirst 5 rows:")
print(df.head().to_string())
print(f"\nColumns: {list(df.columns)}")
input("\nPress Enter for next step...")

# ------------------------------------------------------------
# STEP 2: Handle Missing Values
# ------------------------------------------------------------
print("\n" + "-" * 70)
print("STEP 2: HANDLE MISSING VALUES")
print("Filling empty cells with the average (mean) value of each column")
print("-" * 70)
input("\nPress Enter to run this step...")

missing_before = df.isnull().sum()
missing_before = missing_before[missing_before > 0]
print(f"\nMissing values BEFORE ({len(missing_before)} columns):")
for col, val in missing_before.items():
    print(f"  {col}: {val} missing values")

df = df.fillna(df.select_dtypes(include="number").mean())
missing_after = df.isnull().sum().sum()
print(f"\nRemaining missing values after fix: {missing_after}")
input("\nPress Enter for next step...")

# ------------------------------------------------------------
# STEP 3: Feature Engineering (Log Transformation)
# ------------------------------------------------------------
print("\n" + "-" * 70)
print("STEP 3: FEATURE ENGINEERING")
print("Applying log transformation to make skewed data more normal")
print("-" * 70)
input("\nPress Enter to run this step...")

import numpy as np

print(f"\nBefore log transform - Gr Liv Area: min={df['Gr Liv Area'].min()}, max={df['Gr Liv Area'].max()}")
print(f"Before log transform - SalePrice: min=${df['SalePrice'].min():.0f}, max=${df['SalePrice'].max():.0f}")

df["Gr Liv Area"] = np.log1p(df["Gr Liv Area"])
df["SalePrice"] = np.log1p(df["SalePrice"])

print(f"\nAfter log transform - Gr Liv Area: min={df['Gr Liv Area'].min():.2f}, max={df['Gr Liv Area'].max():.2f}")
print(f"After log transform - SalePrice (log): min={df['SalePrice'].min():.2f}, max={df['SalePrice'].max():.2f}")
input("\nPress Enter for next step...")

# ------------------------------------------------------------
# STEP 4: Outlier Detection
# ------------------------------------------------------------
print("\n" + "-" * 70)
print("STEP 4: OUTLIER DETECTION")
print("Removing unusual house prices using Z-score method")
print("-" * 70)
input("\nPress Enter to run this step...")

from scipy import stats

z_scores = np.abs(stats.zscore(df["SalePrice"]))
outliers = (z_scores > 3).sum()
print(f"\nOutliers detected (Z-score > 3): {outliers} rows")
print(f"Dataset size before: {len(df)} rows")

df = df[z_scores <= 3]
print(f"Dataset size after:  {len(df)} rows ({len(df)/2930*100:.1f}% kept)")
input("\nPress Enter for next step...")

# ------------------------------------------------------------
# STEP 5: Train/Test Split
# ------------------------------------------------------------
print("\n" + "-" * 70)
print("STEP 5: TRAIN/TEST SPLIT")
print("Splitting data: 80% for training, 20% for testing")
print("-" * 70)
input("\nPress Enter to run this step...")

from sklearn.model_selection import train_test_split

X = df.drop(columns=["SalePrice"])
y = df["SalePrice"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining data:   {X_train.shape[0]} rows")
print(f"Testing data:    {X_test.shape[0]} rows")
input("\nPress Enter for next step...")

# ------------------------------------------------------------
# STEP 6: Model Training
# ------------------------------------------------------------
print("\n" + "-" * 70)
print("STEP 6: MODEL TRAINING")
print("Training a Linear Regression model on the data")
print("-" * 70)
input("\nPress Enter to run this step...")

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

numerical_cols = X_train.select_dtypes(exclude=["object", "category"]).columns
preprocessor = SimpleImputer(strategy="mean")
pipeline = Pipeline(steps=[
    ("imputer", preprocessor),
    ("model", LinearRegression())
])

print(f"Training on {len(numerical_cols)} numerical features...")
pipeline.fit(X_train[numerical_cols], y_train)
print("Model training completed!")
input("\nPress Enter for next step...")

# ------------------------------------------------------------
# STEP 7: Evaluation
# ------------------------------------------------------------
print("\n" + "-" * 70)
print("STEP 7: MODEL EVALUATION")
print("Checking how accurate the model is on unseen data")
print("-" * 70)
input("\nPress Enter to run this step...")

from sklearn.metrics import mean_squared_error, r2_score

y_pred = pipeline.predict(X_test[numerical_cols])
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Convert back from log scale
y_test_actual = np.expm1(y_test)
y_pred_actual = np.expm1(y_pred)

print("\nModel Performance:")
print(f"  Mean Squared Error (MSE): {mse:.4f}")
print(f"  R-Squared Score:          {r2:.4f}")
print(f"  Accuracy (rough):         {r2*100:.1f}%")
print("\nSample predictions:")
results = pd.DataFrame({"Actual Price": y_test_actual.head(10), "Predicted Price": y_pred_actual[:10]})
print(results.to_string())

# ------------------------------------------------------------
# DONE
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("  PIPELINE COMPLETE!")
print("=" * 70)
input("\nPress Enter to exit...")
