# Technical Documentation

## Dataset

### Ames Housing Dataset

The project uses the **Ames Housing Dataset**, a modern alternative to the Boston Housing dataset. It contains residential property sales records from Ames, Iowa, USA.

- **Source:** [Kaggle: House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- **Format:** ZIP-compressed CSV (`data/archive.zip`)
- **Rows:** 2,930
- **Columns:** 82 (80 features + `Order` + `PID`)
- **Target variable:** `SalePrice` (continuous, in USD)

### Feature Categories

| Category | Example Columns | Count |
|---|---|---|
| **Lot & Location** | `Lot Area`, `Lot Frontage`, `MS Zoning`, `Neighborhood` | ~8 |
| **Building Structure** | `MS SubClass`, `Year Built`, `Year Remod/Add`, `1st Flr SF`, `Total Bsmt SF` | ~15 |
| **Interior Features** | `Gr Liv Area`, `Full Bath`, `Bedroom AbvGr`, `Kitchen AbvGr`, `TotRms AbvGrd` | ~10 |
| **Amenities** | `Fireplaces`, `Garage Cars`, `Garage Area`, `Wood Deck SF`, `Pool Area` | ~8 |
| **Condition & Quality** | `Overall Qual`, `Overall Cond`, `Exter Qual`, `Kitchen Qual` | ~6 |
| **Sale Information** | `Mo Sold`, `Yr Sold`, `Sale Type`, `Sale Condition` | ~4 |
| **Categorical** | `Street`, `Alley`, `Roof Style`, `Heating`, `Central Air` | ~30 |

## Preprocessing Pipeline

The preprocessing steps are applied in sequence before model training:

```
Raw Data
  │
  ▼
┌──────────────────────────────────────────────┐
│ Step 1: Data Ingestion                       │
│ - Extract archive.zip                        │
│ - Read AmesHousing.csv into DataFrame        │
└──────────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────────────┐
│ Step 2: Handle Missing Values                │
│ - Numeric columns: fill NaN with column mean │
│ - Strategy: FillMissingValuesStrategy("mean")│
└──────────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────────────┐
│ Step 3: Feature Engineering                  │
│ - Log transformation (log1p):                │
│   • Gr Liv Area (skewed feature)             │
│   • SalePrice (target variable)              │
└──────────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────────────┐
│ Step 4: Outlier Detection & Removal          │
│ - Z-score method on SalePrice                │
│ - Threshold: |z| > 3                         │
│ - Removes ~80-100 outlier rows               │
└──────────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────────────┐
│ Step 5: Train/Test Split                     │
│ - 80% training, 20% testing                  │
│ - random_state=42 for reproducibility       │
│ - Target column: SalePrice (log-transformed) │
└──────────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────────────┐
│ Step 6: Model Building                       │
│ - Scikit-learn Pipeline:                     │
│   ① ColumnTransformer                        │
│     - Num: SimpleImputer(strategy="mean")    │
│     - Cat: SimpleImputer + OneHotEncoder     │
│   ② LinearRegression                         │
└──────────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────────────┐
│ Step 7: Model Evaluation                     │
│ - Inverse-transform predictions (expm1)      │
│ - Metrics: MSE, R-squared                    │
└──────────────────────────────────────────────┘
```

## Missing Value Handling

### Strategy: FillMissingValuesStrategy (mean)

All numeric columns with missing values are filled using the column mean:

```python
strategy = FillMissingValuesStrategy(method="mean")
handler = MissingValueHandler(strategy)
filled_df = handler.handle_missing_values(df)
```

**Alternative strategies available:**
- `method="median"` — Median imputation
- `method="mode"` — Mode imputation
- `method="constant"` — Fill with user-specified constant
- `DropMissingValuesStrategy(axis=0)` — Drop rows with NaNs
- `DropMissingValuesStrategy(axis=1)` — Drop columns with NaNs

### Columns with Missing Values (in raw data)

Typical columns with NaNs in the Ames dataset include `Lot Frontage`, `Mas Vnr Area`, `Mas Vnr Type`, `BsmtQual`, `BsmtCond`, `BsmtFin Type 1`, `BsmtFin SF 1`, `Garage Yr Blt`, `Garage Cars`, `Garage Area`, and various categorical basement/garage attributes.

## Outlier Removal

### Method: Z-Score Detection

```python
detector = ZScoreOutlierDetection(threshold=3)
outlier_detector = OutlierDetector(detector)
outliers = outlier_detector.detect_outliers(df_numeric)
cleaned_df = outlier_detector.handle_outliers(df_numeric, method="remove")
```

- Applied to `SalePrice` column after log transformation
- Data points with |z| > 3 are identified as outliers
- Outliers are **removed** (not capped/clipped)

### Alternative: IQR Outlier Detection

```python
iqr_detector = IQROutlierDetection()
```

Uses the interquartile range: points below Q1 − 1.5×IQR or above Q3 + 1.5×IQR are outliers.

## Feature Engineering

### Log Transformation

Applied to two columns:

```python
df["Gr Liv Area"] = np.log1p(df["Gr Liv Area"])    # ln(1 + x)
df["SalePrice"] = np.log1p(df["SalePrice"])
```

**Rationale:** Both `Gr Liv Area` and `SalePrice` exhibit right-skewed distributions. Log transformation normalizes the distributions, which improves linear regression performance since the model assumes normally distributed residuals.

**Inverse transform for predictions:**
```python
predicted_price = np.expm1(predicted_log_saleprice)  # e^x - 1
```

### Available Feature Engineering Strategies

| Strategy | Purpose |
|---|---|
| `LogTransformation` | Normalize skewed features |
| `StandardScaling` | Z-score normalization (mean 0, std 1) |
| `MinMaxScaling` | Scale to range [0, 1] |
| `OneHotEncoding` | Convert categorical features to binary vectors |

## Model Training

### Pipeline Architecture

```python
Pipeline(steps=[
    ("preprocessor", ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="mean"), numerical_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), categorical_cols)
        ]
    )),
    ("model", LinearRegression())
])
```

Key details:
- **Numerical preprocessing:** Missing values → mean imputation
- **Categorical preprocessing:** Missing values → most frequent → one-hot encoding (with `handle_unknown="ignore"` for unseen categories)
- **No scaling** applied to numerical features before LinearRegression (coefficients remain interpretable)
- **MLflow autologging** captures model parameters and metrics automatically

### Training Configuration

| Parameter | Value |
|---|---|
| Model | `LinearRegression` |
| Test size | 0.2 (20%) |
| Random state | 42 |
| Feature columns | 38 numerical (no categorical after log transform) |
| Training samples | ~1,875 (after outlier removal) |
| Test samples | ~469 |

## Evaluation Metrics

### Metrics Computed

**Mean Squared Error (MSE):**

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

Where $y_i$ is the log-transformed actual sale price and $\hat{y}_i$ is the predicted log-transformed sale price.

**R-squared (R²):**

$$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$

Proportion of variance explained by the model. Ranges from 0 to 1 (higher is better).

### Additional Metrics (via MLflow autologging)

| Metric | Value (latest run) |
|---|---|
| `training_r2_score` | 0.7715 |
| `training_mean_squared_error` | 0.0316 |
| `training_root_mean_squared_error` | 0.1776 |
| `training_mean_absolute_error` | 0.1334 |

### Interpreting Metrics

Because the model trains on log-transformed `SalePrice`, the metrics are computed on the log scale. To interpret errors in dollar terms:

```python
# Typical dollar error (approximate)
rmse_log = 0.1776
# At median house price (~$160,000):
# log(160000) ≈ 11.98
# Error in log space: 0.1776
# Error in dollar space at that price: exp(11.98) * 0.1776 ≈ $27,500
```

This is a rough approximation; exact interpretation requires computing metrics on inverse-transformed predictions.

## Model Artifacts

Each MLflow run stores:

| Artifact | Description | Format |
|---|---|---|
| `model.pkl` | Serialized scikit-learn Pipeline | Pickle |
| `MLmodel` | MLflow model metadata | YAML |
| `conda.yaml` | Conda environment specification | YAML |
| `python_env.yaml` | Python virtual environment spec | YAML |
| `requirements.txt` | Python package requirements | Text |
| `estimator.html` | Model visualization (HTML) | HTML |
