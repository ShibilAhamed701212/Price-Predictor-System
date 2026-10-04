# Architecture Documentation

## High-Level Architecture

The Price Predictor System follows a **layered architecture** with strict separation of concerns:

```
┌──────────────────────────────────────────────────────────────┐
│                   Entry Points (CLI)                         │
│  run_pipeline.py  │  run_deployment.py  │  predict.py        │
└──────────────────────────┬───────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────┐
│                  Pipeline Orchestration                      │
│  pipelines/training_pipeline.py  │  pipelines/deployment.py  │
│  ZenML @pipeline decorators       │  Step composition        │
└──────────────────────────┬───────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────┐
│                   Pipeline Steps (ZenML)                     │
│  steps/*.py  │  @step decorators  │  Experiment tracker      │
│  Caching     │  Artifact config   │  MLflow autolog          │
└──────────────────────────┬───────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────┐
│              Core Domain Logic (src/)                        │
│  Strategy Pattern │ Factory Pattern │ ABC interfaces         │
│  Data processing  │ Model training  │ Evaluation             │
└──────────────────────────┬───────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────┐
│                    Data Storage                              │
│  data/archive.zip  │  ZenML Artifact Store  │  MLflow DB     │
└──────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

### 1. Core Domain Logic (`src/`)

Each module implements an abstract base class defining a strategy interface, with concrete implementations and a context class.

| Module | Pattern | Strategy Interface | Concrete Strategies |
|---|---|---|---|
| `ingest_data.py` | Factory | `DataIngestor.ingest()` | `ZipDataIngestor` |
| `handle_missing_values.py` | Strategy | `MissingValueHandlingStrategy.handle()` | `DropMissingValuesStrategy`, `FillMissingValuesStrategy` |
| `feature_engineering.py` | Strategy | `FeatureEngineeringStrategy.apply_transformation()` | `LogTransformation`, `StandardScaling`, `MinMaxScaling`, `OneHotEncoding` |
| `outlier_detection.py` | Strategy | `OutlierDetectionStrategy.detect_outliers()` | `ZScoreOutlierDetection`, `IQROutlierDetection` |
| `data_splitter.py` | Strategy | `DataSplittingStrategy.split_data()` | `SimpleTrainTestSplitStrategy` |
| `model_building.py` | Strategy | `ModelBuildingStrategy.build_and_train_model()` | `LinearRegressionStrategy` |
| `model_evaluator.py` | Strategy | `ModelEvaluationStrategy.evaluate_model()` | `RegressionModelEvaluationStrategy` |

### 2. Pipeline Steps (`steps/`)

Thin wrappers that bridge ZenML step decorators with the core logic modules:

```mermaid
graph LR
    subgraph steps
        DIS[data_ingestion_step] -->|uses| ID[src/ingest_data.py]
        MVS[handle_missing_values_step] -->|uses| HM[src/handle_missing_values.py]
        FES[feature_engineering_step] -->|uses| FE[src/feature_engineering.py]
        ODS[outlier_detection_step] -->|uses| OD[src/outlier_detection.py]
        DSS[data_splitter_step] -->|uses| DS[src/data_splitter.py]
        MBS[model_building_step] -->|uses| MB[src/model_building.py]
        MES[model_evaluator_step] -->|uses| ME[src/model_evaluator.py]
    end
```

Steps handle cross-cutting concerns:
- **MLflow autologging** — Enabled in `model_building_step`
- **Artifact configuration** — Outputs tagged as `ModelArtifact` or `DataArtifact`
- **ZenML caching** — Controlled per-step via `enable_cache`
- **Experiment tracker binding** — Steps reference the active stack's experiment tracker

### 3. Pipeline Orchestration (`pipelines/`)

Defines the execution graph and data flow between steps.

**Training Pipeline** (`training_pipeline.py`):
- Composes 7 steps in sequence
- Registers the model under ZenML's `Model(name="prices_predictor")`
- Returns the trained pipeline object for downstream consumption

**Deployment Pipeline** (`deployment_pipeline.py`):
- Runs the full training pipeline
- Deploys the trained model via `mlflow_model_deployer_step`
- Provides an `inference_pipeline` for batch prediction against the deployed service

### 4. Entry Points (Root scripts)

| Script | Purpose |
|---|---|
| `run_pipeline.py` | Click CLI that executes `ml_pipeline()` and prints MLflow tracking URI |
| `run_deployment.py` | Click CLI for deployment pipeline with `--stop-service` flag |
| `predict.py` | Standalone inference script loading local ZenML artifact |
| `sample_predict.py` | REST API client for deployed MLflow model server |
| `run_step_by_step.py` | Educational walkthrough with manual step-by-step execution |
| `run_screenshots.bat` | Windows batch launcher for `run_step_by_step.py` |

## Data Flow

### Training Data Flow

```mermaid
sequenceDiagram
    participant ZIP as archive.zip
    participant DI as Data Ingestion
    participant MV as Handle Missing Values
    participant FE as Feature Engineering
    participant OD as Outlier Detection
    participant DS as Data Splitter
    participant MB as Model Building
    participant ME as Model Evaluation
    participant ML as MLflow

    ZIP->>DI: Extract CSV
    DI->>DI: Read into DataFrame<br/>(2930 rows × 82 cols)
    DI->>MV: Raw DataFrame
    MV->>MV: Fill NaN with mean
    MV->>FE: Clean DataFrame
    FE->>FE: log1p(Gr Liv Area, SalePrice)
    FE->>OD: Transformed DataFrame
    OD->>OD: Remove Z-score > 3 outliers
    OD->>DS: Cleaned DataFrame
    DS->>DS: 80/20 train-test split
    DS->>MB: X_train, y_train
    DS->>ME: X_test, y_test
    MB->>MB: Fit Pipeline<br/>(Imputer + LinearRegression)
    MB->>ML: autolog: params, metrics, model.pkl
    MB->>ME: Trained Pipeline
    ME->>ME: Predict + Evaluate<br/>(MSE, R²)
    ME->>ML: log evaluation metrics
```

### Inference Data Flow

```mermaid
sequenceDiagram
    participant USER as User / Client
    participant MS as MLflow Model Server<br/>(port 8000)
    participant MODEL as Serialized Pipeline
    participant RESP as Prediction Response

    USER->>MS: POST /invocations<br/>{house features JSON}
    MS->>MODEL: Load model.pkl
    MODEL->>MODEL: Preprocess (impute missing)
    MODEL->>MODEL: Predict log(SalePrice)
    MODEL->>MS: Return prediction
    MS->>USER: 200 OK<br/>{prediction: float[]}
```

## Model Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Training
    Training --> Evaluated: pipeline run
    Evaluated --> Logged: MLflow autolog
    Logged --> Deployed: deployment pipeline
    Deployed --> Serving: MLflow server start
    Serving --> Predictions: inference requests
    Predictions --> Monitoring: performance tracking
    Monitoring --> Retraining: degradation detected
    Retraining --> Training: new data
```

**Current state:** The model passes through Training → Evaluated → Logged states. Deployment is available on Linux/macOS (MLflow daemon). Windows deployment is limited due to daemon restrictions.

## Experiment Tracking Workflow

```mermaid
flowchart LR
    RUN["Pipeline Run"] -->|@step decorator| MB["model_building_step"]
    MB -->|mlflow.sklearn.autolog()| ENABLE["Enable Autologging"]
    ENABLE --> FIT["pipeline.fit()"]
    FIT --> PARAMS["Log Parameters<br/>model type, columns"]
    FIT --> METRICS["Log Metrics<br/>MSE, R², MAE, RMSE"]
    FIT --> MODEL["Log Model Artifact<br/>model.pkl"]
    MODEL --> UI["MLflow UI<br/>http://127.0.0.1:5000"]

    subgraph "Per Run"
        PARAMS
        METRICS
        MODEL
    end
```

## Design Decisions

| Decision | Rationale |
|---|---|
| **Strategy Pattern** for ML components | Enables runtime algorithm switching, testability in isolation, and clean separation of algorithm from context |
| **ZenML over Airflow** | Lightweight pipeline orchestration with native MLflow integration, artifact versioning, and local execution without infrastructure |
| **Log transformation** of target variable | Stabilizes variance in sale prices (log-normal distribution → approximately normal), improving linear regression assumptions |
| **Z-score outlier detection** over IQR | Simpler threshold interpretation for normally distributed features; configurable sensitivity via threshold parameter |
| **SimpleImputer + OneHotEncoder pipeline** | Scikit-learn `Pipeline` ensures consistent preprocessing between training and inference, preventing data leakage |
| **SQLite tracking store** | Zero-configuration experiment persistence suitable for local development; upgradeable to PostgreSQL for production |
| **ZenML artifact lookup in predict.py** | Loads the latest `sklearn_pipeline` artifact by name; `--model-path` covers a standalone pickle |
