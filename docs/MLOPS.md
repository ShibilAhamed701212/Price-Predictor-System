# MLOps Documentation

## Overview

This document describes the MLOps practices implemented in the Price Predictor System. The project is a **local ML pipeline** — it is not deployed to production infrastructure. This document covers what exists, what is simulated, and what would be required for production readiness.

## Current MLOps Capabilities

### ZenML Pipeline Orchestration

ZenML provides the core orchestration layer:

| Capability | Status | Details |
|---|---|---|
| **Pipeline Composition** | ✅ Implemented | `@pipeline` decorator composes 7 sequential steps |
| **Step Isolation** | ✅ Implemented | Each `@step` is independent with defined inputs/outputs |
| **Artifact Caching** | ✅ Implemented | Steps with unchanged inputs are automatically skipped on re-run |
| **Artifact Versioning** | ✅ Implemented | ZenML tracks artifact versions in local SQLite store |
| **Stack Configuration** | ✅ Implemented | MLflow stack with experiment tracker + model deployer |
| **Model Registry** | ✅ Implemented | ZenML `Model(name="prices_predictor")` registers pipeline outputs |
| **Containerization** | ❌ Not configured | Docker settings in `config.yaml` but no Dockerfile |
| **Remote Execution** | ❌ Not configured | Runs only on local orchestrator |

### MLflow Experiment Tracking

MLflow is integrated via ZenML's MLflow experiment tracker:

```mermaid
graph TD
    subgraph "MLflow Tracking"
        UI[MLflow UI<br/>http://127.0.0.1:5000]
        DB[(SQLite Store<br/>mlflow.db)]
        ARTIFACTS[(Artifact Store<br/>model.pkl)]
    end

    subgraph "Pipeline Run"
        PIPELINE[Training Pipeline]
        STEP[model_building_step]
        LOG[mlflow.sklearn.autolog()]
    end

    PIPELINE -->|execute| STEP
    STEP -->|enable autolog| LOG
    LOG -->|log params| DB
    LOG -->|log metrics| DB
    LOG -->|log model| ARTIFACTS
    UI -->|read| DB
    UI -->|browse| ARTIFACTS
```

#### What MLflow Tracks

| Category | Data Captured | Example |
|---|---|---|
| **Parameters** | Model type, column configuration | `model_type = linear_regression` |
| **Metrics** | MSE, R², MAE, RMSE (training) | `training_r2_score = 0.7715` |
| **Model Artifacts** | Serialized Pipeline object | `model.pkl` (~10 KB) |
| **Environment** | Python version, package versions | `python_env.yaml` |
| **Source Code** | Entry point script | `run_pipeline.py` |
| **Tags** | User-defined metadata | Auto-generated run names |

#### Accessing Experiment Data

```bash
# Launch the MLflow UI
mlflow ui --backend-store-uri 'sqlite:///<path-to-zenml-local-stores>/mlflow.db'

# Browse runs: http://127.0.0.1:5000
# Compare runs, view metrics, download artifacts
```

### How Autologging Works

In `steps/model_building_step.py`:

```python
mlflow.sklearn.autolog()

# After this call, the following fit() operation automatically:
# - Logs model hyperparameters
# - Logs evaluation metrics
# - Saves the model as an MLflow artifact

pipeline.fit(X_train, y_train)

mlflow.end_run()
```

**Important:** Autologging is active only within the scope of `model_building_step`. Other steps do not use MLflow directly.

### Model Deployment (Local)

The deployment pipeline (`run_deployment.py`) attempts to:

1. Re-run the training pipeline
2. Deploy the model via `mlflow_model_deployer_step`
3. Start a local MLflow model server as a daemon process
4. Run an inference pipeline against the deployed service

**Current Limitation:** MLflow's daemon functionality is not supported on Windows. The deployment pipeline will log a warning and fail to start the server. The code structure exists and is functional on Linux/macOS.

### Inference Pipeline Components

For local inference without a running server:

- **`predict.py`** — Loads the serialized `Pipeline` directly from ZenML's artifact store
- **`sample_predict.py`** — Sends HTTP requests to a running MLflow model server at `http://127.0.0.1:8000`

## Observability

### Current State

| Observability Aspect | Implementation |
|---|---|
| **Run Metrics** | MLflow UI — compare runs, view metric curves |
| **Step Logs** | Python `logging` module output to console |
| **Pipeline Status** | ZenML console output (step started/finished, duration) |
| **Model Artifacts** | MLflow artifact browser |
| **Error Handling** | Python try/except in step implementations |

### What Is Missing

- **Structured logging** (JSON format, log aggregation)
- **Metric alerts** (no automated notification on degradation)
- **Dashboard** (no persistent metrics dashboard beyond MLflow UI)
- **Drift detection** (no statistical comparison of data distributions)
- **Data quality checks** (no schema validation, no anomaly detection in data)

## Quality Evaluation

### Pipeline Success Criteria

The pipeline is considered successful when:
1. All 7 steps complete without errors
2. Model evaluation produces valid MSE and R-squared metrics
3. MLflow records the run with all artifacts

### Current Limitations

| Limitation | Impact |
|---|---|
| **Single model** (Linear Regression only) | No baseline comparison, no ensemble |
| **No cross-validation** | Single 80/20 split may not represent true generalization |
| **No hyperparameter tuning** | Default LinearRegression hyperparameters only |
| **Categorical column handling** | Pipeline logs "Categorical columns: []" — one-hot encoding path is untested because no categorical columns survive log transformation |
| **Fixed outlier threshold** | Z-score threshold of 3 is arbitrary and not optimized |
| **No feature selection** | All 38 numeric columns used regardless of predictive value |
| **Log-scale metrics only** | MSE and R² reported on log-transformed values, making dollar-interpretation non-trivial |
| **Deterministic caching** | Cache hits from ZenML may mask data drift or concept drift |

## Production Readiness Assessment

### Gaps for Production Deployment

| Requirement | Current State | Production Target |
|---|---|---|
| **CI/CD** | None | GitHub Actions / GitLab CI for automated testing |
| **Testing** | None | Unit tests for `src/`, integration tests for `steps/`, end-to-end pipeline test |
| **Data Validation** | None | Great Expectations or Pandera for schema and distribution checks |
| **Feature Store** | None | Feast or Tecton for feature definitions and serving |
| **Model Registry** | ZenML Model list | MLflow Model Registry with staging/production stages |
| **Containerization** | None | Docker image with reproducible environment |
| **Orchestration** | ZenML local | Scheduled runs with Airflow / Prefect / Kubeflow |
| **Monitoring** | Manual MLflow UI | Evidently / WhyLabs for drift detection and model performance |
| **Alerting** | None | Slack/Email alerts on metric degradation |
| **Deployment** | Manual / local daemon | Kubernetes (Kserve, Seldon) or Serverless (AWS Lambda) |
| **Reproducibility** | Pip freeze + ZenML caching | DVC + MLflow + Container |
| **Security** | None | Authentication, secrets management, audit logging |
| **Scalability** | Single-node | Distributed processing for large datasets |

### What Would Be Needed for Production

1. **Data pipeline improvements:**
   - Schema validation and data quality checks
   - Automated data drift detection
   - Feature store integration

2. **Model pipeline improvements:**
   - Cross-validation and hyperparameter tuning
   - Multiple model benchmarks (Random Forest, XGBoost, LightGBM)
   - Model selection logic (best model promoted to registry)
   - A/B testing framework

3. **Infrastructure improvements:**
   - Docker containerization
   - CI/CD pipeline with automated testing
   - Model serving on Kubernetes
   - Monitoring dashboard with alerting
   - Retraining trigger based on performance degradation or schedule

4. **Observability improvements:**
   - Structured logging (e.g., ELK stack)
   - Metrics dashboard (e.g., Grafana)
   - Model performance monitoring
   - Data and concept drift detection

## Design Patterns for MLOps

The project uses software engineering design patterns that benefit MLOps maintainability:

| Pattern | Application | MLOps Benefit |
|---|---|---|
| **Strategy** | ML components (imputation, engineering, detection, splitting, building, evaluation) | Swappable algorithms without pipeline code changes. Test each strategy in isolation. |
| **Factory** | Data ingestion | Extensible to new data formats (CSV, Parquet, databases, APIs) without modifying pipeline steps. |
| **Template Method** | Analysis modules | Consistent structure for different types of analysis (univariate, bivariate, multivariate). |
