# Price Predictor System

An end-to-end machine learning pipeline that ingests dataset features, trains a model, tracks artifacts and metadata, and serves predictions locally. 

Built with **ZenML** for pipeline orchestration and **MLflow** for experiment tracking and model deployment.

## 🛠 Tech Stack
* **Language:** Python
* **Machine Learning:** scikit-learn
* **MLOps Pipeline:** ZenML
* **Tracking & Serving:** MLflow

## ⚙️ Prerequisites
* Python 3.8+
* `pip` (Python package manager)

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ShibilAhamed701212/Price-Predictor-System.git
   cd Price-Predictor-System
   ```

2. **Set up a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize ZenML and MLflow integrations:**
   ```bash
   zenml init
   zenml integration install mlflow -y
   ```

## 🧠 Usage: Running the Pipeline

To execute the end-to-end pipeline (data ingestion, preprocessing, training, and evaluation):

```bash
python run_pipeline.py
```

You can view the tracked experiments, parameters, and metrics by starting the MLflow UI:

```bash
mlflow ui --backend-store-uri 'file:///path/to/your/mlruns'
```

## 🌐 Deployment & Inference

This project uses a local MLflow prediction daemon. **Note:** There is no web-facing authentication; it is designed strictly for local inference.

**Start the prediction server:**
```bash
python run_deployment.py
```

The server will run locally at `http://127.0.0.1:8000/invocations`.

**Send a test prediction (cURL):**
```bash
curl -X POST http://127.0.0.1:8000/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "dataframe_split": {
      "columns": ["Order","PID","MS SubClass","Lot Frontage","Lot Area","Overall Qual","Overall Cond","Year Built","Year Remod/Add","Mas Vnr Area","BsmtFin SF 1","BsmtFin SF 2","Bsmt Unf SF","Total Bsmt SF","1st Flr SF","2nd Flr SF","Low Qual Fin SF","Gr Liv Area","Bsmt Full Bath","Bsmt Half Bath","Full Bath","Half Bath","Bedroom AbvGr","Kitchen AbvGr","TotRms AbvGrd","Fireplaces","Garage Yr Blt","Garage Cars","Garage Area","Wood Deck SF","Open Porch SF","Enclosed Porch","3Ssn Porch","Screen Porch","Pool Area","Misc Val","Mo Sold","Yr Sold"],
      "data": [[1,5286,20,80.0,9600,5,7,1961,1961,0.0,700.0,0.0,150.0,850.0,856,854,0,1710.0,1,0,1,0,3,1,7,2,1961,2,500.0,210.0,0,0,0,0,0,0,5,2010]]
    }
  }'
```

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
