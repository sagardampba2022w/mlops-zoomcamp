# MLflow Tracking & Model Registry with SQLite Backend

This repository demonstrates how to set up and use **MLflow** with a **SQLite backend** for experiment tracking and model registry, featuring examples of manual tracking, hyperparameter optimization, autologging, and model management.

---

## 📦 Requirements

- Python 3.x
- `mlflow`
- `scikit-learn`
- `xgboost`
- `hyperopt`
- `pandas`, `numpy`
- `matplotlib` (for visualizations)

---

## 🚀 Getting Started

### 1. Initialize MLflow UI with SQLite Backend

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

To set a custom path for run artifacts:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

This launches the UI at [http://localhost:5000](http://localhost:5000).

---

## 🔗 Setting Tracking URI in Python

```python
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
```

---

## 🧪 Experiment Tracking

### Create or Set Experiment

```python
mlflow.set_experiment("nyc-taxi-experiment")
```

### Track a Run Manually

```python
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

with mlflow.start_run():
    mlflow.set_tag("developer", "Qfl3x")
    mlflow.log_param("train-data-path", "data/green_tripdata_2021-01.parquet")
    mlflow.log_param("val-data-path", "data/green_tripdata_2021-02.parquet")

    alpha = 0.01
    mlflow.log_param("alpha", alpha)

    model = Lasso(alpha)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    rmse = mean_squared_error(y_val, y_pred, squared=False)
    mlflow.log_metric("rmse", rmse)
```

---

## 🔍 Hyperparameter Optimization with `hyperopt`

```python
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import xgboost as xgb

def objective(params):
    with mlflow.start_run():
        mlflow.set_tag("model", "xgboost")
        mlflow.log_params(params)

        booster = xgb.train(
            params=params,
            dtrain=xgb.DMatrix(X_train, label=y_train),
            num_boost_round=1000,
            evals=[(xgb.DMatrix(X_val, label=y_val), 'validation')],
            early_stopping_rounds=50
        )
        preds = booster.predict(xgb.DMatrix(X_val))
        rmse = mean_squared_error(y_val, preds, squared=False)
        mlflow.log_metric("rmse", rmse)

        return {"loss": rmse, "status": STATUS_OK}

search_space = {
    'max_depth': hp.quniform('max_depth', 4, 100, 1),
    'learning_rate': hp.loguniform('learning_rate', -3, 0),
    'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
    'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
    'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
    'objective': 'reg:linear',
    'seed': 42
}

best = fmin(fn=objective, space=search_space, algo=tpe.suggest, max_evals=50, trials=Trials())
```

---

## ⚙️ Autologging

Enable MLflow's autologging for various frameworks:

```python
mlflow.autolog()
# or specific to framework
mlflow.xgboost.autolog()
```

---

## 💾 Saving & Loading Models

### Log a Model

```python
mlflow.<framework>.log_model(model, artifact_path="models_mlflow")
```

### Save Extra Artifacts

```python
mlflow.log_artifact("vectorizer.pkl", artifact_path="extra_artifacts")
```

### Load a Model

```python
logged_model = 'runs:/<run-id>/models'
xgboost_model = mlflow.xgboost.load_model(logged_model)
```

---

## 🧰 MLflow Client API

### Initialize Client

```python
from mlflow.tracking import MlflowClient

client = MlflowClient(tracking_uri="sqlite:///mlflow.db")
```

### Search Runs

```python
from mlflow.entities import ViewType

runs = client.search_runs(
    experiment_ids=["1"],
    filter_string="metrics.rmse < 7",
    run_view_type=ViewType.ACTIVE_ONLY,
    order_by=["metrics.rmse ASC"],
    max_results=5
)
```

---

## 🏷️ Model Registry

### Register a Model

```python
run_id = "your-run-id"
model_uri = f"runs:/{run_id}/models"
mlflow.register_model(model_uri=model_uri, name="nyc-taxi-regressor")
```

### List Model Versions

```python
versions = client.get_latest_versions("nyc-taxi-regressor")
for v in versions:
    print(f"Version: {v.version}, Stage: {v.current_stage}")
```

### Promote Model

```python
client.transition_model_version_stage(
    name="nyc-taxi-regressor",
    version=4,
    stage="Staging",
    archive_existing_versions=False
)
```

### Update Model Description

```python
from datetime import datetime

client.update_model_version(
    name="nyc-taxi-regressor",
    version=4,
    description=f"Updated on {datetime.today().date()}"
)
```

---

## 🧠 Visualizing in MLflow UI

- **Metrics**: Compare RMSE across runs.
- **Parallel Coordinates / Contour Plots**: Visualize how hyperparameters affect metrics.
- **Artifacts**: Inspect saved models and extra files.
- **Model Registry**: Promote or archive models by stage (Staging, Production, Archived).

---

## 📚 References

- [MLflow Docs](https://mlflow.org/docs/latest/index.html)
- [MLflow Tracking API](https://mlflow.org/docs/latest/python_api/mlflow.tracking.html)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [Hyperopt Docs](http://hyperopt.github.io/hyperopt/)
- [XGBoost Docs](https://xgboost.readthedocs.io/en/latest/)
