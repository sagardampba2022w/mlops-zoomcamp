## 🚀 How It Works

- Downloads trip data for a specified month and taxi type  
- Uses a trained MLflow model to predict trip durations  
- Saves results to a `.parquet` file in the `output/` directory  
- Automatically processes data for the **previous month** relative to the input date  

## 🛠 Setup Instructions

### 1. Install Dependencies

Use Pipenv to install all dependencies listed in the Pipfile.

```bash
pipenv install
``` 

### 2. Activate the Environment

Start a Pipenv shell to activate the virtual environment.
```bash
pipenv shell
```

### 3. Install Required Libraries

Install additional required libraries: pandas, prefect, mlflow, scikit-learn, pyarrow.
pipenv install pandas prefect mlflow scikit-learn pyarrow

## 🧭 Run a Batch Prediction Job

Use the `score.py` script to run a batch prediction by specifying the taxi type, year, month, and MLflow run ID.

python score.py <taxi_type> <year> <month> <mlflow_run_id>

### Example

Run the script with:
- Taxi type: green  
- Year: 2021  
- Month: 4  
- Run ID: 03f97fa2fa6d4db1a910a0dc13658a7a  

python score.py green 2021 4 03f97fa2fa6d4db1a910a0dc13658a7a

This will process data for **March 2021**, the month before the one provided.
