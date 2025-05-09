import os
import pickle

import mlflow
from flask import Flask, request, jsonify
# Set MLflow tracking URI to use the local SQLite database
mlflow.set_tracking_uri("sqlite:///mlflow.db")

#RUN_ID = os.getenv('RUN_ID')
RUN_ID = '03f97fa2fa6d4db1a910a0dc13658a7a'  # Replace with your actual run ID

#logged_model = f's3://mlflow-models-alexey/1/{RUN_ID}/artifacts/model'
logged_model = f'runs:/{RUN_ID}/model'
model = mlflow.pyfunc.load_model(logged_model)


def prepare_features(ride):
    features = {}
    features['PU_DO'] = '%s_%s' % (ride['PULocationID'], ride['DOLocationID'])
    features['trip_distance'] = ride['trip_distance']
    return features


def predict(features):
    preds = model.predict(features)
    return float(preds[0])


app = Flask('duration-prediction')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()

    features = prepare_features(ride)
    pred = predict(features)

    result = {
        'duration': pred,
        'model_version': RUN_ID
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)