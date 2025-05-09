# 1. Web-Service 

pipenv install scikit-learn==1.0.2 flask

Or

pipenv --python $(which python3)
pip install scikit-learn flask

then run venv using pipenv shell

### 1.1 Create a web service using Flask

```python
from flask import Flask, request, jsonify

app = Flask('duration-prediction')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()

    features = prepare_features(ride)
    pred = predict(features)

    result = {
        'duration': pred
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)

for testing
```
url = 'http://localhost:9696/predict'
response = requests.post(url, json=ride)
print(response.json())
```

for production testing
```python

gunicorn --bind=0.0.0.0:9696 predict:app
```
and then run test.py

# Packaging the app to Docker
```bash
docker build -t ride-duration-prediction-service:v1 .
```
# 2. Run the web service
```bash
docker run -it --rm -p 9696:9696  ride-duration-prediction-service:v1
```
