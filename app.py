from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Cargar modelos y escalador
model_path = 'C:/Users/kathy/Diabetes Modelos/'

with open(model_path + 'model_rf.pkl', 'rb') as rf_file:
    rf_model = pickle.load(rf_file)

with open(model_path + 'model_svm.pkl', 'rb') as svm_file:
    svm_model = pickle.load(svm_file)

with open(model_path + 'model_lr.pkl', 'rb') as lr_file:
    lr_model = pickle.load(lr_file)

with open(model_path + 'scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array(data['features']).reshape(1, -1)
    features_scaled = scaler.transform(features)

    # Obtener predicciones
    prediction_rf = rf_model.predict(features_scaled)[0]
    prediction_svm = svm_model.predict(features_scaled)[0]
    prediction_lr = lr_model.predict(features_scaled)[0]

    return jsonify({
        'RandomForest': int(prediction_rf),
        'SVM': int(prediction_svm),
        'LogisticRegression': int(prediction_lr)
    })

if __name__ == '__main__':
    app.run(debug=True)
