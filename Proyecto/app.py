from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

# Cargar el modelo y el escalador
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los valores del formulario
        features = [float(x) for x in request.form.values()]
        features_array = np.array([features])
        
        # Escalar los datos
        features_scaled = scaler.transform(features_array)
        
        # Hacer la predicción
        prediction = model.predict(features_scaled)[0]
        
        # Devolver resultado
        result = "Positivo (Tiene diabetes)" if prediction == 1 else "Negativo (No tiene diabetes)"
        return render_template('index.html', prediction_text=f'Resultado de la predicción: {result}')
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)