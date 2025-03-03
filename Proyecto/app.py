from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

# Cargar el modelo y el escalador
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

app = Flask(__name__)

# Definir rangos válidos para los valores ingresados
RANGOS = {
    "embarazos": (0, 17),
    "glucosa": (0, 199),
    "presion": (0, 122),
    "espesor_piel": (0, 99),
    "insulina": (0, 846),
    "imc": (0, 67.1),
    "diabetes_pedigree": (0.08, 2.42),
    "edad": (21, 81)
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los valores del formulario
        genero = float(request.form['genero'])
        features = []
        
        for key in RANGOS:
            valor = float(request.form[key])
            min_val, max_val = RANGOS[key]
            
            if valor < min_val or valor > max_val:
                return render_template('index.html', prediction_text=f'Error: {key} debe estar entre {min_val} y {max_val}.')
            
            features.append(valor)
        
        # Agregar género a los datos
        features.insert(0, genero)  # Insertar el género al inicio
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
