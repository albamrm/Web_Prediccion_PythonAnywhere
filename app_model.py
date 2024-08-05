from flask import Flask, jsonify, request, render_template
import os
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Cambiar al directorio del script
# os.chdir(os.path.dirname(__file__))

app = Flask(__name__)
app.config['DEBUG'] = True

# Cargar el modelo y el scaler
model_path = 'home/sabadosteam/ad_model.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Cargar el scaler (si es necesario, asumiendo que se usó un scaler)
scaler = StandardScaler()

# Endpoint para la landing page
@app.route('/')
def home():
    return render_template('index.html')

# Endpoint para realizar predicciones
@app.route('/api/v1/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos del request
        data = request.get_json()
        input_data = pd.DataFrame(data, index=[0])
        
        # Preprocesar los datos de entrada (escalado)
        input_scaled = scaler.transform(input_data)
        
        # Realizar la predicción
        prediction = model.predict(input_scaled)
        
        # Retornar la predicción en formato JSON
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)})

# Endpoint para reentrenar el modelo
@app.route('/api/v1/retrain', methods=['POST'])
def retrain():
    try:
        # Cargar nuevos datos del request
        data = request.get_json()
        df = pd.DataFrame(data)
        
        # Separar características y variable objetivo
        X = df.drop(columns='species')
        y = df['species']
        
        # Escalar los datos
        X_scaled = scaler.fit_transform(X)
        
        # Dividir en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # Reentrenar el modelo
        model.fit(X_train, y_train)
        
        # Guardar el modelo reentrenado
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        return jsonify({'message': 'Model retrained successfully'})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run()
