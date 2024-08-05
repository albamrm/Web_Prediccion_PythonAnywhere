from flask import Flask, jsonify, request
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import subprocess

# os.chdir(os.path.dirname(__file__))

path_base = "/home/sabadosteam/mysite/"

app = Flask(__name__)
app.config['DEBUG'] = True

# Enruta la landing page (endpoint /)
@app.route('/', methods=['GET'])
def hello():
    return "Bienvenido a mi API del modelo pingüinos"

# Enruta la funcion al endpoint /api/v1/predict

@app.route('/api/v1/predict', methods=['GET'])
def predict(): # Ligado al endpoint '/api/v1/predict', con el método GET

    model = pickle.load(open(path_base + 'ad_model.pkl','rb'))
    long1 = request.args.get('l1', None)
    long2 = request.args.get('l2', None)
    long3 = request.args.get('l3', None)
    genero = request.args.get('genero', None)

    print(long1,long2, long3, genero)
    print(type(genero))

    if long1 is None or long2 is None or long3 is None or genero is None:
        return "Args empty, the data are not enough to predict, STUPID!!!!"
    else:
        prediction = model.predict([[float(long1),float(long2),float(long3),str(genero)]])
    
    return jsonify({'predictions': prediction[0]})

# Endpoint para reentrenar el modelo
@app.route('/api/v1/retrain', methods=['GET'])
def retrain(): # Rutarlo al endpoint '/api/v1/retrain/', metodo GET
    if os.path.exists(path_base + "data/penguins.csv"):
        data = pd.read_csv(path_base + 'data/penguins.csv')
        
        # Separar características y variable objetivo
        X = data.drop(columns='species')
        y = data['species']
        
        # Escalar los datos
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Dividir en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # Reentrenar el modelo
        model.fit(X_train, y_train)
        
        # Guardar el modelo reentrenado
        pickle.dump(model, open(path_base + 'ad_model.pkl', 'wb'))

        return f"Model retrained."
    else:
        return f"<h2>New data for retrain NOT FOUND. Nothing done!</h2>"

@app.route('/webhook_2024', methods=['POST'])
def webhook():
    # Ruta al repositorio donde se realizará el pull
    path_repo = '/home/sabadosteam/sabadosteam'
    servidor_web = '/var/www/sabadosteam_pythonanywhere_com_wsgi.py' 

    # Comprueba si la solicitud POST contiene datos JSON
    if request.is_json:
        payload = request.json
        # Verifica si la carga útil (payload) contiene información sobre el repositorio
        if 'repository' in payload:
            # Extrae el nombre del repositorio y la URL de clonación
            repo_name = payload['repository']['name']
            clone_url = payload['repository']['clone_url']
            
            # Cambia al directorio del repositorio
            try:
                os.chdir(path_repo)
            except FileNotFoundError:
                return jsonify({'message': 'El directorio del repositorio no existe'}), 404

            # Realiza un git pull en el repositorio
            try:
                subprocess.run(['git', 'pull', clone_url], check=True)
                subprocess.run(['touch', servidor_web], check=True) # Trick to automatically reload PythonAnywhere WebServer
                return jsonify({'message': f'Se realizó un git pull en el repositorio {repo_name}'}), 200
            except subprocess.CalledProcessError:
                return jsonify({'message': f'Error al realizar git pull en el repositorio {repo_name}'}), 500
        else:
            return jsonify({'message': 'No se encontró información sobre el repositorio en la carga útil (payload)'}), 400
    else:
        return jsonify({'message': 'La solicitud no contiene datos JSON'}), 400


if __name__ == '__main__':
    app.run()