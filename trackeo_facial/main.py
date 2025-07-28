# main.py (modificado para Flask en lugar de menú en terminal)

from flask import Flask, render_template, send_from_directory, request, redirect, url_for
import os
import cv2
import pandas as pd
from datetime import datetime
from scripts.helpers import distancia, detectar_microexpresiones, mostrar_imagen_ajustada
import mediapipe as mp

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('data', exist_ok=True)

@app.route('/get_csv')
def get_csv():
    return send_from_directory('data', 'emociones_imagen.csv')

@app.route('/', methods=['GET', 'POST'])
def index():
    emociones_detectadas = None
    detalles = {}
    imagen_filename = None

    if request.method == 'POST':
        if 'image' not in request.files:
            return "No se envió archivo"

        file = request.files['image']
        if file.filename == '':
            return "Ningún archivo seleccionado"

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        emociones_detectadas, detalles = procesar_imagen(filepath)
        imagen_filename = file.filename

    return render_template('index.html', emociones=emociones_detectadas, detalles=detalles, imagen=imagen_filename)

def procesar_imagen(ruta_imagen):
    imagen = cv2.imread(ruta_imagen)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
    results = face_mesh.process(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))

    emociones_detectadas = []
    valores_medidos = {}
    confianza = {}

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            ih, iw, _ = imagen.shape
            resultado = detectar_microexpresiones(face_landmarks.landmark, (ih, iw), mostrar_detalles=True)
            emociones_detectadas = resultado.get('emociones', [])
            valores_medidos = resultado.get('valores', {})
            confianza = resultado.get('confianza', {})

        # Guardar resultados
        texto_emocion = ", ".join(emociones_detectadas) if emociones_detectadas else "Neutral"
        datos_guardar = {
            'Hora': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            'Imagen': [os.path.basename(ruta_imagen)],
            'Emociones': [texto_emocion],
            'Apertura_Boca': [valores_medidos.get('apertura_boca', 0)],
            'Anchura_Boca': [valores_medidos.get('anchura_boca', 0)],
            'Elevacion_Cejas': [valores_medidos.get('elevacion_cejas', 0)]
        }

        archivo_csv = "data/emociones_imagen.csv"
        if os.path.exists(archivo_csv):
            df_existente = pd.read_csv(archivo_csv)
            df_nuevo = pd.DataFrame(datos_guardar)
            df_total = pd.concat([df_existente, df_nuevo], ignore_index=True)
        else:
            df_total = pd.DataFrame(datos_guardar)
        df_total.to_csv(archivo_csv, index=False)

        return texto_emocion, valores_medidos

    return "No se detectó rostro", {}

if __name__ == '__main__':
    app.run(debug=True)
