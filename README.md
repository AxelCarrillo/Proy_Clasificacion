# Proy_Clasificacion

# Paso 1: Definición de Requisitos

Problema identificado: Necesidad de analizar expresiones faciales en contextos de entrevistas para analizar los comportamiento de la cara.
Tecnología elegida: MediaPipe de Google para detección de landmarks faciales
Tipo de análisis: Procesamiento de imágenes estáticas (.jpg, .png)
Salida esperada: Clasificación de emociones con valores numéricos de confianza.

# Paso 2: Arquitectura del Sistema/Web

Backend: Python con Flask para interfaz de la web
Procesamiento: OpenCV + MediaPipe para análisis facial
Almacenamiento: CSV para guardar los analicis que se han realizado posteriormente
Frontend: Un Html basico para poder subri las imagenes y que se haga el analisis.

# 3: Investigación de MediaPipe

Se investigo mas afondo sobre la libreria de MediaPipi
Se identificaron los landmarks faciales clave para expresiones que nos interesaban en las entrevistas

Definimos metricas para:
Apertura de boca: Distancia vertical entre labios
Anchura de boca: Distancia horizontal de comisuras
Elevación de cejas: Posición vertical de las cejas

# Paso 4: Instalación de Dependencias

Creamos un archivo requirements.txt con las librerías necesarias:
  *mediapipe
  *opencv-python
  *numpy
  *pandas
  *flask

# Paso 5: Creación de Funciones Auxiliares
Funciones clave como:

distancia(): Calcular distancias entre puntos faciales
detectar_microexpresiones(): Algoritmo principal de análisis
mostrar_imagen_ajustada(): Visualización de resultado

