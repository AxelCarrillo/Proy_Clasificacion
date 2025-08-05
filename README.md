# Proy_Clasificacion

## Materia:
Extracción de conocimientos en bases de datos

## Docente:
Salvador Hernández Mendoza

## Alumnos:
Carlos Axel Carrillo Rocha, Brandon Castro Morales, Gustavo Del Razo Rivera, Brallan Josue Tolentino Velasco

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

# Paso 6: Algoritmo de Clasificación
Implementaron lógica para detectar:

Sonrisa forzada: Análisis de simetría labial
Nerviosismo: Tensión en músculos faciales, ojos semi abierto
Sorpresa: Elevación de cejas y apertura ocular 
Estado neutral: Valores dentro de rangos normales sin expreciones

# Paso 7: Migración de Terminal a Web

Versión inicial: Sistema de menú en terminal con teclas, eliguiendo las fotos con numeros
Evolución: Interfaz web con Flask para mayor usabilidad
Funcionalidad: Subida de archivos y visualización de resultados de como esta la persona, definiendo sus expreciones. 

# Paso 8: Sistema de Almacenamiento

Guardado automático en data/emociones_imagen.csv
Campos registrados: Hora, Imagen, Emociones, Métricas numéricas
Historial de tocas las fotos o imagenes enviadas y analisaadas.

# Paso 9: Pipeline de Procesamiento

Carga de imagen: Usuario sube su imagen 
Conversión de color: BGR a RGB para MediaPipe
Detección facial: Face Mesh identifica 468 landmarks
Cálculo de métricas: Distancias y proporciones clave
Clasificación: Algoritmo determina las  emociones
Almacenamiento: Resultados guardados en CSV.

# Paso 10: Métricas de Confianza para deteccion

Sistema de puntuación para cada emoción detectada
Valores numéricos específicos para cada métrica facial
Registro de detalles para análisis posterior.

# Paso 11: Optimización del Algoritmo

Ajuste de umbrales para cada tipo de expresión
Mejora en la precisión de detección porque, si no detectaba casi siempre lo mismo
Manejo de problemas (rostros parciales, ángulos, iluminación)

# Paso 12: Casos de Uso Específicos

Entrevistas laborales: Detección de nerviosismo
Análisis de honestidad: Identificación de microexpresiones
Estudios de comportamiento: Datos de los comportamients que realizan con el rostro.

# Paso 13: Interfaz de Usuario

Formulario simple para subir imágenes
Visualización inmediata de resultados 
Opción de descarga de datos históricos para posteriores analisis.

# Paso 14: Pruebas

Se llevaron acabo distintas pruebas con angulos
Imagenes de diferentes personas para detectar sus expreciones
Diferentes tipos de expreciones para el analisis del sistema.
