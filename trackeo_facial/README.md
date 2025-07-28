# 🧠 Detección de Microexpresiones Faciales

Este proyecto utiliza visión por computadora y MediaPipe para detectar microexpresiones faciales como nerviosismo, sorpresa o sonrisas forzadas a partir de imágenes estáticas. Los resultados se guardan automáticamente para su análisis posterior con pandas o numpy.

---

## 📂 Estructura del Proyecto

trackeo_facial/
│
├── assets/ # Carpeta con imágenes a analizar (.jpg o .png)
├── data/ # Resultados guardados como CSV
│ └── emociones_imagen.csv
├── scripts/
│ ├── helpers.py # Funciones auxiliares (distancia, detección)
│ └── detector_expresiones.py (si se usa cámara)
├── main.py # Menú para elegir imagen y analizar emociones
├── requirements.txt # Dependencias del proyecto
└── README.md