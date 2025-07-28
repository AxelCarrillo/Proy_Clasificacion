import numpy as np
import cv2
import mediapipe as mp
import pandas as pd
import time
from datetime import datetime

def distancia(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# Índices de landmarks más precisos para MediaPipe Face Mesh
class FacialLandmarks:
    # Ojos
    LEFT_EYE_TOP = 159
    LEFT_EYE_BOTTOM = 145
    LEFT_EYE_LEFT = 33
    LEFT_EYE_RIGHT = 133
    
    RIGHT_EYE_TOP = 386
    RIGHT_EYE_BOTTOM = 374
    RIGHT_EYE_LEFT = 362
    RIGHT_EYE_RIGHT = 263
    
    # Cejas
    LEFT_EYEBROW_INNER = 70
    LEFT_EYEBROW_OUTER = 107
    LEFT_EYEBROW_TOP = 55
    
    RIGHT_EYEBROW_INNER = 300
    RIGHT_EYEBROW_OUTER = 336
    RIGHT_EYEBROW_TOP = 285
    
    # Boca
    MOUTH_TOP = 13
    MOUTH_BOTTOM = 14
    MOUTH_LEFT = 78
    MOUTH_RIGHT = 308
    MOUTH_TOP_LIP = 12
    MOUTH_BOTTOM_LIP = 15
    
    # Nariz
    NOSE_TIP = 1
    NOSE_BRIDGE = 6
    
    # Mejillas
    LEFT_CHEEK = 116
    RIGHT_CHEEK = 345

class EmotionDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Calibración inicial para normalizar medidas
        self.face_width_baseline = None
        self.calibration_frames = 0
        self.max_calibration_frames = 30
        
    def calibrar_rostro(self, landmarks, shape):
        """Calibra las medidas base del rostro para normalización"""
        ih, iw = shape
        
        # Usar la distancia entre las esquinas externas de los ojos como referencia
        left_eye_corner = landmarks[FacialLandmarks.LEFT_EYE_LEFT]
        right_eye_corner = landmarks[FacialLandmarks.RIGHT_EYE_RIGHT]
        
        face_width = distancia(
            (left_eye_corner.x * iw, left_eye_corner.y * ih),
            (right_eye_corner.x * iw, right_eye_corner.y * ih)
        )
        
        if self.face_width_baseline is None:
            self.face_width_baseline = face_width
        else:
            # Promedio móvil para estabilizar
            self.face_width_baseline = (self.face_width_baseline * 0.9 + face_width * 0.1)
        
        self.calibration_frames += 1
        return self.calibration_frames >= self.max_calibration_frames
    
    def detectar_emociones(self, landmarks, shape):
        """Detecta múltiples emociones con mayor precisión"""
        if not self.calibrar_rostro(landmarks, shape):
            return ["Calibrando..."]
        
        ih, iw = shape
        emociones = []
        confianza = {}
        
        # Factor de normalización basado en el ancho del rostro
        norm_factor = self.face_width_baseline / 100.0 if self.face_width_baseline else 1.0
        
        # 1. SORPRESA - Apertura de boca y elevación de cejas
        mouth_top = landmarks[FacialLandmarks.MOUTH_TOP]
        mouth_bottom = landmarks[FacialLandmarks.MOUTH_BOTTOM]
        apertura_boca = distancia(
            (mouth_top.x * iw, mouth_top.y * ih),
            (mouth_bottom.x * iw, mouth_bottom.y * ih)
        ) / norm_factor
        
        # Elevación de cejas
        left_brow = landmarks[FacialLandmarks.LEFT_EYEBROW_TOP]
        right_brow = landmarks[FacialLandmarks.RIGHT_EYEBROW_TOP]
        left_eye_top = landmarks[FacialLandmarks.LEFT_EYE_TOP]
        right_eye_top = landmarks[FacialLandmarks.RIGHT_EYE_TOP]
        
        elevacion_cejas = (
            distancia((left_brow.x * iw, left_brow.y * ih), (left_eye_top.x * iw, left_eye_top.y * ih)) +
            distancia((right_brow.x * iw, right_brow.y * ih), (right_eye_top.x * iw, right_eye_top.y * ih))
        ) / (2 * norm_factor)
        
        if apertura_boca > 15 and elevacion_cejas > 18:
            emociones.append("Sorpresa")
            confianza["Sorpresa"] = min(95, (apertura_boca + elevacion_cejas) * 2)
        
        # 2. FELICIDAD - Sonrisa genuina vs forzada
        mouth_left = landmarks[FacialLandmarks.MOUTH_LEFT]
        mouth_right = landmarks[FacialLandmarks.MOUTH_RIGHT]
        ancho_sonrisa = distancia(
            (mouth_left.x * iw, mouth_left.y * ih),
            (mouth_right.x * iw, mouth_right.y * ih)
        ) / norm_factor
        
        # Curvatura de la boca (esquinas hacia arriba)
        mouth_top_lip = landmarks[FacialLandmarks.MOUTH_TOP_LIP]
        curvatura_boca = (mouth_top_lip.y - (mouth_left.y + mouth_right.y) / 2) * ih / norm_factor
        
        # Activación de músculos alrededor de los ojos (sonrisa genuina)
        left_eye_height = distancia(
            (landmarks[FacialLandmarks.LEFT_EYE_TOP].x * iw, landmarks[FacialLandmarks.LEFT_EYE_TOP].y * ih),
            (landmarks[FacialLandmarks.LEFT_EYE_BOTTOM].x * iw, landmarks[FacialLandmarks.LEFT_EYE_BOTTOM].y * ih)
        ) / norm_factor
        
        if ancho_sonrisa > 45 and curvatura_boca < -2:
            if left_eye_height < 8:  # Ojos entrecerrados por sonrisa genuina
                emociones.append("Felicidad genuina")
                confianza["Felicidad genuina"] = min(90, ancho_sonrisa + abs(curvatura_boca) * 10)
            else:
                emociones.append("Sonrisa forzada")
                confianza["Sonrisa forzada"] = min(85, ancho_sonrisa)
        
        # 3. TENSIÓN/ESTRÉS - Múltiples indicadores
        tension_score = 0
        
        # Fruncimiento de cejas
        distancia_cejas = distancia(
            (landmarks[FacialLandmarks.LEFT_EYEBROW_INNER].x * iw, landmarks[FacialLandmarks.LEFT_EYEBROW_INNER].y * ih),
            (landmarks[FacialLandmarks.RIGHT_EYEBROW_INNER].x * iw, landmarks[FacialLandmarks.RIGHT_EYEBROW_INNER].y * ih)
        ) / norm_factor
        
        if distancia_cejas < 35:
            tension_score += 30
        
        # Tensión en la mandíbula (boca apretada)
        if apertura_boca < 3 and ancho_sonrisa < 35:
            tension_score += 25
        
        # Elevación excesiva de cejas (ansiedad)
        if elevacion_cejas > 25:
            tension_score += 20
        
        # Asimetría facial (indicador de tensión)
        left_cheek = landmarks[FacialLandmarks.LEFT_CHEEK]
        right_cheek = landmarks[FacialLandmarks.RIGHT_CHEEK]
        nose_tip = landmarks[FacialLandmarks.NOSE_TIP]
        
        asimetria = abs(
            distancia((left_cheek.x * iw, left_cheek.y * ih), (nose_tip.x * iw, nose_tip.y * ih)) -
            distancia((right_cheek.x * iw, right_cheek.y * ih), (nose_tip.x * iw, nose_tip.y * ih))
        ) / norm_factor
        
        if asimetria > 3:
            tension_score += 15
        
        if tension_score > 40:
            emociones.append("Tensión/Estrés")
            confianza["Tensión/Estrés"] = min(95, tension_score)
        
        # 4. ENOJO - Cejas bajas y fruncidas
        if elevacion_cejas < 10 and distancia_cejas < 30 and apertura_boca < 5:
            emociones.append("Enojo")
            confianza["Enojo"] = min(90, (40 - distancia_cejas) * 2)
        
        # 5. TRISTEZA - Esquinas de la boca hacia abajo
        if curvatura_boca > 2 and ancho_sonrisa < 40:
            emociones.append("Tristeza")
            confianza["Tristeza"] = min(85, curvatura_boca * 15)
        
        # 6. CONCENTRACIÓN - Cejas ligeramente fruncidas, boca neutra
        if 30 < distancia_cejas < 38 and 3 < apertura_boca < 8 and 35 < ancho_sonrisa < 45:
            emociones.append("Concentración")
            confianza["Concentración"] = 70
        
        return emociones, confianza

def main():
    detector = EmotionDetector()
    cap = cv2.VideoCapture(0)
    
    # Configurar ventana
    cv2.namedWindow("Detector de Expresiones Avanzado", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Detector de Expresiones Avanzado", 1000, 700)
    
    parpadeos = 0
    tiempo_inicio = time.time()
    resultados = []
    frame_count = 0
    
    # Variables para suavizado de detección
    historial_emociones = []
    max_historial = 5
    
    print("🎯 Detector de Expresiones Avanzado iniciado")
    print("📍 Mantén tu rostro centrado para calibrar...")
    print("🔧 Presiona ESC para salir")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)  # Espejo para mejor UX
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.face_mesh.process(frame_rgb)
        
        frame_count += 1
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                ih, iw, _ = frame.shape
                landmarks = face_landmarks.landmark
                
                # Detectar parpadeos
                l_eye_h = distancia(
                    (landmarks[FacialLandmarks.LEFT_EYE_TOP].x * iw, landmarks[FacialLandmarks.LEFT_EYE_TOP].y * ih),
                    (landmarks[FacialLandmarks.LEFT_EYE_BOTTOM].x * iw, landmarks[FacialLandmarks.LEFT_EYE_BOTTOM].y * ih)
                )
                r_eye_h = distancia(
                    (landmarks[FacialLandmarks.RIGHT_EYE_TOP].x * iw, landmarks[FacialLandmarks.RIGHT_EYE_TOP].y * ih),
                    (landmarks[FacialLandmarks.RIGHT_EYE_BOTTOM].x * iw, landmarks[FacialLandmarks.RIGHT_EYE_BOTTOM].y * ih)
                )
                
                eye_avg = (l_eye_h + r_eye_h) / 2
                if eye_avg < 4:
                    parpadeos += 1
                    time.sleep(0.1)
                
                # Detectar emociones
                resultado_emociones = detector.detectar_emociones(landmarks, (ih, iw))
                
                if len(resultado_emociones) == 2:
                    emociones_detectadas, confianza = resultado_emociones
                else:
                    emociones_detectadas = resultado_emociones
                    confianza = {}
                
                # Suavizar detecciones
                historial_emociones.append(emociones_detectadas)
                if len(historial_emociones) > max_historial:
                    historial_emociones.pop(0)
                
                # Emociones más frecuentes en el historial
                todas_emociones = [emo for frame_emos in historial_emociones for emo in frame_emos]
                emociones_frecuentes = list(set([emo for emo in todas_emociones if todas_emociones.count(emo) >= 2]))
                
                if not emociones_frecuentes:
                    emociones_frecuentes = ["Neutral"]
                
                # Dibujar landmarks (opcional, comentar para mejor rendimiento)
                # detector.mp_drawing.draw_landmarks(frame, face_landmarks, detector.mp_face_mesh.FACEMESH_CONTOURS)
                
                # Mostrar información en pantalla con mejor diseño
                y_offset = 40
                for i, emotion in enumerate(emociones_frecuentes):
                    conf_text = f" ({confianza.get(emotion, 'N/A')}%)" if emotion in confianza else ""
                    texto = f"{emotion}{conf_text}"
                    
                    # Colores según emoción
                    color = (0, 255, 0)  # Verde por defecto
                    if "Tensión" in emotion or "Estrés" in emotion:
                        color = (0, 0, 255)  # Rojo
                    elif "Felicidad" in emotion:
                        color = (0, 255, 255)  # Amarillo
                    elif "Sorpresa" in emotion:
                        color = (255, 0, 255)  # Magenta
                    elif "Enojo" in emotion:
                        color = (0, 0, 200)  # Rojo oscuro
                    elif "Tristeza" in emotion:
                        color = (128, 128, 128)  # Gris
                    
                    cv2.putText(frame, texto, (20, y_offset + i * 30), 
                               cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)
                
                # Barra de estado
                status_color = (0, 255, 0) if detector.calibration_frames >= detector.max_calibration_frames else (0, 255, 255)
                status_text = "✅ Calibrado" if detector.calibration_frames >= detector.max_calibration_frames else f"📊 Calibrando... {detector.calibration_frames}/{detector.max_calibration_frames}"
                cv2.putText(frame, status_text, (20, ih - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        else:
            cv2.putText(frame, "❌ No se detecta rostro", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)
        
        # Información general
        cv2.putText(frame, f"Parpadeos: {parpadeos}", (20, ih - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Frame: {frame_count}", (20, ih - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow("Detector de Expresiones Avanzado", frame)
        
        # Guardar datos cada 10 segundos
        if time.time() - tiempo_inicio >= 10 and detector.calibration_frames >= detector.max_calibration_frames:
            frecuencia_parpadeos = parpadeos / 10
            texto_emociones = ", ".join(emociones_frecuentes) if emociones_frecuentes else "Neutral"
            
            # Evaluación más sofisticada
            estado = "Tranquilo"
            if frecuencia_parpadeos > 2.5:
                estado = "Nervioso"
            elif "Tensión" in texto_emociones or "Estrés" in texto_emociones:
                estado = "Estresado"
            elif "Felicidad" in texto_emociones:
                estado = "Positivo"
            elif "Enojo" in texto_emociones:
                estado = "Agitado"
            
            resultado = [
                datetime.now().strftime('%H:%M:%S'),
                parpadeos,
                round(frecuencia_parpadeos, 2),
                texto_emociones,
                estado
            ]
            resultados.append(resultado)
            
            print(f"📊 [{resultado[0]}] {parpadeos} parpadeos (freq: {frecuencia_parpadeos:.1f}/s) - {texto_emociones} - Estado: {estado}")
            
            parpadeos = 0
            tiempo_inicio = time.time()
        
        if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Guardar resultados
    if resultados:
        df = pd.DataFrame(resultados, columns=["Hora", "Parpadeos", "Frecuencia", "Emociones", "Evaluación"])
        
        # Crear directorio si no existe
        import os
        os.makedirs("data", exist_ok=True)
        
        filename = f"data/emociones_entrevista_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(filename, index=False)
        print(f"\n✅ Datos guardados en: {filename}")
        print(f"📈 Total de registros: {len(resultados)}")
    else:
        print("\n⚠️  No se guardaron datos (sesión muy corta)")

if __name__ == "__main__":
    main()