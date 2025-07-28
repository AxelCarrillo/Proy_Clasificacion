import numpy as np
import cv2

def distancia(p1, p2):
    """Calcula la distancia euclidiana entre dos puntos"""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def normalizar_coordenadas(landmarks, ancho_img, alto_img):
    """Convierte coordenadas normalizadas de MediaPipe a píxeles"""
    coordenadas = []
    for landmark in landmarks:
        x = int(landmark.x * ancho_img)
        y = int(landmark.y * alto_img)
        coordenadas.append((x, y))
    return coordenadas

def detectar_microexpresiones(landmarks, shape, mostrar_detalles=False):
    """
    Detecta microexpresiones en una imagen estática
    
    Args:
        landmarks: Puntos faciales detectados por MediaPipe (face_landmarks.landmark)
        shape: Tupla (altura, ancho) de la imagen
        mostrar_detalles: Si True, muestra valores numéricos para debug
    
    Returns:
        dict: Diccionario con emociones detectadas y sus valores
    """
    alto_img, ancho_img = shape
    resultados = {
        'emociones': [],
        'valores': {},
        'confianza': {}
    }
    
    # DEBUG: Verificar que lleguen los landmarks
    if mostrar_detalles:
        print(f"Número de landmarks recibidos: {len(landmarks)}")
        print(f"Dimensiones imagen: {ancho_img}x{alto_img}")
    
    # Verificar que tenemos landmarks válidos
    if not landmarks or len(landmarks) < 468:
        print("ERROR: No se recibieron landmarks válidos o están incompletos")
        resultados['emociones'].append("Error: Sin landmarks")
        return resultados

    # Índices de puntos clave para MediaPipe Face Mesh
    INDICES_BOCA = {
        'superior': 13,
        'inferior': 14,
        'izquierdo': 78,
        'derecho': 308,
        'comisuras': [61, 291]  # Comisuras de la boca
    }
    
    INDICES_OJOS = {
        'ojo_izq_interior': 133,
        'ojo_izq_exterior': 33,
        'ojo_der_interior': 362,
        'ojo_der_exterior': 263,
        'parpado_sup_izq': 159,
        'parpado_inf_izq': 145,
        'parpado_sup_der': 386,
        'parpado_inf_der': 374
    }
    
    INDICES_CEJAS = {
        'ceja_izq_interior': 70,
        'ceja_izq_exterior': 46,
        'ceja_der_interior': 300,
        'ceja_der_exterior': 276
    }

    try:
        # === ANÁLISIS DE LA BOCA ===
        
        # 1. Apertura vertical de la boca (asombro)
        boca_sup = landmarks[13]  # MOUTH_TOP
        boca_inf = landmarks[14]  # MOUTH_BOTTOM
        
        # Convertir coordenadas normalizadas a píxeles
        x1, y1 = boca_sup.x * ancho_img, boca_sup.y * alto_img
        x2, y2 = boca_inf.x * ancho_img, boca_inf.y * alto_img
        
        apertura_vertical = abs(y2 - y1)  # Usar valor absoluto
        resultados['valores']['apertura_boca'] = apertura_vertical
        
        if mostrar_detalles:
            print(f"Apertura boca: {apertura_vertical:.2f} píxeles")
        
        # Detectar asombro
        if apertura_vertical > 8:
            resultados['emociones'].append("Asombro")
            confianza = min(apertura_vertical / 20, 1.0)
            resultados['confianza']['Asombro'] = confianza
            if mostrar_detalles:
                print(f"ASOMBRO detectado con confianza: {confianza:.2f}")
        
        # 2. Anchura de la boca
        boca_izq = landmarks[78]   # MOUTH_LEFT
        boca_der = landmarks[308]  # MOUTH_RIGHT
        
        x1, y1 = boca_izq.x * ancho_img, boca_izq.y * alto_img
        x2, y2 = boca_der.x * ancho_img, boca_der.y * alto_img
        
        anchura_boca = abs(x2 - x1)
        resultados['valores']['anchura_boca'] = anchura_boca
        
        if mostrar_detalles:
            print(f"Anchura boca: {anchura_boca:.2f} píxeles")
        
        # === ANÁLISIS DE CEJAS ===
        
        # 3. Elevación de cejas
        ceja_izq = landmarks[70]    # EYEBROW_LEFT
        ojo_ref = landmarks[133]    # EYE_LEFT_INNER
        
        x1, y1 = ceja_izq.x * ancho_img, ceja_izq.y * alto_img
        x2, y2 = ojo_ref.x * ancho_img, ojo_ref.y * alto_img
        
        elevacion_cejas = abs(y1 - y2)  # Distancia vertical
        resultados['valores']['elevacion_cejas'] = elevacion_cejas
        
        if mostrar_detalles:
            print(f"Elevación cejas: {elevacion_cejas:.2f} píxeles")
        
        # Detectar tensión/nerviosismo por cejas elevadas
        if elevacion_cejas > 15:
            resultados['emociones'].append("Tension")
            resultados['confianza']['Tension'] = 0.6
            if mostrar_detalles:
                print("TENSION detectada por cejas elevadas")
        
        # Detectar asombro intenso (cejas + boca)
        if elevacion_cejas > 20 and apertura_vertical > 12:
            if "Asombro intenso" not in resultados['emociones']:
                resultados['emociones'].append("Asombro intenso")
                resultados['confianza']['Asombro intenso'] = 0.8
                if mostrar_detalles:
                    print("ASOMBRO INTENSO detectado")
        
        # === ANÁLISIS DE SONRISA Y FELICIDAD ===
        
        # Detectar diferentes tipos de sonrisa/felicidad
        if anchura_boca > 35:
            # Verificar curvatura básica comparando comisuras
            comisura_izq = landmarks[61]   # Comisura izquierda
            comisura_der = landmarks[291]  # Comisura derecha
            
            altura_comisuras = (comisura_izq.y + comisura_der.y) / 2
            altura_centro = (boca_sup.y + boca_inf.y) / 2
            
            curvatura = (altura_centro - altura_comisuras) * alto_img
            resultados['valores']['curvatura_boca'] = curvatura
            
            # Análisis de ojos para sonrisa genuina
            ojo_izq_sup = landmarks[159]  # Párpado superior izquierdo
            ojo_izq_inf = landmarks[145]  # Párpado inferior izquierdo
            apertura_ojo = abs((ojo_izq_sup.y - ojo_izq_inf.y) * alto_img)
            resultados['valores']['apertura_ojo'] = apertura_ojo
            
            if curvatura > 3 and apertura_ojo < 8:
                resultados['emociones'].append("Feliz")
                resultados['confianza']['Feliz'] = 0.85
                if mostrar_detalles:
                    print(f"FELICIDAD detectada - curvatura: {curvatura:.2f}, ojos: {apertura_ojo:.2f}")
            elif curvatura > 1:
                resultados['emociones'].append("Contento")
                resultados['confianza']['Contento'] = 0.7
                if mostrar_detalles:
                    print(f"CONTENTO detectado - curvatura: {curvatura:.2f}")
        
        # === ANÁLISIS DE ENOJO ===
        
        # Detectar cejas fruncidas (enojo)
        ceja_der = landmarks[300]  # Ceja derecha interior
        ojo_der_ref = landmarks[362]  # Ojo derecho interior
        
        x1, y1 = ceja_der.x * ancho_img, ceja_der.y * alto_img
        x2, y2 = ojo_der_ref.x * ancho_img, ojo_der_ref.y * alto_img
        
        distancia_ceja_der = abs(y1 - y2)
        
        # Promedio de ambas cejas para mejor precisión
        elevacion_cejas_promedio = (elevacion_cejas + distancia_ceja_der) / 2
        resultados['valores']['elevacion_cejas_promedio'] = elevacion_cejas_promedio
        
        # Detectar labios apretados (señal de enojo)
        labio_sup_centro = landmarks[12]  # Labio superior centro
        labio_inf_centro = landmarks[15]  # Labio inferior centro
        
        grosor_labios = abs((labio_sup_centro.y - labio_inf_centro.y) * alto_img)
        resultados['valores']['grosor_labios'] = grosor_labios
        
        # Enojo: cejas bajas + labios apretados + boca no sonriente
        if elevacion_cejas_promedio < 12 and grosor_labios < 3 and curvatura < 0:
            resultados['emociones'].append("Enojado") 
            resultados['confianza']['Enojado'] = 0.75
            if mostrar_detalles:
                print(f"ENOJO detectado - cejas: {elevacion_cejas_promedio:.2f}, labios: {grosor_labios:.2f}")
        
        # === ANÁLISIS DE NERVIOSISMO ===
        
        # Nerviosismo: múltiples indicadores
        indicadores_nervios = 0
        
        # 1. Cejas ligeramente elevadas (tensión)
        if 15 < elevacion_cejas_promedio < 25:
            indicadores_nervios += 1
            
        # 2. Boca ligeramente abierta (respiración ansiosa)
        if 3 < apertura_vertical < 10:
            indicadores_nervios += 1
            
        # 3. Ojos ligeramente más abiertos (alerta)
        if apertura_ojo > 9:
            indicadores_nervios += 1
            
        # 4. Labios no completamente relajados
        if 2 < grosor_labios < 5:
            indicadores_nervios += 1
            
        resultados['valores']['indicadores_nervios'] = indicadores_nervios
        
        if indicadores_nervios >= 2:
            if indicadores_nervios >= 3:
                resultados['emociones'].append("Muy nervioso")
                resultados['confianza']['Muy nervioso'] = 0.8
            else:
                resultados['emociones'].append("Nervioso")
                resultados['confianza']['Nervioso'] = 0.6
            
            if mostrar_detalles:
                print(f"NERVIOSISMO detectado - indicadores: {indicadores_nervios}/4")
        
        # === ANÁLISIS DE TRISTEZA ===
        
        # Detectar comisuras hacia abajo (tristeza)
        if 'curvatura_boca' in resultados['valores']:
            curvatura = resultados['valores']['curvatura_boca']
            
            # Tristeza: comisuras hacia abajo + cejas ligeramente caídas
            if curvatura < -1 and elevacion_cejas_promedio < 18:
                if curvatura < -3:
                    resultados['emociones'].append("Muy triste")
                    resultados['confianza']['Muy triste'] = 0.75
                else:
                    resultados['emociones'].append("Triste")
                    resultados['confianza']['Triste'] = 0.65
                    
                if mostrar_detalles:
                    print(f"TRISTEZA detectada - curvatura: {curvatura:.2f}")
        
        # === ANÁLISIS DE SORPRESA/MIEDO ===
        
        # Distinguir entre asombro positivo y miedo
        if apertura_vertical > 8:
            # Miedo: boca abierta + cejas muy elevadas + ojos muy abiertos
            if elevacion_cejas_promedio > 25 and apertura_ojo > 12:
                resultados['emociones'].append("Miedo")
                resultados['confianza']['Miedo'] = 0.7
                if mostrar_detalles:
                    print("MIEDO detectado por combinación extrema")
        
        # === ANÁLISIS DE DISGUSTO ===
        
        # Disgusto: nariz arrugada + labio superior elevado
        nariz_tip = landmarks[1]   # Punta de la nariz
        labio_sup_izq = landmarks[37]  # Labio superior izquierdo
        
        elevacion_labio_sup = abs((nariz_tip.y - labio_sup_izq.y) * alto_img)
        resultados['valores']['elevacion_labio_sup'] = elevacion_labio_sup
        
        if elevacion_labio_sup < 15 and curvatura < -1 and grosor_labios < 4:
            resultados['emociones'].append("Disgusto")
            resultados['confianza']['Disgusto'] = 0.65
            if mostrar_detalles:
                print(f"DISGUSTO detectado - elevación labio: {elevacion_labio_sup:.2f}")
        
        # === ANÁLISIS DE CONCENTRACIÓN/DETERMINACIÓN ===
        
        if (12 < elevacion_cejas_promedio < 18 and 
            apertura_vertical < 5 and 
            3 < grosor_labios < 6):
            resultados['emociones'].append("Concentrado")
            resultados['confianza']['Concentrado'] = 0.6
            if mostrar_detalles:
                print("CONCENTRACIÓN detectada")
        
    except (IndexError, AttributeError) as e:
        print(f"ERROR al analizar landmarks: {e}")
        resultados['emociones'].append("Error en análisis")
        return resultados
    
    # Si no se detectaron emociones específicas, evaluar como neutral
    if not resultados['emociones'] or resultados['emociones'] == ["Error en análisis"]:
        # Evaluar qué tan neutral está realmente
        neutralidad = 0
        factores_neutralidad = []
        
        if 'apertura_boca' in resultados['valores']:
            if resultados['valores']['apertura_boca'] < 8:
                neutralidad += 25
                factores_neutralidad.append("boca relajada")
                
        if 'elevacion_cejas_promedio' in resultados['valores']:
            if 12 <= resultados['valores']['elevacion_cejas_promedio'] <= 20:
                neutralidad += 25
                factores_neutralidad.append("cejas normales")
                
        if 'curvatura_boca' in resultados['valores']:
            if -1 <= resultados['valores']['curvatura_boca'] <= 1:
                neutralidad += 25
                factores_neutralidad.append("expresión equilibrada")
                
        if 'grosor_labios' in resultados['valores']:
            if resultados['valores']['grosor_labios'] > 3:
                neutralidad += 25
                factores_neutralidad.append("labios relajados")
        
        if neutralidad >= 75:
            resultados['emociones'] = ["Expresión neutra"]
            resultados['confianza']['Expresión neutra'] = neutralidad / 100
            if mostrar_detalles:
                print(f"NEUTRALIDAD detectada ({neutralidad}%) - factores: {factores_neutralidad}")
        else:
            resultados['emociones'] = ["Expresión ambigua"]
            resultados['confianza']['Expresión ambigua'] = 0.3
    
    # Mostrar detalles para debugging
    if mostrar_detalles:
        print(f"=== VALORES DE ANÁLISIS ===")
        for clave, valor in resultados['valores'].items():
            print(f"{clave}: {valor:.2f}")
        print(f"Emociones detectadas: {resultados['emociones']}")
    
    return resultados

def redimensionar_imagen(imagen, ancho_max=800, alto_max=600):
    """
    Redimensiona la imagen manteniendo la proporción para mostrarla en ventana
    """
    altura, ancho = imagen.shape[:2]
    
    # Calcular factor de escala
    factor_ancho = ancho_max / ancho
    factor_altura = alto_max / altura
    factor_escala = min(factor_ancho, factor_altura, 1.0)  # No agrandar
    
    if factor_escala < 1.0:
        nuevo_ancho = int(ancho * factor_escala)
        nueva_altura = int(altura * factor_escala)
        imagen_redimensionada = cv2.resize(imagen, (nuevo_ancho, nueva_altura), 
                                         interpolation=cv2.INTER_AREA)
        return imagen_redimensionada, factor_escala
    
    return imagen, 1.0

def dibujar_landmarks_clave(imagen, landmarks, mostrar_indices=False):
    """
    Dibuja los puntos clave en la imagen para visualización
    """
    altura, ancho = imagen.shape[:2]
    
    # Puntos clave a resaltar
    puntos_importantes = [13, 14, 78, 308, 61, 291, 133, 33, 362, 263, 
                         159, 145, 386, 374, 70, 46, 300, 276]
    
    for i, punto in enumerate(puntos_importantes):
        if punto < len(landmarks):
            landmark = landmarks[punto]
            x = int(landmark.x * ancho)
            y = int(landmark.y * altura)
            
            # Dibujar punto
            cv2.circle(imagen, (x, y), 2, (0, 255, 0), -1)
            
            # Mostrar índice si se solicita
            if mostrar_indices:
                cv2.putText(imagen, str(punto), (x+3, y-3), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    return imagen

def mostrar_imagen_ajustada(imagen, titulo="Imagen", esperar_tecla=True):
    """
    Muestra la imagen redimensionada en una ventana
    """
    imagen_mostrar, factor = redimensionar_imagen(imagen)
    
    cv2.imshow(titulo, imagen_mostrar)
    
    if esperar_tecla:
        print(f"Imagen redimensionada al {factor*100:.1f}% del tamaño original")
        print("Presiona cualquier tecla para continuar...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return imagen_mostrar