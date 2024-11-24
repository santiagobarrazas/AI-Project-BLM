import cv2
import mediapipe as mp
import numpy as np
import pickle
import pandas as pd
import math

# Cargar el modelo entrenado desde el archivo .pkl
with open('modelo_xgboost.pkl', 'rb') as f:
    model = pickle.load(f)

# Inicializar MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Inicializar la cámara
cap = cv2.VideoCapture(1)

# Para almacenar el frame anterior y calcular los deltas
prev_values = None

# Definir las columnas de las características (con los ángulos y las posiciones de caderas)
columnas_relevantes = [
    'LEFT_HIP_x', 'LEFT_HIP_y',
    'RIGHT_HIP_x', 'RIGHT_HIP_y','shoulder_distance', 'hip_distance',
    'LEFT_KNEE_ANGLE', 'RIGHT_KNEE_ANGLE',
]

# Función para calcular el ángulo entre tres puntos
def calculate_angle(a, b, c):
    # A, B, C son puntos (x, y)
    ax, ay = a
    bx, by = b
    cx, cy = c
    
    # Calcular las distancias
    ab = np.sqrt((bx - ax) ** 2 + (by - ay) ** 2)
    bc = np.sqrt((cx - bx) ** 2 + (cy - by) ** 2)
    ac = np.sqrt((cx - ax) ** 2 + (cy - ay) ** 2)

    # Calcular el coseno del ángulo utilizando la ley de cosenos
    cos_angle = (ab**2 + bc**2 - ac**2) / (2 * ab * bc)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))  # Restringir valores a [-1, 1] para evitar errores numéricos
    return np.degrees(angle)

# Función para calcular los deltas entre frames (solo para las posiciones)
def calcular_deltas(current_values, prev_values):
    deltas = []
    if prev_values is not None:
        for current, prev in zip(current_values, prev_values):
            deltas.append(current - prev)
    else:
        deltas = current_values  # Si no hay frame anterior, solo toma las posiciones actuales
    return deltas

# Función para calcular la distancia entre los hombros
def calcular_distance(landmarks, left, right):
    left = landmarks[left]
    right = landmarks[right]
    
    x_diff = left.x - right.x
    y_diff = left.y - right.y
    return np.sqrt(x_diff**2 + y_diff**2)

# Función para calcular las posiciones y los ángulos relevantes
def calcular_posiciones_y_angulos(landmarks):
    # Coordenadas de las caderas
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    
    # Coordenadas de las rodillas
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
    right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]

    # Coordenadas de los tobillos (usados para calcular el ángulo de la rodilla)
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

    # Calcular ángulos de las rodillas
    left_knee_angle = calculate_angle(
        (left_hip.x, left_hip.y), (left_knee.x, left_knee.y), (left_ankle.x, left_ankle.y)
    )
    right_knee_angle = calculate_angle(
        (right_hip.x, right_hip.y), (right_knee.x, right_knee.y), (right_ankle.x, right_ankle.y)
    )

    # Distancia entre los hombros
    shoulder_distance = calcular_distance(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER)
    hip_distance = calcular_distance(landmarks, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP)
    
    return {
        "LEFT_HIP_x": left_hip.x, "LEFT_HIP_y": left_hip.y,
        "RIGHT_HIP_x": right_hip.x, "RIGHT_HIP_y": right_hip.y,
        "LEFT_KNEE_ANGLE": left_knee_angle, "RIGHT_KNEE_ANGLE": right_knee_angle,
        "shoulder_distance": shoulder_distance, "hip_distance": hip_distance,
    }

# Bucle para capturar video en tiempo real
prev = None
while True:
    # Capturar un frame de la cámara
    ret, frame = cap.read()
    if not ret:
        break

    

    # Convertir a RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detectar los puntos de la pose con MediaPipe
    results = pose.process(frame_rgb)

    # Verificar si se detectaron los puntos
    if results.pose_landmarks:
        # Calcular las posiciones y ángulos
        current_values = calcular_posiciones_y_angulos(results.pose_landmarks.landmark)

        # Calcular los deltas solo para las posiciones
        deltas = calcular_deltas(list(current_values.values())[:-2], list(prev_values.values())[:-2] if prev_values else None)  # No calculamos delta para los ángulos y shoulder_distance
        prev_values = current_values  # Actualizar el frame anterior con los valores actuales

        # Convertir los deltas en un dataframe (igual que en el entrenamiento)
        delta_df = pd.DataFrame([deltas + [current_values["LEFT_KNEE_ANGLE"], current_values["RIGHT_KNEE_ANGLE"]]],
                                columns=columnas_relevantes)

        # Realizar la predicción con el modelo XGBoost
        prediction = model.predict(delta_df)
        movement = prediction[0]

        # Mostrar el resultado
        movement_dict = {
            0: 'Still', 1: 'Approach', 2: 'Back', 3: 'Jump', 4: 'Sit', 
            5: 'Stand', 6: 'Turn Left', 7: 'Turn Right', 8: 'Walk Left', 9: 'Walk Right'
        }
        if(movement != 0):
            prev = movement_dict[movement]
            cv2.putText(frame, f"Movimiento: {movement_dict[movement]}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, f"Movimiento: {prev}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Dibujar los puntos clave de los hombros en el frame
        h, w, c = frame.shape
        left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        # Convertir las coordenadas (x, y) a píxeles
        left_shoulder_x = int(left_shoulder.x * w)
        left_shoulder_y = int(left_shoulder.y * h)
        right_shoulder_x = int(right_shoulder.x * w)
        right_shoulder_y = int(right_shoulder.y * h)

        # Dibujar círculos en los puntos relevantes
        puntos_relevantes = [
            (left_shoulder_x, left_shoulder_y),
            (right_shoulder_x, right_shoulder_y),
            (int(current_values["LEFT_HIP_x"] * w), int(current_values["LEFT_HIP_y"] * h)),
            (int(current_values["RIGHT_HIP_x"] * w), int(current_values["RIGHT_HIP_y"] * h)),
            (int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x * w), int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y * h)),
            (int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x * w), int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y * h)),
            (int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x * w), int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y * h)),
            (int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x * w), int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y * h))
        ]

        for punto in puntos_relevantes:
            cv2.circle(frame, punto, 5, (0, 0, 255), -1)  # Dibujar cada punto relevante en rojo

    # Mostrar el frame con la predicción y los puntos de los hombros
    cv2.imshow('Pose Detection', frame)

    # Salir del bucle con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
