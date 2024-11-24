import cv2
import mediapipe as mp
import numpy as np
import pickle

# Cargar el modelo de predicción
with open('best_xgb_model.pkl', 'rb') as file:
    action_model = pickle.load(file)

# Inicializar MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Función para calcular el ángulo entre tres puntos
def calculate_angle(p1, p2, p3):
    """
    Calculate the angle at p2 formed by the line segments p1-p2 and p3-p2.
    Arguments:
    - p1, p2, p3: Arrays or lists containing the coordinates [x, y, z].
    Returns:
    - Angle in degrees.
    """
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)

    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)

    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return None

    angle_radians = np.arccos(dot_product / (magnitude_v1 * magnitude_v2))
    angle_degrees = np.degrees(angle_radians)
    return angle_degrees

# Función para normalizar las coordenadas
def normalize_landmarks(landmarks, width, height):
    """
    Normaliza las coordenadas x, y de los landmarks en relación con el ancho y alto de la imagen.
    """
    normalized_landmarks = []
    for lm in landmarks:
        normalized_landmarks.append([
            lm.x * width,  # Escalar x
            lm.y * height, # Escalar y
            lm.z           # La profundidad se deja sin escalar
        ])
    return normalized_landmarks

# Inicializar captura de video
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar el video.")
        break

    height, width, _ = frame.shape  # Obtener dimensiones del marco

    # Convertir a RGB para MediaPipe
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Procesar landmarks y calcular ángulos
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Normalizar landmarks
        normalized_landmarks = normalize_landmarks(landmarks, width, height)

        # Extraer coordenadas relevantes
        keypoints = {
            "LEFT_HIP": normalized_landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
            "LEFT_KNEE": normalized_landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
            "LEFT_ANKLE": normalized_landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value],
            "RIGHT_HIP": normalized_landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
            "RIGHT_KNEE": normalized_landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
            "RIGHT_ANKLE": normalized_landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value],
        }

        # Calcular ángulos de las rodillas
        left_knee_angle = calculate_angle(
            keypoints["LEFT_HIP"], keypoints["LEFT_KNEE"], keypoints["LEFT_ANKLE"]
        )
        right_knee_angle = calculate_angle(
            keypoints["RIGHT_HIP"], keypoints["RIGHT_KNEE"], keypoints["RIGHT_ANKLE"]
        )

        # Preparar características para el modelo
        row = []
        for lm in normalized_landmarks:
            row.extend(lm)
        if left_knee_angle is not None and right_knee_angle is not None:
            row.extend([left_knee_angle, right_knee_angle])  # Añadir ángulos como características
        else:
            continue  # Skip the frame if angles are not valid

        # Realizar predicción
        X_input = np.array(row).reshape(1, -1)
        try:
            y_pred = action_model.predict(X_input)
            action = y_pred[0]
        except Exception as e:
            action = "Error"

        # Mostrar información en el video
        cv2.putText(frame, f"Action: {action}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Left Knee Angle: {left_knee_angle:.2f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"Right Knee Angle: {right_knee_angle:.2f}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Dibujar landmarks en la imagen
    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Mostrar el video en tiempo real
    cv2.imshow('Real-Time Action Detection', frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
pose.close()
