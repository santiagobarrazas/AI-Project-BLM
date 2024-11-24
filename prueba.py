import cv2
import mediapipe as mp
import numpy as np
import pickle
import pandas as pd

# Load the trained model and scaler
with open('xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize the camera
cap = cv2.VideoCapture(1)

# Store previous frame's values
prev_values = None

# Define feature columns (matching the training script)
feature_columns = [
    'LEFT_KNEE_ANGLE', 'RIGHT_KNEE_ANGLE', 'LEFT_HIP_ANGLE', 'RIGHT_HIP_ANGLE',
    'shoulder_distance', 'hip_distance', 'knee_distance', 'ankle_distance',
    'center_velocity_x', 'center_velocity_y', 'center_acceleration_x', 'center_acceleration_y',
    'LEFT_KNEE_ANGLE_velocity', 'RIGHT_KNEE_ANGLE_velocity',
    'LEFT_HIP_ANGLE_velocity', 'RIGHT_HIP_ANGLE_velocity',
    'step_length', 'body_height_left', 'body_height_right',
    'vertical_displacement', 'vertical_velocity', 'body_rotation',
    'vertical_displacement_rolling_mean', 'vertical_displacement_rolling_std',
    'body_rotation_rolling_mean', 'body_rotation_rolling_std',
    'step_length_rolling_mean', 'step_length_rolling_std', 'hand_distance_x'
]

def calculate_angle(p1, p2, p3):
    """Calculate angle between three points."""
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0
    angle_radians = np.arccos(np.clip(dot_product / (magnitude_v1 * magnitude_v2), -1.0, 1.0))
    return np.degrees(angle_radians)

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def extract_features(landmarks, prev_values, frame_time=0.033):
    """Extract all features from landmarks."""
    # Extract coordinates
    points = {}
    for name in [
        'LEFT_SHOULDER', 'RIGHT_SHOULDER',
        'LEFT_HIP', 'RIGHT_HIP',
        'LEFT_KNEE', 'RIGHT_KNEE',
        'LEFT_ANKLE', 'RIGHT_ANKLE',
        'LEFT_ELBOW', 'RIGHT_ELBOW',
        'LEFT_WRIST', 'RIGHT_WRIST'
    ]:
        landmark = landmarks[getattr(mp_pose.PoseLandmark, name)]
        points[name] = (landmark.x, landmark.y)

    # Calculate joint angles
    features = {
        'LEFT_KNEE_ANGLE': calculate_angle(points['LEFT_HIP'], points['LEFT_KNEE'], points['LEFT_ANKLE']),
        'RIGHT_KNEE_ANGLE': calculate_angle(points['RIGHT_HIP'], points['RIGHT_KNEE'], points['RIGHT_ANKLE']),
        'LEFT_HIP_ANGLE': calculate_angle(points['LEFT_SHOULDER'], points['LEFT_HIP'], points['LEFT_KNEE']),
        'RIGHT_HIP_ANGLE': calculate_angle(points['RIGHT_SHOULDER'], points['RIGHT_HIP'], points['RIGHT_KNEE'])
    }

    # Calculate distances
    features.update({
        'shoulder_distance': calculate_distance(points['LEFT_SHOULDER'], points['RIGHT_SHOULDER']),
        'hip_distance': calculate_distance(points['LEFT_HIP'], points['RIGHT_HIP']),
        'knee_distance': calculate_distance(points['LEFT_KNEE'], points['RIGHT_KNEE']),
        'ankle_distance': calculate_distance(points['LEFT_ANKLE'], points['RIGHT_ANKLE']),
        'step_length': calculate_distance(points['LEFT_ANKLE'], points['RIGHT_ANKLE']),
        'body_height_left': calculate_distance(points['LEFT_SHOULDER'], points['LEFT_ANKLE']),
        'body_height_right': calculate_distance(points['RIGHT_SHOULDER'], points['RIGHT_ANKLE'])
    })

    # Calculate center of mass
    center_x = (points['LEFT_HIP'][0] + points['RIGHT_HIP'][0]) / 2
    center_y = (points['LEFT_HIP'][1] + points['RIGHT_HIP'][1]) / 2
    
    # Calculate body rotation
    features['body_rotation'] = np.degrees(np.arctan2(
        points['RIGHT_SHOULDER'][1] - points['LEFT_SHOULDER'][1],
        points['RIGHT_SHOULDER'][0] - points['LEFT_SHOULDER'][0]
    ) - np.arctan2(
        points['RIGHT_HIP'][1] - points['LEFT_HIP'][1],
        points['RIGHT_HIP'][0] - points['LEFT_HIP'][0]
    ))

    if prev_values is None:
        # Initialize velocities and accelerations with zeros
        features.update({
            'center_velocity_x': 0, 'center_velocity_y': 0,
            'center_acceleration_x': 0, 'center_acceleration_y': 0,
            'LEFT_KNEE_ANGLE_velocity': 0, 'RIGHT_KNEE_ANGLE_velocity': 0,
            'LEFT_HIP_ANGLE_velocity': 0, 'RIGHT_HIP_ANGLE_velocity': 0,
            'vertical_displacement': 0, 'vertical_velocity': 0,
            'vertical_displacement_rolling_mean': 0, 'vertical_displacement_rolling_std': 0,
            'body_rotation_rolling_mean': features['body_rotation'], 'body_rotation_rolling_std': 0,
            'step_length_rolling_mean': features['step_length'], 'step_length_rolling_std': 0,
            'hand_distance_x': 0,
        })
    else:
        # Calculate velocities and accelerations
# Initialize velocities and accelerations with zeros if the keys are missing
        # Calculate center velocities first
        features['center_velocity_x'] = (center_x - prev_values.get('center_x', center_x)) / frame_time
        features['center_velocity_y'] = (center_y - prev_values.get('center_y', center_y)) / frame_time

        # Calculate center accelerations using the newly calculated velocities
        features['center_acceleration_x'] = (features['center_velocity_x'] - prev_values.get('center_velocity_x', 0)) / frame_time
        features['center_acceleration_y'] = (features['center_velocity_y'] - prev_values.get('center_velocity_y', 0)) / frame_time

        features['hand_distance_x'] = points['RIGHT_WRIST'][0] - points['LEFT_WRIST'][0]

        # Calculate remaining features
        features.update({
            'LEFT_KNEE_ANGLE_velocity': (features['LEFT_KNEE_ANGLE'] - prev_values.get('LEFT_KNEE_ANGLE', features['LEFT_KNEE_ANGLE'])) / frame_time,
            'RIGHT_KNEE_ANGLE_velocity': (features['RIGHT_KNEE_ANGLE'] - prev_values.get('RIGHT_KNEE_ANGLE', features['RIGHT_KNEE_ANGLE'])) / frame_time,
            'LEFT_HIP_ANGLE_velocity': (features['LEFT_HIP_ANGLE'] - prev_values.get('LEFT_HIP_ANGLE', features['LEFT_HIP_ANGLE'])) / frame_time,
            'RIGHT_HIP_ANGLE_velocity': (features['RIGHT_HIP_ANGLE'] - prev_values.get('RIGHT_HIP_ANGLE', features['RIGHT_HIP_ANGLE'])) / frame_time,
            'vertical_displacement': center_y - prev_values.get('center_y', center_y),
            'vertical_velocity': (center_y - prev_values.get('center_y', center_y)) / frame_time
        })


        
        # Update rolling statistics (simple moving average for real-time)
        alpha = 0.2  # Smoothing factor
        features.update({
            'vertical_displacement_rolling_mean': alpha * features['vertical_displacement'] + 
                (1 - alpha) * prev_values['vertical_displacement_rolling_mean'],
            'vertical_displacement_rolling_std': np.std([features['vertical_displacement'], 
                prev_values['vertical_displacement_rolling_mean']]),
            'body_rotation_rolling_mean': alpha * features['body_rotation'] + 
                (1 - alpha) * prev_values['body_rotation_rolling_mean'],
            'body_rotation_rolling_std': np.std([features['body_rotation'], 
                prev_values['body_rotation_rolling_mean']]),
            'step_length_rolling_mean': alpha * features['step_length'] + 
                (1 - alpha) * prev_values['step_length_rolling_mean'],
            'step_length_rolling_std': np.std([features['step_length'], 
                prev_values['step_length_rolling_mean']])
        })

    # Store current center values for next frame
    features.update({
        'center_x': center_x,
        'center_y': center_y
    })

    return features

# Movement labels
movement_labels = {
    0: 'Still', 1: 'approach', 2: 'back', 3: 'jump', 
    4: 'turn_left', 5: 'walk_left', 6: 'turn_right', 
    7: 'walk_right', 8: 'sit', 9: 'stand'
}

# Real-time processing loop
prev_movement = "Still"
movement_buffer = []  # For smoothing predictions
buffer_size = 5  # Number of frames to average

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        # Extract features
        landmarks = results.pose_landmarks.landmark
        features = extract_features(landmarks, prev_values)
        prev_values = features

        # Prepare features for prediction
        feature_values = [features[col] for col in feature_columns]
        feature_df = pd.DataFrame([feature_values], columns=feature_columns)
        
        # Scale features
        scaled_features = scaler.transform(feature_df)
        
        # Predict movement
        prediction = model.predict(scaled_features)[0]
        movement = movement_labels.get(prediction, 'Unknown')
        
        # Smooth predictions using a buffer
        movement_buffer.append(movement)
        if len(movement_buffer) > buffer_size:
            movement_buffer.pop(0)
        
        # Get most common movement in buffer
        if movement != 'Still':
            prev_movement = movement
        
        # Display prediction
        cv2.putText(frame, f"Movement: {prev_movement}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw skeleton
        h, w, _ = frame.shape
        for connection in mp_pose.POSE_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]
            
            start_point = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
            end_point = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
            
            cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

        # Draw key points
        for idx in [
            mp_pose.PoseLandmark.LEFT_SHOULDER,
            mp_pose.PoseLandmark.RIGHT_SHOULDER,
            mp_pose.PoseLandmark.LEFT_HIP,
            mp_pose.PoseLandmark.RIGHT_HIP,
            mp_pose.PoseLandmark.LEFT_KNEE,
            mp_pose.PoseLandmark.RIGHT_KNEE,
            mp_pose.PoseLandmark.LEFT_ANKLE,
            mp_pose.PoseLandmark.RIGHT_ANKLE
        ]:
            x = int(landmarks[idx].x * w)
            y = int(landmarks[idx].y * h)
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

    # Show the frame
    cv2.imshow('Enhanced Motion Detection', frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()