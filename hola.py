import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import pickle

def calculate_angle(p1, p2, p3):
    """Calculate angle between three points."""
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return None
    angle_radians = np.arccos(dot_product / (magnitude_v1 * magnitude_v2))
    return np.degrees(angle_radians)

def calculate_center_point(points):
    """Calculate centroid of multiple points."""
    return np.mean(points, axis=0)

def calculate_velocity(positions, frame_time):
    """Calculate velocity from position differences."""
    return np.gradient(positions, frame_time, axis=0)

def calculate_acceleration(velocities, frame_time):
    """Calculate acceleration from velocities."""
    return np.gradient(velocities, frame_time, axis=0)

# Read input data
input_file = "merged_output.csv"
output_file = "enhanced_motion_features.csv"
frame_time = 0.033  # 30 FPS

df = pd.read_csv(input_file)

# Define key points for calculations
keypoints = {
    'LEFT_SHOULDER': ['LEFT_SHOULDER_x', 'LEFT_SHOULDER_y'],
    'RIGHT_SHOULDER': ['RIGHT_SHOULDER_x', 'RIGHT_SHOULDER_y'],
    'LEFT_HIP': ['LEFT_HIP_x', 'LEFT_HIP_y'],
    'RIGHT_HIP': ['RIGHT_HIP_x', 'RIGHT_HIP_y'],
    'LEFT_KNEE': ['LEFT_KNEE_x', 'LEFT_KNEE_y'],
    'RIGHT_KNEE': ['RIGHT_KNEE_x', 'RIGHT_KNEE_y'],
    'LEFT_ANKLE': ['LEFT_ANKLE_x', 'LEFT_ANKLE_y'],
    'RIGHT_ANKLE': ['RIGHT_ANKLE_x', 'RIGHT_ANKLE_y'],
    'LEFT_ELBOW': ['LEFT_ELBOW_x', 'LEFT_ELBOW_y'],
    'RIGHT_ELBOW': ['RIGHT_ELBOW_x', 'RIGHT_ELBOW_y'],
    'LEFT_WRIST': ['LEFT_WRIST_x', 'LEFT_WRIST_y'],
    'RIGHT_WRIST': ['RIGHT_WRIST_x', 'RIGHT_WRIST_y']
}

# Calculate joint angles
print("Calculating joint angles...")
for row_idx, row in df.iterrows():
    # Knee angles
    df.loc[row_idx, 'LEFT_KNEE_ANGLE'] = calculate_angle(
        [row[keypoints['LEFT_HIP'][0]], row[keypoints['LEFT_HIP'][1]]],
        [row[keypoints['LEFT_KNEE'][0]], row[keypoints['LEFT_KNEE'][1]]],
        [row[keypoints['LEFT_ANKLE'][0]], row[keypoints['LEFT_ANKLE'][1]]]
    )
    df.loc[row_idx, 'RIGHT_KNEE_ANGLE'] = calculate_angle(
        [row[keypoints['RIGHT_HIP'][0]], row[keypoints['RIGHT_HIP'][1]]],
        [row[keypoints['RIGHT_KNEE'][0]], row[keypoints['RIGHT_KNEE'][1]]],
        [row[keypoints['RIGHT_ANKLE'][0]], row[keypoints['RIGHT_ANKLE'][1]]]
    )
    
    # Hip angles
    df.loc[row_idx, 'LEFT_HIP_ANGLE'] = calculate_angle(
        [row[keypoints['LEFT_SHOULDER'][0]], row[keypoints['LEFT_SHOULDER'][1]]],
        [row[keypoints['LEFT_HIP'][0]], row[keypoints['LEFT_HIP'][1]]],
        [row[keypoints['LEFT_KNEE'][0]], row[keypoints['LEFT_KNEE'][1]]]
    )
    df.loc[row_idx, 'RIGHT_HIP_ANGLE'] = calculate_angle(
        [row[keypoints['RIGHT_SHOULDER'][0]], row[keypoints['RIGHT_SHOULDER'][1]]],
        [row[keypoints['RIGHT_HIP'][0]], row[keypoints['RIGHT_HIP'][1]]],
        [row[keypoints['RIGHT_KNEE'][0]], row[keypoints['RIGHT_KNEE'][1]]]
    )

# Calculate center of mass (approximation using hip midpoint)
print("Calculating center of mass...")
df['center_x'] = (df['LEFT_HIP_x'] + df['RIGHT_HIP_x']) / 2
df['center_y'] = (df['LEFT_HIP_y'] + df['RIGHT_HIP_y']) / 2

# Calculate distances
print("Calculating joint distances...")
for joint_pair in [
    ('shoulder', ['LEFT_SHOULDER', 'RIGHT_SHOULDER']),
    ('hip', ['LEFT_HIP', 'RIGHT_HIP']),
    ('knee', ['LEFT_KNEE', 'RIGHT_KNEE']),
    ('ankle', ['LEFT_ANKLE', 'RIGHT_ANKLE'])
]:
    name, (left, right) = joint_pair
    df[f'{name}_distance'] = np.sqrt(
        (df[keypoints[left][0]] - df[keypoints[right][0]])**2 +
        (df[keypoints[left][1]] - df[keypoints[right][1]])**2
    )

# Calculate velocities and accelerations
print("Calculating velocities and accelerations...")
# Center of mass velocity and acceleration
center_positions = df[['center_x', 'center_y']].values
center_velocities = calculate_velocity(center_positions, frame_time)
center_accelerations = calculate_acceleration(center_velocities, frame_time)

df['center_velocity_x'] = center_velocities[:, 0]
df['center_velocity_y'] = center_velocities[:, 1]
df['center_acceleration_x'] = center_accelerations[:, 0]
df['center_acceleration_y'] = center_accelerations[:, 1]

# Calculate angular velocities
print("Calculating angular velocities...")
for angle in ['LEFT_KNEE_ANGLE', 'RIGHT_KNEE_ANGLE', 'LEFT_HIP_ANGLE', 'RIGHT_HIP_ANGLE']:
    df[f'{angle}_velocity'] = np.gradient(df[angle], frame_time)

# Calculate step length (distance between ankles)
df['step_length'] = np.sqrt(
    (df['LEFT_ANKLE_x'] - df['RIGHT_ANKLE_x'])**2 +
    (df['LEFT_ANKLE_y'] - df['RIGHT_ANKLE_y'])**2
)

# Calculate body height (shoulder to ankle distance)
df['body_height_left'] = np.sqrt(
    (df['LEFT_SHOULDER_x'] - df['LEFT_ANKLE_x'])**2 +
    (df['LEFT_SHOULDER_y'] - df['LEFT_ANKLE_y'])**2
)
df['body_height_right'] = np.sqrt(
    (df['RIGHT_SHOULDER_x'] - df['RIGHT_ANKLE_x'])**2 +
    (df['RIGHT_SHOULDER_y'] - df['RIGHT_ANKLE_y'])**2
)

# Calculate movement features
print("Calculating movement features...")
# Vertical displacement (useful for jump detection)
df['vertical_displacement'] = np.gradient(df['center_y'], frame_time)
df['vertical_velocity'] = np.gradient(df['vertical_displacement'], frame_time)

# Body rotation (angle between shoulders and hips)
df['body_rotation'] = np.arctan2(
    df['RIGHT_SHOULDER_y'] - df['LEFT_SHOULDER_y'],
    df['RIGHT_SHOULDER_x'] - df['LEFT_SHOULDER_x']
) - np.arctan2(
    df['RIGHT_HIP_y'] - df['LEFT_HIP_y'],
    df['RIGHT_HIP_x'] - df['LEFT_HIP_x']
)
df['body_rotation'] = np.degrees(df['body_rotation'])

# Calculate rolling statistics for temporal context
window_size = 5  # Adjust based on your needs
for col in ['vertical_displacement', 'body_rotation', 'step_length']:
    df[f'{col}_rolling_mean'] = df[col].rolling(window=window_size, center=True).mean()
    df[f'{col}_rolling_std'] = df[col].rolling(window=window_size, center=True).std()

# Fill NaN values
df.fillna(0, inplace=True)

# Save enhanced features
print("Saving enhanced features...")
df.to_csv(output_file, index=False)

# Prepare for model training
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
    'step_length_rolling_mean', 'step_length_rolling_std'
]

X = df[feature_columns]
y = df['annotation']

# Map labels to integers
label_map = {
    'Still': 0, 'approach': 1, 'back': 2, 'jump': 3, 'turn_left': 4,
    'walk_left': 5, 'turn_right': 6, 'walk_right': 7, 'sit': 8, 'stand': 9
}
y = y.map(label_map)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model with optimized parameters
print("Training model...")
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='mlogloss'
)

model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=list(label_map.keys())))

# Save model and scaler
print("Saving model and scaler...")
with open('xgboost_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)