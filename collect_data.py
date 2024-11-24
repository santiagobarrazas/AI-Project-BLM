import os
import cv2
import mediapipe as mp
import pandas as pd

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=False)
mp_drawing = mp.solutions.drawing_utils

INPUT_DIR = 'video/'
OUTPUT_DIR = 'output/'

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    frame_number = 0
    rows = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = pose.process(frame_rgb)
        
        if results.pose_landmarks:
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                rows.append({
                    'frame': frame_number,
                    'landmark': mp_pose.PoseLandmark(idx).name,
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                })

        frame_number += 1

    cap.release()

    if rows:
        df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Processed and saved: {output_path}")

def process_videos(input_dir, output_dir):
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_dir)
                output_path = os.path.join(output_dir, relative_path, os.path.splitext(file)[0] + '.csv')

                print(f"Processing: {input_path}")
                process_video(input_path, output_path)

if __name__ == "__main__":
    process_videos(INPUT_DIR, OUTPUT_DIR)
