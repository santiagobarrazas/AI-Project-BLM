import os
import json
import pandas as pd

json_folder = "./annotations"
csv_folder = "./output"
output_folder = "./labeled_output"

os.makedirs(output_folder, exist_ok=True)

for json_file in os.listdir(json_folder):
    if json_file.endswith(".json"):
        json_path = os.path.join(json_folder, json_file)
        csv_path = os.path.join(csv_folder, json_file.replace(".json", ".csv"))
        
        if not os.path.exists(csv_path):
            print(f"El archivo CSV para {json_file} no existe. Saltando...")
            continue
        
        with open(json_path, 'r') as f:
            json_data = json.load(f)

        categories = json_data["categories"]
        
        annotated_frames = {ann["image_id"] for ann in json_data["annotations"]}
        annotated_frames_categories = {ann["category_id"] for ann in json_data["annotations"]}

        csv_data = pd.read_csv(csv_path)

        csv_data["annotation"] = csv_data["frame"].apply(
            lambda frame: "Still" if frame not in annotated_frames else categories[json_data["annotations"][list(annotated_frames)[0]]["category_id"]-1]["name"]
        )
        
        output_csv_path = os.path.join(output_folder, os.path.basename(csv_path))
        csv_data.to_csv(output_csv_path, index=False)
        print(f"Procesado: {json_file} -> {output_csv_path}")

print("Procesamiento completado.")