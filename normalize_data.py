import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Definir rutas
input_folder = "labeled_output"
output_folder = "normalized_output"

# Crear la carpeta de salida si no existe
os.makedirs(output_folder, exist_ok=True)

# Inicializar el escalador
scaler = MinMaxScaler()

# Procesar todos los archivos CSV en la carpeta de entrada
for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        # Cargar el archivo CSV
        df = pd.read_csv(input_path)
        
        # Identificar columnas numéricas, excluyendo "frame"
        numeric_columns = df.select_dtypes(include=["number"]).columns
        numeric_columns = numeric_columns.drop("frame", errors="ignore")  # Excluir "frame" si existe
        
        # Normalizar las columnas numéricas si existen
        if not numeric_columns.empty:
            if {"LEFT_HIP_x", "RIGHT_HIP_x", "LEFT_HIP_y", "RIGHT_HIP_y"}.issubset(df.columns):
                # Calcular el centro de las caderas
                left_hip_x = df["LEFT_HIP_x"]
                left_hip_y = df["LEFT_HIP_y"]
                right_hip_x = df["RIGHT_HIP_x"]
                right_hip_y = df["RIGHT_HIP_y"]

                hip_center_x = (left_hip_x + right_hip_x) / 2
                hip_center_y = (left_hip_y + right_hip_y) / 2

                # Normalizar landmarks respecto al centro de las caderas
                for col in ["x", "y", "z"]:
                    cols_to_normalize = [f"{landmark}_{col}" for landmark in df.columns if f"_{col}" in landmark]
                    for normalized_col in cols_to_normalize:
                        if col == "x":
                            df[normalized_col] = (df[normalized_col] - hip_center_x) * 100
                        elif col == "y":
                            df[normalized_col] = (df[normalized_col] - hip_center_y) * 100
                        elif col == "z":
                            df[normalized_col] = df[normalized_col] * 100
            else:
                # Normalización estándar con MinMaxScaler
                df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
        
        # Guardar el archivo normalizado en la carpeta de salida
        df.to_csv(output_path, index=False)

print(f"Todos los archivos han sido procesados y guardados en '{output_folder}'.")
