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
        
        # Normalizar las columnas numéricas
        if not numeric_columns.empty:
            df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
        
        # Guardar el archivo normalizado en la carpeta de salida
        df.to_csv(output_path, index=False)

print(f"Todos los archivos han sido procesados y guardados en '{output_folder}'.")
