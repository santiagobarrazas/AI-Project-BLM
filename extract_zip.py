import os
import zipfile
import shutil

def process_zip_files(zip_folder, output_folder):
    """
    Procesa los archivos zip dentro de la carpeta especificada.

    - Extrae cada archivo zip.
    - Cambia el nombre del archivo `instances_default.json` dentro de la carpeta `annotations`.
    - Mueve los archivos JSON renombrados a una carpeta centralizada.

    Args:
        zip_folder (str): Ruta de la carpeta que contiene los archivos zip.
        output_folder (str): Ruta de la carpeta donde se guardarán los JSON procesados.
    """
    # Asegúrate de que la carpeta de salida exista
    os.makedirs(output_folder, exist_ok=True)

    # Itera sobre cada archivo en la carpeta de zips
    for filename in os.listdir(zip_folder):
        if filename.endswith('.zip'):
            zip_path = os.path.join(zip_folder, filename)
            zip_name = os.path.splitext(filename)[0]  # Nombre sin la extensión

            # Crear una carpeta temporal para extraer el contenido
            extract_path = os.path.join(zip_folder, f"temp_{zip_name}")
            os.makedirs(extract_path, exist_ok=True)

            try:
                # Extraer el zip
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)

                # Ruta de la carpeta annotations dentro de la extracción
                annotations_path = os.path.join(extract_path, 'annotations')

                # Verifica si existe la carpeta annotations
                if os.path.isdir(annotations_path):
                    json_file_path = os.path.join(annotations_path, 'instances_default.json')

                    if os.path.isfile(json_file_path):
                        # Renombrar el archivo JSON
                        new_json_name = f"{zip_name}.json"
                        new_json_path = os.path.join(output_folder, new_json_name)
                        shutil.move(json_file_path, new_json_path)
                        print(f"Procesado: {new_json_name}")
                    else:
                        print(f"No se encontró 'instances_default.json' en {annotations_path}")
                else:
                    print(f"No se encontró la carpeta 'annotations' en {zip_name}")

            except Exception as e:
                print(f"Error procesando {zip_name}: {e}")

            finally:
                # Eliminar la carpeta temporal
                shutil.rmtree(extract_path)

if __name__ == "__main__":
    zip_folder = "D:/Descargas/DataAnnotations"
    output_folder = "D:/Proyectos/AI-Project-BLM/annotations"

    process_zip_files(zip_folder, output_folder)
    print("Procesamiento completado.")
