import os
import pandas as pd

# Folder containing the CSV files
folder_path = 'normalized_output'

# List to store processed DataFrames
dataframes = []

# Iterate over all CSV files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        file_path = os.path.join(folder_path, file_name)
        videoid = os.path.splitext(file_name)[0]  # Extract the videoid from the file name
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Extract the annotation for the frame
        df_annotation = df[['frame', 'annotation']].drop_duplicates(subset=['frame'])
        
        # Pivot the rest of the data
        pivot_df = df.pivot(index='frame', columns='landmark', values=['x', 'y', 'z'])
        
        # Flatten the multi-index columns
        pivot_df.columns = [f"{col[1]}_{col[0]}" for col in pivot_df.columns]
        
        # Add the annotation and videoid columns
        pivot_df = pivot_df.merge(df_annotation, on='frame')
        pivot_df['videoid'] = videoid
        
        # Append to the list
        dataframes.append(pivot_df)

# Concatenate all DataFrames
final_df = pd.concat(dataframes, ignore_index=True)

# Save the final DataFrame to a CSV file
final_output_path = 'merged_output.csv'
final_df.to_csv(final_output_path, index=False)

print(f"Data merged and saved to {final_output_path}")
