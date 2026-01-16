import os
import pandas as pd

def get_max_id(csv_paths):
    """Devuelve el máximo 'id' encontrado en los CSVs (int). Si no hay ninguno devuelve 0."""
    max_id = 0
    for p in csv_paths:
        if not os.path.exists(p):
            continue
        try:
            # leer solo la columna 'id' es más rápido y evita problemas de memoria
            df = pd.read_csv(p, usecols=['id'], low_memory=False)
        except Exception as e:
            print(f"Warning leyendo {p}: {e}")
            continue
        if 'id' not in df.columns or df['id'].dropna().empty:
            continue
        # convertir a numérico y obtener el máximo válido
        col_max = pd.to_numeric(df['id'], errors='coerce').dropna().astype(int).max()
        if pd.notna(col_max):
            max_id = max(max_id, int(col_max))
    
    new_id = max_id  # Incrementar ID para cada nuevo modelo max_id + 1
    
    return int(new_id)