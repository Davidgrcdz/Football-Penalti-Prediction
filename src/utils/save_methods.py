import os
import pandas as pd
import torch


def save_model(model, path: str, config: dict = None, model_id: int = None, preproc: dict = None):
    obj = {'model_state': model.state_dict(), 'model_id': model_id, 'timestamp': pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}
    if config is not None:
        obj['config'] = config
    if preproc is not None:
        obj['preproc'] = preproc
    torch.save(obj, path)
    print(f"Modelo guardado en {path}\n")


def save_train_test_ids(embedding_name, train_ids, test_ids, seed, output_dir):
    """
    Guarda los IDs de train y test en un CSV para análisis posterior.
    
    Args:
        embedding_name: Nombre del embedding (archivo CSV)
        train_ids: Lista de IDs de videos usados para entrenamiento
        test_ids: Lista de IDs de videos usados para prueba
        video_labels: Diccionario con las etiquetas por video_ID
        seed: Semilla utilizada para la división de los datos
        output_dir: Directorio donde guardar los CSVs de splits
        add_timestamp: Si se debe añadir timestamp al nombre del archivo
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Crear DataFrames para train y test
    train_data = []
    for vid_id in train_ids:
        row = {
            'video_ID': vid_id,
            'split': 'train',
            'embedding': embedding_name,
            'seed': seed
        }
        train_data.append(row)
    
    test_data = []
    for vid_id in test_ids:
        row = {
            'video_ID': vid_id,
            'split': 'test',
            'embedding': embedding_name,
            'seed': seed
        }
        test_data.append(row)
    
    # Combinar en un solo DataFrame
    split_df = pd.concat([pd.DataFrame(train_data), pd.DataFrame(test_data)])
    
    output_file = os.path.join(output_dir, f"split_{seed}_{embedding_name.replace('.csv', '')}.csv")
    
    # Si existe el archivo, no guardar de nuevo
    if os.path.exists(output_file):
        print(f"El archivo {output_file} ya existe. No se sobrescribe.")
        return output_file

    # Guardar en CSV
    split_df.to_csv(output_file, index=False)
    print(f"IDs de train/test guardados en: {output_file}")
    
    # Guardar en un archivo consolidado
    all_splits_file = os.path.join(output_dir, f"splits_seed{seed}.csv")
    if os.path.exists(all_splits_file):
        # Línea vacía entre entradas
        with open(all_splits_file, 'a') as f:
            f.write("\n")
        split_df.to_csv(all_splits_file, mode='a', header=False, index=False)
    else:
        split_df.to_csv(all_splits_file, index=False)
    
    return output_file