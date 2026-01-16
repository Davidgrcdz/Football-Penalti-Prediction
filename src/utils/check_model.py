import os
import torch
import re

from src.utils.find_best import find_model

def check_model_config(model_path: str, device: torch.device):
    """
    Función universal para cargar un checkpoint, buscar sus métricas históricas
    y mostrar toda la información relevante: config, métricas y estructura.
    """
    print(f"--- Chequeando configuración para: {os.path.basename(model_path)} ---")

    # --- 1. Cargar Checkpoint ---
    try:
        # Carga el archivo del modelo. Es importante usar weights_only=False
        # porque tus checkpoints contienen diccionarios de configuración, no solo pesos.
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo del modelo en '{model_path}'")
        return
    except Exception as e:
        print(f"❌ Error al cargar el archivo del modelo: {e}")
        return

    # --- 2. Buscar métricas históricas con tu función find_model ---
    # (Asegúrate de que la función find_model ya ha sido definida en el notebook)
    best_val, metric_name, source_file, model_name, metrics = find_model(model_path=model_path)
    
    print(f"\nModelo: {model_name}")
    if best_val is not None:
        print(f"Mejor {metric_name} histórico = {best_val:.4f} (encontrado en {os.path.basename(source_file)})")
        print("Métricas históricas:", metrics)
    else:
        print("⚠️ No se encontraron métricas históricas para este ID de modelo.")

    # --- 3. Extraer y mostrar información del checkpoint ---
    model_id = checkpoint.get('model_id', 'No ID')
    timestamp = checkpoint.get('timestamp', 'No timestamp')
    config = checkpoint.get('config', {})
    # Busca la seed primero en las métricas históricas, si no, en la config del checkpoint
    seed = metrics.get('seed', config.get('seed', 'No disponible'))
    
    match = re.match(r"([a-zA-Z]+)", os.path.basename(model_path))
    model_type_abbr = match.group(1).upper() if match else "DESCONOCIDO"

    model_full_names = {
        'TCN': 'Temporal Convolutional Network (TCN)',
        'LSTM': 'Long Short-Term Memory (LSTM)',
        'MLP': 'Multi-Layer Perceptron (MLP)',
        'TraN': 'Transformer Encoder (Transformer)',
    }
    model_type = model_full_names.get(model_type_abbr, model_type_abbr)


    print(f"\nTipo de Modelo: {model_type}")
    print(f"Modelo ID: {model_id}")
    print(f"Timestamp de guardado: {timestamp}")
    print(f"Seed: {seed}")
    
    print("\nConfiguración guardada en el checkpoint:")
    if config:
        for key, value in config.items():
            print(f"  - {key}: {value}")
    else:
        print("  (No se encontró configuración en el checkpoint)")

    # --- 4. Mostrar estructura de capas ---
    # Busca el state_dict con varios nombres posibles para mayor compatibilidad
    state_dict = checkpoint.get('model_state', checkpoint.get('model_state_dict', {}))
    print("\nEstructura de capas (state_dict):")
    if state_dict:
        for key, tensor in state_dict.items():
            print(f"  - {key}: {tensor.shape}")
    else:
        print("  (No se encontró state_dict en el checkpoint)")
