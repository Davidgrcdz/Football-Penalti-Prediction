import pandas as pd
from typing import List, Optional, Dict
import glob, re, os



def find_best_configs(
    csv_files: List[str],
    output_file: Optional[str],
    model_name: str,
    f1_threshold: float = 0.55
):
    """
    Lee históricos, filtra por best_f1 >= umbral, elimina duplicados por config
    y devuelve SOLO los hiperparámetros necesarios para reentrenar, añadiendo
    la columna 'source_params' para indicar el origen.
    """
    # Columnas de hiperparámetros que definen una configuración única.
    # Se añade 'source_params' para la trazabilidad.
    if model_name == "TCN":
        CONFIG_COLS = ["channels", "dilations", "kernel_size", "convs_per_block", "activation",
                        "use_weight_norm", "wn_on_skip", "pooling", "fc_hidden",
                        "batch_size", "optimizer", "momentum_sgd", "lr", "num_blocks",
                        "weight_decay", "norm", "dropout", "ln_block_end", "source_params"
        ]

    elif model_name == "LSTM":
        CONFIG_COLS = ["num_layers", "layer_sizes","bidirectional","norm","dropout","dropout_layers","norm_layers",
                        "pooling","fc_hidden","activation","norm_post_lstm","batch_size","optimizer",
                        "momentum_sgd","learning_rate","weight_decay","native_mode", "source_params"
        ]
    
    elif model_name == "MLP":
        # NUEVOO para MLP: antes era igual pero se tenía do_layer y norm_layer en vez de como se tiene ahora
        CONFIG_COLS = ["hidden_layers", "norm_layers", "dropout_layers", "dropout_rate", "activation",
                        "base_dim", "batch_size", "num_layers", "pooling", "norm", "optimizer", "momentum_sgd",
                        "learning_rate", "weight_decay", "use_pca", "pca_frac", "source_params"]

    
    elif model_name == "Transformer":
        CONFIG_COLS = ["model_dims", "num_heads", "num_layers", "dim_feedforward", "dropout", 
                        "activation", "norm_type", "pos_encoding", "pooling", "use_cls_token", "lr",
                        "weight_decay", "batch_size", "optimizer", "momentum_sgd", "norm", "source_params"]

    if not csv_files:
        print("⚠️ La lista de archivos CSV está vacía.")
        return None

    dfs = []
    print(f"Procesando {len(csv_files)} archivo(s) CSV...")
    for fp in csv_files:
        try:
            df = pd.read_csv(fp, low_memory=False)
            # ¡NUEVO! Añadimos la columna con el nombre del archivo de origen
            df['source_params'] = os.path.basename(fp)
            if 'best_f1' in df.columns:
                dfs.append(df)
            else:
                print(f"⚠️ '{fp}' no tiene 'best_f1' → ignorado.")
        except FileNotFoundError:
            print(f"❌ No encontrado: {fp}")
        except Exception as e:
            print(f"❌ Error leyendo '{fp}': {e}")

    if not dfs:
        print("❌ No se pudo leer ninguna configuración válida.")
        return None

    # 1) Combinar todos los DataFrames
    combined = pd.concat(dfs, ignore_index=True, sort=False)

    # 2) Filtrar por el umbral de F1-score
    combined['best_f1'] = pd.to_numeric(combined['best_f1'], errors='coerce')
    best = combined[combined['best_f1'] >= f1_threshold].copy()
    if best.empty:
        print(f"⚠️ No hay configs con best_f1 >= {f1_threshold}.")
        return None

    # 3) ¡CLAVE! Ordenar por 'best_f1' para que al eliminar duplicados nos quedemos con el mejor
    best = best.sort_values(by='best_f1', ascending=False)

    # 4) Mantener solo las columnas de configuración que realmente existen en los CSVs
    config_cols_present = [c for c in CONFIG_COLS if c in best.columns]
    if not config_cols_present:
        print("❌ Ninguna columna de hiperparámetros presente en los CSVs.")
        return None

    # Columnas de hiperparámetros (sin 'source_params') para la lógica de duplicados
    hp_cols_only = [c for c in config_cols_present if c != 'source_params']
    
    best_cfgs = best[config_cols_present].copy()

    # 5) Rellenar NaN con un placeholder para poder usar drop_duplicates
    for c in hp_cols_only:
        if best_cfgs[c].isnull().any():
            best_cfgs[c] = best_cfgs[c].fillna('NaN_placeholder')

    # 6) Eliminar duplicados por configuración. Gracias al ordenamiento, 'first' es el mejor.
    unique_cfgs = best_cfgs.drop_duplicates(subset=hp_cols_only, keep='first').copy()

    # 7) Restaurar los valores NaN
    unique_cfgs.replace('NaN_placeholder', pd.NA, inplace=True)

    # (Opcional) Ordenar las columnas para una mejor legibilidad
    unique_cfgs = unique_cfgs[config_cols_present]

    print(f"✅ Encontradas {len(unique_cfgs)} configuraciones únicas con su origen.")

    # 8) Guardar el archivo de presets si se especifica una ruta
    if output_file:
        outdir = os.path.dirname(output_file)
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        unique_cfgs.to_csv(output_file, index=False)
        print(f"Guardado en '{output_file}'")

    return unique_cfgs






# --- Buscar best F1 en CSVs bajo results/ ---

def find_model(model_id=None, model_path=None, config=None, results_root="results"):
    """
    Buscar en CSVs bajo results_root el mejor valor de 'best_f1' asociado al modelo indicado.
    Devuelve: (best_val, 'best_f1', source_filepath, model_full_name, metrics_dict)
      - model_full_name: string construido a partir de columnas disponibles (model, target_embedding, id)
      - metrics_dict: dict con keys ['train_loss','accuracy','epochs_trained','f1_macro','best_f1','best_epoch','best_state_loss']
    Estrategia de matching:
      - Sólo match por id (columna 'id'). Se extrae id del nombre del archivo si no se pasa explicitamente.
    """
    csv_files = glob.glob(os.path.join(results_root, "**", "*.csv"), recursive=True)

    # intentar extraer id del nombre del archivo (p.ej. lstmNNN_ o mlpNNN_)
    if model_id is None and model_path is not None:
        m = re.search(r"(?:lstm|mlp|tcn)(\d+)_", os.path.basename(model_path), re.I)
        if m:
            try:
                model_id = int(m.group(1))
            except Exception:
                model_id = None

    if model_id is None:
        # sin id no hacemos búsqueda (según la nueva regla)
        return None, None, None, None, {k: None for k in ['train_loss','accuracy','epochs_trained','f1_macro','best_f1','best_epoch','best_state_loss']}

    best_val = None
    best_src = None
    best_row = None

    for f in csv_files:
        try:
            df = pd.read_csv(f, low_memory=False)
        except Exception:
            continue

        if 'best_f1' not in df.columns or 'id' not in df.columns:
            continue

        # match por id (estrictamente)
        try:
            matches = df[df['id'] == int(model_id)]
        except Exception:
            matches = df[df['id'].astype(str) == str(model_id)]

        if matches.empty:
            continue

        # Asegurar columna numericada y descartar NaNs
        matches = matches.copy()
        matches['__best_f1_num'] = pd.to_numeric(matches['best_f1'], errors='coerce')
        matches = matches.dropna(subset=['__best_f1_num'])
        if matches.empty:
            continue

        # seleccionar la fila con mayor best_f1 dentro de este archivo
        idxmax = matches['__best_f1_num'].idxmax()
        row = matches.loc[idxmax]
        file_best = float(row['__best_f1_num'])

        if best_val is None or file_best > best_val:
            best_val = file_best
            best_src = f
            best_row = row

    if best_val is None:
        return None, None, None, None, {k: None for k in ['train_loss','accuracy','epochs_trained','f1_macro','best_f1','best_epoch','best_state_loss']}


    # Extraer métricas solicitadas con fallback a None
    metrics = {}
    for col in ['train_loss','accuracy','epochs_trained','f1_macro','best_f1','best_epoch','best_state_loss', 'seed']:
        if col in best_row.index:
            val = best_row.get(col)
            try:
                if pd.isna(val):
                    metrics[col] = None
                else:
                    if col in ('epochs_trained','best_epoch','seed'):
                        metrics[col] = int(val)
                    else:
                        metrics[col] = float(val) if pd.notna(val) else None
            except Exception:
                metrics[col] = val
        else:
            metrics[col] = None
    
    model_name = os.path.basename(model_path)  # sin extensión

    return best_val, 'best_f1', best_src, model_name, metrics








def topK_from_each_topx_csv(
    input_dir: str,
    output_dir: str,
    model_name: str,
    top_k: int = 5,
    dedup: bool = True
):
    """
    Busca en input_dir todos los CSV cuyo nombre contenga 'top<numero>%' (case-insensitive).
    Para cada uno:
      - ordena por best_f1 desc
      - (opcional) drop_duplicates por hiperparámetros del modelo
      - guarda top_k en output_dir con sufijo '_top5.csv'
    """

    model_config_cols: Dict[str, List[str]] = {
    "TCN": [
        "channels","dilations","kernel_size","convs_per_block","activation",
        "use_weight_norm","wn_on_skip","pooling","fc_hidden",
        "batch_size","optimizer","momentum_sgd","lr","num_blocks",
        "weight_decay","norm","dropout","ln_block_end"
    ],
    "LSTM": [
        "num_layers","layer_sizes","bidirectional","norm","dropout","dropout_layers","norm_layers",
        "pooling","fc_hidden","activation","norm_post_lstm","batch_size","optimizer",
        "momentum_sgd","learning_rate","weight_decay","native_mode"
    ],
    "MLP": [
        "hidden_layers","norm_layers","dropout_layers","dropout_rate","activation",
        "base_dim","batch_size","num_layers","pooling","norm","optimizer","momentum_sgd",
        "learning_rate","weight_decay","use_pca","pca_frac"
    ],
    "Transformer": [
        "model_dims","num_heads","num_layers","dim_feedforward","dropout",
        "activation","norm_type","pos_encoding","pooling","use_cls_token","lr",
        "weight_decay","batch_size","optimizer","momentum_sgd","norm"
    ],
}

    if model_name not in model_config_cols:
        raise ValueError(f"model_name debe ser uno de: {list(model_config_cols.keys())}")

    os.makedirs(output_dir, exist_ok=True)

    # CSVs con topX% en el nombre
    all_csv = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    rx = re.compile(r"top\s*\d+\s*%", re.IGNORECASE)
    target_csv = [fp for fp in all_csv if rx.search(os.path.basename(fp))]

    if not target_csv:
        print(f"⚠️ No encontré CSVs con 'topX%' en el nombre dentro de: {input_dir}")
        return

    hp_cols = model_config_cols[model_name]

    for fp in target_csv:
        name = os.path.basename(fp)
        try:
            df = pd.read_csv(fp, low_memory=False)
        except Exception as e:
            print(f"Error leyendo {name}: {e}")
            continue

        if "best_f1" not in df.columns:
            print(f"{name}: no tiene columna 'best_f1' -> saltado")
            continue

        df["best_f1"] = pd.to_numeric(df["best_f1"], errors="coerce")
        df = df.dropna(subset=["best_f1"]).sort_values("best_f1", ascending=False)

        # columnas de config presentes en ese CSV
        hp_present = [c for c in hp_cols if c in df.columns]
        if not hp_present:
            print(f"{name}: no tiene columnas de hiperparámetros esperadas para {model_name} -> saltado")
            continue

        out_cols = hp_present + ["best_f1"]
        if "source_params" in df.columns:
            out_cols.append("source_params")

        out = df[out_cols].copy()

        if dedup:
            # proteger NaN para drop_duplicates
            for c in hp_present:
                if out[c].isnull().any():
                    out[c] = out[c].astype("object").where(~out[c].isnull(), other="__NA__")
            out = out.drop_duplicates(subset=hp_present, keep="first")
            out = out.replace("__NA__", pd.NA)

        out = out.head(top_k).reset_index(drop=True)

        out_name = os.path.splitext(name)[0] + f"_top{top_k}.csv"
        out_path = os.path.join(output_dir, out_name)
        out.to_csv(out_path, index=False)
        print(f"{name} -> {out_name} (filas={len(out)})")



