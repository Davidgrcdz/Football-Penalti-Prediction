import os
import ast
import numpy as np
import pandas as pd
from typing import List


def parse_list_or_nan(x, n_classes: int = 3):
    """
    Convierte una cadena tipo "[0.65, 0.70, 0.55]" en np.array de floats.
    Si falla, devuelve un array de NaNs de tamaño n_classes.
    """
    if pd.isna(x):
        return np.full(n_classes, np.nan, dtype=float)
    try:
        vals = ast.literal_eval(str(x))
        vals = list(vals)
        if len(vals) < n_classes:
            vals = (vals + [np.nan] * n_classes)[:n_classes]
        return np.array(vals, dtype=float)
    except Exception:
        return np.full(n_classes, np.nan, dtype=float)


def compare_model_results(model_type: str,
                          historical_files: List[str],
                          final_file_path: str,
                          output_dir: str,
                          n_classes: int = 3):
    """
    Compara resultados históricos (V1+V2) con los resultados finales (V3) para un modelo dado.
    Genera dos CSV:

    1) {model_type}_training_comparison_stats.csv
       - Comparación por 'target_embedding' usando estadísticas de best_f1:
         media, máximo y desviación típica, tanto en V1+V2 como en V3, y sus diferencias.

    2) {model_type}_training_comparison_id.csv
       - Comparación por 'id' (misma configuración):
         best_f1_v1_v2, best_f1_v3, diferencia
         f1_per_class_v1_v2_fmt, f1_per_class_v3_fmt, diff_f1_per_class (todo en formato string "a / b / c").
    """
    print(f"--- Iniciando comparación para el modelo: {model_type} ---")
    os.makedirs(output_dir, exist_ok=True)

    # ---------- 1. Carga de datos ----------
    try:
        df_final = pd.read_csv(final_file_path, low_memory=False)
        dfs_hist = [pd.read_csv(p, low_memory=False)
                    for p in historical_files if os.path.exists(p)]
        if not dfs_hist:
            print("⚠️ No se encontraron archivos históricos. No se puede realizar la comparación.")
            return
        df_hist_all = pd.concat(dfs_hist, ignore_index=True)
    except FileNotFoundError as e:
        print(f"❌ Error: No se pudo encontrar el archivo: {e.filename}")
        return
    except Exception as e:
        print(f"❌ Error al cargar los datos: {e}")
        return

    print(f"Archivos históricos cargados: {len(dfs_hist)}")
    print(f"Archivo final cargado: {os.path.basename(final_file_path)}")

    # ---------- 2. Comparación agregada por embedding (stats de best_f1) ----------
    print("\n--- Comparación por embedding (media, max, std de best_f1) sobre IDs coincidentes... ---")

    # IDs presentes en V3
    final_ids = df_final["id"].dropna().unique()
    print(f"Se comparará sobre los {len(final_ids)} IDs únicos presentes en el archivo final (V3).")

    # Filtrar histórico a esos IDs para comparación justa
    df_hist_matched = df_hist_all[df_hist_all["id"].isin(final_ids)]

    # Estadísticas en históricos (V1+V2)
    stats_hist = (
        df_hist_matched
        .groupby("target_embedding")["best_f1"]
        .agg(["mean", "max", "std"])
        .reset_index()
    )
    stats_hist.columns = ["target_embedding",
                          "mean_f1_v1_v2",
                          "max_f1_v1_v2",
                          "std_f1_v1_v2"]

    # Estadísticas en V3
    stats_final = (
        df_final
        .groupby("target_embedding")["best_f1"]
        .agg(["mean", "max", "std"])
        .reset_index()
    )
    stats_final.columns = ["target_embedding",
                           "mean_f1_v3",
                           "max_f1_v3",
                           "std_f1_v3"]

    # Unir y calcular diferencias
    comp_stats = pd.merge(stats_hist, stats_final,
                          on="target_embedding", how="inner")
    comp_stats["mean_diff"] = comp_stats["mean_f1_v3"] - comp_stats["mean_f1_v1_v2"]
    comp_stats["max_diff"] = comp_stats["max_f1_v3"] - comp_stats["max_f1_v1_v2"]

    comp_stats.fillna(0, inplace=True)

    out_stats_filename = f"{model_type.lower()}_training_comparison_stats.csv"
    out_stats_path = os.path.join(output_dir, out_stats_filename)
    (
        comp_stats
        .round(5)
        .sort_values("mean_diff", ascending=False)
        .to_csv(out_stats_path, index=False)
    )
    print(f"✅ Comparación estadística por embedding guardada en: {out_stats_path}")

    # ---------- 3. Comparación detallada por ID (incluyendo f1_per_class) ----------
    print("\n--- Comparación por ID (configuración exacta) con F1 macro y F1 por clase... ---")

    cols_needed = ["id", "target_embedding", "best_f1", "best_f1_per_class"]

    # Históricos: quedarnos con el mejor best_f1 por ID
    df_hist_unified = df_hist_all.dropna(subset=["id"]).copy()
    df_hist_unified["id"] = pd.to_numeric(df_hist_unified["id"], errors="coerce")
    df_hist_unified = df_hist_unified.dropna(subset=["id"])
    df_hist_unified["id"] = df_hist_unified["id"].astype(int)

    df_hist_unified = df_hist_unified.sort_values("best_f1", ascending=False)
    df_hist_unified = df_hist_unified.drop_duplicates(subset="id")

    # Datos finales (V3)
    df_final_ids = df_final.copy()
    df_final_ids["id"] = pd.to_numeric(df_final_ids["id"], errors="coerce")
    df_final_ids = df_final_ids.dropna(subset=["id"])
    df_final_ids["id"] = df_final_ids["id"].astype(int)

    final_data_by_id = df_final_ids[cols_needed].copy()

    # Merge por ID
    final_comparison = pd.merge(
        final_data_by_id,
        df_hist_unified[cols_needed],
        on="id",
        how="inner",
        suffixes=("_v3", "_v1_v2")
    )

    # Diferencia en F1 macro
    final_comparison["difference"] = (
        final_comparison["best_f1_v3"] - final_comparison["best_f1_v1_v2"]
    ).round(5)

    # Formatear listas F1 por clase
    def format_list_str(x):
        arr = parse_list_or_nan(x, n_classes)
        return " / ".join(f"{v:.3f}" for v in arr)

    def compute_diff_list_str(row):
        v12 = parse_list_or_nan(row["best_f1_per_class_v1_v2"], n_classes)
        v3  = parse_list_or_nan(row["best_f1_per_class_v3"], n_classes)
        diff = v3 - v12
        return " / ".join(f"{d:.3f}" for d in diff)

    final_comparison["f1_per_class_v1_v2_fmt"] = final_comparison["best_f1_per_class_v1_v2"].apply(format_list_str)
    final_comparison["f1_per_class_v3_fmt"]    = final_comparison["best_f1_per_class_v3"].apply(format_list_str)
    final_comparison["diff_f1_per_class"]      = final_comparison.apply(compute_diff_list_str, axis=1)

    final_comparison = final_comparison.rename(columns={"target_embedding_v3": "embedding"})

    final_columns = [
        "id",
        "embedding",
        "best_f1_v1_v2",
        "best_f1_v3",
        "difference",
        "f1_per_class_v1_v2_fmt",
        "f1_per_class_v3_fmt",
        "diff_f1_per_class",
    ]

    out_id_filename = f"{model_type.lower()}_training_comparison_id.csv"
    out_id_path = os.path.join(output_dir, out_id_filename)
    (
        final_comparison[final_columns]
        .sort_values("id")
        .to_csv(out_id_path, index=False)
    )
    print(f"✅ Comparación por ID guardada en: {out_id_path}\n")

    return out_stats_path, out_id_path
