# =======================================================
#  OPTUNA LSTM — VERSIÓN LIMPIA, COHERENTE + METADATA
# =======================================================

import os
import numpy as np
import pandas as pd
import optuna
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, normalize
from sklearn.metrics import f1_score

from src.lstm.model import FlexibleLSTM

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
DATA_DIR    = "Gait_Embeddings_good"
OUT_DIR     = "results/LSTM/Optuna_FINAL"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED        = 42
EPOCHS      = 300
TEST_SIZE   = 0.1
N_SPLITS    = 5
N_TRIALS    = 50  # nº de trials por embedding

BATCH_SIZE_CHOICES = [16, 32, 48, 64, 96, 128]
ES_PATIENCE  = 40
MIN_EPOCHS   = 50
ES_MIN_DELTA = 1e-3

os.makedirs(OUT_DIR, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)


# -------------------------------------------------------
# COLLATE
# -------------------------------------------------------
def collate_sequences(batch):
    seqs, labels = zip(*batch)
    lengths = torch.tensor([s.size(0) for s in seqs], dtype=torch.long)
    padded  = pad_sequence(seqs, batch_first=True)
    labels  = torch.tensor(labels, dtype=torch.long)
    return padded, lengths, labels


# ========================================================================
#    FUNCIÓN CLAVE — GENERA UNA CONFIG LSTM SIEMPRE 100% VÁLIDA
# ========================================================================
def sample_valid_lstm_config(trial):
    """
    Coherente con FlexibleLSTM:

    - num_layers >= 1
    - layer_sizes longitud == num_layers
    - si num_layers == 1:
        * native_mode = True
        * dropout = 0.0
        * dropout_layers = []
        * norm_layers = []
    - si uniform == False:
        * native_mode = False
    - si native_mode = True:
        * dropout_layers = []
        * norm_layers = []
        * dropout = dropout_raw
    - si native_mode = False y num_layers > 1:
        * dropout_layers puede estar vacío → dropout = 0.0
        * si dropout_layers no vacío → dropout > 0.0
    """

    all_sizes = [64, 128, 192, 256, 384, 448, 512, 640, 768, 896]

    # ---------------- NUM LAYERS ----------------
    num_layers = trial.suggest_int("num_layers", 1, 4)

    # ---------------- PATRÓN DE ARQUITECTURA ----------------
    pattern = trial.suggest_categorical(
        "architecture_pattern",
        ["uniform", "increasing", "decreasing"]
    )

    # ---------------- LAYER SIZES ----------------
    if pattern == "uniform":
        idx = trial.suggest_int("size_idx", 0, len(all_sizes) - 1)
        layer_sizes = [all_sizes[idx]] * num_layers

    elif pattern == "increasing":
        f = trial.suggest_int("first_size_idx", 0, len(all_sizes) - num_layers)
        layer_sizes = [all_sizes[f + i] for i in range(num_layers)]

    else:  # decreasing
        f = trial.suggest_int("first_size_idx", num_layers - 1, len(all_sizes) - 1)
        layer_sizes = [all_sizes[f - i] for i in range(num_layers)]

    # ---------------- IS UNIFORM ----------------
    uniform = all(h == layer_sizes[0] for h in layer_sizes)

    # ---------------- NATIVE MODE ----------------
    if num_layers == 1:
        native_mode = True
    else:
        if uniform:
            native_mode = trial.suggest_categorical("native_mode", [True, False])
        else:
            native_mode = False

    # ---------------- DROPOUT RAW ----------------
    dropout_raw = float(
        trial.suggest_categorical("dropout_raw", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    )

    # ---------------- DROPOUT / NORM LAYERS ----------------
    if num_layers == 1:
        dropout = 0.0
        dropout_layers = []
        norm_layers = []

    else:
        if native_mode:
            dropout_layers = []
            norm_layers = []
            dropout = dropout_raw
        else:
            # índices válidos: 1..num_layers-1
            dropout_flags = [
                trial.suggest_categorical(f"dropout_layer_{i}", [True, False])
                for i in range(1, num_layers)
            ]
            dropout_layers = [i for i, f in enumerate(dropout_flags, start=1) if f]

            norm_flags = [
                trial.suggest_categorical(f"norm_layer_{i}", [True, False])
                for i in range(1, num_layers)
            ]
            norm_layers = [i for i, f in enumerate(norm_flags, start=1) if f]

            if len(dropout_layers) == 0:
                dropout = 0.0
            else:
                if dropout_raw == 0.0:
                    dropout_fix = float(
                        trial.suggest_categorical("dropout_fix", [0.1, 0.2, 0.3, 0.4, 0.5])
                    )
                    dropout = dropout_fix
                else:
                    dropout = dropout_raw

    return {
        "layer_sizes": layer_sizes,
        "num_layers": num_layers,
        "dropout": dropout,
        "dropout_layers": dropout_layers,
        "norm_layers": norm_layers,
        "native_mode": native_mode,
    }


# ========================================================================
#   OBJECTIVE (entreno por fold)
# ========================================================================
def objective(trial, df_train):

    lstm_cfg = sample_valid_lstm_config(trial)

    bidirectional = trial.suggest_categorical("bidirectional", [False, True])
    pooling       = trial.suggest_categorical("pooling", ["mean", "max", "last", "attention"])
    fc_hidden     = trial.suggest_categorical("fc_hidden", [None, 128, 256, 512, 768, 1024])
    norm_post     = trial.suggest_categorical("norm_post_lstm", [True, False])

    lr           = trial.suggest_categorical("lr", [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 1e-3])
    weight_decay = trial.suggest_categorical("weight_decay", [0.0, 1e-7, 1e-6, 1e-5])
    batch_size   = trial.suggest_categorical("batch_size", BATCH_SIZE_CHOICES)
    optimizer_nm = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "SGD", "RMSprop"])
    norm_mode    = trial.suggest_categorical("norm", ["minmax", "L2"])

    activation = None
    if fc_hidden is not None:
        activation = trial.suggest_categorical(
            "activation", ["relu", "tanh", "gelu", "leakyrelu", "elu", "silu"])

    momentum_sgd = 0.0
    if optimizer_nm == "SGD":
        momentum_sgd = trial.suggest_categorical("momentum_sgd", [0.0, 0.7, 0.8, 0.9])

    # ----------------   DATA SPLIT   ----------------
    feat_cols = [c for c in df_train.columns if c.startswith("feat_")]
    video_ids = list(df_train["video_ID"].unique())
    labels_ids = [
        int(df_train[df_train["video_ID"] == vid]["shoot_zone"].iloc[0])
        for vid in video_ids
    ]

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    loss_fn = nn.CrossEntropyLoss()

    results_f1 = []

    # -------------------------------------------------------
    #                  K-FOLD TRAINING
    # -------------------------------------------------------
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(video_ids, labels_ids), start=1):

        tr_ids = [video_ids[i] for i in tr_idx]
        va_ids = [video_ids[i] for i in va_idx]

        df_tr = df_train[df_train["video_ID"].isin(tr_ids)]
        df_va = df_train[df_train["video_ID"].isin(va_ids)]

        train_seqs = [grp[feat_cols].values.astype(np.float32) for _, grp in df_tr.groupby("video_ID")]
        train_labs = [int(grp["shoot_zone"].iloc[0]) for _, grp in df_tr.groupby("video_ID")]

        val_seqs   = [grp[feat_cols].values.astype(np.float32) for _, grp in df_va.groupby("video_ID")]
        val_labs   = [int(grp["shoot_zone"].iloc[0]) for _, grp in df_va.groupby("video_ID")]

        # Normalización
        if norm_mode == "minmax":
            scaler = MinMaxScaler().fit(np.vstack(train_seqs))
            train_seqs = [scaler.transform(s) for s in train_seqs]
            val_seqs   = [scaler.transform(s) for s in val_seqs]
        else:
            train_seqs = [normalize(s, norm="l2", axis=1) for s in train_seqs]
            val_seqs   = [normalize(s, norm="l2", axis=1) for s in val_seqs]

        train_loader = DataLoader(
            [(torch.from_numpy(s), y) for s, y in zip(train_seqs, train_labs)],
            batch_size=batch_size, shuffle=True, collate_fn=collate_sequences
        )
        val_loader = DataLoader(
            [(torch.from_numpy(s), y) for s, y in zip(val_seqs, val_labs)],
            batch_size=batch_size, shuffle=False, collate_fn=collate_sequences
        )

        # -------------------------------------------------------
        #     CREAR MODELO 100% COHERENTE CON lstm_cfg
        # -------------------------------------------------------
        model = FlexibleLSTM(
            input_dim=len(feat_cols),
            hidden_dim=None,
            num_layers=lstm_cfg["num_layers"],
            bidirectional=bidirectional,
            dropout=lstm_cfg["dropout"],
            dropout_layers=lstm_cfg["dropout_layers"],
            norm_layers=lstm_cfg["norm_layers"],
            fc_hidden=fc_hidden,
            num_classes=3,
            activation=activation,
            pooling=pooling,
            norm_post_lstm=norm_post,
            layer_sizes=lstm_cfg["layer_sizes"],
            native_mode=lstm_cfg["native_mode"],
        ).to(DEVICE)

        # ---------------- OPTIMIZER ----------------
        if optimizer_nm == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_nm == "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_nm == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum_sgd, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Entrenamiento por fold
        best_f1 = 0.0
        patience = 0

        for ep in range(1, EPOCHS + 1):
            model.train()
            sum_loss = 0.0
            count    = 0

            for xb, lengths, yb in train_loader:
                xb, lengths, yb = xb.to(DEVICE), lengths.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()

                logits = model(xb, lengths)
                loss   = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()

                sum_loss += loss.item() * yb.size(0)
                count    += yb.size(0)
            avg_loss = sum_loss / max(1, count)

            # Validación
            model.eval()
            preds, labs = [], []
            with torch.no_grad():
                for xb, lengths, yb in val_loader:
                    xb, lengths, yb = xb.to(DEVICE), lengths.to(DEVICE), yb.to(DEVICE)
                    logits = model(xb, lengths)
                    preds.append(torch.argmax(logits, dim=1).cpu().numpy())
                    labs.append(yb.cpu().numpy())

            preds = np.concatenate(preds)
            labs  = np.concatenate(labs)
            f1 = f1_score(labs, preds, average="macro")

            if ep % 10 == 0 or ep == 1:
                print(f"Fold {fold_idx} | Epoch {ep} | Loss: {avg_loss:.4f} | F1: {f1:.4f} | Best F1: {best_f1:.4f}")

            # EARLY STOPPING
            if f1 > best_f1 + ES_MIN_DELTA:
                best_f1 = f1
                patience = 0
            else:
                patience += 1

            if patience >= ES_PATIENCE and ep >= MIN_EPOCHS:
                break

        results_f1.append(best_f1)

    return float(np.mean(results_f1))




# ========================================================================
#   OPTIMIZE ALL EMBEDDINGS + CLEAN CSV + BEST PARAMS (estilo MLP/TCN)
# ========================================================================
def optimize_embeddings():
    pruner  = optuna.pruners.HyperbandPruner(
        min_resource=MIN_EPOCHS,
        max_resource=EPOCHS * N_SPLITS,
        reduction_factor=2
    )
    sampler = optuna.samplers.TPESampler(seed=SEED)

    results = []  # para best_params_LSTM.csv

    all_sizes = [64, 128, 192, 256, 384, 448, 512, 640, 768, 896]

    for fname in os.listdir(DATA_DIR):
        if not fname.endswith(".csv"):
            continue

        print("\n" + "="*70)
        print(f"Optimizando LSTM → {fname}")
        print("="*70)

        base = os.path.splitext(fname)[0]
        out_trials = os.path.join(OUT_DIR, f"LSTM_Optuna_{base}.csv")

        if os.path.exists(out_trials):
            print("↩️ Ya existe. Lo salto.")
            continue

        df = pd.read_csv(os.path.join(DATA_DIR, fname), low_memory=False)

        vids = df["video_ID"].unique()
        labs = [int(df[df["video_ID"] == vid]["shoot_zone"].iloc[0]) for vid in vids]

        train_ids, _ = train_test_split(
            vids, test_size=TEST_SIZE, stratify=labs, random_state=SEED
        )
        df_train = df[df["video_ID"].isin(train_ids)]

        study = optuna.create_study(
            direction="maximize",
            study_name=f"lstm_{fname}",
            pruner=pruner,
            sampler=sampler
        )

        study.optimize(lambda t: objective(t, df_train),
                       n_trials=N_TRIALS, n_jobs=1)

        # ======================================================
        #     LIMPIEZA INTEGRADA (igual filosofía que TCN)
        # ======================================================
        df_raw = study.trials_dataframe()
        clean_rows = []

        for _, r in df_raw.iterrows():
            # algunos trials pueden no tener value usable
            if "value" in r.index and pd.isna(r["value"]):
                continue

            # params_* -> dict plano
            params = {
                k.replace("params_", ""): v
                for k, v in r.items()
                if isinstance(k, str) and k.startswith("params_") and pd.notna(v)
            }

            # si faltan claves mínimas
            if "num_layers" not in params or "architecture_pattern" not in params:
                continue

            num_layers = int(params["num_layers"])
            pattern = str(params["architecture_pattern"])

            # -------- layer_sizes --------
            if pattern == "uniform":
                if "size_idx" not in params:
                    continue
                idx = int(params["size_idx"])
                layer_sizes = [all_sizes[idx]] * num_layers

            elif pattern == "increasing":
                if "first_size_idx" not in params:
                    continue
                f = int(params["first_size_idx"])
                layer_sizes = [all_sizes[f + i] for i in range(num_layers)]

            else:  # decreasing
                if "first_size_idx" not in params:
                    continue
                f = int(params["first_size_idx"])
                layer_sizes = [all_sizes[f - i] for i in range(num_layers)]

            uniform = all(h == layer_sizes[0] for h in layer_sizes)

            # -------- native_mode --------
            if num_layers == 1:
                native_mode = True
            elif not uniform:
                native_mode = False
            else:
                native_mode = bool(params.get("native_mode", False))

            # -------- dropout_layers / norm_layers (solo modular) --------
            if num_layers > 1 and not native_mode:
                dropout_layers = sorted([
                    int(k.split("_")[-1])
                    for k, v in params.items()
                    if isinstance(k, str) and k.startswith("dropout_layer_") and bool(v)
                ])
                norm_layers = sorted([
                    int(k.split("_")[-1])
                    for k, v in params.items()
                    if isinstance(k, str) and k.startswith("norm_layer_") and bool(v)
                ])
            else:
                dropout_layers = []
                norm_layers = []

            # -------- dropout final --------
            raw_dropout = float(params.get("dropout_raw", 0.0))
            raw_fix = float(params.get("dropout_fix", 0.0)) if "dropout_fix" in params else 0.0

            if num_layers == 1:
                dropout = 0.0
            elif native_mode:
                dropout = raw_dropout
            else:
                if len(dropout_layers) == 0:
                    dropout = 0.0
                else:
                    dropout = raw_dropout if raw_dropout > 0 else raw_fix

            # -------- fila limpia base --------
            clean = {
                "layer_sizes": str(layer_sizes),
                "dropout_layers": str(dropout_layers),
                "norm_layers": str(norm_layers),
                "num_layers": num_layers,
                "native_mode": native_mode,
                "dropout": dropout,
                "architecture_pattern": pattern,

                "number": r.get("number", None),
                "value": r.get("value", None),
            }

            # metadata (si existe)
            for meta in ["state", "datetime_start", "datetime_complete", "duration"]:
                if meta in df_raw.columns:
                    clean[meta] = r.get(meta, None)

            # copiar el resto de hiperparámetros “reales”
            # (evita indices internos y auxiliares de reconstrucción)
            skip = {"size_idx", "first_size_idx", "dropout_raw", "dropout_fix", "native_mode"}
            for k, v in params.items():
                if isinstance(k, str) and (k.startswith("dropout_layer_") or k.startswith("norm_layer_")):
                    continue
                if k in skip:
                    continue
                clean[k] = v

            clean_rows.append(clean)

        clean_df = pd.DataFrame(clean_rows)

        # Orden columnas como MLP/TCN: meta primero
        first_cols = ["number", "value", "state", "datetime_start", "datetime_complete", "duration"]
        clean_df = clean_df[
            [c for c in first_cols if c in clean_df.columns] +
            [c for c in clean_df.columns if c not in first_cols]
        ]

        clean_df.to_csv(out_trials, index=False)

        # ======================================================
        #     BEST PARAMS (igual que MLP/TCN)
        # ======================================================
        if clean_df.empty or "value" not in clean_df.columns:
            print("⚠️ No hay trials válidos para este embedding.")
            continue

        best_num = study.best_trial.number
        best_row = clean_df[clean_df["number"] == best_num].iloc[0]

        summary = {"embedding": fname, "best_f1_cv": float(best_row["value"])}
        for k, v in best_row.items():
            if k in ["number", "value", "state", "datetime_start", "datetime_complete", "duration"]:
                continue
            summary[k] = v

        results.append(summary)
        print(f"\nMejores parámetros para {fname}: F1_cv = {summary['best_f1_cv']:.4f}")

    # ======================================================
    #     CSV GLOBAL best_params_LSTM.csv
    # ======================================================
    if results:
        summary_df = pd.DataFrame(results)
        order = ["embedding", "best_f1_cv"] + [c for c in summary_df.columns if c not in ["embedding", "best_f1_cv"]]
        summary_df = summary_df[order]
        summary_df["best_f1_cv"] = summary_df["best_f1_cv"].round(4)
        summary_df.to_csv(os.path.join(OUT_DIR, "best_params_LSTM.csv"), index=False)

    return results
