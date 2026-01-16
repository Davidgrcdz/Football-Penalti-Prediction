
# =========================
# Optuna + Hyperband (TCN) — versión limpia con dropout por bloque
# =========================
import os
import numpy as np
import pandas as pd
import optuna
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.amp import autocast, GradScaler

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, normalize
from sklearn.metrics import f1_score

from src.tcn.model import FlexibleTCN  # importa tu modelo donde corresponda

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
DATA_DIR   = "Gait_Embeddings_good"
OUT_DIR    = "results/TCN/Optuna"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED       = 42
EPOCHS     = 300
TEST_SIZE  = 0.1
N_SPLITS   = 5

ES_PATIENCE   = 40
MIN_EPOCHS    = 50
ES_MIN_DELTA  = 1e-3

os.makedirs(OUT_DIR, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)

# ------------------------------------------------------------------
# collate sequences
# ------------------------------------------------------------------
def collate_sequences(batch):
    seqs, labels = zip(*batch)
    lengths = torch.tensor([s.size(0) for s in seqs], dtype=torch.long)
    padded  = pad_sequence(seqs, batch_first=True)  # (B,T,D)
    labels  = torch.tensor(labels, dtype=torch.long)
    return padded, lengths, labels

# ------------------------------------------------------------------
# Helpers del espacio de búsqueda / reconstrucción
# ------------------------------------------------------------------
def make_dilations(trial, L):
    mode = trial.suggest_categorical("dilation_mode", ["pow2", "linear"])
    if mode == "pow2":
        start = trial.suggest_categorical("dilation_start", [1, 2])
        return [start * (2 ** i) for i in range(L)]
    else:
        d0   = trial.suggest_int("dilation_d0", 1, 3)
        step = trial.suggest_int("dilation_step", 1, 3)
        return [d0 + i * step for i in range(L)]

def build_channels(trial):
    base_sizes = [64, 96, 128, 192, 256, 384, 512, 640, 768, 896]
    L = trial.suggest_int("num_blocks", 1, 6)
    pattern = trial.suggest_categorical("channels_pattern", ["uniform", "increasing", "decreasing"])

    if pattern == "uniform":
        idx = trial.suggest_int("ch_idx", 0, len(base_sizes) - 1)
        return [base_sizes[idx]] * L
    elif pattern == "increasing":
        max_first = len(base_sizes) - L
        first_idx = trial.suggest_int("ch_first_idx", 0, max_first)
        return [base_sizes[first_idx + i] for i in range(L)]
    else:
        min_first = L - 1
        first_idx = trial.suggest_int("ch_first_idx", min_first, len(base_sizes) - 1)
        return [base_sizes[first_idx - i] for i in range(L)]

# Helpers para reconstrucción desde CSV limpio
def make_dilations_csv(params, L):
    mode = params["dilation_mode"]
    if mode == "pow2":
        start = int(params["dilation_start"])
        return [start * (2 ** i) for i in range(L)]
    else:
        d0   = int(params["dilation_d0"])
        step = int(params["dilation_step"])
        return [d0 + i * step for i in range(L)]

def build_channels_csv(params):
    base_sizes = [64, 96, 128, 192, 256, 384, 512, 640, 768, 896] 
    L        = int(params["num_blocks"])
    pattern  = params["channels_pattern"]

    if pattern == "uniform":
        idx = int(params["ch_idx"])
        return [base_sizes[idx]] * L
    elif pattern == "increasing":
        first_idx = int(params["ch_first_idx"])
        return [base_sizes[first_idx + i] for i in range(L)]
    else:
        first_idx = int(params["ch_first_idx"])
        return [base_sizes[first_idx - i] for i in range(L)]

def _to_bool(x):
    """Convierte cualquier cosa de Optuna (True/False, 'True'/'False', 1/0) a bool real."""
    return str(x).lower() in ("true", "1", "1.0")


# ------------------------------------------------------------------
# OBJETIVO DE OPTUNA
# ------------------------------------------------------------------
def objective(trial, df_train, num_classes=3):

    # Arquitectura TCN
    channels  = build_channels(trial)
    L         = len(channels)
    dilations = make_dilations(trial, L)

    convs_per_block = trial.suggest_int("convs_per_block", 1, 3)
    kernel_size     = trial.suggest_categorical("kernel_size", [2, 3, 5, 7, 9])
    activation      = trial.suggest_categorical(
        "activation", ["relu","gelu","prelu","silu","mish","leakyrelu","hardswish","elu"]
    )

    # LayerNorm por bloque
    ln_enable = [trial.suggest_categorical(f"ln_use_b{i}", [True, False]) for i in range(L)]

    # Dropout por bloque (enable/disable + p global)
    dr_enable = [trial.suggest_categorical(f"dr_use_b{i}", [False, True]) for i in range(L)]
    raw_dr_p  = trial.suggest_categorical("dr_p", [0.1, 0.2, 0.3, 0.4])
    dr_p      = raw_dr_p if any(dr_enable) else 0.0

    # Clasificador final
    pooling   = trial.suggest_categorical("pooling", ["mean", "max", "last", "attention"])
    fc_hidden = trial.suggest_categorical("fc_hidden", [None, 64, 128, 256, 512, 768])

    # Optimización
    lr           = trial.suggest_categorical("lr", [1e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3])
    weight_decay = trial.suggest_categorical("weight_decay", [0.0, 1e-7, 2e-7, 5e-7, 1e-6, 2e-6, 1e-5])
    batch_size   = trial.suggest_categorical("batch_size", [16, 32, 48, 64, 96, 128])
    optimizer_nm = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "SGD", "RMSprop"])
    momentum_sgd = trial.suggest_categorical("momentum_sgd", [0.0, 0.7, 0.8, 0.9]) if optimizer_nm == "SGD" else 0.0

    norm = trial.suggest_categorical("norm", ["minmax", "L2"])

    # Datos
    video_ids  = list(df_train.groupby("video_ID").groups.keys())
    labels_ids = [int(df_train[df_train["video_ID"] == vid]["shoot_zone"].iloc[0]) for vid in video_ids]
    feat_cols  = [c for c in df_train.columns if c.startswith("feat_")]

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold_f1s = []
    global_step_offset = 0
    criterion = nn.CrossEntropyLoss()

    is_cuda = torch.cuda.is_available()
    print_name = trial.study.study_name if hasattr(trial, "study") else "TCN_Study"

    for fold, (idx_tr, idx_va) in enumerate(skf.split(video_ids, labels_ids), start=1):

        tr_ids = [video_ids[i] for i in idx_tr]
        va_ids = [video_ids[i] for i in idx_va]

        df_tr = df_train[df_train["video_ID"].isin(tr_ids)]
        df_va = df_train[df_train["video_ID"].isin(va_ids)]

        train_seqs = [grp[feat_cols].values.astype(np.float32) for _, grp in df_tr.groupby("video_ID")]
        train_labs = [int(grp["shoot_zone"].iloc[0]) for _, grp in df_tr.groupby("video_ID")]

        val_seqs = [grp[feat_cols].values.astype(np.float32) for _, grp in df_va.groupby("video_ID")]
        val_labs = [int(grp["shoot_zone"].iloc[0]) for _, grp in df_va.groupby("video_ID")]

        # Normalización
        if norm == "minmax":
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

        # Construir listas reales para LN y Dropout
        ln_block_end = ln_enable  # ya es lista de bool

        if dr_p == 0.0:
            dr_enable = [False] * L
        dropout_list = [(dr_p if on else 0.0) for on in dr_enable]

        # Modelo
        model = FlexibleTCN(
            input_dim=len(feat_cols),
            channels=channels,
            kernel_size=kernel_size,
            dilations=dilations,
            convs_per_block=convs_per_block,
            dropout=dropout_list,       # 👈 lista tipo [0.2, 0.2, 0.0]
            activation=activation,
            use_weight_norm=trial.suggest_categorical("use_weight_norm", [True, False]),
            wn_on_skip=trial.suggest_categorical("wn_on_skip", [False, True]),
            ln_block_end=ln_block_end,
            pooling=pooling,
            fc_hidden=fc_hidden,
            num_classes=num_classes,
        ).to(DEVICE)

        # Optimizador
        if optimizer_nm == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_nm == "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_nm == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum_sgd, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)

        scaler = GradScaler(enabled=is_cuda)

        best_f1 = 0.0
        best_state = None
        patience_counter = 0

        # ------------------------ TRAIN ------------------------
        for epoch in range(1, EPOCHS + 1):
            model.train()
            for padded, lengths, labels in train_loader:
                padded, lengths, labels = padded.to(DEVICE), lengths.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad(set_to_none=True)

                with autocast(device_type="cuda", dtype=torch.float16, enabled=is_cuda):
                    logits = model(padded, lengths)
                    loss = criterion(logits, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            # -------------------- VALIDACIÓN --------------------
            model.eval()
            preds_all, labs_all = [], []
            with torch.no_grad():
                for padded, lengths, labels in val_loader:
                    padded, lengths, labels = padded.to(DEVICE), lengths.to(DEVICE), labels.to(DEVICE)
                    with autocast(device_type="cuda", dtype=torch.float16, enabled=is_cuda):
                        logits = model(padded, lengths)
                        loss = criterion(logits, labels)
                    preds_all.extend(torch.argmax(logits, dim=1).cpu().numpy())
                    labs_all.extend(labels.cpu().numpy())

            val_f1 = f1_score(labs_all, preds_all, average="macro")

            if epoch % 10 == 0:
                print(
                    f"[{print_name}] Fold {fold}/{N_SPLITS} | "
                    f"Ep {epoch}/{EPOCHS} | F1 {val_f1:.4f} | bestF1 {best_f1:.4f}"
                )

            # Pruner
            trial.report(val_f1, global_step_offset + epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            # Early stopping
            if epoch >= MIN_EPOCHS:
                if (val_f1 - best_f1) > ES_MIN_DELTA:
                    best_f1 = val_f1
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= ES_PATIENCE:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        fold_f1s.append(best_f1)
        global_step_offset += EPOCHS

    return float(np.mean(fold_f1s))


# ------------------------------------------------------------------
# OPTIMIZACIÓN GLOBAL Y LIMPIEZA DE CSV
# ------------------------------------------------------------------
def optimize_embeddings():

    pruner  = optuna.pruners.HyperbandPruner(
        min_resource=MIN_EPOCHS,
        max_resource=EPOCHS * N_SPLITS,
        reduction_factor=2
    )
    sampler = optuna.samplers.TPESampler(seed=SEED)

    results = []

    for fname in os.listdir(DATA_DIR):
        if not fname.endswith(".csv"):
            continue

        base = os.path.splitext(fname)[0]
        out_trials = os.path.join(OUT_DIR, f"TCN_Optuna_{base}.csv")

        if os.path.exists(out_trials):
            print(f"↩️ Ya existe {out_trials}. Lo salto.")
            continue

        print("\n" + "="*70)
        print(f"Optimizando (TCN) {fname}")
        print("="*70)

        df = pd.read_csv(os.path.join(DATA_DIR, fname))

        vids = list(df.groupby("video_ID").groups.keys())
        labs = [int(df[df["video_ID"] == vid]["shoot_zone"].iloc[0]) for vid in vids]

        train_ids, _ = train_test_split(
            vids, test_size=TEST_SIZE, stratify=labs, random_state=SEED
        )
        df_train = df[df["video_ID"].isin(train_ids)]

        study = optuna.create_study(
            direction="maximize",
            study_name=f"tcn_{fname}",
            pruner=pruner,
            sampler=sampler
        )

        study.optimize(lambda t: objective(t, df_train), n_trials=50, timeout=10800)

        # ===================== LIMPIEZA DE TRIALS =====================
        df_raw = study.trials_dataframe()
        clean_rows = []

        for _, r in df_raw.iterrows():
            params = {
                k.replace("params_", ""): v
                for k, v in r.items()
                if k.startswith("params_")
            }

            L        = int(params["num_blocks"])
            channels = build_channels_csv(params)
            dilations = make_dilations_csv(params, L)

            # Reconstruir ln_block_end y dropout como listas
            ln_flags = [_to_bool(params.get(f"ln_use_b{i}", False)) for i in range(L)]
            dr_flags = [_to_bool(params.get(f"dr_use_b{i}", False)) for i in range(L)]
            dr_p     = float(params.get("dr_p", 0.0))
            
            # corrección: si dr_p == 0.0, aunque los flags digan que sí, en la práctica NO hubo dropout
            if dr_p == 0.0:
                dr_flags = [False] * L

            dropout_list = [(dr_p if dr_flags[i] else 0.0) for i in range(L)]

            clean_cfg = {
                "channels": str(channels),
                "dilations": str(dilations),
                "ln_block_end": str(ln_flags),  # ej. "[True, False, True]"
                "dropout": str(dropout_list),   # ej. "[0.2, 0.2, 0.0]"
            }

            # Copiar hiperparámetros sin llaves internas de índices
            for key, val in params.items():
                if key.startswith("ln_use_b") or key.startswith("dr_use_b"):
                    continue
                if key in ["ch_idx", "ch_first_idx", "dilation_start", "dilation_d0", "dilation_step"]:
                    continue
                if key == "dr_p":
                    continue  # ya lo hemos incorporado en "dropout"
                clean_cfg[key] = val

            clean_cfg["value"]  = r["value"]
            clean_cfg["number"] = r["number"]
            for meta in ["state", "datetime_start", "datetime_complete", "duration"]:
                if meta in df_raw.columns:
                    clean_cfg[meta] = r[meta]

            clean_rows.append(clean_cfg)

        clean_df = pd.DataFrame(clean_rows)
        first_cols = ["number", "value", "state", "datetime_start", "datetime_complete", "duration"]
        clean_df = clean_df[
            [c for c in first_cols if c in clean_df.columns] +
            [c for c in clean_df.columns if c not in first_cols]
        ]

        clean_df.to_csv(out_trials, index=False)

        # ================== Resumen best params ==================
        best_num = study.best_trial.number
        best_row = clean_df[clean_df["number"] == best_num].iloc[0]

        row_summary = {"embedding": fname, "best_f1_cv": best_row["value"]}
        for k, v in best_row.items():
            if k in ["number", "value", "state", "datetime_start", "datetime_complete", "duration"]:
                continue
            row_summary[k] = v

        results.append(row_summary)
        print(f"\nMejores parámetros para {fname}: F1_cv = {row_summary['best_f1_cv']:.4f}")

    # ================== Resumen global ==================
    if results:
        summary_df = pd.DataFrame(results)
        best_cols = ["embedding", "best_f1_cv"] + \
                    [c for c in summary_df.columns if c not in ["embedding", "best_f1_cv"]]
        summary_df = summary_df[best_cols]
        summary_df.to_csv(os.path.join(OUT_DIR, "best_params_TCN.csv"), index=False)

    return results
