# =========================
# Optuna + Hyperband (Transformer)
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

from src.transformer.model import FlexibleTransformer

# ---------- Config ----------
DATA_DIR   = "Gait_Embeddings_good"
OUT_DIR    = "results/Transformer/Optuna2"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED       = 42
EPOCHS     = 300
TEST_SIZE  = 0.10
N_SPLITS   = 5

ES_PATIENCE   = 40
MIN_EPOCHS    = 50
ES_MIN_DELTA  = 1e-3


# ---------- Collate ----------
def collate_transformer(batch):
    """
    batch: lista de (tensor_seq, label)
    Devuelve:
      - batch_padded: (B, T_max, D)
      - mask: (B, T_max) bool (True=valido)
      - labels: (B,)
    """
    seqs, labels = zip(*batch)
    lengths = [s.size(0) for s in seqs]
    max_len = max(lengths)
    feat_dim = seqs[0].size(1)

    batch_padded = torch.zeros(len(seqs), max_len, feat_dim, dtype=seqs[0].dtype)
    mask = torch.zeros(len(seqs), max_len, dtype=torch.bool)
    for i, s in enumerate(seqs):
        L = s.size(0)
        batch_padded[i, :L] = s
        mask[i, :L] = True

    labels = torch.tensor(labels, dtype=torch.long)
    return batch_padded, mask, labels


# ---------- Objective ----------
def objective(trial, df_train, num_classes=3):
    feat_cols = [c for c in df_train.columns if c.startswith("feat_")]
    input_dim = len(feat_cols)
    is_cuda   = torch.cuda.is_available()

    # -------------------------
    # Espacio de búsqueda modelo
    # -------------------------
    base_dims = [64, 96, 128, 192, 256, 384, 512, 640, 768]

    num_heads = trial.suggest_categorical("num_heads", [2, 4, 8, 16])

    # Filtro opcional por divisibilidad (aquí todos son múltiplos de 2/4/8/16)
    dims_space = [d for d in base_dims if d % num_heads == 0]

    num_layers = trial.suggest_int("num_layers", 1, 4)

    pattern = trial.suggest_categorical("model_dim_pattern",
                                        ["uniform", "increasing", "decreasing"])

    if pattern == "uniform":
        dim_idx = trial.suggest_int("model_dim_idx", 0, len(dims_space) - 1)
        model_dims = [dims_space[dim_idx]] * num_layers

    elif pattern == "increasing":
        max_first = len(dims_space) - num_layers
        first_idx = trial.suggest_int("model_dim_first_idx", 0, max_first)
        model_dims = [dims_space[first_idx + i] for i in range(num_layers)]

    else:  # decreasing
        min_first = num_layers - 1
        first_idx = trial.suggest_int("model_dim_first_idx", min_first, len(dims_space) - 1)
        model_dims = [dims_space[first_idx - i] for i in range(num_layers)]

    dim_feedforward = trial.suggest_categorical("dim_feedforward", [128, 256, 384, 512, 768])
    dropout = trial.suggest_categorical("dropout", [0.0, 0.1, 0.2, 0.3, 0.4])
    activation = trial.suggest_categorical("activation", ["relu", "gelu", "silu", "mish", "prelu", "leakyrelu", "hardswish", "elu"])
    norm_type = trial.suggest_categorical("norm_type", ["prenorm", "postnorm"])
    pos_encoding = trial.suggest_categorical("pos_encoding", ["learned", "sinusoidal"])
    pooling = trial.suggest_categorical("pooling", ["cls", "mean", "max"])
    use_cls_token = (pooling == "cls")

    # -------------------------
    # Hiperparámetros optimización
    # -------------------------
    lr = trial.suggest_categorical("lr", [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3])
    weight_decay = trial.suggest_categorical("weight_decay",[0.0, 1e-7, 1e-6, 1e-5])
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 48, 64, 96, 128])
    optimizer_nm = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "SGD", "RMSprop"])
    momentum_sgd = trial.suggest_categorical("momentum_sgd", [0.0, 0.7, 0.8, 0.9]) if optimizer_nm == "SGD" else 0.0

    norm_in = trial.suggest_categorical("norm", ["minmax", "L2"])

    # -------------------------
    # K-Fold estratificado por video_ID
    # -------------------------
    video_ids = list(df_train.groupby("video_ID").groups.keys())
    labels_ids = [int(df_train[df_train["video_ID"] == vid]["shoot_zone"].iloc[0])
                  for vid in video_ids]

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    fold_f1s = []
    global_step_offset = 0
    study_name = trial.study.study_name if hasattr(trial, "study") else "Transformer_Study"

    for fold, (idx_tr, idx_va) in enumerate(skf.split(video_ids, labels_ids), start=1):
        tr_ids = [video_ids[i] for i in idx_tr]
        va_ids = [video_ids[i] for i in idx_va]

        df_tr = df_train[df_train["video_ID"].isin(tr_ids)]
        df_va = df_train[df_train["video_ID"].isin(va_ids)]

        def seqs_and_labels(df):
            seqs, labs = [], []
            for vid, grp in df.groupby("video_ID"):
                arr = grp[feat_cols].values.astype(np.float32)
                seqs.append(arr)
                labs.append(int(grp["shoot_zone"].iloc[0]))
            return seqs, labs

        train_seqs, train_labs = seqs_and_labels(df_tr)
        val_seqs, val_labs = seqs_and_labels(df_va)

        # normalización
        if norm_in == "minmax":
            scaler = MinMaxScaler().fit(np.vstack(train_seqs))
            train_seqs = [scaler.transform(s) for s in train_seqs]
            val_seqs   = [scaler.transform(s) for s in val_seqs]
        else:
            train_seqs = [normalize(s, norm="l2", axis=1) for s in train_seqs]
            val_seqs   = [normalize(s, norm="l2", axis=1) for s in val_seqs]

        # DataLoaders
        train_list = [(torch.from_numpy(s), y) for s, y in zip(train_seqs, train_labs)]
        val_list   = [(torch.from_numpy(s), y) for s, y in zip(val_seqs, val_labs)]
        train_loader = DataLoader(train_list, batch_size=batch_size,
                                  shuffle=True, collate_fn=collate_transformer)
        val_loader   = DataLoader(val_list, batch_size=batch_size,
                                  shuffle=False, collate_fn=collate_transformer)

        # max_seq_len para PE
        max_seq_len = max(len(s) for s in train_seqs)

        # Modelo
        model = FlexibleTransformer(
            input_dim=input_dim,
            model_dim=model_dims,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_type=norm_type,
            pos_encoding=pos_encoding,
            use_cls_token=use_cls_token,
            pooling=pooling,
            num_classes=num_classes,
            max_seq_len=max_seq_len
        ).to(DEVICE)

        # Optimizador
        if optimizer_nm == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                         weight_decay=weight_decay)
        elif optimizer_nm == "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                          weight_decay=weight_decay)
        elif optimizer_nm == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                        momentum=momentum_sgd,
                                        weight_decay=weight_decay)
        else:
            optimizer = torch.optim.RMSprop(model.parameters(), lr=lr,
                                            weight_decay=weight_decay)

        criterion = nn.CrossEntropyLoss()
        scaler = GradScaler(enabled=is_cuda)

        best_f1_fold = 0.0
        best_state_fold = None
        patience_counter = 0

        # -------------------------
        # Training loop por fold
        # -------------------------
        for epoch in range(1, EPOCHS + 1):
            model.train()
            for xb, mask, yb in train_loader:
                xb, mask, yb = xb.to(DEVICE), mask.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad(set_to_none=True)

                with autocast(device_type="cuda", dtype=torch.float16, enabled=is_cuda):
                    logits = model(xb, mask)
                    loss = criterion(logits, yb)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            # Validación
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                with autocast(device_type="cuda", dtype=torch.float16, enabled=is_cuda):
                    for xb, mask, yb in val_loader:
                        xb, mask, yb = xb.to(DEVICE), mask.to(DEVICE), yb.to(DEVICE)
                        logits = model(xb, mask)
                        preds = torch.argmax(logits, dim=1).cpu().numpy()
                        all_preds.extend(preds)
                        all_labels.extend(yb.cpu().numpy())

            val_f1 = f1_score(all_labels, all_preds, average="macro")

            if epoch == 1 or epoch % 10 == 0:
                print(
                    f"[{study_name}] Fold {fold}/{N_SPLITS} | "
                    f"Ep {epoch:03d}/{EPOCHS} | F1 {val_f1:.4f} | bestF1 {best_f1_fold:.4f}"
                )

            step = global_step_offset + epoch
            trial.report(val_f1, step=step)
            if trial.should_prune():
                raise optuna.TrialPruned()

            if epoch >= MIN_EPOCHS:
                if (val_f1 - best_f1_fold) > ES_MIN_DELTA:
                    best_f1_fold = val_f1
                    best_state_fold = {k: v.detach().cpu().clone()
                                       for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= ES_PATIENCE:
                    print(f"[{study_name}] Fold {fold}/{N_SPLITS} -> Early stopping en época {epoch}")
                    break

        if best_state_fold is not None:
            model.load_state_dict(best_state_fold)

        fold_f1s.append(best_f1_fold)
        global_step_offset += EPOCHS

    return float(np.mean(fold_f1s))


# ---------- Optimize embeddings ----------
def optimize_embeddings():
    pruner  = optuna.pruners.HyperbandPruner(
        min_resource=MIN_EPOCHS,
        max_resource=EPOCHS * N_SPLITS,
        reduction_factor=2
    )
    sampler = optuna.samplers.TPESampler(seed=SEED)

    results_rows = []

    for fname in os.listdir(DATA_DIR):
        if not fname.endswith(".csv"):
            continue

        print("\n" + "="*70)
        print(f"Optimizando (Transformer) {fname}")
        print("="*70)

        base = os.path.splitext(fname)[0]
        out_trials = os.path.join(OUT_DIR, f"Transformer_Optuna_{base}.csv")
        if os.path.exists(out_trials):
            print(f"↩️ Ya existe {out_trials}. Lo salto.")
            continue

        df = pd.read_csv(os.path.join(DATA_DIR, fname))

        # Split 90/10 por video_ID
        vids = list(df.groupby("video_ID").groups.keys())
        labs = [int(df[df["video_ID"] == vid]["shoot_zone"].iloc[0]) for vid in vids]
        train_ids, _ = train_test_split(
            vids, test_size=TEST_SIZE, stratify=labs, random_state=SEED
        )
        df_train = df[df["video_ID"].isin(train_ids)]

        study = optuna.create_study(
            direction="maximize",
            study_name=f"transformer_{fname}",
            pruner=pruner,
            sampler=sampler
        )

        study.optimize(lambda t: objective(t, df_train),
                       n_trials=50,
                       n_jobs=1,
                       timeout=10800)

        # ------------- Limpieza de trials -------------
        df_raw = study.trials_dataframe()
        clean_rows = []

        base_dims = [64, 96, 128, 192, 256, 384, 512, 640, 768]

        for _, r in df_raw.iterrows():
            params = {k.replace("params_", ""): v
                      for k, v in r.items() if k.startswith("params_")}

            L = int(params["num_layers"])
            pattern = params["model_dim_pattern"]
            num_heads = int(params["num_heads"])

            # dims_space (aquí todos los base_dims sirven, pero lo dejamos por claridad)
            dims_space = [d for d in base_dims if d % num_heads == 0]

            if pattern == "uniform":
                idx = int(params["model_dim_idx"])
                dims = [dims_space[idx]] * L
            elif pattern == "increasing":
                f = int(params["model_dim_first_idx"])
                dims = [dims_space[f + i] for i in range(L)]
            else:
                f = int(params["model_dim_first_idx"])
                dims = [dims_space[f - i] for i in range(L)]

            clean = {
                "model_dims": str(dims)
            }

            # Copiar el resto de hiperparámetros salvo los de índices/patrón
            for k, v in params.items():
                if k in ["model_dim_pattern", "model_dim_idx", "model_dim_first_idx"]:
                    continue
                clean[k] = v

            clean["number"] = r["number"]
            clean["value"]  = r["value"]

            for meta in ["state", "datetime_start", "datetime_complete", "duration"]:
                if meta in df_raw.columns:
                    clean[meta] = r[meta]

            clean_rows.append(clean)

        clean_df = pd.DataFrame(clean_rows)
        first_cols = ["number", "value", "state", "datetime_start", "datetime_complete", "duration"]
        clean_df = clean_df[[c for c in first_cols if c in clean_df.columns] +
                            [c for c in clean_df.columns if c not in first_cols]]

        clean_df.to_csv(out_trials, index=False)

        # ------------- Best params por embedding -------------
        best_num = study.best_trial.number
        best_row = clean_df[clean_df["number"] == best_num].iloc[0]

        row_summary = {
            "embedding": fname,
            "best_f1_cv": best_row["value"]
        }
        for k, v in best_row.items():
            if k in ["number", "value", "state", "datetime_start", "datetime_complete", "duration"]:
                continue
            row_summary[k] = v

        results_rows.append(row_summary)

        print(f"\nMejores parámetros para {fname}: F1_cv = {row_summary['best_f1_cv']:.4f}")

    if results_rows:
        summary_df = pd.DataFrame(results_rows)
        best_cols = ["embedding", "best_f1_cv"] + \
                    [c for c in summary_df.columns if c not in ["embedding", "best_f1_cv"]]
        summary_df = summary_df[best_cols]
        summary_df.to_csv(os.path.join(OUT_DIR, "best_params_Transformer.csv"), index=False)
        return summary_df

    return None
