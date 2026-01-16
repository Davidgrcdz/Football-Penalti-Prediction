# =========================
# Optuna MLP (Objective + Optimize) - VERSIÓN UNIFICADA
# =========================
import os
import numpy as np
import pandas as pd
import optuna
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, normalize
from sklearn.metrics import f1_score

from src.mlp.model import FlexibleMLP

# ---------------------------
# Directorios principales
# ---------------------------
DATA_DIR = "Gait_Embeddings_good"
OUT_DIR  = "results/MLP/OPTUNA DEFINITIVO"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------
# Configuración global
# ---------------------------
SEED         = 42
EPOCHS       = 300
N_SPLITS     = 5
TEST_SIZE    = 0.1
MIN_EPOCHS   = 50
ES_PATIENCE  = 40
ES_MIN_DELTA = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {DEVICE}")


# ============================================================
#                  Objective FUNCTION
# ============================================================
def objective(trial, df_train):
    feat_cols = [c for c in df_train.columns if c.startswith("feat_")]

    # -------------------------
    # Hiperparámetros MLP
    # -------------------------
    pooling = trial.suggest_categorical("pooling", ["mean", "max"])
    norm_in = trial.suggest_categorical("norm", ["minmax", "L2"])
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 48, 64, 96, 128])
    num_layers = trial.suggest_int("num_layers", 1, 6)
    base_dim   = trial.suggest_categorical("base_dim", [1024, 896, 768, 640, 512, 384, 256, 192, 128])
    pattern    = trial.suggest_categorical("pattern", ["uniform", "increasing", "decreasing"])
    activation = trial.suggest_categorical("activation", ["relu","tanh","gelu","leakyrelu","elu","silu"])

    # <<< NUEVO: PCA
    use_pca   = trial.suggest_categorical("use_pca", [True, False])
    pca_frac  = None
    if use_pca:
        pca_frac = trial.suggest_categorical("pca_frac", [0.5, 0.6, 0.7, 0.8, 0.9])

    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "SGD", "RMSprop"])
    momentum = trial.suggest_categorical("momentum_sgd", [0.0, 0.7, 0.8, 0.9]) if optimizer_name == "SGD" else None

    lr = trial.suggest_categorical("lr", [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3])
    weight_decay = trial.suggest_categorical("weight_decay", [0.0, 1e-7, 1e-6, 1e-5, 1e-4])

    # -------------------------
    # Arquitectura MLP
    # -------------------------
    def build_hidden_layers(base, L, patt):
        if patt == "uniform":
            return [base] * L
        if patt == "increasing":
            return [max(64, base // (2**(L - i - 1))) for i in range(L)]
        if patt == "decreasing":
            return [max(64, base // (2**i)) for i in range(L)]

    hidden_layers = build_hidden_layers(base_dim, num_layers, pattern)

    # <<< NUEVO: norm_layers y dropout_layers multinivel
    norm_layers = [i for i in range(1, num_layers) if trial.suggest_categorical(f"norm_layer_{i}", [True, False])]
    dropout_layers = [i for i in range(1, num_layers) if trial.suggest_categorical(f"dropout_layer_{i}", [True, False])]

    
    if len(dropout_layers) == 0:
        dropout = 0.0
    else:
        dropout = trial.suggest_categorical("dropout", [0.1, 0.2, 0.3, 0.4, 0.5])


    # -------------------------
    # Crear X,y con pooling
    # -------------------------
    X, y = [], []

    for vid, grp in df_train.groupby("video_ID"):
        arr = grp[feat_cols].values.astype(np.float32)
        pooled = arr.mean(axis=0) if pooling == "mean" else arr.max(axis=0)
        X.append(pooled)
        y.append(int(grp["shoot_zone"].iloc[0]))

    X = np.stack(X)
    y = np.array(y)

    # <<< NUEVO: aplicar PCA si toca
    if use_pca:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=pca_frac, random_state=SEED).fit(X)
        X = pca.transform(X)

    # -------------------------
    # K-Fold CV
    # -------------------------
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold_f1s = []
    global_step_offset = 0
    is_cuda = torch.cuda.is_available()

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        # normalización
        if norm_in == "minmax":
            scaler = MinMaxScaler().fit(X_tr)
            X_tr = scaler.transform(X_tr)
            X_va = scaler.transform(X_va)
        else:
            X_tr = normalize(X_tr, norm="l2")
            X_va = normalize(X_va, norm="l2")

        # DataLoader
        train_ds = TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).long())
        val_ds   = TensorDataset(torch.from_numpy(X_va).float(), torch.from_numpy(y_va).long())
        train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_ld   = DataLoader(val_ds, batch_size=batch_size)

        # Modelo
        model = FlexibleMLP(
            input_dim=X_tr.shape[1],
            hidden_layers=hidden_layers,
            activation=activation,
            dropout=dropout,
            normalization_layers=norm_layers,
            dropout_layers=dropout_layers
        ).to(DEVICE)

                # -------------------------
        # Optimizer
        # -------------------------
        if optimizer_name == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        elif optimizer_name == "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        elif optimizer_name == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

        else:
            optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)


        loss_fn = nn.CrossEntropyLoss()
        scaler = GradScaler(enabled=is_cuda)

        best_f1 = 0.0
        best_state = None
        patience = 0

        # ===== TRAINING LOOP =====
        for ep in range(1, EPOCHS + 1):
            model.train()
            train_loss_sum = 0.0
            train_count = 0

            for xb, yb in train_ld:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad(set_to_none=True)

                with autocast(device_type="cuda", dtype=torch.float16, enabled=is_cuda):
                    logits = model(xb)
                    loss = loss_fn(logits, yb)

                train_loss_sum += loss.item() * xb.size(0)
                train_count    += xb.size(0)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            # train loss (unused but kept for consistency)
            train_loss = train_loss_sum / max(1, train_count)

            # Validation
            model.eval()
            preds_all, labs_all = [], []
            with torch.no_grad():
                with autocast(device_type="cuda", dtype=torch.float16, enabled=is_cuda):
                    for xb, yb in val_ld:
                        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                        logits = model(xb)
                        preds_all.extend(torch.argmax(logits, dim=1).cpu().numpy())
                        labs_all.extend(yb.cpu().numpy())

            val_f1 = f1_score(labs_all, preds_all, average="macro")

            if ep % 10 == 0 or ep == 1:
                print(f"[Fold {fold}] Ep {ep:03d}/{EPOCHS} | Train Loss {train_loss:.4f} | Val F1 {val_f1:.4f}")

            # Pruning
            trial.report(val_f1, global_step_offset + ep)
            if trial.should_prune():
                raise optuna.TrialPruned()

            # Early stopping
            if ep >= MIN_EPOCHS:
                if (val_f1 - best_f1) > ES_MIN_DELTA:
                    best_f1 = val_f1
                    best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
                    patience = 0
                else:
                    patience += 1
                if patience >= ES_PATIENCE:
                    print(f"[Fold {fold}] Early stopping en época {ep}")
                    break
        
        if best_state is not None:
            model.load_state_dict(best_state)

        fold_f1s.append(best_f1)
        global_step_offset += EPOCHS

    return float(np.mean(fold_f1s))




# ============================================================
#            optimize_embeddings() MLP
# ============================================================
def optimize_embeddings():
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=MIN_EPOCHS,
        max_resource=EPOCHS * N_SPLITS,
        reduction_factor=2
    )
    sampler = optuna.samplers.TPESampler(seed=SEED)

    results = []

    for fname in os.listdir(DATA_DIR):
        if not fname.endswith(".csv"):
            continue

        print("\n" + "="*70)
        print(f"Optimizando MLP → {fname}")
        print("="*70)

        base = os.path.splitext(fname)[0]
        out_trials = os.path.join(OUT_DIR, f"MLP_Optuna_{base}.csv")

        if os.path.exists(out_trials):
            print("↩️ Ya existe. Lo salto.")
            continue

        df = pd.read_csv(os.path.join(DATA_DIR, fname))

        # Split inicial 90/10
        vids = list(df["video_ID"].unique())
        labs = [int(df[df["video_ID"] == vid]["shoot_zone"].iloc[0]) for vid in vids]

        train_ids, _ = train_test_split(
            vids, test_size=TEST_SIZE, stratify=labs, random_state=SEED
        )
        df_train = df[df["video_ID"].isin(train_ids)]

        # Estudio
        study = optuna.create_study(
            direction="maximize",
            study_name=f"mlp_{fname}",
            pruner=pruner,
            sampler=sampler
        )

        study.optimize(lambda t: objective(t, df_train),
                       n_trials=50, n_jobs=1, timeout=7200)

        # Guardar trials
        df_raw = study.trials_dataframe()
        clean_rows = []

        # reconstrucción post-process (MISMA QUE ARRIBA)
        def build_hidden_layers(base, L, patt):
            if patt == "uniform":
                return [base] * L
            if patt == "increasing":
                return [max(64, base // (2**(L - i - 1))) for i in range(L)]
            if patt == "decreasing":
                return [max(64, base // (2**i)) for i in range(L)]

        for _, r in df_raw.iterrows():
            # params_* → dict plano
            params = {
                k.replace("params_", ""): v
                for k, v in r.items()
                if k.startswith("params_")
            }

            L       = int(params["num_layers"])
            base    = int(params["base_dim"])
            pattern = params["pattern"]

            hidden_layers = build_hidden_layers(base, L, pattern)

            # reconstruir listas de capas con norm / dropout
            norm_layers = [
                i for i in range(1, L)
                if bool(params.get(f"norm_layer_{i}", False))
            ]
            dropout_layers = [
                i for i in range(1, L)
                if bool(params.get(f"dropout_layer_{i}", False))
            ]

            clean = {
                "hidden_layers": str(hidden_layers),
                "norm_layers": str(norm_layers),
                "dropout_layers": str(dropout_layers),
            }

            # copiar el resto de hiperparámetros,
            # excluyendo los flags norm_layer_i / dropout_layer_i
            for k, v in params.items():
                if k.startswith("norm_layer_") or k.startswith("dropout_layer_"):
                    continue
                clean[k] = v

            clean["number"] = r["number"]
            clean["value"]  = r["value"]

            # Campos meta
            for meta in ["state", "datetime_start", "datetime_complete", "duration"]:
                if meta in df_raw.columns:
                    clean[meta] = r[meta]

            clean_rows.append(clean)

        clean_df = pd.DataFrame(clean_rows)
        order = ["number", "value", "state", "datetime_start", "datetime_complete", "duration"]
        clean_df = clean_df[[c for c in order if c in clean_df.columns] +
                            [c for c in clean_df.columns if c not in order]]

        clean_df.to_csv(out_trials, index=False)

        # Best params
        best_num = study.best_trial.number
        best_row = clean_df[clean_df["number"] == best_num].iloc[0]

        summary = {"embedding": fname, "best_f1_cv": best_row["value"]}
        for k, v in best_row.items():
            if k not in ["number", "value", "state", "datetime_start", "datetime_complete", "duration"]:
                summary[k] = v

        results.append(summary)

    # Guardar resumen global
    sum_df = pd.DataFrame(results)
    order = ["embedding", "best_f1_cv"] + [c for c in sum_df.columns if c not in ["embedding", "best_f1_cv"]]
    sum_df = sum_df[order]
    sum_df["best_f1_cv"] = sum_df["best_f1_cv"].round(4)
    sum_df.to_csv(os.path.join(OUT_DIR, "best_params_MLP.csv"), index=False)

    return sum_df
