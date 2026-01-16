import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from torch.amp import autocast, GradScaler

from src.utils.compute_class_weights import compute_class_weights_from_loader



def run_training_mlp(model, train_loader, test_loader, epochs, lr, weight_decay, optimizer_name, momentum_sgd=None,
    use_class_weights=False,
    class_weights=None,
    label_smoothing=0.0,
    clip_grad_norm=None
):
    device = next(model.parameters()).device
    is_cuda = torch.cuda.is_available()

    # ------------ OPTIMIZER ------------
    name = optimizer_name.lower()
    if name == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum_sgd or 0.0, weight_decay=weight_decay)
    elif name == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"[MLP] Optimizador no soportado: {optimizer_name}")

    
    # ------------ CLASS WEIGHTS ------------
    cw_tensor = None
    if use_class_weights:
        if class_weights is not None:
            cw_tensor = torch.as_tensor(class_weights, dtype=torch.float32, device=device)
            print(f"[CW] provided: {cw_tensor.detach().cpu().tolist()}")
        else:
            cw_tensor, weights = compute_class_weights_from_loader(
                train_loader=train_loader,
                num_classes=3,              # ajusta si cambias nº de clases
                device=device,
                method="sklearn",           # o "manual"
                normalize=False             # True si quieres media=1
            )


    loss_fn_train = nn.CrossEntropyLoss(weight=cw_tensor, label_smoothing=label_smoothing)
    loss_fn_test  = nn.CrossEntropyLoss()

    scaler = GradScaler(enabled=is_cuda)

    # ------------ HISTORY ------------
    history = {"train_loss": [], "test_loss": [], "test_acc": [], "test_f1": [], "f1_per_class": []}

    best_f1_checkpoint = -1.0
    best_epoch = 0
    best_state = None
    train_loss_best_state = None
    best_acc = None
    test_loss_best_state = None
    best_f1_per_class = None

    # ============================================================
    #                 EARLY STOPPING HÍBRIDO
    # ============================================================
    loss_threshold = 0.75
    low_loss_stop = 0.12
    min_delta_loss = 0.001
    min_delta_f1   = 0.001

    patience_loss = 100
    patience_f1   = 100

    best_loss = float("inf")
    best_f1_es = -1.0

    no_improve_loss = 0
    no_improve_f1 = 0
    consecutive_low = 0
    early_stopped = False

    t0 = time.perf_counter()

    # ======================================================
    #                   TRAINING LOOP
    # ======================================================
    for ep in range(1, epochs + 1):

        # -------- TRAIN --------
        model.train()
        sum_loss = 0.0
        total = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=is_cuda):
                logits = model(xb)
                loss = loss_fn_train(logits, yb)

            scaler.scale(loss).backward()

            if clip_grad_norm is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), float(clip_grad_norm))

            scaler.step(optimizer)
            scaler.update()

            sum_loss += loss.item() * yb.size(0)
            total += yb.size(0)

        train_loss = sum_loss / max(1, total)
        history["train_loss"].append(train_loss)

        # -------- TEST --------
        model.eval()
        sum_test = 0.0
        total_test = 0
        preds, labels = [], []

        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                with autocast("cuda", enabled=is_cuda):
                    logits = model(xb)
                    tloss = loss_fn_test(logits, yb)

                sum_test += tloss.item() * yb.size(0)
                total_test += yb.size(0)
                preds.append(logits.argmax(1).cpu().numpy())
                labels.append(yb.cpu().numpy())

        test_loss = sum_test / max(1, total_test)
        y_pred = np.concatenate(preds)
        y_true = np.concatenate(labels)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()

        history["test_loss"].append(test_loss)
        history["test_acc"].append(acc)
        history["test_f1"].append(f1)
        history["f1_per_class"].append(f1_per_class)

        if ep % 10 == 0 or ep == 1:
            print(f"[MLP] Ep {ep:03d}/{epochs} | "
                  f"Train {train_loss:.4f} | Test {test_loss:.4f} | F1 {f1:.4f}")

        # ---------- CHECKPOINT (best F1) ----------
        if train_loss < loss_threshold and f1 > best_f1_checkpoint:
            best_f1_checkpoint = f1
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            train_loss_best_state = train_loss
            test_loss_best_state = test_loss
            best_acc = acc
            best_f1_per_class = f1_per_class

        # ---------- EARLY STOP: pérdida muy baja ----------
        if train_loss < low_loss_stop:
            consecutive_low += 1
            if consecutive_low >= 5:
                print(f"[MLP] -> Early stopping: pérdida muy baja ({train_loss:.6f})")
                break
        else:
            consecutive_low = 0

        # =====================================================
        #             EARLY STOP HÍBRIDO (nuevo)
        # =====================================================

        # --- LOSS ---
        if train_loss < best_loss - min_delta_loss:
            best_loss = train_loss
            no_improve_loss = 0
        else:
            no_improve_loss += 1

        # --- F1 ---
        if f1 > best_f1_es + min_delta_f1:
            best_f1_es = f1
            no_improve_f1 = 0
        else:
            no_improve_f1 += 1

        # --- Condición híbrida ---
        if no_improve_loss >= patience_loss and no_improve_f1 >= patience_f1:
            print(f"[MLP] -> Early stopping híbrido (loss y f1 estancados)")
            early_stopped = True
            break

    # END LOOP
    elapsed = time.perf_counter() - t0

    # Si no hubo checkpoint
    if best_state is None:
        best_state = {k: v.detach().cpu().clone()
                      for k, v in model.state_dict().items()}
        train_loss_best_state = train_loss
        best_epoch = ep
        best_f1_checkpoint = f1
        test_loss_best_state = test_loss
        best_acc = acc
        best_f1_per_class = f1_per_class

    # Guardar history
    history.update({
        "epochs_trained": ep,
        "best_epoch": best_epoch,
        "best_f1": best_f1_checkpoint,
        "best_f1_per_class": best_f1_per_class,
        "best_acc": best_acc,
        "train_loss_best_state": train_loss_best_state,
        "test_loss_best_state": test_loss_best_state,
        "early_stopped": early_stopped,
        "duration_sec": elapsed,
        "class_weights": weights
    })

    return history, best_state
