import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight

def compute_class_weights_from_loader(
    train_loader,
    num_classes: int,
    device,
    method: str = "sklearn",   # "sklearn" o "manual"
    normalize: bool = False,   # opcional: media=1
):
    """
    Asume que el batch del DataLoader es una tupla/lista y que yb es el ÚLTIMO elemento.
    Compatible con MLP: (xb, yb)
    Compatible con LSTM/TCN/Transformer: (xb, lengths, yb) o similar
    """
    ys = []
    with torch.no_grad():
        for batch in train_loader:
            yb = batch[-1]
            yb = yb.view(-1).detach().cpu()
            ys.append(yb)

    y = torch.cat(ys, dim=0).numpy().astype(int)

    counts = np.bincount(y, minlength=num_classes)
    counts = np.maximum(counts, 1)

    if method.lower() == "manual":
        w = counts.max().astype(np.float64) / counts.astype(np.float64)
    elif method.lower() == "sklearn":
        classes = np.arange(num_classes)
        w = compute_class_weight(class_weight="balanced", classes=classes, y=y).astype(np.float64)
    else:
        raise ValueError(f"method no soportado: {method}")

    if normalize:
        w = w / w.mean()

    cw_tensor = torch.tensor(w, dtype=torch.float32, device=device)

    print(f"[CW] method={method} counts={counts.tolist()} weights={w.tolist()}")

    return cw_tensor, w
