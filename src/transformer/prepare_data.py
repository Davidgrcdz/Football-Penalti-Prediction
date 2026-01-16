import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, normalize
from sklearn.model_selection import train_test_split


def collate_transformer(batch):
    """
    Crea un batch de secuencias de longitud variable con padding.
    Devuelve:
      - batch_padded: tensor (B, T, D)
      - mask: tensor (B, T) con True en posiciones válidas
      - labels: tensor (B,)
    """
    seqs, labels = zip(*batch)
    lengths = [s.size(0) for s in seqs]
    max_len = max(lengths)
    feat_dim = seqs[0].size(1)

    # Padding manual (para crear máscara)
    batch_padded = torch.zeros(len(seqs), max_len, feat_dim, dtype=seqs[0].dtype)
    mask = torch.zeros(len(seqs), max_len, dtype=torch.bool)
    for i, s in enumerate(seqs):
        batch_padded[i, :s.size(0)] = s
        mask[i, :s.size(0)] = True

    labels = torch.tensor(labels, dtype=torch.long)
    return batch_padded, mask, labels




def prepare_transformer_data(df, norm, test_size, seed, return_preproc=False):
    """
    Prepara los datos para el Transformer:
      - Agrupa por video_ID
      - Normaliza (MinMax o L2)
      - Split estratificado
      - Devuelve listas [(tensor_seq, label)]
    """
    feat_cols = [c for c in df.columns if c.startswith('feat_')]
    video_seqs = {}
    video_labels = {}

    for vid, grp in df.groupby('video_ID'):
        arr = grp[feat_cols].values.astype(np.float32)
        video_seqs[vid] = arr
        video_labels[vid] = int(grp['shoot_zone'].iloc[0])

    video_ids = list(video_seqs.keys())
    labels = [video_labels[vid] for vid in video_ids]

    train_ids, test_ids = train_test_split(
        video_ids,
        test_size=test_size,
        stratify=labels,
        random_state=seed
    )

    train_seqs = [video_seqs[vid] for vid in train_ids]
    train_labels = [video_labels[vid] for vid in train_ids]
    test_seqs = [video_seqs[vid] for vid in test_ids]
    test_labels = [video_labels[vid] for vid in test_ids]

    # Normalización
    if norm.lower() == 'minmax':
        all_train_frames = np.vstack(train_seqs)
        scaler = MinMaxScaler().fit(all_train_frames)
        train_seqs = [scaler.transform(seq) for seq in train_seqs]
        test_seqs = [scaler.transform(seq) for seq in test_seqs]
    elif norm.lower() == 'l2':
        train_seqs = [normalize(seq, norm='l2', axis=1) for seq in train_seqs]
        test_seqs = [normalize(seq, norm='l2', axis=1) for seq in test_seqs]
    else:
        print(f"⚠️ Normalización '{norm}' no reconocida. Usando datos sin normalizar.")

    train_list = [(torch.from_numpy(seq), label) for seq, label in zip(train_seqs, train_labels)]
    test_list = [(torch.from_numpy(seq), label) for seq, label in zip(test_seqs, test_labels)]

    if return_preproc:
        return train_list, test_list, train_ids, test_ids, scaler
    return train_list, test_list, train_ids, test_ids