import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, normalize
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence


def collate_sequences(batch):
    """
    Recibe una lista de (tensor_seq, label).
    Devuelve:
      - padded: tensor (B, T_max, D)
      - lengths: tensor (B,)
      - labels: tensor (B,)
    """
    seqs, labels = zip(*batch)
    lengths = torch.tensor([s.size(0) for s in seqs], dtype=torch.long)
    padded = pad_sequence(seqs, batch_first=True)
    labels = torch.tensor(labels, dtype=torch.long)
    return padded, lengths, labels


def prepare_lstm_data(df, norm, test_size, seed, return_preproc=False):
    """
    1) Mantiene las secuencias temporales para cada video_ID
    2) Split 90/10 estratificado POR VIDEO
    3) Normalización (MinMax o L2) *solo* con parámetros del train
    4) Devuelve listas de (secuencia, etiqueta) para train/test y los IDs
    """
    # 1) Agrupar por video y obtener secuencias
    feat_cols = [c for c in df.columns if c.startswith('feat_')]
    video_seqs = {}
    video_labels = {}
    
    for vid, grp in df.groupby('video_ID'):
        arr = grp[feat_cols].values.astype(np.float32)  # (T, D)
        video_seqs[vid] = arr
        video_labels[vid] = int(grp['shoot_zone'].iloc[0])
    
    # 2) Split estratificado por video
    video_ids = list(video_seqs.keys())
    labels = [video_labels[vid] for vid in video_ids]
    
    train_ids, test_ids = train_test_split(
        video_ids, 
        test_size=test_size, 
        stratify=labels, 
        random_state=seed
    )
    
    # 3) Separar secuencias de train y test
    train_seqs = [video_seqs[vid] for vid in train_ids]
    train_labels = [video_labels[vid] for vid in train_ids]
    test_seqs = [video_seqs[vid] for vid in test_ids]
    test_labels = [video_labels[vid] for vid in test_ids]

    # 4) Normalización (fit en train, transform en ambos)
    if norm == 'minmax':
        # Aplanar todas las secuencias de train para ajustar el scaler
        all_train_frames = np.vstack(train_seqs)
        scaler = MinMaxScaler().fit(all_train_frames)
        # Aplicar a cada secuencia individualmente
        train_seqs = [scaler.transform(seq) for seq in train_seqs]
        test_seqs = [scaler.transform(seq) for seq in test_seqs]
    elif norm == 'L2' or norm == 'l2':
        # Normalizar cada frame individualmente
        train_seqs = [normalize(seq, norm='l2', axis=1) for seq in train_seqs]
        test_seqs = [normalize(seq, norm='l2', axis=1) for seq in test_seqs]

    # 5) Convertir a tensores
    train_list = [(torch.from_numpy(seq), label) for seq, label in zip(train_seqs, train_labels)]
    test_list = [(torch.from_numpy(seq), label) for seq, label in zip(test_seqs, test_labels)]

    if return_preproc:
        return train_list, test_list, train_ids, test_ids, scaler
    return train_list, test_list, train_ids, test_ids