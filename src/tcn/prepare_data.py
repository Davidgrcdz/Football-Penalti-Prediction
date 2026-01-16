import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import MinMaxScaler, normalize
from sklearn.model_selection import train_test_split



def collate_sequences(batch):
    """
    Crea un batch de secuencias de longitud variable, añadiendo padding
    y devolviendo las longitudes originales.
    """
    seqs, labels = zip(*batch)
    lengths = torch.tensor([s.size(0) for s in seqs], dtype=torch.long)
    padded = pad_sequence(seqs, batch_first=True)
    labels = torch.tensor(labels, dtype=torch.long)
    return padded, lengths, labels

def prepare_tcn_data(df, norm, test_size, seed, return_preproc=False):
    """
    Prepara los datos para la TCN, agrupando por vídeo y haciendo un split
    estratificado para evitar fugas de datos.
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

    if norm.lower() == 'minmax':
        all_train_frames = np.vstack(train_seqs)
        scaler = MinMaxScaler().fit(all_train_frames)
        train_seqs = [scaler.transform(seq) for seq in train_seqs]
        test_seqs = [scaler.transform(seq) for seq in test_seqs]
    elif norm.lower() == 'l2':
        train_seqs = [normalize(seq, norm='l2', axis=1) for seq in train_seqs]
        test_seqs = [normalize(seq, norm='l2', axis=1) for seq in test_seqs]
    else:
        print(f"Norma desconocida '{norm}', se omite la normalización.")

    train_list = [(torch.from_numpy(seq), label) for seq, label in zip(train_seqs, train_labels)]
    test_list = [(torch.from_numpy(seq), label) for seq, label in zip(test_seqs, test_labels)]
    
    if return_preproc:
        return train_list, test_list, train_ids, test_ids, scaler
    return train_list, test_list, train_ids, test_ids