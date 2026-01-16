import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, normalize
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from torch.utils.data import TensorDataset


# Método mejorado: divide los datos por video_ID para saber qué vídeos van a train/test
def prepare_mlp_data(df, pooling, norm, test_size, use_pca, pca_frac, seed, return_preproc=False):
    """
    1) Pooling temporal (mean o max) para cada video_ID -> vector (D,)
    2) Split 90/10 estratificado POR VIDEO
    3) Normalización (MinMax o L2) *solo* con parámetros del train
    4) PCA (.fit en train, .transform en train y test)
    5) Devolver TensorDatasets y los IDs de train/test
    """
    # 1) Agrupar por video y obtener (X, y) con pooling
    feat_cols = [c for c in df.columns if c.startswith('feat_')]
    video_data = {}
    video_labels = {}
    scaler = None
    pca = None
    
    for vid, grp in df.groupby('video_ID'):
        arr = grp[feat_cols].values.astype(np.float32)
        if pooling == 'mean':
            vec = arr.mean(axis=0)
        elif pooling == 'max':
            vec = arr.max(axis=0)
        else:
            raise ValueError(f"Pooling no reconocido: {pooling}")
        video_data[vid] = vec
        video_labels[vid] = int(grp['shoot_zone'].iloc[0])
    
    # 2) Split estratificado por video
    video_ids = list(video_data.keys())
    labels = [video_labels[vid] for vid in video_ids]
    
    train_ids, test_ids = train_test_split(video_ids, test_size=test_size, stratify=labels, random_state=seed)
    
    # 3) Construir matrices X_tr, X_te
    X_tr = np.stack([video_data[vid] for vid in train_ids])
    y_tr = np.array([video_labels[vid] for vid in train_ids])
    X_te = np.stack([video_data[vid] for vid in test_ids])
    y_te = np.array([video_labels[vid] for vid in test_ids])

    # 4) Normalización (fit en train, transform en ambos)
    if norm == 'minmax':
        scaler = MinMaxScaler().fit(X_tr)
        X_tr = scaler.transform(X_tr)
        X_te = scaler.transform(X_te)
    elif norm == 'L2':
        X_tr = normalize(X_tr, norm='l2')
        X_te = normalize(X_te, norm='l2')

    # 5) PCA si se requiere
    if use_pca:
        pca = PCA(n_components=pca_frac, random_state=seed).fit(X_tr)
        X_tr = pca.transform(X_tr)
        X_te = pca.transform(X_te)
        print(f"PCA redujo a {pca.n_components_} componentes "
              f"({pca.explained_variance_ratio_.sum():.2%} varianza explicada)")

    # 6) Crear TensorDatasets
    tr_ds = TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).long())
    te_ds = TensorDataset(torch.from_numpy(X_te).float(), torch.from_numpy(y_te).long())
    
    if return_preproc:
        return tr_ds, te_ds, train_ids, test_ids, scaler, pca
    return tr_ds, te_ds, train_ids, test_ids


