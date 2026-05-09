import pandas as pd
from pathlib import Path
from omegaconf import DictConfig

def load_and_merge_features(cfg: DictConfig):
    """
    Lädt alle 5 Feature-Dateien und merged sie zu einem großen Feature-Matrix.
    Behält Metadaten (eyes, site, subject_id) separat.
    """
    dfs = []
    for filename in cfg.data.feature_files:
        path = Path(cfg.data.feature_dir) / filename
        if path.suffix == '.csv':
            df = pd.read_csv(path)
        else:
            df = pd.read_hdf(path)
        
        # Metadaten trennen
        meta_cols = ['eyes', 'site', 'subject_id']
        feat_cols = [c for c in df.columns if c not in meta_cols]
        
        df_feat = df[feat_cols].copy()
        prefix = filename.split('.')[0]
        df_feat.columns = [f"{prefix}_{c}" for c in df_feat.columns]
        
        dfs.append(df_feat)

    # Alle Features mergen
    X = pd.concat(dfs, axis=1)

    # Labels und Groups laden
    labels_path = Path(cfg.data.feature_dir) / "labels.csv"
    labels_df = pd.read_csv(labels_path)
    
    y = labels_df['tinnitus'].values
    groups = labels_df['subject_id'].values

    print(f"✅ Features geladen: {X.shape[1]} Spalten, {X.shape[0]} Samples")
    return X, y, groups
