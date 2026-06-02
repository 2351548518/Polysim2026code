"""特征数据加载模块。

当前实现采用“全量预加载”策略：
读取 CSV 后一次性加载所有 npy 特征到内存，
以换取训练与评估阶段更快的迭代速度。
"""

import random
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


class LoadData(Dataset):
    """
    视听双模态数据集（全量驻留内存）。

    设计要点：
    - 仅负责特征加载与样本索引，不包含缺失模态逻辑；
    - 面向快速训练/全量评估，减少磁盘 IO 成本。
    """

    def __init__(
        self,
        csv_path: str,
        config,
        audio_encoder: str,
        face_encoder: str,
        modality: str = "audiovisual",
    ):
        # ---- sanity check ----
        assert modality == "audiovisual", (
            "This loader supports audiovisual data only."
        )

        self.audio_encoder = audio_encoder
        self.face_encoder = face_encoder
        self.modality = modality
        self.config = config

        # ---- read CSV once ----
        df = pd.read_csv(csv_path)
        self.num_samples = len(df)

        # ---- resolve paths ONCE ----
        def resolve_feat_path(p):
            # CSV 可能混有不同操作系统路径分隔符，统一为 '/'.
            p = str(p).replace("\\", "/")
            if p.startswith("./"):
                p = p[2:]
            # 统一挂到 config.home_dir 下，确保运行目录变化时依然可定位。
            return str((Path(config.home_dir) / p).resolve())

        audio_paths = [
            resolve_feat_path(p)
            for p in df[audio_encoder]
        ]
        face_paths = [
            resolve_feat_path(p)
            for p in df[face_encoder]
        ]
        labels = df["label"].astype(int).to_numpy()

        # ---- load EVERYTHING into memory ----
        audio_feats = []
        face_feats = []

        for i in tqdm(
            range(self.num_samples),
            desc=f"Loading features from {Path(csv_path).name}",
            total=self.num_samples,
        ):
            audio_feats.append(
                np.load(audio_paths[i]).astype("float32")
            )
            face_feats.append(
                np.load(face_paths[i]).astype("float32")
            )

        # ---- stack for cache-friendly access ----
        self.audio_feats = np.stack(audio_feats)   # (N, Da)
        self.face_feats = np.stack(face_feats)     # (N, Df)
        self.labels = labels                       # (N,)

    def _audio_augment(self, audio):
        if not getattr(self.config, 'audio_aug_enabled', False):
            return audio

        p = getattr(self.config, 'audio_aug_prob', 0.3)
        if random.random() > p:
            return audio

        audio = audio.copy()

        noise_std = getattr(self.config, 'audio_noise_std', 0.02)
        noise = np.random.randn(*audio.shape).astype(np.float32) * noise_std
        audio = audio + noise

        dropout_prob = getattr(self.config, 'audio_dropout_prob', 0.1)
        mask = np.random.rand(*audio.shape) > dropout_prob
        audio = audio * mask

        scale_range = getattr(self.config, 'audio_scale_range', (0.9, 1.1))
        scale = np.random.uniform(scale_range[0], scale_range[1])
        audio = audio * scale

        return audio

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        audio = self.audio_feats[idx]
        face = self.face_feats[idx]
        label = self.labels[idx]

        audio = self._audio_augment(audio)

        return (audio, face, label)


if __name__ == "__main__":
    import torch
    from copy_polysim.config import ExperimentConfig

    config = ExperimentConfig()
    torch.manual_seed(config.seed)

    dataset = LoadData(
        csv_path="./feature_tracker/v1_test_English.csv",
        config=config,
        audio_encoder="ecappa_feats_path",
        face_encoder="facenet_feats_path",
        modality="audiovisual",
    )

    a, f, y = dataset[0]
    print(a.shape, f.shape, y)
