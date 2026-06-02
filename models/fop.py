import torch
import torch.nn as nn

from .model import (
    EmbedBranch,
    LinearFusion,
    GatedFusion,
    LSTMFusion,
    CrossAttentionFusion,
)


# --------------------------------------------------
# Main model
# --------------------------------------------------


class FOP(nn.Module):
    """FOP 基线模型。

    结构为：
    - face/audio 两个嵌入分支；
    - 一种融合策略（linear/gated/concat/lstm/cross_attention）；
    - 一个最终分类头。
    """

    def __init__(self, config, face_dim, audio_dim):
        super().__init__()

        emb_dim = config.embedding_dim
        num_classes = config.resolved_num_classes

        self.face_branch = EmbedBranch(
            face_dim,
            emb_dim,
            config=config,
        )
        self.audio_branch = EmbedBranch(
            audio_dim,
            emb_dim,
            config=config,
        )
        self.reliability_eps = getattr(config, "reliability_eps", 1e-6)
        self.face_feature_boost = getattr(config, "face_feature_boost", 1.0)
        self.fusion_type = config.fusion

        # 轻量可靠性打分：每个模态输出一个 [0, 1] 分数。
        self.face_reliability = nn.Linear(emb_dim, 1)
        self.audio_reliability = nn.Linear(emb_dim, 1)

        # --------------------------------------------------
        # Fusion selection
        # --------------------------------------------------
        if config.fusion == "linear":
            self.fusion = LinearFusion()
            fusion_dim = emb_dim

        elif config.fusion == "gated":
            self.fusion = GatedFusion(emb_dim)
            fusion_dim = emb_dim

        elif config.fusion == "concat":
            self.fusion = None
            fusion_dim = emb_dim * 2

        elif config.fusion == "lstm":
            self.fusion = LSTMFusion(
                emb_dim,
                num_layers=getattr(config, "fusion_lstm_layers", 1),
                dropout=getattr(config, "fusion_dropout", 0.1),
            )
            fusion_dim = emb_dim

        elif config.fusion == "cross_attention":
            self.fusion = CrossAttentionFusion(
                emb_dim,
                num_heads=getattr(config, "fusion_transformer_heads", 4),
                dropout=getattr(config, "fusion_dropout", 0.1),
            )
            fusion_dim = emb_dim

        else:
            raise ValueError(f"Unknown fusion type: {config.fusion}")

        # --------------------------------------------------
        # Classifier
        # --------------------------------------------------
        self.classifier = nn.Linear(fusion_dim, num_classes)

    def forward(self, face, audio):
        # 将两种模态投影到同维度嵌入空间。
        if self.fusion_type == "cross_attention":
            face_e, face_tokens = self.face_branch(face, return_tokens=True)
            audio_e, audio_tokens = self.audio_branch(audio, return_tokens=True)
        else:
            face_e = self.face_branch(face)
            audio_e = self.audio_branch(audio)
            face_tokens = face_e.unsqueeze(1)
            audio_tokens = audio_e.unsqueeze(1)

        # -------------------------
        # Reliability weighting
        # -------------------------
        # 当前任务仅考虑 face 缺失：face 全零时将其可靠性显式置 0。
        face_missing = (face.abs().sum(dim=1, keepdim=True) == 0)

        r_face = torch.sigmoid(self.face_reliability(face_e))
        r_audio = torch.sigmoid(self.audio_reliability(audio_e))

        r_face = r_face.masked_fill(face_missing, 0.0)

        denom = r_face + r_audio + self.reliability_eps
        face_scale = (r_face / denom)
        audio_scale = (r_audio / denom)
        face_w = face_scale * face_e
        audio_w = audio_scale * audio_e

        face_tokens_w = face_tokens * face_scale.unsqueeze(1)
        audio_tokens_w = audio_tokens * audio_scale.unsqueeze(1)

        # face 可用时给轻量增益，提升 AV 条件下的人脸参与度。
        face_present = (~face_missing).float()
        face_w = face_w * (1.0 + face_present * (self.face_feature_boost - 1.0))
        face_tokens_w = face_tokens_w * (1.0 + face_present.unsqueeze(1) * (self.face_feature_boost - 1.0))

        # --------------------------------------------------
        # Fusion
        # --------------------------------------------------
        if self.fusion is None:
            # concat 模式：直接拼接两路嵌入。
            fused = torch.cat([face_w, audio_w], dim=1)
        elif self.fusion_type == "cross_attention":
            fused, face_t, audio_t = self.fusion(face_tokens_w, audio_tokens_w)
        else:
            # linear/gated 模式：调用融合模块。
            fused, face_t, audio_t = self.fusion(face_w, audio_w)

        # 基于融合表示做分类。
        logits = self.classifier(fused)

        if self.fusion is None:
            face_t, audio_t = face_w, audio_w

        return fused, logits, face_t, audio_t
