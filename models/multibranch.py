import torch
import torch.nn as nn

from .model import (
    EmbedBranch,
    LinearFusion,
    GatedFusion,
    LSTMFusion,
    CrossAttentionFusion,
)

class MultiBranchFOP(nn.Module):
    """
    多分支多模态模型。

    相比基础 FOP，该模型同时提供三条监督路径：
    - 人脸分支分类头（face head）
    - 语音分支分类头（audio head）
    - 融合分支分类头（fusion head）

    融合策略可选：linear / gated / concat / lstm / cross_attention。
    """

    def __init__(self, config, face_dim, audio_dim):
        super().__init__()

        self.config = config
        emb = config.embedding_dim
        num_classes = config.resolved_num_classes

        # -------------------------
        # Embedding branches
        # -------------------------
        self.face_branch = EmbedBranch(
            face_dim,
            emb,
            config=config,
        )
        self.audio_branch = EmbedBranch(
            audio_dim,
            emb,
            config=config,
        )
        self.reliability_eps = getattr(config, "reliability_eps", 1e-6)
        self.face_feature_boost = getattr(config, "face_feature_boost", 1.0)
        self.fusion_type = config.fusion

        # 轻量可靠性打分：每个模态一个标量分数。
        self.face_reliability = nn.Linear(emb, 1)
        self.audio_reliability = nn.Linear(emb, 1)

        # -------------------------
        # Unimodal classifiers
        # -------------------------
        self.face_classifier = nn.Linear(emb, num_classes)
        self.audio_classifier = nn.Linear(emb, num_classes)

        # -------------------------
        # Fusion
        # -------------------------
        if config.fusion == "linear":
            self.fusion = LinearFusion()
            fusion_dim = emb

        elif config.fusion == "gated":
            self.fusion = GatedFusion(emb)
            fusion_dim = emb

        elif config.fusion == "concat":
            self.fusion = None
            fusion_dim = emb * 2

        elif config.fusion == "lstm":
            self.fusion = LSTMFusion(
                emb,
                num_layers=getattr(config, "fusion_lstm_layers", 1),
                dropout=getattr(config, "fusion_dropout", 0.1),
            )
            fusion_dim = emb

        elif config.fusion == "cross_attention":
            self.fusion = CrossAttentionFusion(
                emb,
                num_heads=getattr(config, "fusion_transformer_heads", 4),
                dropout=getattr(config, "fusion_dropout", 0.1),
            )
            fusion_dim = emb

        else:
            raise ValueError(f"Unknown fusion type: {config.fusion}")

        self.fusion_classifier = nn.Linear(fusion_dim, num_classes)

        # Learnable Missing Token for face modality
        # 当 face 缺失（全零）时，用可学习的 token 替代
        self.face_missing_token = nn.Parameter(torch.randn(1, 1, face_dim) * 0.02)

    def _apply_missing_token(self, face):
        """将全零的 face 替换为 Learnable Missing Token。

        Args:
            face: [B, D] 或 [B, T, D] 的输入特征

        Returns:
            face_replaced: 替换后的 face
            face_missing: BoolTensor，标识哪些样本的 face 是缺失的
        """
        if face.dim() == 2:
            face_missing = (face.abs().sum(dim=1, keepdim=True) == 0)
            if face_missing.any():
                B, D = face.shape
                missing_face = self.face_missing_token.expand(B, -1, -1).squeeze(1)
                face = torch.where(face_missing, missing_face, face)
        elif face.dim() == 3:
            face_missing = (face.abs().sum(dim=2, keepdim=True) == 0)
            if face_missing.any():
                B, T, D = face.shape
                missing_face = self.face_missing_token.expand(B, T, -1)
                face = torch.where(face_missing.unsqueeze(-1), missing_face, face)
        else:
            face_missing = torch.zeros(face.size(0), dtype=torch.bool, device=face.device)
        return face, face_missing

    def forward(self, face, audio, return_intermediates=False):
        # -------------------------
        # Missing Token Replacement
        # -------------------------
        face, face_missing = self._apply_missing_token(face)

        # -------------------------
        # Embeddings
        # -------------------------
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

        # face 可用时给轻量增益。
        face_present = (~face_missing).float()
        face_w = face_w * (1.0 + face_present * (self.face_feature_boost - 1.0))
        face_tokens_w = face_tokens_w * (1.0 + face_present.unsqueeze(1) * (self.face_feature_boost - 1.0))

        # -------------------------
        # Unimodal logits
        # -------------------------
        face_logits = self.face_classifier(face_w)
        audio_logits = self.audio_classifier(audio_w)

        # -------------------------
        # Fusion
        # -------------------------
        if self.fusion is None:
            fused = torch.cat([face_w, audio_w], dim=1)
        elif self.fusion_type == "cross_attention":
            fused, _, _ = self.fusion(face_tokens_w, audio_tokens_w)
        else:
            fused, _, _ = self.fusion(face_w, audio_w)

        fusion_logits = self.fusion_classifier(fused)

        result = {
            "face_logits": face_logits,
            "audio_logits": audio_logits,
            "fusion_logits": fusion_logits,
            "face_embed": face_w,
            "audio_embed": audio_w,
            "fusion_embed": fused,
            "face_reliability": r_face,
            "audio_reliability": r_audio,
        }

        if return_intermediates:
            result["face_tokens_weighted"] = face_tokens_w
            result["audio_tokens_weighted"] = audio_tokens_w
            result["face_e_raw"] = face_e
            result["audio_e_raw"] = audio_e
            result["face_missing"] = face_missing

        return result

    def forward_from_intermediates(self, face_e, audio_e, face_tokens, audio_tokens,
                                    face_missing_mask):
        """从预计算的中间特征重跑 reliability + fusion + classifiers。

        用于 cross-modal dropout consistency：drop 中间特征后只重跑轻量的融合层，
        而非整个 EmbedBranch encoder。
        """
        r_face = torch.sigmoid(self.face_reliability(face_e))
        r_audio = torch.sigmoid(self.audio_reliability(audio_e))

        if face_missing_mask is not None:
            r_face = r_face.masked_fill(face_missing_mask, 0.0)

        denom = r_face + r_audio + self.reliability_eps
        face_scale = (r_face / denom)
        audio_scale = (r_audio / denom)

        face_w = face_scale * face_e
        audio_w = audio_scale * audio_e

        face_tokens_w = face_tokens * face_scale.unsqueeze(1)
        audio_tokens_w = audio_tokens * audio_scale.unsqueeze(1)

        face_present = (~face_missing_mask).float() if face_missing_mask is not None else 1.0
        face_w = face_w * (1.0 + face_present * (self.face_feature_boost - 1.0))
        face_tokens_w = face_tokens_w * (1.0 + face_present.unsqueeze(1) * (self.face_feature_boost - 1.0)) \
            if face_missing_mask is not None else face_tokens_w * self.face_feature_boost

        face_logits = self.face_classifier(face_w)
        audio_logits = self.audio_classifier(audio_w)

        if self.fusion is None:
            fused = torch.cat([face_w, audio_w], dim=1)
        elif self.fusion_type == "cross_attention":
            fused, _, _ = self.fusion(face_tokens_w, audio_tokens_w)
        else:
            fused, _, _ = self.fusion(face_w, audio_w)

        fusion_logits = self.fusion_classifier(fused)

        return {
            "face_logits": face_logits,
            "audio_logits": audio_logits,
            "fusion_logits": fusion_logits,
            "face_embed": face_w,
            "audio_embed": audio_w,
            "fusion_embed": fused,
            "face_reliability": r_face,
            "audio_reliability": r_audio,
        }
