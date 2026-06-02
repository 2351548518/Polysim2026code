"""模型基础组件。

本文件提供：
1) 单模态嵌入分支；
2) 线性融合、门控融合与 LSTM 融合模块；
3) 若干可复用的前馈块。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------
# Utility blocks
# --------------------------------------------------

def fc_block(in_dim, out_dim, p=0.5):
    """标准全连接块：Linear + BN + ReLU + Dropout。"""
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(p),
    )


def lstm_block(input_size, hidden_size, num_layers=1, dropout=0.1, bidirectional=False):
    """标准 LSTM 块。"""
    return nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout if num_layers > 1 else 0.0,
        batch_first=True,
        bidirectional=bidirectional,
    )


def transformer_block(emb_dim, num_heads=4, num_layers=1, dropout=0.1):
    """标准 Transformer Encoder 块。"""
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=emb_dim,
        nhead=num_heads,
        dim_feedforward=emb_dim * 4,
        dropout=dropout,
        batch_first=True,
        activation="gelu",
        # 使用 post-norm 可避免 nested tensor 相关的运行时提示。
        norm_first=False,
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=num_layers)


class EmbedBranch(nn.Module):
    """单模态特征映射分支。

    输入为原始特征向量，输出为 L2 归一化后的嵌入向量。
    """

    def __init__(
        self,
        feat_dim,
        emb_dim,
        config,
    ):
        super().__init__()
        encoder_type = getattr(config, "branch_encoder", "mlp")
        lstm_layers = getattr(config, "branch_lstm_layers", 1)
        transformer_layers = getattr(config, "branch_transformer_layers", 1)
        transformer_heads = getattr(config, "branch_transformer_heads", 4)
        num_tokens = getattr(config, "branch_num_tokens", 4)
        dropout = getattr(config, "branch_dropout", 0.1)
        self.encoder_type = encoder_type
        self.num_tokens = num_tokens
        self.emb_dim = emb_dim

        if encoder_type == "mlp":
            self.fc = fc_block(feat_dim, emb_dim, p=dropout)

        elif encoder_type == "lstm":
            # 先映射到 emb 维，再在长度为 1 的序列上做 LSTM 编码。
            self.proj = nn.Sequential(
                nn.Linear(feat_dim, emb_dim),
                nn.LayerNorm(emb_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

            self.lstm = lstm_block(
                input_size=emb_dim,
                hidden_size=emb_dim,
                num_layers=lstm_layers,
                dropout=dropout,
                bidirectional=False,
            )

            self.post = nn.LayerNorm(emb_dim)

        elif encoder_type == "transformer":
            # 真正的多 token 化：将单向量映射为 [T, D] token 序列。
            self.token_proj = nn.Sequential(
                nn.Linear(feat_dim, emb_dim * num_tokens),
                nn.LayerNorm(emb_dim * num_tokens),
                nn.GELU(),
                nn.Dropout(dropout),
            )

            self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, emb_dim))
            self.transformer = transformer_block(
                emb_dim=emb_dim,
                num_heads=transformer_heads,
                num_layers=transformer_layers,
                dropout=dropout,
            )
            self.post = nn.LayerNorm(emb_dim)

        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

    def forward(self, x, return_tokens=False):
        # 归一化有助于稳定相似度计算与融合学习。
        if self.encoder_type == "mlp":
            x = self.fc(x)
            tokens = x.unsqueeze(1)
        elif self.encoder_type == "lstm":
            x = self.proj(x)
            x = x.unsqueeze(1)  # [B, 1, emb]
            x, _ = self.lstm(x)
            x = self.post(x[:, -1, :])
            tokens = x.unsqueeze(1)

        else:
            x = self.token_proj(x)
            tokens = x.view(x.size(0), self.num_tokens, self.emb_dim)
            tokens = self.transformer(tokens + self.pos_embed)
            x = self.post(tokens.mean(dim=1))

        x = F.normalize(x, dim=1)
        tokens = F.normalize(tokens, dim=-1)

        if return_tokens:
            return x, tokens

        return x


# --------------------------------------------------
# Linear fusion
# --------------------------------------------------

class LinearFusion(nn.Module):
    """可学习加权求和融合。

    通过两个可学习标量，动态控制 face/audio 两路嵌入的占比。
    """

    def __init__(self):
        super().__init__()
        self.w_face = nn.Parameter(torch.rand(1))
        self.w_audio = nn.Parameter(torch.rand(1))

    def forward(self, face, audio):
        fused = self.w_face * face + self.w_audio * audio
        return fused, face, audio


# --------------------------------------------------
# Gated fusion
# --------------------------------------------------

class ForwardBlock(nn.Module):
    """门控网络内部使用的小型前馈块。"""

    def __init__(self, in_dim, out_dim, p=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
        )

    def forward(self, x):
        return self.block(x)


class GatedFusion(nn.Module):
    """门控融合。

    根据拼接后的双模态特征预测逐维 gate，
    再对两路投影结果做逐维加权插值。
    """

    def __init__(self, emb_dim, mid_dim=128):
        super().__init__()

        self.attention = nn.Sequential(
            ForwardBlock(emb_dim * 2, mid_dim),
            nn.Linear(mid_dim, emb_dim),
        )

        self.face_proj = nn.Linear(emb_dim, emb_dim)
        self.audio_proj = nn.Linear(emb_dim, emb_dim)

    def forward(self, face, audio):
        # 先拼接再预测 gate，gate 取值范围在 (0, 1)。
        concat = torch.cat([face, audio], dim=1)
        gate = torch.sigmoid(self.attention(concat))

        # 对两路特征做轻量非线性变换。
        face_t = torch.tanh(self.face_proj(face))
        audio_t = torch.tanh(self.audio_proj(audio))

        # gate 越大越偏向 face_t，越小越偏向 audio_t。
        fused = gate * face_t + (1.0 - gate) * audio_t
        return fused, face_t, audio_t


class LSTMFusion(nn.Module):
    """LSTM 融合。

    将 face/audio 两路嵌入视作长度为 2 的序列，
    用 LSTM 编码后取最后时刻作为融合表示。
    """

    def __init__(self, emb_dim, num_layers=1, dropout=0.1):
        super().__init__()
        self.lstm = lstm_block(
            input_size=emb_dim,
            hidden_size=emb_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=False,
        )
        self.post = nn.LayerNorm(emb_dim)

    def forward(self, face, audio):
        seq = torch.stack([face, audio], dim=1)  # [B, 2, emb]
        out, _ = self.lstm(seq)
        fused = self.post(out[:, -1, :])
        return fused, face, audio


class CrossAttentionFusion(nn.Module):
    """Cross Attention 融合。

    让 face 作为 query 去关注 audio，让 audio 作为 query 去关注 face，
    最后将两路上下文向量融合。
    """

    def __init__(self, emb_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.face_attn = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.audio_attn = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.face_norm = nn.LayerNorm(emb_dim)
        self.audio_norm = nn.LayerNorm(emb_dim)
        self.out_norm = nn.LayerNorm(emb_dim)

    def forward(self, face, audio):
        # 兼容 [B, D] 与 [B, T, D] 两种输入。
        face_seq = face.unsqueeze(1) if face.dim() == 2 else face
        audio_seq = audio.unsqueeze(1) if audio.dim() == 2 else audio

        face_ctx, _ = self.face_attn(face_seq, audio_seq, audio_seq)
        audio_ctx, _ = self.audio_attn(audio_seq, face_seq, face_seq)

        face_ctx = self.face_norm(face_ctx + face_seq)
        audio_ctx = self.audio_norm(audio_ctx + audio_seq)

        face_vec = face_ctx.mean(dim=1)
        audio_vec = audio_ctx.mean(dim=1)
        fused = self.out_norm(0.5 * (face_vec + audio_vec))

        return fused, face_vec, audio_vec
