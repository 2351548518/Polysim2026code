import torch
import torch.nn as nn
import torch.nn.functional as F


class OrthogonalProjectionLoss(nn.Module):
    """正交投影损失（OPL）。

    目标：
    - 同类样本在嵌入空间内更接近；
    - 异类样本相似度更低。
    """

    def forward(self, feats, labels):
        # 归一化后，点积可近似看作余弦相似度。
        feats = F.normalize(feats, dim=1)
        labels = labels.unsqueeze(1)

        # 构建同类/异类掩码矩阵。
        mask = labels.eq(labels.T)
        eye = torch.eye(len(labels), device=labels.device).bool()

        pos = (mask & ~eye).float()
        neg = (~mask).float()

        # 计算 batch 内两两样本相似度。
        dot = feats @ feats.T

        # 正样本对相似度希望越高；负样本对相似度希望越低。
        pos_mean = (pos * dot).sum() / (pos.sum() + 1e-6)
        neg_mean = (neg * dot).abs().sum() / (neg.sum() + 1e-6)

        # 组合成最终损失项。
        loss = (1 - pos_mean) + 0.7 * neg_mean
        return loss


class CenterLoss(nn.Module):
    """Center Loss - 最小化类内距离，增强类间可分性。

    核心思想：
    - 为每个类别维护一个可学习的中心向量
    - 拉近样本特征与其对应类中心的距离
    - 与 CrossEntropyLoss 结合使用效果更好
    """

    def __init__(self, num_classes, emb_dim):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, emb_dim))
        nn.init.xavier_uniform_(self.centers)

    def forward(self, embeddings, labels):
        batch_centers = self.centers.to(embeddings.device)[labels]
        loss = F.mse_loss(embeddings, batch_centers, reduction='mean')
        return loss
