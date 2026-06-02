from dataclasses import dataclass
from pathlib import Path
import logging


@dataclass
class ExperimentConfigAll:

    home_dir: str = str(Path(__file__).resolve().parent / "feats")
    #home_dir: str = str('2026polysim/feats')
    """特征文件的根目录"""

    audio_encoder: str = "ecappa_feats_path"
    """音频特征列名，对应CSV中的特征路径"""

    face_encoder: str = "facenet_feats_path"
    """人脸特征列名，对应CSV中的特征路径"""

    exp_name: str = "center_loss_all_missinglearning"
    """实验名称，用于checkpoint和日志文件命名"""

    seed: int = 1
    """随机种子，保证实验可复现"""

    device: str = "cuda"
    """训练设备，cuda或cpu"""

    lr: float = 5e-05
    """学习率"""

    batch_size: int = 32
    """批大小"""

    max_epochs: int = 24
    """最大训练轮数"""

    num_workers: int = 0
    """数据加载的进程数，0表示主进程加载"""

    alpha: float = 0
    """正交投影损失(OPL)的权重，0表示不启用"""

    embedding_dim: int = 512
    """embedding维度，模态融合后的特征维度"""

    model_type: str = "multibranch"
    """模型类型，当前使用multibranch多分支模型"""

    fusion: str = "cross_attention"
    """融合策略：linear/gated/concat/lstm/cross_attention"""

    branch_encoder: str = "transformer"
    """分支编码器类型：transformer/lstm/linear"""

    branch_lstm_layers: int = 1
    """LSTM分支的层数"""

    branch_dropout: float = 0.1
    """分支编码器的dropout比例"""

    branch_transformer_layers: int = 1
    """Transformer分支的层数"""

    branch_transformer_heads: int = 4
    """Transformer分支的注意力头数"""

    branch_num_tokens: int = 6
    """分支的token数量（用于Transformer编码器）"""

    fusion_lstm_layers: int = 1
    """融合层LSTM的层数"""

    fusion_dropout: float = 0.1
    """融合层的dropout比例"""

    fusion_transformer_heads: int = 4
    """融合层CrossAttention的注意力头数"""

    loss_face: float = 1.0
    """人脸分支损失的权重"""

    loss_audio: float = 1.0
    """音频分支损失的权重"""

    loss_fusion: float = 1.0
    """融合分支损失的权重"""

    p_av: float =1.0
    """完整AV双模态数据的采样概率"""

    p_a_only: float = 0.0
    """Audio-only数据的采样概率（face被置零/替换为Missing Token）"""

    kd_enabled: bool = True
    """是否启用知识蒸馏（从完整模态到缺失模态的蒸馏）"""

    kd_temperature: float = 2.0
    """知识蒸馏的温度参数"""

    lambda_kd: float = 0.2
    """知识蒸馏损失的权重"""

    reliability_eps: float = 1e-6
    """可靠性打分的最小值，防止除零"""

    face_feature_boost: float = 0.95
    """当face可用时，给其特征的放大系数（>1增强，<1削弱）"""

    face_rel_target: float = 0.7
    """人脸可靠性分数的目标值，用于正则化"""

    lambda_face_rel: float = 0.1
    """人脸可靠性正则化的权重"""

    version: str = "v1"
    """数据集版本：v1/v2/v3，不同版本有不同的类别数"""

    debug: bool = False
    """调试模式开关"""

    log_level = logging.DEBUG if debug else logging.INFO
    """日志级别"""

    save_train_log: bool = True
    """是否保存训练日志到文件"""

    log_dir: str = "./log"
    """日志文件保存目录"""

    stage1_end: int = 0
    """Stage1（完整模态训练阶段）的结束轮次"""

    stage2_start: int = 0
    """Stage2（引入缺失模态阶段）的开始轮次"""

    stage2_end: int = 0
    """Stage2的结束轮次"""

    stage3_start: int = 0
    """Stage3（蒸馏/精调阶段）的开始轮次"""

    dropout_min: float = 0.0
    """逐样本dropout的最小比例"""

    dropout_max: float = 0.5
    """逐样本dropout的最大比例"""

    dropout_ramp_epochs: int = 30
    """dropout概率从min增加到max的轮数"""

    lambda_consistency: float = 0.0
    """一致性正则化损失的权重"""

    consistency_on_branches: bool = False
    """一致性正则化是否作用于分支特征"""

    lambda_sd_logits: float = 0.0
    """logits蒸馏损失权重"""

    lambda_sd_embed: float = 0.0
    """embedding蒸馏损失权重"""

    sd_temperature: float = 2.0
    """自蒸馏温度参数"""

    stage3_dropout_fixed: float = 0.5
    """Stage3固定使用的dropout比例"""

    warmup_epochs: int = 0
    """学习率warmup的轮数"""

    weight_decay: float = 0.0001
    """权重衰减（L2正则化）"""

    grad_clip_norm: float = 0.0
    """梯度裁剪的范数阈值，0表示不裁剪"""

    lr_scheduler: str = ""
    """学习率调度器类型：cosine/step/empty"""

    center_loss_weight: float = 0.15
    """Center Loss的权重，用于拉近类内距离"""

    label_smoothing: float = 0.1
    """标签平滑系数，0表示不使用，例如0.1表示将硬标签软化为10%的均匀分布"""

    use_class_weights: bool = True
    """是否使用类别加权loss，根据各类别样本数设置权重"""

    audio_aug_enabled: bool = False
    """是否启用音频数据增强"""

    audio_aug_prob: float = 0.2
    """音频增强的应用概率"""

    audio_noise_std: float = 0.01
    """音频噪声注入的标准差"""

    audio_dropout_prob: float = 0.1
    """音频随机维度dropout的概率"""

    audio_scale_range: tuple = (0.9, 1.1)
    """音频随机缩放的范围"""

    @property
    def resolved_num_classes(self):
        """根据数据集版本返回类别数"""
        if self.version == "v1":
            return 70
        elif self.version == "v2":
            return 84
        elif self.version == "v3":
            return 36
        else:
            raise ValueError(f"Unknown version '{self.version}'")

    @property
    def train_csv(self):
        """训练数据集CSV路径"""
        return "./csv_files/v1_train_English.csv"




