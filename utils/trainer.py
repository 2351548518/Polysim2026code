import torch
import torch.nn.functional as F
import random
import csv
from collections import Counter
from tqdm import tqdm
from .losses import OrthogonalProjectionLoss, CenterLoss


def compute_class_weights(csv_path, num_classes):
    """根据训练集CSV计算类别加权权重。

    使用 sqrt(N / n_i) 的形式，N为总样本数，n_i为第i类样本数。
    这样可以避免极端权重值，同时给少数类适当加权。
    """
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        labels = [int(float(row['label'])) for row in reader]

    counter = Counter(labels)
    total = len(labels)

    weights = []
    for i in range(num_classes):
        count = counter.get(i, 0)
        if count == 0:
            weights.append(0.0)
        else:
            weights.append((total / (num_classes * count)) ** 0.5)

    return torch.tensor(weights, dtype=torch.float32)


class Trainer:
    def __init__(self, model, config, teacher_model=None):
        """训练器：封装一次 epoch 的前向、损失与优化步骤。

        Args:
            model: 主模型（student）。
            config: ExperimentConfig 实例。
            teacher_model: Stage 3 的 frozen teacher（可选）。
        """
        self.model = model.to(config.device)
        self.config = config
        self.teacher_model = teacher_model

        use_class_weights = getattr(config, 'use_class_weights', False)
        label_smoothing = getattr(config, 'label_smoothing', 0.0)

        if use_class_weights:
            weights = compute_class_weights(config.train_csv, config.resolved_num_classes)
            weights = weights.to(config.device)
            self.ce = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
        else:
            self.ce = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        self.opl = OrthogonalProjectionLoss()
        self.center_loss = CenterLoss(
            num_classes=config.resolved_num_classes,
            emb_dim=config.embedding_dim
        )
        wd = getattr(config, "weight_decay", 0.0)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=config.lr, weight_decay=wd)
        self.scheduler = None  # 由 main.py 在需要时设置

    # ------------------------------------------------------------------
    # Stage / schedule helpers
    # ------------------------------------------------------------------

    def _get_current_stage(self, epoch):
        """根据 config 中的 epoch 边界判断当前训练阶段。

        Returns:
            0 = legacy（单阶段，与原版一致）
            1 = 全模态预训练
            2 = 渐进 modality dropout + 一致性约束
            3 = self-distillation
        """
        cfg = self.config
        s1 = getattr(cfg, "stage1_end", 0)
        s2s = getattr(cfg, "stage2_start", 0)
        s2e = getattr(cfg, "stage2_end", 0)
        s3s = getattr(cfg, "stage3_start", 0)

        if s1 > 0 and epoch <= s1:
            return 1
        if s2s > 0 and s2e > 0 and s2s <= epoch <= s2e:
            return 2
        if s3s > 0 and epoch >= s3s:
            return 3
        return 0

    def _compute_face_dropout_prob(self, epoch):
        """Stage 2 的线性 ramp dropout 概率；Stage 3 固定值。"""
        cfg = self.config
        stage = self._get_current_stage(epoch)

        if stage == 2:
            s2s = getattr(cfg, "stage2_start", 0)
            ramp = getattr(cfg, "dropout_ramp_epochs", 30)
            min_p = getattr(cfg, "dropout_min", 0.0)
            max_p = getattr(cfg, "dropout_max", 0.5)
            progress = min(1.0, (epoch - s2s) / max(ramp, 1))
            return min_p + (max_p - min_p) * progress
        elif stage == 3:
            fixed = getattr(cfg, "stage3_dropout_fixed", 0.5)
            if fixed < 0:
                return getattr(cfg, "dropout_max", 0.5)
            return fixed
        return 0.0

    # ------------------------------------------------------------------
    # Legacy batch-level helpers (Stage 0)
    # ------------------------------------------------------------------

    def _sample_mode(self):
        """按配置概率采样本 batch 的输入模式（仅 AV / A-only）。"""
        p_av = getattr(self.config, "p_av", 0.4)
        p_a_only = getattr(self.config, "p_a_only", 0.5)

        total = p_av + p_a_only
        if total <= 0:
            return "av"

        x = random.random() * total
        if x < p_av:
            return "av"
        return "a_only"

    @staticmethod
    def _mask_by_mode(face, audio, mode):
        """根据模式构造输入：AV / A-only。"""
        if mode == "a_only":
            return torch.zeros_like(face), audio
        return face, audio

    # ------------------------------------------------------------------
    # Per-sample dropout (Stage 2 / 3)
    # ------------------------------------------------------------------

    def _per_sample_mask(self, face, audio, dropout_prob):
        """Per-sample modality dropout：每个样本独立决定是否 drop face。

        Returns:
            face_masked: 部分 face 被置零的 face 张量
            audio: 不变的 audio 张量
            drop_mask: bool [B]，True 表示该样本 face 被 drop
        """
        B = face.size(0)
        drop_mask = torch.rand(B, device=face.device) < dropout_prob
        face_masked = face.clone()
        face_masked[drop_mask] = 0.0
        return face_masked, audio, drop_mask

    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------

    def _kd_loss(self, teacher_logits, student_logits):
        """KL(teacher || student) with temperature scaling。"""
        t = getattr(self.config, "kd_temperature", 2.0)
        teacher_prob = F.softmax(teacher_logits.detach() / t, dim=1)
        student_log_prob = F.log_softmax(student_logits / t, dim=1)
        return F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * (t * t)

    def _face_reliability_floor_loss(self, model, face, audio, out_av=None):
        """约束 AV 输入下的 face 可靠性不低于阈值，避免塌缩到纯音频。"""
        target = getattr(self.config, "face_rel_target", 0.35)

        face_missing = (face.abs().sum(dim=1, keepdim=True) == 0)
        face_present = (~face_missing).float()

        if out_av is not None and isinstance(out_av, dict):
            r_face = out_av["face_reliability"]
        else:
            face_e = model.face_branch(face)
            r_face = torch.sigmoid(model.face_reliability(face_e))

        penalty = torch.relu(target - r_face) ** 2
        denom = face_present.sum().clamp_min(1.0)
        return (penalty * face_present).sum() / denom

    def _face_reliability_floor_loss_from_subset(self, r_face, face_present_mask):
        """对 face-present 子集的可靠性下限约束（per-sample dropout 版本）。"""
        target = getattr(self.config, "face_rel_target", 0.35)
        r_face_present = r_face[face_present_mask]
        face_present_float = face_present_mask.float().unsqueeze(1)
        if face_present_mask.sum() == 0:
            return 0.0
        penalty = torch.relu(target - r_face_present) ** 2
        return penalty.mean()

    def _consistency_loss(self, out_full, out_dropped, drop_mask):
        """一致性损失：||z_full - z_drop||^2 on fusion_embed。"""
        if not drop_mask.any():
            return 0.0
        if drop_mask.sum() < 3:
            return 0.0

        z_full = out_full["fusion_embed"]
        z_drop = out_dropped["fusion_embed"][drop_mask]
        loss = F.mse_loss(z_full, z_drop, reduction="mean")

        if getattr(self.config, "consistency_on_branches", False):
            z_full_a = out_full["audio_embed"]
            z_drop_a = out_dropped["audio_embed"][drop_mask]
            loss = loss + F.mse_loss(z_full_a, z_drop_a, reduction="mean")

        return getattr(self.config, "lambda_consistency", 0.0) * loss

    def _self_distillation_loss(self, student_logits, student_embed,
                                teacher_logits, teacher_embed):
        """Self-distillation：KL logits + MSE embed。"""
        loss = 0.0
        t = getattr(self.config, "sd_temperature", 2.0)

        lambda_logits = getattr(self.config, "lambda_sd_logits", 0.0)
        if lambda_logits > 0:
            teacher_prob = F.softmax(teacher_logits.detach() / t, dim=1)
            student_log_prob = F.log_softmax(student_logits / t, dim=1)
            loss = loss + lambda_logits * F.kl_div(
                student_log_prob, teacher_prob, reduction="batchmean"
            ) * (t * t)

        lambda_embed = getattr(self.config, "lambda_sd_embed", 0.0)
        if lambda_embed > 0:
            loss = loss + lambda_embed * F.mse_loss(
                student_embed, teacher_embed.detach(), reduction="mean"
            )

        return loss

    # ------------------------------------------------------------------
    # Stage 0: legacy training step (unchanged from original)
    # ------------------------------------------------------------------

    def _train_step_legacy(self, face, audio, labels, alpha):
        """与原版完全一致的训练步骤：batch-level mode sampling + 2 diagnostic passes。"""
        mode = self._sample_mode()
        face_main, audio_main = self._mask_by_mode(face, audio, mode)

        # Forward with sampled mode
        out = self.model(face_main, audio_main)

        # Diagnostic forward passes (AV + A-only)
        out_av_stats = self.model(face, audio)
        out_a_only_stats = self.model(torch.zeros_like(face), audio)

        # Accumulate diagnostic stats
        if isinstance(out_av_stats, dict):
            logits_av_stats = out_av_stats["fusion_logits"]
            logits_a_stats = out_a_only_stats["fusion_logits"]
            r_face_stats = out_av_stats["face_reliability"]
            r_audio_stats = out_av_stats["audio_reliability"]
        else:
            _, logits_av_stats, _, _ = out_av_stats
            _, logits_a_stats, _, _ = out_a_only_stats
            face_e_stats = self.model.face_branch(face)
            audio_e_stats = self.model.audio_branch(audio)
            r_face_stats = torch.sigmoid(self.model.face_reliability(face_e_stats))
            r_audio_stats = torch.sigmoid(self.model.audio_reliability(audio_e_stats))

        bsz = labels.size(0)
        mad_batch = (logits_av_stats.detach() - logits_a_stats.detach()).abs().mean().item()

        self._legacy_rel_face_sum += r_face_stats.detach().mean().item() * bsz
        self._legacy_rel_audio_sum += r_audio_stats.detach().mean().item() * bsz
        self._legacy_rel_count += bsz
        self._legacy_mad_sum += mad_batch * bsz
        self._legacy_mad_count += bsz

        # Compute loss
        loss = self._compute_legacy_loss(out, mode, face, audio, labels, alpha,
                                         logits_av_stats, logits_a_stats, out_av_stats)
        return loss

    def _compute_legacy_loss(self, out, mode, face, audio, labels, alpha,
                             logits_av_stats, logits_a_stats, out_av_stats):
        """原版损失计算逻辑。"""
        if isinstance(out, dict):
            loss_face = self.ce(out["face_logits"], labels)
            loss_audio = self.ce(out["audio_logits"], labels)
            loss_fusion = self.ce(out["fusion_logits"], labels)

            loss = (
                self.config.loss_face * loss_face
                + self.config.loss_audio * loss_audio
                + self.config.loss_fusion * loss_fusion
            )

            if getattr(self.config, "kd_enabled", True):
                loss_kd = self._kd_loss(
                    teacher_logits=logits_av_stats,
                    student_logits=logits_a_stats,
                )
                loss = loss + getattr(self.config, "lambda_kd", 0.5) * loss_kd

            if mode == "av":
                loss_face_rel = self._face_reliability_floor_loss(
                    self.model, face, audio, out_av=out_av_stats
                )
                loss = loss + getattr(self.config, "lambda_face_rel", 0.0) * loss_face_rel

            if alpha > 0:
                loss = loss + alpha * self.opl(out["fusion_embed"], labels)

            center_loss_weight = getattr(self.config, 'center_loss_weight', 0.0)
            if center_loss_weight > 0 and isinstance(out, dict):
                loss_center = self.center_loss(out["fusion_embed"], labels)
                loss = loss + center_loss_weight * loss_center

        else:
            fused, logits, _, _ = out
            loss = self.ce(logits, labels)

            if getattr(self.config, "kd_enabled", True):
                loss_kd = self._kd_loss(
                    teacher_logits=logits_av_stats,
                    student_logits=logits_a_stats,
                )
                loss = loss + getattr(self.config, "lambda_kd", 0.5) * loss_kd

            if mode == "av":
                loss_face_rel = self._face_reliability_floor_loss(
                    self.model, face, audio
                )
                loss = loss + getattr(self.config, "lambda_face_rel", 0.0) * loss_face_rel

            if alpha > 0:
                loss = loss + alpha * self.opl(fused, labels)

        return loss

    # ------------------------------------------------------------------
    # Stage 1: full AV pretraining (1 forward pass, no KD)
    # ------------------------------------------------------------------

    def _train_step_stage1(self, face, audio, labels, alpha):
        """Stage 1：全 AV，无 dropout，无 KD。1 次 forward pass。"""
        out = self.model(face, audio)

        loss_face = self.ce(out["face_logits"], labels)
        loss_audio = self.ce(out["audio_logits"], labels)
        loss_fusion = self.ce(out["fusion_logits"], labels)

        loss = (
            self.config.loss_face * loss_face
            + self.config.loss_audio * loss_audio
            + self.config.loss_fusion * loss_fusion
        )

        # Face reliability floor（face 总是 present）
        loss_face_rel = self._face_reliability_floor_loss(
            self.model, face, audio, out_av=out
        )
        loss = loss + getattr(self.config, "lambda_face_rel", 0.0) * loss_face_rel

        if alpha > 0:
            loss = loss + alpha * self.opl(out["fusion_embed"], labels)

        center_loss_weight = getattr(self.config, 'center_loss_weight', 0.0)
        if center_loss_weight > 0:
            loss_center = self.center_loss(out["fusion_embed"], labels)
            loss = loss + center_loss_weight * loss_center

        return loss

    # ------------------------------------------------------------------
    # Stage 2: progressive dropout + consistency
    # ------------------------------------------------------------------

    def _train_step_stage2(self, face, audio, labels, alpha, p_drop):
        """Stage 2：per-sample dropout + consistency + KD。"""
        face_masked, _, drop_mask = self._per_sample_mask(face, audio, p_drop)
        present_mask = ~drop_mask

        # Forward 1: mixed batch
        out_dropped = self.model(face_masked, audio)

        # Forward 2: full AV for dropped subset (consistency loss)
        out_full_subset = None
        if drop_mask.any() and getattr(self.config, "lambda_consistency", 0.0) > 0:
            out_full_subset = self.model(face[drop_mask], audio[drop_mask])

        # Classification losses on mixed batch
        loss_face = self.ce(out_dropped["face_logits"], labels)
        loss_audio = self.ce(out_dropped["audio_logits"], labels)
        loss_fusion = self.ce(out_dropped["fusion_logits"], labels)

        loss = (
            self.config.loss_face * loss_face
            + self.config.loss_audio * loss_audio
            + self.config.loss_fusion * loss_fusion
        )

        # KD: AV logits -> A-only logits (on dropped subset)
        if getattr(self.config, "kd_enabled", True) and drop_mask.any():
            # Use out_full_subset as teacher if available, else compute separately
            if out_full_subset is not None:
                teacher_logits = out_full_subset["fusion_logits"]
            else:
                # Compute full AV forward for dropped subset
                out_full_kd = self.model(face[drop_mask], audio[drop_mask])
                teacher_logits = out_full_kd["fusion_logits"]

            student_logits = out_dropped["fusion_logits"][drop_mask]
            loss_kd = self._kd_loss(teacher_logits, student_logits)
            loss = loss + getattr(self.config, "lambda_kd", 0.2) * loss_kd

        # Face reliability floor (on present subset)
        if present_mask.any():
            loss_face_rel = self._face_reliability_floor_loss_from_subset(
                out_dropped["face_reliability"], present_mask
            )
            loss = loss + getattr(self.config, "lambda_face_rel", 0.0) * loss_face_rel

        # Consistency loss
        if out_full_subset is not None:
            loss_consistency = self._consistency_loss(out_full_subset, out_dropped, drop_mask)
            loss = loss + loss_consistency

        if alpha > 0:
            loss = loss + alpha * self.opl(out_dropped["fusion_embed"], labels)

        center_loss_weight = getattr(self.config, 'center_loss_weight', 0.0)
        if center_loss_weight > 0:
            loss_center = self.center_loss(out_dropped["fusion_embed"], labels)
            loss = loss + center_loss_weight * loss_center

        return loss

    # ------------------------------------------------------------------
    # Stage 3: self-distillation
    # ------------------------------------------------------------------

    def _train_step_stage3(self, face, audio, labels, alpha, p_drop):
        """Stage 3：frozen teacher self-distillation + per-sample dropout。"""
        face_masked, _, drop_mask = self._per_sample_mask(face, audio, p_drop)
        present_mask = ~drop_mask

        # Student forward: mixed batch
        out_student = self.model(face_masked, audio)

        # Teacher forward: full AV (frozen)
        with torch.no_grad():
            out_teacher = self.teacher_model(face, audio)

        # Classification losses on mixed student batch
        loss_face = self.ce(out_student["face_logits"], labels)
        loss_audio = self.ce(out_student["audio_logits"], labels)
        loss_fusion = self.ce(out_student["fusion_logits"], labels)

        loss = (
            self.config.loss_face * loss_face
            + self.config.loss_audio * loss_audio
            + self.config.loss_fusion * loss_fusion
        )

        # Self-distillation on dropped subset
        if drop_mask.any():
            loss_sd_drop = self._self_distillation_loss(
                out_student["fusion_logits"][drop_mask],
                out_student["fusion_embed"][drop_mask],
                out_teacher["fusion_logits"][drop_mask],
                out_teacher["fusion_embed"][drop_mask],
            )
            loss = loss + loss_sd_drop

        # Self-distillation on present subset (lighter guidance)
        if present_mask.any() and getattr(self.config, "lambda_sd_logits", 0.0) > 0:
            loss_sd_present = self._self_distillation_loss(
                out_student["fusion_logits"][present_mask],
                out_student["fusion_embed"][present_mask],
                out_teacher["fusion_logits"][present_mask],
                out_teacher["fusion_embed"][present_mask],
            )
            loss = loss + loss_sd_present

        # Face reliability floor (on present subset)
        if present_mask.any():
            loss_face_rel = self._face_reliability_floor_loss_from_subset(
                out_student["face_reliability"], present_mask
            )
            loss = loss + getattr(self.config, "lambda_face_rel", 0.0) * loss_face_rel

        # Optional consistency loss (reuses teacher as full-AV reference)
        if drop_mask.any() and getattr(self.config, "lambda_consistency", 0.0) > 0:
            loss_consistency = self._consistency_loss(
                {"fusion_embed": out_teacher["fusion_embed"][drop_mask],
                 "audio_embed": out_teacher["audio_embed"][drop_mask]},
                out_student,
                drop_mask,
            )
            loss = loss + loss_consistency

        if alpha > 0:
            loss = loss + alpha * self.opl(out_student["fusion_embed"], labels)

        center_loss_weight = getattr(self.config, 'center_loss_weight', 0.0)
        if center_loss_weight > 0:
            loss_center = self.center_loss(out_student["fusion_embed"], labels)
            loss = loss + center_loss_weight * loss_center

        return loss

    # ------------------------------------------------------------------
    # Legacy epoch stats accumulator (used only in Stage 0)
    # ------------------------------------------------------------------

    def train_epoch(self, loader, alpha, logger=None, epoch=None):
        self.model.train()
        total_loss = 0.0

        stage = self._get_current_stage(epoch if epoch is not None else 0)
        p_drop = self._compute_face_dropout_prob(epoch if epoch is not None else 0)

        # Reset legacy accumulators for Stage 0
        self._legacy_rel_face_sum = 0.0
        self._legacy_rel_audio_sum = 0.0
        self._legacy_rel_count = 0
        self._legacy_mad_sum = 0.0
        self._legacy_mad_count = 0

        pbar = tqdm(
            loader,
            desc=f"Epoch {epoch} [S{stage}]",
            disable=not self.config.debug,
            leave=False,
        )

        for audio, face, labels in pbar:
            audio = audio.to(self.config.device, non_blocking=True)
            face = face.to(self.config.device, non_blocking=True)
            labels = labels.to(self.config.device, non_blocking=True)

            if stage == 0:
                loss = self._train_step_legacy(face, audio, labels, alpha)
            elif stage == 1:
                loss = self._train_step_stage1(face, audio, labels, alpha)
            elif stage == 2:
                loss = self._train_step_stage2(face, audio, labels, alpha, p_drop)
            elif stage == 3:
                loss = self._train_step_stage3(face, audio, labels, alpha, p_drop)
            else:
                raise ValueError(f"Unknown stage: {stage}")

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_clip = getattr(self.config, "grad_clip_norm", 0.0)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            self.opt.step()

            total_loss += loss.item()

        epoch_loss = total_loss / len(loader)

        if stage == 0:
            epoch_stats = {
                "face_reliability": self._legacy_rel_face_sum / max(self._legacy_rel_count, 1),
                "audio_reliability": self._legacy_rel_audio_sum / max(self._legacy_rel_count, 1),
                "av_aonly_logits_mad": self._legacy_mad_sum / max(self._legacy_mad_count, 1),
            }
        else:
            epoch_stats = {
                "face_reliability": 0.0,
                "audio_reliability": 0.0,
                "av_aonly_logits_mad": 0.0,
            }

        return epoch_loss, epoch_stats