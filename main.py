import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import logging
import torch
from torch.utils.data import DataLoader
from dataclasses import fields

from config_all import ExperimentConfigAll

from utils.featLoader import LoadData
from utils.trainer import Trainer

from models.multibranch import MultiBranchFOP


def collect_full_config(config):
    snapshot = {}
    for f in fields(config):
        if f.name.startswith("_"):
            continue
        value = getattr(config, f.name)
        if callable(value):
            continue
        snapshot[f.name] = value
    for key, value in vars(config.__class__).items():
        if key.startswith("_") or key in snapshot:
            continue
        if isinstance(value, property) or callable(value):
            continue
        snapshot[key] = value
    return snapshot


def save_checkpoint(model, optimizer, config, epoch, metric_value, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "metric": metric_value,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer else None,
        "config": collect_full_config(config),
    }
    torch.save(checkpoint, save_path)


def setup_logger(config):
    logger = logging.getLogger("Experiment")
    logger.setLevel(config.log_level)
    if logger.handlers:
        logger.handlers.clear()
    formatter = logging.Formatter("[%(levelname)s][%(name)s] %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if getattr(config, "save_train_log", True):
        log_path = (
            f"{config.log_dir}/"
            f"{config.version}_"
            f"all_"
            f"alpha{config.alpha}_"
            f"{config.model_type}_"
            f"{config.fusion}_"
            f"{config.exp_name}.log"
        )
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger, log_path


def make_loader(csv_path, config, shuffle=False, logger=None):
    dataset = LoadData(
        csv_path=csv_path,
        config=config,
        audio_encoder=config.audio_encoder,
        face_encoder=config.face_encoder,
        modality="audiovisual",
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return dataset, loader


def main():
    config = ExperimentConfigAll()
    torch.manual_seed(config.seed)
    if config.device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    logger, log_path = setup_logger(config)
    logger.info("=== Experiment started ===")
    if log_path is not None:
        logger.info("Train log file: %s", log_path)

    full_config = collect_full_config(config)
    logger.info("Full config snapshot for reproducibility:")
    for key in sorted(full_config.keys()):
        logger.info("  %s = %r", key, full_config[key])

    logger.info(
        "Seed=%d | Device=%s | Model=%s | Fusion=%s | Version=%s | Train_Lang=English | #Classes=%d | Alpha=%.3f",
        config.seed,
        config.device,
        config.model_type,
        config.fusion,
        config.version,
        config.resolved_num_classes,
        config.alpha,
    )

    train_csv = config.train_csv


    logger.info("Train CSV: %s", train_csv)
    

    _, train_loader = make_loader(train_csv, config, shuffle=True, logger=logger)

    audio, face, _ = next(iter(train_loader))
    logger.info("Feature dimensions | Audio=%d | Face=%d", audio.shape[1], face.shape[1])

    model = MultiBranchFOP(
        config=config,
        face_dim=face.shape[1],
        audio_dim=audio.shape[1]
    )

    logger.info("Model initialized | Params=%.2fM", sum(p.numel() for p in model.parameters()) / 1e6)

    trainer = Trainer(model, config, teacher_model=None)

    if getattr(config, "lr_scheduler", "") == "cosine":
        trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            trainer.opt, T_max=config.max_epochs, eta_min=1e-6
        )
        logger.info("LR scheduler: CosineAnnealingLR (T_max=%d, eta_min=1e-6)", config.max_epochs)

    alpha = config.alpha
    logger.info("=== Training with alpha=%.3f ===", alpha)

    save_path = (
        f"./checkpoints/"
        f"{config.version}_"
        f"all_"
        f"alpha{alpha}_"
        f"{config.model_type}_"
        f"{config.fusion}_"
        f"{config.exp_name}.pt"
    )

    for epoch in range(config.max_epochs):
        warmup_epochs = getattr(config, "warmup_epochs", 0)
        if warmup_epochs > 0 and epoch < warmup_epochs:
            warmup_factor = (epoch + 1) / warmup_epochs
            for pg in trainer.opt.param_groups:
                pg["lr"] = config.lr * warmup_factor

        loss, train_stats = trainer.train_epoch(train_loader, alpha, epoch=epoch)

        if trainer.scheduler is not None:
            trainer.scheduler.step()

        p_drop = trainer._compute_face_dropout_prob(epoch)

        logger.info(
            "[α=%.3f] Epoch %03d | Loss %.4f | "
            "Rel(face/audio)=%.4f/%.4f | MAD=%.6f | p_drop=%.3f",
            alpha,
            epoch,
            loss,
            train_stats["face_reliability"],
            train_stats["audio_reliability"],
            train_stats["av_aonly_logits_mad"],
            p_drop,
        )

    save_checkpoint(
        model=model,
        optimizer=trainer.opt,
        config=config,
        epoch=config.max_epochs - 1,
        metric_value=0.0,
        save_path=save_path,
    )

    logger.info("=== Experiment finished ===")


if __name__ == "__main__":
    main()
