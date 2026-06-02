import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import numpy as np
import pandas as pd

from config_all import ExperimentConfigAll
from models.multibranch import MultiBranchFOP
from utils.post_process import run_all_post_processing

SUBMISSION_DIR = "csv_files/submission_combined"


def load_npy(csv_file, feats_dir, device, audio_col, face_col):
    """读取 CSV 中指定列的特征路径并加载为 GPU/CPU 张量。"""
    if audio_col not in csv_file.columns:
        if "ecappa_feats_path" in csv_file.columns:
            audio_col = "ecappa_feats_path"
        else:
            raise KeyError(f"Missing audio feature column: {audio_col}")
    if face_col not in csv_file.columns:
        if "facenet_feats_path" in csv_file.columns:
            face_col = "facenet_feats_path"
        else:
            raise KeyError(f"Missing face feature column: {face_col}")

    def resolve_path(p):
        p = str(p).replace("\\", "/")
        if p.startswith("./"):
            p = p[2:]
        return p if os.path.isabs(p) else os.path.join(feats_dir, p)

    audio_feats = [np.load(resolve_path(i)) for i in csv_file[audio_col]]
    face_feats = [np.load(resolve_path(i)) for i in csv_file[face_col]]
    audio_feats = np.asarray(audio_feats)
    face_feats = np.asarray(face_feats)
    audio_feats = torch.from_numpy(audio_feats).to(device)
    face_feats = torch.from_numpy(face_feats).to(device)
    return audio_feats, face_feats


def extract_logits(model_out):
    """从模型输出中提取 logits。"""
    if isinstance(model_out, dict):
        return model_out["fusion_logits"]
    return model_out[1]


def main():
    config = ExperimentConfigAll()
    config.debug = False
    device = torch.device(config.device)
    torch.manual_seed(config.seed)

    SPLIT = "test"
    FEATS_DIR = "./test_set/feat"

    english_csv = pd.read_csv(f"./test_set/csv/comp/v1_test_English.csv")
    urdu_csv = pd.read_csv(f"./test_set/csv/comp/v1_test_Urdu.csv")

    # ── 加载原始 numpy 特征（P3/P5 后处理用）──
    def _resolve_raw(p):
        p = str(p).replace('\\', '/')
        if p.startswith('./'):
            p = p[2:]
        return p if os.path.isabs(p) else os.path.join(FEATS_DIR, p)

    en_a_raw = np.asarray([np.load(_resolve_raw(p)) for p in english_csv['ecappa_feats_path']], dtype=np.float32)
    en_f_raw = np.asarray([np.load(_resolve_raw(p)) for p in english_csv['facenet_feats_path']], dtype=np.float32)
    ur_a_raw = np.asarray([np.load(_resolve_raw(p)) for p in urdu_csv['ecappa_feats_path']], dtype=np.float32)
    ur_f_raw = np.asarray([np.load(_resolve_raw(p)) for p in urdu_csv['facenet_feats_path']], dtype=np.float32)

    # ── 加载模型输入特征 ──
    english_audio_feats, english_face_feats = load_npy(
        english_csv, FEATS_DIR, device,
        config.audio_encoder, config.face_encoder,
    )
    urdu_audio_feats, urdu_face_feats = load_npy(
        urdu_csv, FEATS_DIR, device,
        config.audio_encoder, config.face_encoder,
    )

    face_dim = english_face_feats.shape[1]
    audio_dim = english_audio_feats.shape[1]

    # ── 加载模型 ──
    model = MultiBranchFOP(
        config=config, face_dim=face_dim, audio_dim=audio_dim
    ).to(device)

    checkpoint_path = (
        f"./checkpoints/{config.version}_all_"
        f"alpha{config.alpha}_{config.model_type}_{config.fusion}_{config.exp_name}.pt"
    )
    ckpt = torch.load(checkpoint_path, map_location=device)
    print(f"Loaded checkpoint from {checkpoint_path}")
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with torch.no_grad():
        p3_dict = model(english_face_feats, english_audio_feats)
        p3_logits = p3_dict["fusion_logits"]
        p3_probs = torch.softmax(p3_logits, dim=1)
        p3_pred = p3_logits.argmax(dim=1).detach().cpu().numpy()
        p3_prob = p3_probs.max(dim=1).values.detach().cpu().numpy()

        p4_dict = model(english_face_feats * 0.0, english_audio_feats)
        p4_logits = p4_dict["fusion_logits"]
        p4_probs = torch.softmax(p4_logits, dim=1)
        p4_pred = p4_logits.argmax(dim=1).detach().cpu().numpy()
        p4_prob = p4_probs.max(dim=1).values.detach().cpu().numpy()
        p4_audio_embed = p4_dict["audio_embed"].detach()

        p5_dict = model(urdu_face_feats, urdu_audio_feats)
        p5_logits = p5_dict["fusion_logits"]
        p5_probs = torch.softmax(p5_logits, dim=1)
        p5_pred = p5_logits.argmax(dim=1).detach().cpu().numpy()
        p5_prob = p5_probs.max(dim=1).values.detach().cpu().numpy()

        p6_dict = model(urdu_face_feats * 0.0, urdu_audio_feats)
        p6_logits = p6_dict["fusion_logits"]
        p6_probs = torch.softmax(p6_logits, dim=1)
        p6_pred = p6_logits.argmax(dim=1).detach().cpu().numpy()
        p6_prob = p6_probs.max(dim=1).values.detach().cpu().numpy()

    p3_pred, p4_pred, p5_pred, p6_pred = run_all_post_processing(
        en_f_raw, en_a_raw, ur_f_raw, ur_a_raw,
        p3_pred, p3_probs, p3_prob,
        p4_pred, p4_prob, p4_audio_embed,
        p5_pred, p5_probs, p5_prob,
        p6_pred, p6_prob,
        english_audio_feats, urdu_audio_feats,
        model=model, device=device,
        n_classes=config.resolved_num_classes, face_dim=face_dim,
    )

    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    submission_en = pd.DataFrame()
    submission_en["key"] = english_csv["key"]
    submission_en["p3"] = p3_pred
    submission_en["p4"] = p4_pred
    submission_en.to_csv(
        f"{SUBMISSION_DIR}/submission_{config.version}_{SPLIT}_English_English.csv",
        index=None
    )

    submission_ur = pd.DataFrame()
    submission_ur["key"] = urdu_csv["key"]
    submission_ur["p5"] = p5_pred
    submission_ur["p6"] = p6_pred
    submission_ur.to_csv(
        f"{SUBMISSION_DIR}/submission_{config.version}_{SPLIT}_English_Urdu.csv",
        index=None
    )

    print(f"\nSubmission files generated:")
    print(f"  - {SUBMISSION_DIR}/submission_{config.version}_{SPLIT}_English_English.csv")
    print(f"  - {SUBMISSION_DIR}/submission_{config.version}_{SPLIT}_English_Urdu.csv")


if __name__ == "__main__":
    main()
