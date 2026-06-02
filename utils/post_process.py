import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


P3_CONFIDENCE_THRESHOLD = 0.3
P5_CONFIDENCE_THRESHOLD = 0.4
P3_RATIO_THRESHOLD = 1.1
P5_RATIO_THRESHOLD = 1.0

def build_prototypes(f_raw, a_raw, labels, n_classes):
    def l2(x):
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    fp = np.zeros((n_classes, f_raw.shape[1]), dtype=np.float32)
    ap = np.zeros((n_classes, a_raw.shape[1]), dtype=np.float32)
    cnt = np.zeros(n_classes)
    for i, lab in enumerate(labels):
        fp[lab] += f_raw[i]
        ap[lab] += a_raw[i]
        cnt[lab] += 1
    cnt = np.maximum(cnt, 1)
    return l2(fp / cnt[:, None]), l2(ap / cnt[:, None]), cnt


def post_process_by_raw_prototype(f_raw, a_raw, preds, probs_tensor,
                                    probs_np, confidence_threshold, ratio_threshold,
                                    proto_f=None, proto_a=None, score_mode="sum",
                                    verbose=False):
    N_CLASSES = probs_tensor.shape[1]

    def l2(x):
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

    if proto_f is not None and proto_a is not None:
        f_proto = proto_f
        a_proto = proto_a
    else:
        f_proto, a_proto, _ = build_prototypes(f_raw, a_raw, preds, N_CLASSES)

    f_norm = l2(f_raw)
    a_norm = l2(a_raw)

    low_mask = probs_np < confidence_threshold
    low_idx = np.where(low_mask)[0]
    n_low = len(low_idx)

    if n_low == 0:
        if verbose:
            print(f"  Skip: {n_low} low-conf")
        return preds.copy(), {"n_total": len(preds), "n_low_conf": 0, "n_changed": 0}

    f_sim = f_norm[low_idx] @ f_proto.T
    a_sim = a_norm[low_idx] @ a_proto.T

    new_preds = preds.copy()
    n_changed = 0

    for k, idx in enumerate(low_idx):
        pred = preds[idx]

        if score_mode == "product":
            select_scores = (f_sim[k] + 1e-6) * (a_sim[k] + 1e-6)
        else:
            select_scores = f_sim[k] + a_sim[k]

        select_scores[pred] = -np.inf
        best_other = int(np.argmax(select_scores))

        pred_score = f_sim[k, pred] + a_sim[k, pred]
        best_score = f_sim[k, best_other] + a_sim[k, best_other]
        ratio = best_score / max(pred_score, 1e-6)

        if ratio > ratio_threshold:
            new_preds[idx] = best_other
            n_changed += 1

    if verbose:
        print(f"  Low-conf: {n_low}, Changed: {n_changed}")
    return new_preds, {"n_total": len(preds), "n_low_conf": n_low, "n_changed": n_changed}


def _build_audio_centroids(train_csv_path, audio_col='ecappa_feats_path',
                            label_col='label', feats_dir="./feats", verbose=True):
    train_df = pd.read_csv(train_csv_path)

    label_to_feats = {}
    for _, row in train_df.iterrows():
        feat_path = str(row[audio_col]).replace("\\", "/")
        resolved = feat_path if os.path.isabs(feat_path) else os.path.join(feats_dir, feat_path.lstrip("./"))
        try:
            feat = np.load(resolved).astype(np.float32).flatten()
            label = int(row[label_col])
            label_to_feats.setdefault(label, []).append(feat)
        except Exception:
            continue

    labels_sorted = sorted(label_to_feats.keys())
    feat_dim = next(iter(label_to_feats.values()))[0].shape[0]
    centroids = np.zeros((len(labels_sorted), feat_dim), dtype=np.float32)

    for i, label in enumerate(labels_sorted):
        centroids[i] = np.mean(label_to_feats[label], axis=0)

    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids_norm = centroids / np.where(norms == 0, 1.0, norms)

    if verbose:
        print(f"[AudioCentroid] Built {len(labels_sorted)} class centroids "
              f"from {len(train_df)} training samples, dim={centroids_norm.shape[1]}")

    return centroids_norm, labels_sorted


PERTURB_GAMMA_EN = 3.0
PERTURB_GAMMA_UR = 1.5
PERTURB_PER_CLASS = True
PERTURB_PER_CLASS_MODE = "norm_sqrt"
PERTURB_PER_CLASS_CLAMP = (0.3, 3.0)

def _compute_drift_from_pseudo_labels(train_centroids, test_audio_feats,
                                       pseudo_labels, n_classes, verbose=True):
    test_feats_np = F.normalize(test_audio_feats, dim=1).cpu().numpy().astype(np.float32)

    test_centroids = np.zeros((n_classes, test_feats_np.shape[1]), dtype=np.float32)
    counts = np.zeros(n_classes, dtype=np.int32)
    for i, lab in enumerate(pseudo_labels):
        c = int(lab)
        test_centroids[c] += test_feats_np[i]
        counts[c] += 1

    for c in range(n_classes):
        if counts[c] == 0:
            test_centroids[c] = train_centroids[c]
        else:
            test_centroids[c] /= counts[c]

    test_norms = np.linalg.norm(test_centroids, axis=1, keepdims=True)
    test_norms[test_norms == 0] = 1.0
    test_centroids_l2 = test_centroids / test_norms

    delta = test_centroids_l2 - train_centroids

    if verbose:
        delta_norms = np.linalg.norm(delta, axis=1)
        angles = np.degrees(np.arccos(np.clip(
            (train_centroids * test_centroids_l2).sum(axis=1), -1, 1)))
        zero_count = int((counts == 0).sum())
        print(f"  [Drift-Pseudo] delta norm mean={delta_norms.mean():.4f} max={delta_norms.max():.4f}, "
              f"angular mean={angles.mean():.1f}° max={angles.max():.1f}°")
        print(f"  [Drift-Pseudo] test samples/class: min={counts[counts>0].min() if (counts>0).any() else 0}, "
              f"max={counts.max()}, zero={zero_count}/{n_classes}")

    return delta


def _build_per_class_gamma(base_gamma, delta, mode="norm_sqrt", clamp=(0.3, 3.0)):
    if mode == "global":
        return np.full(delta.shape[0], base_gamma, dtype=np.float32)
    delta_norms = np.linalg.norm(delta, axis=1)
    mean_norm = delta_norms.mean()
    if mean_norm == 0:
        return np.full(len(delta_norms), base_gamma, dtype=np.float32)
    rel_scale = delta_norms / mean_norm
    if mode == "norm_sqrt":
        rel_scale = np.sqrt(rel_scale)
    lo, hi = clamp
    rel_scale = np.clip(rel_scale, lo, hi)
    return (base_gamma * rel_scale).astype(np.float32)


def _apply_perturb_drift(centroids_norm, drift_en_path=None, drift_ur_path=None,
                          delta_en=None, delta_ur=None,
                          gamma_en=PERTURB_GAMMA_EN,
                          gamma_ur=PERTURB_GAMMA_UR,
                          per_class=PERTURB_PER_CLASS,
                          per_class_mode=PERTURB_PER_CLASS_MODE,
                          per_class_clamp=PERTURB_PER_CLASS_CLAMP,
                          verbose=True):
    def _l2(x):
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.where(norms == 0, 1.0, norms)

    if delta_en is not None:
        delta_en = delta_en.astype(np.float32)
    else:
        assert drift_en_path is not None, "Must provide either delta_en or drift_en_path"
        delta_en = np.load(drift_en_path).astype(np.float32)

    if delta_ur is not None:
        delta_ur = delta_ur.astype(np.float32)
    else:
        assert drift_ur_path is not None, "Must provide either delta_ur or drift_ur_path"
        delta_ur = np.load(drift_ur_path).astype(np.float32)

    assert delta_en.shape == centroids_norm.shape, \
        f"delta_en shape {delta_en.shape} != centroids {centroids_norm.shape}"
    assert delta_ur.shape == centroids_norm.shape, \
        f"delta_ur shape {delta_ur.shape} != centroids {centroids_norm.shape}"

    gamma_en_vec = _build_per_class_gamma(gamma_en, delta_en,
                                           mode=per_class_mode, clamp=per_class_clamp)
    gamma_ur_vec = _build_per_class_gamma(gamma_ur, delta_ur,
                                           mode=per_class_mode, clamp=per_class_clamp)

    centroids_en = _l2(centroids_norm + gamma_en_vec[:, None] * delta_en)
    centroids_ur = _l2(centroids_norm + gamma_ur_vec[:, None] * delta_ur)

    if verbose:
        for name, delta, gamma_vec in [("English", delta_en, gamma_en_vec),
                                        ("Urdu", delta_ur, gamma_ur_vec)]:
            norms = np.linalg.norm(delta, axis=1)
            angles = np.degrees(np.arccos(np.clip(
                (centroids_norm * (centroids_norm + delta)).sum(axis=1), -1, 1)))
            mode_str = per_class_mode if per_class else "global"
            print(f"[Perturb] {name}: delta norm mean={norms.mean():.4f} max={norms.max():.4f}, "
                  f"angular mean={angles.mean():.1f}° max={angles.max():.1f}°, "
                  f"mode={mode_str}")
            print(f"          per-class gamma: min={gamma_vec.min():.3f} mean={gamma_vec.mean():.3f} "
                  f"max={gamma_vec.max():.3f}")
        print(f"[Perturb] centroids shape={centroids_en.shape}")

    return centroids_en, centroids_ur


def _apply_centroid_correction(predictions, confidences, audio_feats,
                                centroids_norm, labels_sorted,
                                confidence_threshold, margin_threshold,
                                min_audio_sim, task_name="P4", verbose=True):
    predictions = np.array(predictions).astype(np.int64)
    confidences = np.array(confidences).astype(np.float64)
    audio_feats = np.array(audio_feats).astype(np.float32)
    N = len(predictions)

    sims = cosine_similarity(audio_feats, centroids_norm)

    audio_best_idx = np.argmax(sims, axis=1)
    audio_best_sim = sims[np.arange(N), audio_best_idx]

    label_to_idx = {label: i for i, label in enumerate(labels_sorted)}
    audio_model_sim = np.zeros(N, dtype=np.float32)
    for i in range(N):
        pred = predictions[i]
        if pred in label_to_idx:
            audio_model_sim[i] = sims[i, label_to_idx[pred]]

    margin = audio_best_sim - audio_model_sim

    low_conf = confidences < confidence_threshold
    audio_disagrees = predictions != np.array([labels_sorted[idx] for idx in audio_best_idx])
    margin_ok = margin > margin_threshold
    audio_reliable = audio_best_sim > min_audio_sim

    override_mask = low_conf & audio_disagrees & margin_ok & audio_reliable

    corrected = predictions.copy()
    for i in range(N):
        if override_mask[i]:
            corrected[i] = labels_sorted[audio_best_idx[i]]

    n_changed = override_mask.sum()
    n_low_conf = low_conf.sum()
    n_low_conf_disagree = (low_conf & audio_disagrees).sum()

    if verbose:
        print(f"  [{task_name}] Centroid correction (conf<{confidence_threshold}, "
              f"margin>{margin_threshold}, sim>{min_audio_sim}): "
              f"{n_changed}/{N} changed ({n_changed / N * 100:.1f}%), "
              f"low-conf: {n_low_conf}, low-conf+disagree: {n_low_conf_disagree}")

    stats = {
        'n_total': N,
        'n_low_conf': n_low_conf,
        'n_low_conf_disagree': n_low_conf_disagree,
        'n_changed': n_changed,
    }

    return corrected, stats


def _build_lp_graph(embs, k, mutual=False, doubly_stochastic=False, ds_iters=20,
                    precomp_sim=None):
    n = embs.shape[0]
    sim = precomp_sim if precomp_sim is not None else embs @ embs.T
    np.fill_diagonal(sim, 0.0)
    W = np.zeros((n, n), dtype=np.float32)
    top_k = np.argsort(sim, axis=1)[:, -k:]
    rows = np.arange(n)[:, None].repeat(k, axis=1)
    W[rows, top_k] = np.maximum(sim[rows, top_k], 0.0)
    if mutual:
        W = W * (W.T > 0).astype(np.float32)
    W = (W + W.T) * 0.5
    if doubly_stochastic:
        for _ in range(ds_iters):
            W = W / (W.sum(axis=1, keepdims=True) + 1e-8)
            W = W / (W.sum(axis=0, keepdims=True) + 1e-8)
        return W
    deg = W.sum(axis=1)
    d = 1.0 / (np.sqrt(deg) + 1e-8)
    return W * d[:, None] * d[None, :]


def graph_lp(labeled_embs, labeled_labels, unlabeled_embs, n_classes,
             k=15, alpha=0.99, n_iters=100, mutual=False,
             doubly_stochastic=False, precomp_sim=None, return_soft=False,
             raw_anchor_weight=0.0, raw_preds=None):
    n_l = labeled_embs.shape[0]
    all_embs = np.vstack([labeled_embs, unlabeled_embs]).astype(np.float32)
    S = _build_lp_graph(all_embs, k, mutual=mutual,
                        doubly_stochastic=doubly_stochastic,
                        precomp_sim=precomp_sim)
    Y = np.zeros((all_embs.shape[0], n_classes), dtype=np.float32)
    for i, c in enumerate(labeled_labels):
        Y[i, c] = 1.0
    if raw_anchor_weight is not None and raw_preds is not None:
        w = np.asarray(raw_anchor_weight, dtype=np.float32)
        is_scalar = w.ndim == 0
        if is_scalar:
            if w > 0:
                for i, c in enumerate(raw_preds):
                    Y[n_l + i, c] = float(w)
        else:
            for i, c in enumerate(raw_preds):
                if w[i] > 0:
                    Y[n_l + i, c] = float(w[i])
    F = Y.copy()
    for _ in range(n_iters):
        F = alpha * (S @ F) + (1.0 - alpha) * Y
    if return_soft:
        return F[n_l:]
    return F[n_l:].argmax(axis=1)


def graph_lp_ensemble(labeled_embs, labeled_labels, unlabeled_embs, n_classes,
                       k_list, alpha=0.9, n_iters=150, mutual=False,
                       doubly_stochastic=False, precomp_sim=None, verbose=True,
                       raw_anchor_weight=0.0, raw_preds=None, return_soft=False):
    if verbose:
        has_anchor = raw_anchor_weight is not None and np.any(np.asarray(raw_anchor_weight) > 0)
        anchor_info = ""
        if has_anchor:
            w = np.asarray(raw_anchor_weight)
            anchor_info = f", raw_anchor=conf_weighted(mean={w.mean():.4f})" if w.ndim > 0 else f", raw_anchor={w}"
        print(f"  [LP-Ensemble] k_list={k_list}, alpha={alpha}, "
              f"iters={n_iters}, mutual={mutual}, ds={doubly_stochastic}{anchor_info}")

    all_F = []
    for k in k_list:
        F_soft = graph_lp(
            labeled_embs, labeled_labels, unlabeled_embs, n_classes,
            k=k, alpha=alpha, n_iters=n_iters,
            mutual=mutual, doubly_stochastic=doubly_stochastic,
            precomp_sim=precomp_sim, return_soft=True,
            raw_anchor_weight=raw_anchor_weight, raw_preds=raw_preds,
        )
        all_F.append(F_soft)

    F_avg = np.mean(all_F, axis=0)
    if return_soft:
        return F_avg
    return F_avg.argmax(axis=1)


LP_ALPHA = 0.90
LP_N_ITERS = 150

LP_EN_ENSEMBLE = True
LP_EN_K_LIST = [3, 5, 10, 20, 50]
LP_EN_K_SINGLE = 10
LP_RAW_ANCHOR_WEIGHT_EN = 0.03

LP_UR_ENSEMBLE = False
LP_UR_K_LIST = [5, 8, 12, 20, 50]
LP_UR_K_SINGLE = 6
LP_RAW_ANCHOR_WEIGHT_UR = 0.12

LP_SIM_GUARD_MARGIN_EN = 0.01
LP_SIM_GUARD_MARGIN_UR = 0.02
LP_SIM_GUARD_CONF_EN = 0.50
LP_SIM_GUARD_CONF_UR = 0.60
LP_SIM_GUARD_P6_RULE1_MIN_CONF = 0.07
LP_SIM_GUARD_P6_RULE1_TOP1_TOP2_MARGIN = 0.02
LP_SIM_GUARD_P6_RULE1B = True
LP_SIM_GUARD_P6_RULE1B_K = 5
LP_SIM_GUARD_P6_RULE1B_MAX_RAW_CONF = 0.08

LP_CENTROID_KNN_CONF_THRESH = 0.50
LP_CENTROID_KNN_MIN_SIM = 0.22
LP_CENTROID_KNN_MARGIN_EN = 0.04
LP_CENTROID_KNN_MARGIN_UR = 0.08
LP_CENTROID_KNN_TOP1_TOP2_MARGIN_UR = 0.06

LP_SOFT_CONF_THRESH = 0.03

P4_P6_LP_MODE = "train_anchors_perturbed"

def post_process_p4_p6_graph_lp(p4_pred, p6_pred,
                                english_audio_emb, urdu_audio_emb,
                                train_csv_path=None,
                                centroids_norm=None,
                                labels_sorted=None,
                                centroids_en=None,
                                centroids_ur=None,
                                lp_alpha=LP_ALPHA,
                                lp_n_iters=LP_N_ITERS,
                                en_ensemble=LP_EN_ENSEMBLE,
                                en_k_list=LP_EN_K_LIST,
                                en_k_single=LP_EN_K_SINGLE,
                                ur_ensemble=LP_UR_ENSEMBLE,
                                ur_k_list=LP_UR_K_LIST,
                                ur_k_single=LP_UR_K_SINGLE,
                                sim_guard_margin_en=LP_SIM_GUARD_MARGIN_EN,
                                sim_guard_margin_ur=LP_SIM_GUARD_MARGIN_UR,
                                sim_guard_conf_en=LP_SIM_GUARD_CONF_EN,
                                sim_guard_conf_ur=LP_SIM_GUARD_CONF_UR,
                                raw_anchor_weight_en=LP_RAW_ANCHOR_WEIGHT_EN,
                                raw_anchor_weight_ur=LP_RAW_ANCHOR_WEIGHT_UR,
                                sim_guard_p6_rule1_min_conf=LP_SIM_GUARD_P6_RULE1_MIN_CONF,
                                sim_guard_p6_rule1_top1_top2_margin=LP_SIM_GUARD_P6_RULE1_TOP1_TOP2_MARGIN,
                                sim_guard_p6_rule1b=LP_SIM_GUARD_P6_RULE1B,
                                sim_guard_p6_rule1b_k=LP_SIM_GUARD_P6_RULE1B_K,
                                sim_guard_p6_rule1b_max_raw_conf=LP_SIM_GUARD_P6_RULE1B_MAX_RAW_CONF,
                                p4_conf=None,
                                p6_conf=None,
                                p4_ae_embed=None,
                                p4_ae_train_protos=None,
                                ae_lp_en=False,
                                verbose=True):
    if train_csv_path is None:
        train_csv_path = "./csv_files/v1_train_English.csv"

    if centroids_norm is None or labels_sorted is None:
        centroids_norm, labels_sorted = _build_audio_centroids(train_csv_path, verbose=verbose)
    else:
        labels_sorted = list(labels_sorted)
        if verbose:
            print(f"[GraphLP] Using pre-built centroids: "
                  f"{centroids_norm.shape[0]} classes, dim={centroids_norm.shape[1]}")

    n_classes = len(labels_sorted)
    labeled_labels = np.arange(n_classes)

    en_audio_np = F.normalize(english_audio_emb, dim=1).cpu().numpy().astype(np.float32)
    ur_audio_np = F.normalize(urdu_audio_emb, dim=1).cpu().numpy().astype(np.float32)

    p4_centroids = centroids_en if centroids_en is not None else centroids_norm
    p6_centroids = centroids_ur if centroids_ur is not None else centroids_norm

    def _run_lp(centroids, audio_np, task, ensemble, k_list, k_single,
                raw_preds_np=None, raw_weight=0.0, raw_conf_np=None,
                return_soft=False):
        raw_anchor_args = {}
        if raw_preds_np is not None and raw_weight > 0:
            label_to_idx = {int(l): i for i, l in enumerate(labels_sorted)}
            raw_idx = np.array([label_to_idx.get(int(p), 0) for p in raw_preds_np],
                               dtype=np.int64)
            per_sample_weight = raw_weight
            raw_anchor_args = dict(raw_anchor_weight=per_sample_weight,
                                   raw_preds=raw_idx)
        if ensemble:
            soft = graph_lp_ensemble(
                labeled_embs=centroids, labeled_labels=labeled_labels,
                unlabeled_embs=audio_np, n_classes=n_classes,
                k_list=k_list, alpha=lp_alpha, n_iters=lp_n_iters,
                mutual=False, doubly_stochastic=False, verbose=verbose,
                return_soft=True,
                **raw_anchor_args,
            )
            if return_soft:
                return soft
            return soft.argmax(axis=1)
        else:
            soft = graph_lp(
                labeled_embs=centroids, labeled_labels=labeled_labels,
                unlabeled_embs=audio_np, n_classes=n_classes,
                k=k_single, alpha=lp_alpha, n_iters=lp_n_iters,
                mutual=False, doubly_stochastic=False,
                return_soft=True,
                **raw_anchor_args,
            )
            if return_soft:
                return soft
            return soft.argmax(axis=1)

    p4_soft = None
    if ae_lp_en and p4_ae_embed is not None and p4_ae_train_protos is not None:
        p4_ae_np = F.normalize(p4_ae_embed, dim=1).cpu().numpy().astype(np.float32)
        ae_centroids_np = p4_ae_train_protos.astype(np.float32)
        p4_soft = _run_lp(ae_centroids_np, p4_ae_np, "P4", en_ensemble,
                         en_k_list, en_k_single,
                         raw_preds_np=np.array(p4_pred).astype(np.int64),
                         raw_weight=raw_anchor_weight_en,
                         raw_conf_np=np.array(p4_conf) if p4_conf is not None else None,
                         return_soft=True)
        p4_lp = np.array([labels_sorted[i] for i in p4_soft.argmax(axis=1)], dtype=np.int64)
    else:
        p4_soft = _run_lp(p4_centroids, en_audio_np, "P4", en_ensemble,
                         en_k_list, en_k_single,
                         raw_preds_np=np.array(p4_pred).astype(np.int64),
                         raw_weight=raw_anchor_weight_en,
                         raw_conf_np=np.array(p4_conf) if p4_conf is not None else None,
                         return_soft=True)
        p4_lp = np.array([labels_sorted[i] for i in p4_soft.argmax(axis=1)], dtype=np.int64)

    p6_soft = _run_lp(p6_centroids, ur_audio_np, "P6", ur_ensemble,
                     ur_k_list, ur_k_single,
                     raw_preds_np=np.array(p6_pred).astype(np.int64),
                     raw_weight=raw_anchor_weight_ur,
                     return_soft=True)
    p6_lp = np.array([labels_sorted[i] for i in p6_soft.argmax(axis=1)], dtype=np.int64)

    if p4_conf is not None and p6_conf is not None:
        label2idx = {int(l): i for i, l in enumerate(labels_sorted)}

        def _sim_guard(lp_pred, raw_pred, raw_conf, audio_np, centroids,
                       margin, conf_thresh, task_name, rule1_min_conf=0.0,
                       rule1_top1_top2_margin=0.0,
                       rule1b=False, rule1b_k=5, rule1b_max_raw_conf=1.0):
            diff_mask = lp_pred != raw_pred
            n_diff = diff_mask.sum()
            if n_diff == 0:
                return lp_pred
            diff_idx = np.where(diff_mask)[0]
            sims = audio_np[diff_idx] @ centroids.T
            audio_top1_idx = np.argmax(sims, axis=1)
            if rule1_top1_top2_margin > 0:
                audio_top2_sim = np.partition(-sims, 1, axis=1)[:, 1]
                audio_top2_sim = -audio_top2_sim

            n_top1 = 0
            n_raw = 0
            n_rule1b = 0
            for j, idx in enumerate(diff_idx):
                lp_c = lp_pred[idx]
                raw_c = raw_pred[idx]
                top1_c = labels_sorted[audio_top1_idx[j]]
                sim_lp = sims[j, label2idx.get(lp_c, 0)]
                sim_raw = sims[j, label2idx.get(raw_c, 0)]
                sim_top1 = sims[j, audio_top1_idx[j]]

                if rule1b and top1_c == lp_c and raw_c != lp_c:
                    if raw_conf[idx] < rule1b_max_raw_conf:
                        raw_idx_in_sims = label2idx.get(raw_c, 0)
                        rank_raw = (sims[j] > sim_raw).sum() + 1
                        if rank_raw <= rule1b_k:
                            lp_pred[idx] = int(raw_c)
                            n_rule1b += 1
                            continue

                rule1_fire = False
                if (raw_conf[idx] >= rule1_min_conf
                        and top1_c != lp_c
                        and sim_top1 > sim_lp + margin):
                    if rule1_top1_top2_margin > 0:
                        sim_top2 = audio_top2_sim[j]
                        if sim_top1 - sim_top2 > rule1_top1_top2_margin:
                            rule1_fire = True
                    else:
                        rule1_fire = True

                if rule1_fire:
                    lp_pred[idx] = int(top1_c)
                    n_top1 += 1
                elif sim_raw > sim_lp and raw_conf[idx] >= conf_thresh:
                    lp_pred[idx] = int(raw_c)
                    n_raw += 1
            return lp_pred

        p4_lp = _sim_guard(p4_lp, np.array(p4_pred).astype(np.int64),
                           np.array(p4_conf), en_audio_np, p4_centroids,
                           sim_guard_margin_en, sim_guard_conf_en, "P4")
        p6_lp = _sim_guard(p6_lp, np.array(p6_pred).astype(np.int64),
                           np.array(p6_conf), ur_audio_np, p6_centroids,
                           sim_guard_margin_ur, sim_guard_conf_ur, "P6",
                           rule1_min_conf=sim_guard_p6_rule1_min_conf,
                           rule1_top1_top2_margin=sim_guard_p6_rule1_top1_top2_margin,
                           rule1b=sim_guard_p6_rule1b,
                           rule1b_k=sim_guard_p6_rule1b_k,
                           rule1b_max_raw_conf=sim_guard_p6_rule1b_max_raw_conf)

    if p4_conf is not None and p6_conf is not None \
       and p4_soft is not None and p6_soft is not None:
        p4_raw = np.array(p4_pred).astype(np.int64)
        p6_raw = np.array(p6_pred).astype(np.int64)

        def _lp_soft_filter(lp_pred, raw_pred, lp_soft, task):
            diff_mask = lp_pred != raw_pred
            n_diff = diff_mask.sum()
            if n_diff == 0:
                return lp_pred
            lp_conf = lp_soft.max(axis=1)
            low_conf = diff_mask & (lp_conf < LP_SOFT_CONF_THRESH)
            lp_pred[low_conf] = raw_pred[low_conf]
            return lp_pred

        p4_lp = _lp_soft_filter(p4_lp, p4_raw, p4_soft, "P4")
        p6_lp = _lp_soft_filter(p6_lp, p6_raw, p6_soft, "P6")

    if p4_conf is not None and p6_conf is not None:
        label2idx = {int(l): i for i, l in enumerate(labels_sorted)}

        def _centroid_knn_override(pred, raw_pred, raw_conf, audio_np, centroids,
                                    task_name, conf_thresh, min_sim, margin,
                                    top1_top2_margin=0.0):
            unchanged = (pred == raw_pred)
            low_conf = raw_conf < conf_thresh
            candidates = unchanged & low_conf
            if candidates.sum() == 0:
                return pred

            sims = audio_np[candidates] @ centroids.T
            audio_top1_idx = np.argmax(sims, axis=1)
            audio_top1_sim = sims[np.arange(len(audio_top1_idx)), audio_top1_idx]

            if top1_top2_margin > 0:
                audio_top2_sim = np.partition(-sims, 1, axis=1)[:, 1]
                audio_top2_sim = -audio_top2_sim

            candidate_indices = np.where(candidates)[0]
            cur_sim = np.array([sims[j, label2idx.get(pred[idx], 0)]
                                for j, idx in enumerate(candidate_indices)])

            for j, idx in enumerate(candidate_indices):
                top1_c = labels_sorted[audio_top1_idx[j]]
                if (top1_c != pred[idx]
                        and audio_top1_sim[j] > min_sim
                        and audio_top1_sim[j] - cur_sim[j] > margin):
                    if top1_top2_margin > 0:
                        if audio_top1_sim[j] - audio_top2_sim[j] <= top1_top2_margin:
                            continue
                    pred[idx] = int(top1_c)

            return pred

        p4_lp = _centroid_knn_override(p4_lp, np.array(p4_pred).astype(np.int64),
                                        np.array(p4_conf), en_audio_np,
                                        p4_centroids, "P4",
                                        conf_thresh=LP_CENTROID_KNN_CONF_THRESH,
                                        min_sim=LP_CENTROID_KNN_MIN_SIM,
                                        margin=LP_CENTROID_KNN_MARGIN_EN)
        p6_lp = _centroid_knn_override(p6_lp, np.array(p6_pred).astype(np.int64),
                                        np.array(p6_conf), ur_audio_np,
                                        p6_centroids, "P6",
                                        conf_thresh=LP_CENTROID_KNN_CONF_THRESH,
                                        min_sim=LP_CENTROID_KNN_MIN_SIM,
                                        margin=LP_CENTROID_KNN_MARGIN_UR,
                                        top1_top2_margin=LP_CENTROID_KNN_TOP1_TOP2_MARGIN_UR)

    return p4_lp, p6_lp


def post_process_p4_p6_centroid(p4_pred, p6_pred,
                                 english_audio_emb, urdu_audio_emb,
                                 p4_confidence, p6_confidence,
                                 confidence_threshold=0.12,
                                 margin_threshold=0.05,
                                 min_audio_sim=0.3,
                                 train_csv_path=None,
                                 centroids_norm=None,
                                 labels_sorted=None,
                                 verbose=True):
    if train_csv_path is None:
        train_csv_path = "./csv_files/v1_train_English.csv"

    if centroids_norm is None or labels_sorted is None:
        centroids_norm, labels_sorted = _build_audio_centroids(train_csv_path, verbose=verbose)
    else:
        labels_sorted = list(labels_sorted)
        if verbose:
            print(f"[AudioCentroid] Using pre-built centroids: "
                  f"{centroids_norm.shape[0]} classes, dim={centroids_norm.shape[1]}")

    en_audio_np = F.normalize(english_audio_emb, dim=1).cpu().numpy().astype(np.float32)
    ur_audio_np = F.normalize(urdu_audio_emb, dim=1).cpu().numpy().astype(np.float32)

    p4_corrected, p4_stats = _apply_centroid_correction(
        p4_pred, p4_confidence, en_audio_np,
        centroids_norm, labels_sorted,
        confidence_threshold=confidence_threshold,
        margin_threshold=margin_threshold,
        min_audio_sim=min_audio_sim,
        task_name="P4",
        verbose=verbose,
    )

    p6_corrected, p6_stats = _apply_centroid_correction(
        p6_pred, p6_confidence, ur_audio_np,
        centroids_norm, labels_sorted,
        confidence_threshold=confidence_threshold,
        margin_threshold=margin_threshold,
        min_audio_sim=min_audio_sim,
        task_name="P6",
        verbose=verbose,
    )

    return p4_corrected, p6_corrected


def run_all_post_processing(
    en_f_raw, en_a_raw, ur_f_raw, ur_a_raw,
    p3_pred, p3_probs, p3_prob,
    p4_pred, p4_prob, p4_audio_embed,
    p5_pred, p5_probs, p5_prob,
    p6_pred, p6_prob,
    english_audio_feats, urdu_audio_feats,
    model=None, device=None, n_classes=None, face_dim=None,
    train_csv_path=None, verbose=False,
):
    if train_csv_path is None:
        train_csv_path = "./csv_files/v1_train_English.csv"

    if verbose:
        print("\n" + "=" * 60)
        print("STEP 2: P3 Post-Processing (product mode)")
        print("=" * 60)
    p3_pred, p3_stats = post_process_by_raw_prototype(
        en_f_raw, en_a_raw, p3_pred, p3_probs, p3_prob,
        confidence_threshold=P3_CONFIDENCE_THRESHOLD,
        ratio_threshold=P3_RATIO_THRESHOLD,
        score_mode="product", verbose=verbose,
    )
    if verbose:
        print(f"  P3: {p3_stats['n_total']} total | {p3_stats['n_low_conf']} low-conf "
              f"| {p3_stats['n_changed']} changed")

    if verbose:
        print("\n" + "=" * 60)
        print("STEP 3: P5 Post-Processing (Urdu-only)")
        print("=" * 60)
    p5_pred, p5_stats = post_process_by_raw_prototype(
        ur_f_raw, ur_a_raw, p5_pred, p5_probs, p5_prob,
        confidence_threshold=P5_CONFIDENCE_THRESHOLD,
        ratio_threshold=P5_RATIO_THRESHOLD,
        verbose=verbose,
    )
    if verbose:
        print(f"  P5: {p5_stats['n_total']} total | {p5_stats['n_low_conf']} low-conf "
              f"| {p5_stats['n_changed']} changed")

    if verbose:
        print("\n" + "=" * 60)
        print("STEP 3.4: Build train centroids & compute perturb drift from P3/P5 (in-memory)")
        print("=" * 60)
    centroids_norm, labels_sorted = _build_audio_centroids(train_csv_path, verbose=verbose)
    delta_en = _compute_drift_from_pseudo_labels(
        centroids_norm, english_audio_feats, p3_pred, n_classes, verbose=verbose)
    delta_ur = _compute_drift_from_pseudo_labels(
        centroids_norm, urdu_audio_feats, p5_pred, n_classes, verbose=verbose)

    if verbose:
        print("\n" + "=" * 60)
    if P4_P6_LP_MODE == "train_anchors_perturbed":
        if verbose:
            print("STEP 4: P4/P6 Graph LP (train_anchors_perturbed)")
            print("=" * 60)
        centroids_en, centroids_ur = _apply_perturb_drift(
            centroids_norm,
            delta_en=delta_en, delta_ur=delta_ur,
            gamma_en=PERTURB_GAMMA_EN, gamma_ur=PERTURB_GAMMA_UR,
            verbose=verbose,
        )
        p4_pred, p6_pred = post_process_p4_p6_graph_lp(
            p4_pred, p6_pred,
            english_audio_feats, urdu_audio_feats,
            centroids_norm=centroids_norm, labels_sorted=labels_sorted,
            centroids_en=centroids_en, centroids_ur=centroids_ur,
            p4_conf=p4_prob, p6_conf=p6_prob,
            verbose=verbose,
        )
    elif P4_P6_LP_MODE == "train_anchors":
        if verbose:
            print("STEP 4: P4/P6 Graph Label Propagation (train_anchors)")
            print("=" * 60)
        p4_pred, p6_pred = post_process_p4_p6_graph_lp(
            p4_pred, p6_pred,
            english_audio_feats, urdu_audio_feats,
            centroids_norm=centroids_norm, labels_sorted=labels_sorted,
            p4_conf=p4_prob, p6_conf=p6_prob,
            verbose=verbose,
        )
    else:
        if verbose:
            print("STEP 4: P4/P6 Training Centroid Correction [FALLBACK]")
            print("=" * 60)
        p4_pred, p6_pred = post_process_p4_p6_centroid(
            p4_pred, p6_pred,
            english_audio_feats, urdu_audio_feats,
            p4_confidence=p4_prob, p6_confidence=p6_prob,
            confidence_threshold=0.4,
            margin_threshold=0.05,
            min_audio_sim=0.3,
            centroids_norm=centroids_norm, labels_sorted=labels_sorted,
            verbose=verbose,
        )

    return p3_pred, p4_pred, p5_pred, p6_pred
