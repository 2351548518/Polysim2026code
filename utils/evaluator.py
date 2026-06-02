import torch


class Evaluator:
    """评估器：提供向量化准确率计算，并缓存数据集张量。"""

    def __init__(self, model, config):
        self.model = model
        self.config = config

        # Cache tensors (lazy init)
        self._cached = {}

    # --------------------------------------------------
    # Dataset → tensor cache
    # --------------------------------------------------
    def _get_tensors(self, dataset):
        """
        缓存张量，避免重复执行 torch.from_numpy。
        """
        key = id(dataset)
        if key not in self._cached:
            self._cached[key] = (
                torch.from_numpy(dataset.face_feats).float(),
                torch.from_numpy(dataset.audio_feats).float(),
                torch.from_numpy(dataset.labels).long(),
            )
        return self._cached[key]

    # --------------------------------------------------
    # Core accuracy from tensors
    # --------------------------------------------------
    def accuracy_from_tensors(
        self,
        face,
        audio,
        labels,
        head="fusion",   # "fusion" | "face" | "audio"
    ):
        """
        基于张量直接计算准确率（向量化实现）。

        head:
            - "fusion"（默认）
            - "face"
            - "audio"
        """
        self.model.eval()

        with torch.no_grad():
            out = self.model(face, audio)

            # ---------------------------
            # MultiBranchFOP
            # ---------------------------
            if isinstance(out, dict):
                # MultiBranchFOP 支持按分支选择 logits。
                if head == "fusion":
                    logits = out["fusion_logits"]
                elif head == "face":
                    logits = out["face_logits"]
                elif head == "audio":
                    logits = out["audio_logits"]
                else:
                    raise ValueError(f"Unknown head: {head}")

            # ---------------------------
            # Baseline FOP
            # ---------------------------
            else:
                # FOP 仅有单个 logits 输出。
                _, logits, _, _ = out

            preds = logits.argmax(dim=1)
            correct = (preds == labels).sum().item()

        return 100.0 * correct / labels.size(0)

    # --------------------------------------------------
    # Dataset-level accuracy
    # --------------------------------------------------
    def accuracy(self, dataset, head="fusion"):
        """
        数据集级准确率计算（快速路径）。

        head:
            - "fusion"（默认）
            - "face"
            - "audio"（仅 MultiBranchFOP 有效）
        """
        face, audio, labels = self._get_tensors(dataset)

        face = face.to(self.config.device, non_blocking=True)
        audio = audio.to(self.config.device, non_blocking=True)
        labels = labels.to(self.config.device, non_blocking=True)

        return self.accuracy_from_tensors(
            face,
            audio,
            labels,
            head=head,
        )
