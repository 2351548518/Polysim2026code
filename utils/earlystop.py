class EarlyStopping:
    """早停器。

    当监控指标在连续 `patience` 轮内未达到 `min_delta` 的提升时，
    返回 should_stop=True，供外部训练循环决定是否提前结束。
    """

    def __init__(self, patience, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def step(self, score):
        """输入本轮指标并更新状态，返回是否应停止训练。"""
        if self.best_score is None:
            self.best_score = score
            return False

        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop
