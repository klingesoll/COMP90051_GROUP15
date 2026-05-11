"""
Multinomial Logistic Regression trained with mini-batch SGD - from scratch.

按 COMP90051 项目提案 第 4 节 LR 实现细节 重构，相比初版补足以下 4 处：

  [1] 收敛准则      训练集 loss 连续 5 epoch 相对变化 <1e-4 时提前停止，
                    或达到 max_epochs=200。原版用固定 epochs=100，可能
                    过早停止或浪费算力。
  [2] loss 跟踪     每个 epoch 结束后记录全训练集 loss 到 self.loss_history_，
                    供收敛判断与后续学习曲线绘制使用。
  [3] predict_proba 暴露概率输出，供 confusion matrix 与少数类 recall 分析。
  [4] gradient_check 模块外提供数值梯度对照工具，作为算法正确性的金标准。

L2 正则
  采用 lam = 1/(C*n)（sklearn 约定，loss = mean(CE) + 1/(2*C*n) * ||W||^2）。
  数据项 (1/B)*sum 与正则项 (1/(C*n))*W 在数学上对应同一个全集 loss 的
  无偏 SGD 估计；C 越大正则越弱，与 sklearn LogisticRegression 一致。

  注意：因正则强度被 n=17000 稀释，C 网格 {0.01, 0.1, 1.0} 在此数据集上
  几乎无差异。建议在 nested CV 调用处使用 {0.001, 0.01, 0.1} 或更宽的
  范围，确保中间值显著胜出（提案要求）。

数值稳定
  softmax 使用 row-max shift；log-softmax 用同一技巧避免 exp 上溢后再 log。
"""
import numpy as np


class LogisticRegressionSGD:
    """
    多类 logistic regression，mini-batch SGD 训练。

    Parameters
    ----------
    C          : float - 正则强度倒数 (sklearn 约定，C 大则正则弱)
    lr         : float - SGD 学习率（固定，不衰减）
    max_epochs : int   - 训练轮数硬上限（提案: 200）
    batch_size : int   - mini-batch 大小（提案: 256）
    tol        : float - 单 epoch 相对 loss 变化阈值（提案: 1e-4）
    patience   : int   - 连续满足 tol 多少 epoch 才真正停止（提案: 5）
    seed       : int   - 洗牌与权重初始化的随机种子
    """
    def __init__(self, C=1.0, lr=0.01, max_epochs=200, batch_size=256,
                 tol=1e-4, patience=5, seed=42, epochs=None):
        # epochs 是为了和 run_all.py 现有 LR_GRID 兼容的别名；
        # 实际语义等同于 max_epochs（带早停的训练上限）
        if epochs is not None:
            max_epochs = epochs
        self.C          = C
        self.lr         = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.tol        = tol
        self.patience   = patience
        self.seed       = seed

    # ------------------------------------------------------------------ #
    # 数值稳定的 softmax / log-softmax                                    #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _softmax(z):
        """逐行 softmax，row-max shift 避免 exp 溢出。"""
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    @staticmethod
    def _log_softmax(z):
        """逐行 log-softmax，同样 shift 后再做 log-sum-exp。"""
        z = z - z.max(axis=1, keepdims=True)
        return z - np.log(np.exp(z).sum(axis=1, keepdims=True))

    def _normalise(self, X):
        return (X - self._mu) / self._sigma

    def _full_loss(self, X_norm, y_int):
        """全训练集 mean CE + L2，仅用于收敛监控（不参与 SGD 更新）。"""
        log_probs = self._log_softmax(X_norm @ self.W_ + self.b_)
        ce  = -log_probs[np.arange(len(y_int)), y_int].mean()
        reg = (1.0 / (2.0 * self.C * len(y_int))) * np.sum(self.W_ ** 2)
        return ce + reg

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #
    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        X   = np.asarray(X, dtype=np.float64)
        y   = np.asarray(y)

        # --- 在训练数据上拟合 z-score（不用测试 fold 的统计量）---------
        self._mu    = X.mean(axis=0)
        self._sigma = X.std(axis=0) + 1e-8
        X = self._normalise(X)

        n, F          = X.shape
        self.classes_ = np.unique(y)
        K             = len(self.classes_)

        # 把原始标签映射到 0..K-1（兼容非连续类标）
        self._label_map = {int(c): i for i, c in enumerate(self.classes_)}
        y_int = np.array([self._label_map[int(c)] for c in y], dtype=np.int32)

        # 权重零初始化（凸问题，初值不敏感）
        self.W_ = np.zeros((F, K))
        self.b_ = np.zeros(K)

        lam = 1.0 / (self.C * n)   # L2 系数（sklearn 约定的 per-sample 形式）

        # ── 训练循环 + 收敛检查 [改进 1, 2] ─────────────────────────────
        self.loss_history_   = [self._full_loss(X, y_int)]
        self.n_epochs_run_   = 0
        prev_loss   = self.loss_history_[0]
        n_satisfied = 0

        for epoch in range(self.max_epochs):
            perm = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                end = start + self.batch_size
                bx  = X[perm[start:end]]
                by  = y_int[perm[start:end]]
                bs  = len(by)

                # 前向 + 关键化简: dL/dz = y_hat - y (softmax+CE)
                probs = self._softmax(bx @ self.W_ + self.b_)
                grad  = probs.copy()
                grad[np.arange(bs), by] -= 1.0
                grad /= bs                                   # (1/B)*(Y_hat - Y)

                # W 更新 = lr * (data_grad + L2_grad), b 不正则
                self.W_ -= self.lr * (bx.T @ grad + lam * self.W_)
                self.b_ -= self.lr * grad.sum(axis=0)

            # epoch 末尾在全训练集上算一次 loss 用于收敛判断
            curr_loss = self._full_loss(X, y_int)
            self.loss_history_.append(curr_loss)
            self.n_epochs_run_ = epoch + 1

            rel_change = abs(prev_loss - curr_loss) / max(abs(prev_loss), 1e-12)
            if rel_change < self.tol:
                n_satisfied += 1
                if n_satisfied >= self.patience:
                    break
            else:
                n_satisfied = 0
            prev_loss = curr_loss

        return self

    def predict_proba(self, X):
        """[改进 3] 返回 (n, K) 概率矩阵，供 confusion matrix / 概率分析用。"""
        X = self._normalise(np.asarray(X, dtype=np.float64))
        return self._softmax(X @ self.W_ + self.b_)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def get_params(self):
        return {"C": self.C, "lr": self.lr, "max_epochs": self.max_epochs,
                "batch_size": self.batch_size, "tol": self.tol,
                "patience": self.patience}


# ====================================================================== #
# [改进 4] 梯度检验工具                                                   #
# ====================================================================== #

def gradient_check(C=1.0, n=50, F=20, K=16, eps=1e-5, n_checks=20, seed=0):
    """
    用中心差分数值梯度对照解析梯度。
    实现正确时，最大相对误差应当 < 1e-5。

    在小规模合成数据上跑一次就够，作为算法正确性的金标准。
    通过此检验后，所有后续问题都不用再怀疑梯度公式或矩阵 shape。

    Returns
    -------
    max_err : float  - 所有抽样位置中的最大相对误差
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n, F))
    y = rng.integers(0, K, size=n)

    model = LogisticRegressionSGD(C=C, seed=seed)

    # 手动初始化（不跑 SGD），匹配 fit() 内部状态
    model._mu    = X.mean(axis=0)
    model._sigma = X.std(axis=0) + 1e-8
    X_norm = model._normalise(X)
    model.classes_   = np.unique(y)
    model._label_map = {int(c): i for i, c in enumerate(model.classes_)}
    y_int = np.array([model._label_map[int(c)] for c in y], dtype=np.int32)
    model.W_ = rng.normal(0, 0.01, size=(F, len(model.classes_)))
    model.b_ = np.zeros(len(model.classes_))

    # 解析梯度（全批量，便于和 _full_loss 对照）
    probs = model._softmax(X_norm @ model.W_ + model.b_)
    grad_logits = probs.copy()
    grad_logits[np.arange(n), y_int] -= 1.0
    grad_logits /= n
    lam = 1.0 / (model.C * n)
    dW_ana = X_norm.T @ grad_logits + lam * model.W_
    db_ana = grad_logits.sum(axis=0)

    def L():
        return model._full_loss(X_norm, y_int)

    max_err = 0.0
    rng2 = np.random.default_rng(seed + 1)

    # 抽查 W
    for _ in range(n_checks):
        i = rng2.integers(F); j = rng2.integers(len(model.classes_))
        orig = model.W_[i, j]
        model.W_[i, j] = orig + eps; lp = L()
        model.W_[i, j] = orig - eps; lm = L()
        model.W_[i, j] = orig
        num = (lp - lm) / (2 * eps)
        ana = dW_ana[i, j]
        err = abs(num - ana) / max(abs(num), abs(ana), 1e-12)
        max_err = max(max_err, err)

    # 抽查 b
    for _ in range(n_checks):
        j = rng2.integers(len(model.classes_))
        orig = model.b_[j]
        model.b_[j] = orig + eps; lp = L()
        model.b_[j] = orig - eps; lm = L()
        model.b_[j] = orig
        num = (lp - lm) / (2 * eps)
        ana = db_ana[j]
        err = abs(num - ana) / max(abs(num), abs(ana), 1e-12)
        max_err = max(max_err, err)

    return max_err


if __name__ == "__main__":
    # 自测：跑一次梯度检验，确认实现正确
    err = gradient_check()
    print(f"Gradient check max relative error: {err:.2e}  (need < 1e-5)")
    assert err < 1e-5, "Gradient check FAILED"
    print("PASSED")