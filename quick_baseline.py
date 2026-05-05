"""
快速基准测试 —— 用 sklearn 估计三个模型的相对性能
目的：预估手写版能拿到多少 macro-F1，判断结果是否有规律可写
"""
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import f1_score, make_scorer

DATA_DIR = r"C:\Users\Kling\fma\data"

print("Loading data...")
X = np.load(f"{DATA_DIR}/X_medium.npy").astype(np.float32)
y = np.load(f"{DATA_DIR}/y_medium.npy").astype(np.int32)
print(f"  X: {X.shape}, y: {y.shape}, classes: {len(np.unique(y))}")

scorer = make_scorer(f1_score, average="macro", zero_division=0)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "GaussianNB": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GaussianNB()),
    ]),
    "LogisticRegression (C=1)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, C=1.0, solver="lbfgs",
                                   multi_class="multinomial", n_jobs=-1)),
    ]),
    "MLP (proxy for ResNet)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=(256, 256),
                              max_iter=100, random_state=42)),
    ]),
}

print("\n=== 5-fold Stratified CV — macro-F1 ===\n")
results = {}
for name, model in models.items():
    print(f"Running {name}...", end=" ", flush=True)
    scores = cross_validate(model, X, y, cv=cv, scoring=scorer,
                            n_jobs=-1, return_train_score=False)
    f1_scores = scores["test_score"]
    results[name] = f1_scores
    print(f"  {f1_scores.mean():.4f} ± {f1_scores.std():.4f}   "
          f"[min={f1_scores.min():.4f}, max={f1_scores.max():.4f}]")

print("\n=== 各折明细 ===")
for name, scores in results.items():
    fold_str = "  ".join([f"{s:.4f}" for s in scores])
    print(f"{name:35s}  {fold_str}")

print("\n=== 结论参考 ===")
names = list(results.keys())
means = [results[n].mean() for n in names]
best = names[np.argmax(means)]
worst = names[np.argmin(means)]
gap = max(means) - min(means)
print(f"最好: {best}  ({max(means):.4f})")
print(f"最差: {worst}  ({min(means):.4f})")
print(f"差距: {gap:.4f} ({gap*100:.1f} 个百分点)")
if gap > 0.05:
    print("→ 模型间差距明显，结论有内容可写")
else:
    print("→ 模型间差距较小，重点写分层采样/不均衡的影响")
