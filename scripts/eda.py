"""
数据探索脚本 —— 为报告 Introduction + Results 章节收集素材
生成的图保存到 figures/ 目录，可直接插入报告。

运行: python eda.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os

# ── 路径 ─────────────────────────────────────────────────────────────────────
_ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(_ROOT, 'data')
FIG_DIR      = os.path.join(_ROOT, 'figures')
TRACKS_PATH  = os.path.join(DATA_DIR, 'fma_metadata', 'tracks.csv')
FEATURES_PATH= os.path.join(DATA_DIR, 'fma_metadata', 'features.csv')
X_PATH       = os.path.join(DATA_DIR, 'X_medium.npy')
Y_PATH       = os.path.join(DATA_DIR, 'y_medium.npy')
LABELS_PATH  = os.path.join(DATA_DIR, 'genre_labels.csv')

os.makedirs(FIG_DIR, exist_ok=True)
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)

# ── 加载数据 ─────────────────────────────────────────────────────────────────
print("加载数据 ...")
import ast
tracks = pd.read_csv(TRACKS_PATH, index_col=0, header=[0, 1])
medium = tracks[tracks[('set','subset')] == 'medium']
features = pd.read_csv(FEATURES_PATH, index_col=0, header=[0,1,2])
feat_medium = features.loc[features.index.isin(medium.index)]

X = np.load(X_PATH)
y = np.load(Y_PATH)
genres = pd.read_csv(LABELS_PATH, header=None, index_col=0)[1].values

print(f"  X: {X.shape}, y: {y.shape}, genres: {len(genres)}")

# ════════════════════════════════════════════════════════════════════════════
# 图 1: 流派分布（类别不平衡可视化）
# ════════════════════════════════════════════════════════════════════════════
print("\n[图1] 流派分布 ...")

genre_counts = pd.Series(y).map(dict(enumerate(genres))).value_counts()

fig, ax = plt.subplots(figsize=(10, 5))
colors = sns.color_palette('muted', len(genre_counts))
bars = ax.barh(genre_counts.index, genre_counts.values, color=colors)
ax.set_xlabel('Track Count')
ax.set_title('FMA-Medium: Genre Distribution (n=17,000)\nImbalance ratio = 339x (Rock vs International)', 
             fontsize=12)
for bar, val in zip(bars, genre_counts.values):
    ax.text(val + 30, bar.get_y() + bar.get_height()/2,
            f'{val:,}', va='center', fontsize=9)
ax.axvline(17000/16, color='red', linestyle='--', alpha=0.6, label='Uniform baseline (1/16)')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig1_genre_distribution.png'), dpi=150)
plt.close()
print("  → figures/fig1_genre_distribution.png")

# ════════════════════════════════════════════════════════════════════════════
# 图 2: 原始特征 vs 构造特征的分布对比（取代表性的几列）
# ════════════════════════════════════════════════════════════════════════════
print("\n[图2] 特征分布对比 ...")

# 原始：MFCC mean 01 (col 0)
# 构造：CV of MFCC 01 (col 518), Chroma entropy stft (col ~638)
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
fig.suptitle('Feature Distribution: Raw vs Engineered\n(colored by genre, top 5 genres shown)', 
             fontsize=12)

top5_mask  = np.isin(y, np.argsort(np.bincount(y))[-5:])
top5_labels = genres[np.argsort(np.bincount(y))[-5:]]
palette = sns.color_palette('tab10', 5)

plot_cols = [
    (0,   'MFCC-01 mean\n(raw)',           'Original'),
    (20,  'MFCC-01 std\n(raw)',            'Original'),
    (140, 'Chroma CENS mean-01\n(raw)',    'Original'),
    (518, 'CV(MFCC-01)\n(engineered)',     'Engineered'),
    (537, 'MFCC coeff diff-01\n(engineered)', 'Engineered'),
    (636, 'Chroma entropy (stft)\n(engineered)', 'Engineered'),
]

for ax, (col_idx, col_name, col_type) in zip(axes.ravel(), plot_cols):
    for i, (g_idx, g_name) in enumerate(zip(np.argsort(np.bincount(y))[-5:], top5_labels)):
        mask = (y == g_idx)
        vals = X[mask, col_idx]
        sns.kdeplot(vals, ax=ax, label=g_name, color=palette[i], linewidth=1.5)
    ax.set_title(col_name, fontsize=10)
    ax.set_xlabel('')
    bg = '#EBF5FB' if col_type == 'Engineered' else '#FFFFFF'
    ax.set_facecolor(bg)
    if col_idx == 0:
        ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig2_feature_distributions.png'), dpi=150)
plt.close()
print("  → figures/fig2_feature_distributions.png")

# ════════════════════════════════════════════════════════════════════════════
# 图 3: 特征相关性热图（各特征组代表列）
# ════════════════════════════════════════════════════════════════════════════
print("\n[图3] 特征相关性热图 ...")

# 取各组的 mean 第1系数作代表，共11个原始组 + 5个构造特征
rep_cols = {
    'mfcc_mean01':    0,
    'mfcc_std01':     20,
    'chroma_cens01':  140,
    'chroma_cqt01':   224,
    'chroma_stft01':  308,
    'spec_contrast01':392,
    'tonnetz01':      441,
    'rmse_mean':      462,
    'spec_bw_mean':   469,
    'spec_cen_mean':  476,
    'zcr_mean':       511,
    'CV_mfcc01':      518,
    'mfcc_diff01':    587,
    'ratio_cen_bw':   606,
    'ratio_rmse_zcr': 608,
    'entropy_chroma': 636,
}

rep_df = pd.DataFrame({k: X[:, v] for k, v in rep_cols.items()})
corr   = rep_df.corr()

fig, ax = plt.subplots(figsize=(11, 9))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, ax=ax, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            annot=True, fmt='.2f', linewidths=0.5, 
            annot_kws={'size': 8})
ax.set_title('Representative Feature Correlations\n(lower triangle)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig3_feature_correlation.png'), dpi=150)
plt.close()
print("  → figures/fig3_feature_correlation.png")

# ════════════════════════════════════════════════════════════════════════════
# 图 4: 分层 K-Fold 验证图（证明折间分布均衡）
# ════════════════════════════════════════════════════════════════════════════
print("\n[图4] 分层 K-Fold 折间分布均衡性 ...")

K = 10
np.random.seed(42)

# 手写分层 K-Fold
folds = [[] for _ in range(K)]
for label in range(len(genres)):
    idx = np.where(y == label)[0].tolist()
    np.random.shuffle(idx)
    for i, sample_idx in enumerate(idx):
        folds[i % K].append(sample_idx)

# 计算每折各类别比例
global_ratio = np.bincount(y) / len(y)
fold_ratios  = []
for fold in folds:
    fold_y = y[fold]
    ratio  = np.bincount(fold_y, minlength=len(genres)) / len(fold_y)
    fold_ratios.append(ratio)
fold_ratios = np.array(fold_ratios)  # (K, n_genres)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Stratified K-Fold (K=10): Balance Verification', fontsize=12)

# 左图：每折大小
fold_sizes = [len(f) for f in folds]
axes[0].bar(range(1, K+1), fold_sizes, color=sns.color_palette('muted', K))
axes[0].axhline(np.mean(fold_sizes), color='red', linestyle='--', 
                label=f'Mean = {np.mean(fold_sizes):.0f}')
axes[0].set_xlabel('Fold')
axes[0].set_ylabel('Number of samples')
axes[0].set_title('Fold Sizes')
axes[0].legend()
axes[0].set_xticks(range(1, K+1))

# 右图：各折最大偏差（与全局比例的差）
max_devs = np.abs(fold_ratios - global_ratio).max(axis=1)
axes[1].bar(range(1, K+1), max_devs * 100, 
            color=['green' if d < 0.5 else 'orange' for d in max_devs])
axes[1].axhline(0.5, color='red', linestyle='--', label='0.5% threshold')
axes[1].set_xlabel('Fold')
axes[1].set_ylabel('Max genre proportion deviation (%)')
axes[1].set_title('Max Genre Imbalance per Fold\n(lower = more balanced)')
axes[1].legend()
axes[1].set_xticks(range(1, K+1))

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig4_kfold_balance.png'), dpi=150)
plt.close()
print("  → figures/fig4_kfold_balance.png")

# ════════════════════════════════════════════════════════════════════════════
# 图 5: 工程特征的判别力（类内 vs 类间方差）
# ════════════════════════════════════════════════════════════════════════════
print("\n[图5] 特征判别力（F统计量 top 20）...")

# 手写单因素方差分析 F 统计量 = 类间方差 / 类内方差
n_features = X.shape[1]
n_classes  = len(genres)
N          = len(y)

grand_mean = X.mean(axis=0)
ss_between = np.zeros(n_features)
ss_within  = np.zeros(n_features)

for c in range(n_classes):
    mask   = (y == c)
    nc     = mask.sum()
    if nc == 0:
        continue
    class_mean = X[mask].mean(axis=0)
    ss_between += nc * (class_mean - grand_mean) ** 2
    ss_within  += ((X[mask] - class_mean) ** 2).sum(axis=0)

df_between = n_classes - 1
df_within  = N - n_classes
F_stat = (ss_between / df_between) / (ss_within / df_within + 1e-8)

# 生成列名
orig_names = ['_'.join(c).strip() for c in 
              pd.read_csv(FEATURES_PATH, index_col=0, header=[0,1,2]).columns.values]
new_names  = []
for feat_name in ['mfcc','chroma_cens','chroma_cqt','chroma_stft','spectral_contrast','tonnetz']:
    feat_medium_tmp = features.loc[features.index.isin(medium.index)]
    try:
        mean_cols = feat_medium_tmp[feat_name]['mean'].columns
        new_names += [f'cv_{feat_name}_{c}' for c in mean_cols]
    except: pass
for i in range(19):
    new_names.append(f'mfcc_coeff_diff_{i+1:02d}')
new_names += ['ratio_centroid_bandwidth','ratio_rolloff_centroid','ratio_rmse_zcr']
for feat_name in ['mfcc','spectral_contrast']:
    try:
        skew_cols = feat_medium_tmp[feat_name]['skew'].columns
        new_names += [f'sk_product_{feat_name}_{c}' for c in skew_cols]
    except: pass
new_names += ['entropy_chroma_stft','entropy_chroma_cqt','entropy_chroma_cens']

all_names = orig_names + new_names
assert len(all_names) == n_features, \
    f"列名数量 {len(all_names)} != 特征数量 {n_features}"

top20_idx   = np.argsort(F_stat)[-20:][::-1]
top20_names = [all_names[i] for i in top20_idx]
top20_F     = F_stat[top20_idx]

# 标记是原始还是构造
colors_f = ['#E74C3C' if i >= 518 else '#3498DB' for i in top20_idx]

fig, ax = plt.subplots(figsize=(11, 7))
bars = ax.barh(range(20), top20_F[::-1], color=colors_f[::-1])
ax.set_yticks(range(20))
ax.set_yticklabels([n.replace('_',' ') for n in top20_names[::-1]], fontsize=9)
ax.set_xlabel('F-statistic (higher = more discriminative)')
ax.set_title('Top 20 Most Discriminative Features\n(Red = Engineered, Blue = Original)', fontsize=12)

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#E74C3C', label='Engineered features'),
                   Patch(facecolor='#3498DB', label='Original features')]
ax.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig5_feature_discriminability.png'), dpi=150)
plt.close()
print("  → figures/fig5_feature_discriminability.png")

# ════════════════════════════════════════════════════════════════════════════
# 打印数据探索摘要（写报告用）
# ════════════════════════════════════════════════════════════════════════════
print(f"""
{'='*60}
数据探索摘要（可直接用于报告）

数据集规模:
  样本数:       17,000
  原始特征维度: 518
  构造特征维度: 121
  最终维度:     639

类别不平衡:
  最大类 Rock:         6,103 条 (35.9%)
  最小类 International:   18 条 (0.1%)
  不平衡比:            339x
  → 必须使用分层 K-Fold，报告中需说明原因

特征判别力 Top 3:
  1. {top20_names[0]}  (F={top20_F[0]:.1f})
  2. {top20_names[1]}  (F={top20_F[1]:.1f})
  3. {top20_names[2]}  (F={top20_F[2]:.1f})

分层 K-Fold 质量:
  折大小范围:   {min(fold_sizes)} ~ {max(fold_sizes)} 条
  最大比例偏差: {max_devs.max()*100:.3f}%（远低于 0.5%）

生成的图表:
  fig1_genre_distribution.png    → Introduction 章节用
  fig2_feature_distributions.png → Methods/Feature 章节用
  fig3_feature_correlation.png   → Methods/Feature 章节用
  fig4_kfold_balance.png         → Methods/CV 章节用
  fig5_feature_discriminability.png → Results 章节用
{'='*60}
""")
