"""
特征工程探索脚本 —— 自我诊断版
运行方式: python feature_engineering_explore.py
每一步都有断言检查，出错会告诉你哪里错了、为什么错。
"""

import os
import numpy as np
import pandas as pd
import sys

_ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR    = os.path.join(_ROOT, 'data', 'fma_metadata')
TRACKS_PATH  = os.path.join(_DATA_DIR, 'tracks.csv')
FEATURES_PATH = os.path.join(_DATA_DIR, 'features.csv')

# ════════════════════════════════════════════════════════════════════════════
# 工具：带颜色的打印
# ════════════════════════════════════════════════════════════════════════════

def ok(msg):   print(f"  [✓] {msg}")
def warn(msg): print(f"  [!] {msg}")
def err(msg):  print(f"  [✗] {msg}"); sys.exit(1)
def step(msg): print(f"\n{'='*60}\n{msg}\n{'='*60}")

# ════════════════════════════════════════════════════════════════════════════
# STEP 0: 理解数据结构（最重要的一步）
# ════════════════════════════════════════════════════════════════════════════
step("STEP 0: 理解 features.csv 的真实结构")

features = pd.read_csv(FEATURES_PATH, index_col=0, header=[0, 1, 2])
print(f"  形状: {features.shape}")
print(f"  列索引层级: {features.columns.nlevels} 层")
print(f"  层级名称: {features.columns.names}")

# 关键认知：这是「汇总统计」而非「原始帧序列」
print("""
  ┌─────────────────────────────────────────────────────────────┐
  │  重要！features.csv 存储的是每首歌的「统计量汇总」           │
  │                                                             │
  │  列结构: (特征名, 统计量, 系数编号)                          │
  │  例如: ('mfcc', 'mean', '01') = 第1个MFCC系数的均值         │
  │        ('mfcc', 'std',  '01') = 第1个MFCC系数的标准差       │
  │                                                             │
  │  可用统计量: mean, std, skew, kurtosis, min, max, median    │
  │  时间维度已被折叠 → 无法直接算 Δ-MFCC（需要原始音频）       │
  └─────────────────────────────────────────────────────────────┘
""")

# 验证
assert features.columns.nlevels == 3, "列索引应该是3层！"
ok("列结构确认：3层多级索引 (feature, stat, number)")

# ════════════════════════════════════════════════════════════════════════════
# STEP 1: 筛选 medium 子集
# ════════════════════════════════════════════════════════════════════════════
step("STEP 1: 加载 tracks.csv，筛选 medium 子集")

import ast
tracks = pd.read_csv(TRACKS_PATH, index_col=0, header=[0, 1])
medium_mask = tracks[('set', 'subset')] == 'medium'
medium_ids  = tracks[medium_mask].index
y_raw       = tracks.loc[medium_ids, ('track', 'genre_top')]

print(f"  medium 样本数: {len(medium_ids)}")
print(f"  genre_top NaN 数: {y_raw.isna().sum()}")

assert len(medium_ids) >= 10000, f"medium 行数不足！只有 {len(medium_ids)}"
assert y_raw.isna().sum() == 0, "存在无标签样本，需要先剔除"
ok(f"medium 子集 {len(medium_ids)} 条，标签完整")

# 过滤 features，只保留 medium
feat_medium = features.loc[features.index.isin(medium_ids)].copy()
assert feat_medium.shape[0] == len(medium_ids), \
    f"features 中找到 {feat_medium.shape[0]} 条，期望 {len(medium_ids)}"
ok(f"features 对齐完成: {feat_medium.shape}")

# 对齐索引顺序（重要！）
common_ids  = feat_medium.index.intersection(medium_ids)
feat_medium = feat_medium.loc[common_ids]
y_series    = y_raw.loc[common_ids]
assert (feat_medium.index == y_series.index).all(), "索引不对齐！"
ok("索引对齐验证通过")

# ════════════════════════════════════════════════════════════════════════════
# STEP 2: 评估可以构造哪些「非平凡特征」
# ════════════════════════════════════════════════════════════════════════════
step("STEP 2: 非平凡特征构造方案（从 features.csv 出发）")

print("""
  虽然时间维度已折叠，但仍可从统计量中构造大量非平凡特征：

  A) 变异系数 (CV = std/mean)
     物理含义：该特征的相对波动强度
     非平凡性：原始列中没有，是 std 和 mean 的比值

  B) MFCC 系数间差分（伪谱导数）
     物理含义：相邻 MFCC 系数均值之差 → 频谱包络斜率
     非平凡性：类比 Δ-MFCC，但在系数维度而非时间维度
     注意：这不等于时域 Δ-MFCC，但仍有区分度

  C) 跨特征比率（声学物理意义）
     spectral_centroid_mean / spectral_bandwidth_mean → 谱集中度
     rmse_mean / zcr_mean                             → 有声/无声比
     spectral_rolloff_mean / spectral_centroid_mean   → 高频能量比

  D) 统计量之间的交叉项
     skew × kurtosis（分布形状联合指标）
     (max - min) / (std + 1e-8)（归一化动态范围）

  E) Chroma 熵（12个音高bin的信息熵）
     物理含义：音调分布的多样性，Rock vs Classical 差异大
     计算：-sum(p * log(p)) 其中 p = softmax(chroma_mean)
""")

# ════════════════════════════════════════════════════════════════════════════
# STEP 3: 实际构造特征，每步带断言
# ════════════════════════════════════════════════════════════════════════════
step("STEP 3: 实际构造特征")

new_features = {}

# ── A: 变异系数 (CV) ─────────────────────────────────────────────────────
print("  [A] 变异系数 CV = std / mean ...")

for feat_name in ['mfcc', 'chroma_cens', 'chroma_cqt', 'chroma_stft',
                  'spectral_contrast', 'tonnetz']:
    try:
        mean_df = feat_medium[feat_name]['mean']
        std_df  = feat_medium[feat_name]['std']
        # 防止除以零
        cv = std_df.values / (np.abs(mean_df.values) + 1e-8)
        col_names = [f'cv_{feat_name}_{c}' for c in mean_df.columns]
        for i, name in enumerate(col_names):
            new_features[name] = cv[:, i]
    except KeyError as e:
        warn(f"  特征 {feat_name} 不存在: {e}")

ok(f"CV 特征数量: {len(new_features)}")

# ── B: MFCC 系数间差分（伪谱导数）────────────────────────────────────────
print("  [B] MFCC 系数间差分（系数 k+1 减 k）...")

mfcc_mean = feat_medium['mfcc']['mean'].values  # shape: (N, 20)
assert mfcc_mean.shape[1] == 20, f"期望20个MFCC系数，实际{mfcc_mean.shape[1]}"

mfcc_diff = np.diff(mfcc_mean, axis=1)  # shape: (N, 19)
assert mfcc_diff.shape == (len(feat_medium), 19), \
    f"差分形状错误: {mfcc_diff.shape}"

for i in range(mfcc_diff.shape[1]):
    new_features[f'mfcc_coeff_diff_{i+1:02d}'] = mfcc_diff[:, i]

ok(f"MFCC差分特征: {mfcc_diff.shape[1]} 维")

# ── C: 跨特征比率 ─────────────────────────────────────────────────────────
print("  [C] 跨特征比率（声学物理意义）...")

def get_scalar(feat, stat):
    """获取标量特征（如 spectral_centroid 只有1个系数）"""
    arr = feat_medium[feat][stat].values.ravel()
    assert arr.shape == (len(feat_medium),), \
        f"{feat}/{stat} 形状期望 ({len(feat_medium)},) 实际 {arr.shape}"
    return arr

sc_mean  = get_scalar('spectral_centroid',  'mean')
sb_mean  = get_scalar('spectral_bandwidth', 'mean')
sr_mean  = get_scalar('spectral_rolloff',   'mean')
rmse_mean = get_scalar('rmse',              'mean')
zcr_mean  = get_scalar('zcr',              'mean')

new_features['ratio_centroid_bandwidth'] = sc_mean  / (sb_mean  + 1e-8)
new_features['ratio_rolloff_centroid']   = sr_mean  / (sc_mean  + 1e-8)
new_features['ratio_rmse_zcr']           = rmse_mean / (zcr_mean + 1e-8)

ok("跨特征比率: 3 维")

# ── D: skew × kurtosis 联合项 ─────────────────────────────────────────────
print("  [D] skew × kurtosis 联合分布指标...")

for feat_name in ['mfcc', 'spectral_contrast']:
    try:
        skew_df = feat_medium[feat_name]['skew'].values
        kurt_df = feat_medium[feat_name]['kurtosis'].values
        product = skew_df * kurt_df
        n_cols  = product.shape[1]
        for i in range(n_cols):
            new_features[f'sk_product_{feat_name}_{i+1:02d}'] = product[:, i]
    except KeyError as e:
        warn(f"  {feat_name} 无法计算: {e}")

ok(f"skew×kurtosis 特征数量: {sum(1 for k in new_features if 'sk_product' in k)}")

# ── E: Chroma 熵 ──────────────────────────────────────────────────────────
print("  [E] Chroma 信息熵（音调多样性）...")

for chroma_name in ['chroma_stft', 'chroma_cqt', 'chroma_cens']:
    chroma_mean = feat_medium[chroma_name]['mean'].values  # (N, 12)
    assert chroma_mean.shape[1] == 12, \
        f"{chroma_name} 期望12个bin，实际{chroma_mean.shape[1]}"
    # softmax → 概率分布 → 信息熵
    chroma_shifted = chroma_mean - chroma_mean.min(axis=1, keepdims=True)
    exp_vals = np.exp(chroma_shifted)
    probs    = exp_vals / (exp_vals.sum(axis=1, keepdims=True) + 1e-8)
    entropy  = -np.sum(probs * np.log(probs + 1e-8), axis=1)
    assert entropy.shape == (len(feat_medium),), \
        f"熵形状错误: {entropy.shape}"
    new_features[f'entropy_{chroma_name}'] = entropy

ok("Chroma 熵: 3 维（每种 chroma 各一个）")

# ════════════════════════════════════════════════════════════════════════════
# STEP 4: 合并原始特征 + 新特征，最终检查
# ════════════════════════════════════════════════════════════════════════════
step("STEP 4: 合并特征矩阵，最终完整性检查")

# 原始特征（展平多级索引）
X_orig = feat_medium.copy()
X_orig.columns = ['_'.join(col).strip() for col in X_orig.columns.values]

# 新构造特征
new_df = pd.DataFrame(new_features, index=feat_medium.index)

# 合并
X_full = pd.concat([X_orig, new_df], axis=1)

# 检查
assert X_full.shape[0] == len(feat_medium), "行数不一致！"
assert X_full.isna().sum().sum() == 0, \
    f"合并后出现 NaN！数量: {X_full.isna().sum().sum()}\n" \
    f"出问题的列: {X_full.columns[X_full.isna().any()].tolist()[:10]}"
assert X_full.shape[1] > 518, \
    f"新特征数量不对，总列数{X_full.shape[1]}应>518"

print(f"  原始特征:   518 列")
print(f"  新构造特征: {len(new_features)} 列")
print(f"  最终矩阵:   {X_full.shape[0]} 行 × {X_full.shape[1]} 列")
print(f"  NaN 总数:   {X_full.isna().sum().sum()}")

ok("特征矩阵完整，无缺失值")

# ── Winsorize：按列 clip 到 [p1, p99] ────────────────────────────────────
# 动机：CV = std/(|mean|+eps) 在 MFCC 均值接近 0 时会爆炸（实测最大值 541799）。
# 模型内部 z-score 无法充分缓解：单个极端值会把其他 16999 个样本压到均值附近。
# 安全性：winsorize 是无监督、逐列、全量操作，不依赖标签，不引入 fold 间信息泄漏。
# 在 CV pipeline 里不需要再 re-fit winsorize 参数（截断百分位是全局固定的）。
step("STEP 4b: Winsorize — 按列 clip 到 [p1, p99]")

X_arr_raw = X_full.values.astype(np.float32)
p1  = np.percentile(X_arr_raw, 1,  axis=0)
p99 = np.percentile(X_arr_raw, 99, axis=0)
X_arr_clipped = np.clip(X_arr_raw, p1, p99)

# 验证
assert not np.isnan(X_arr_clipped).any(), "Winsorize 后出现 NaN！"
assert not np.isinf(X_arr_clipped).any(), "Winsorize 后出现 Inf！"
mu  = X_arr_clipped.mean(0)
sd  = X_arr_clipped.std(0) + 1e-8
z   = np.abs((X_arr_clipped - mu) / sd)
extreme_after = (z > 10).sum()
extreme_before = (np.abs((X_arr_raw - X_arr_raw.mean(0)) /
                         (X_arr_raw.std(0) + 1e-8)) > 10).sum()
print(f"  Winsorize 前 |z|>10 的值: {extreme_before}")
print(f"  Winsorize 后 |z|>10 的值: {extreme_after}")
print(f"  全局 max 前: {X_arr_raw.max():.1f}  → 后: {X_arr_clipped.max():.1f}")
ok("Winsorize 完成，极端值已消除")

# 标签编码
y_encoded = pd.Categorical(y_series).codes
genres    = pd.Categorical(y_series).categories
print(f"\n  标签分布（共 {len(genres)} 类）:")
for g, code in zip(genres, range(len(genres))):
    count = (y_encoded == code).sum()
    print(f"    [{code:2d}] {g:<22} {count:5d} 条")

# ════════════════════════════════════════════════════════════════════════════
# STEP 5: 保存供后续使用
# ════════════════════════════════════════════════════════════════════════════
step("STEP 5: 保存 X_medium.npy / X_medium_raw_derived.npy / y_medium.npy")

out_dir = os.path.join(_ROOT, 'data')
X_arr = X_arr_clipped          # 使用 winsorize 后的版本（供 EDA / 快速调试）
y_arr = y_encoded.astype(np.int32)

np.save(os.path.join(out_dir, 'X_medium.npy'), X_arr)
np.save(os.path.join(out_dir, 'y_medium.npy'), y_arr)

# ── 新增：保存未 Winsorize 的原始派生特征（供正式 CV 按折处理）─────────────
# 说明：正式 nested CV 中，Winsorization 参数必须只从 training fold 估计
# 再应用到 test fold，避免测试集分布信息泄漏到预处理步骤。
# X_medium_raw_derived.npy = 原始 518 维 + 未 Winsorize 的 121 维派生特征
# nested_cv.py 会在每个外层折内部完成 Winsorize → Z-score 两步归一化。
np.save(os.path.join(out_dir, 'X_medium_raw_derived.npy'), X_arr_raw)

# 保存标签映射
pd.Series(genres).to_csv(os.path.join(out_dir, 'genre_labels.csv'), header=False)

# 最终验证
X_check     = np.load(os.path.join(out_dir, 'X_medium.npy'))
X_raw_check = np.load(os.path.join(out_dir, 'X_medium_raw_derived.npy'))
y_check     = np.load(os.path.join(out_dir, 'y_medium.npy'))
assert X_check.shape     == X_arr.shape,     "X_medium shape mismatch"
assert X_raw_check.shape == X_arr_raw.shape, "X_medium_raw_derived shape mismatch"
assert y_check.shape     == y_arr.shape,     "y shape mismatch"
assert not np.isnan(X_check).any(),     "X_medium 含有 NaN！"
assert not np.isinf(X_check).any(),     "X_medium 含有 Inf！"
assert not np.isnan(X_raw_check).any(), "X_medium_raw_derived 含有 NaN！"

ok(f"X_medium.npy 已保存:             {X_check.shape}  (Winsorized, for EDA)")
ok(f"X_medium_raw_derived.npy 已保存: {X_raw_check.shape}  (raw, for CV per-fold)")
ok(f"y_medium.npy 已保存:             {y_check.shape}")
ok(f"genre_labels.csv 已保存:         {len(genres)} 个类别")

print(f"""
{'='*60}
全部检查通过！

特征工程难度评估：
  实际难度：★★★☆☆（中等，低于预期）

  原因：
  1. features.csv 已经是汇总统计，无法做时域 Δ-MFCC
     （时域 Δ 需要 22GB 的原始音频，不建议下载）
  2. 但从统计量出发可以构造 {len(new_features)} 个有物理意义的新特征
  3. 每一步都可以用断言自检，出错立刻定位

  哪里最容易出错（按概率排序）：
  1. 索引对齐：tracks 和 features 的 track_id 顺序不同
     → 已用 .loc[common_ids] 对齐，断言保护
  2. 形状广播：std/mean 矩阵维度不一致
     → 已逐个断言 .shape
  3. NaN 传播：除以0 / log(0)
     → 已在分母加 1e-8 的 epsilon
  4. 多级列索引访问：features['mfcc']['mean'] vs features[('mfcc','mean')]
     → 两种方式都支持，但混用时会出错，统一用链式访问

{'='*60}
""")
