"""
FMA 数据集审计脚本
检查 tracks.csv 和 features.csv，评估 K-Fold 分层策略。
"""

import os
import sys
import zipfile
import urllib.request
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
ZIP_URL  = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
ZIP_PATH = os.path.join(DATA_DIR, "fma_metadata.zip")
TRACKS_PATH  = os.path.join(DATA_DIR, "fma_metadata", "tracks.csv")
FEATURES_PATH = os.path.join(DATA_DIR, "fma_metadata", "features.csv")


# ─── 0. 下载（若文件不存在）──────────────────────────────────────────────────

def download_metadata():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(ZIP_PATH):
        print(f"[下载] fma_metadata.zip → {ZIP_PATH}")
        print("  文件约 342 MiB，请耐心等待……")

        def _progress(count, block_size, total_size):
            pct = count * block_size / total_size * 100
            sys.stdout.write(f"\r  进度: {min(pct, 100):.1f}%")
            sys.stdout.flush()

        urllib.request.urlretrieve(ZIP_URL, ZIP_PATH, _progress)
        print("\n  下载完成。")
    else:
        print("[下载] fma_metadata.zip 已存在，跳过下载。")


def extract_metadata():
    extract_dir = os.path.join(DATA_DIR, "fma_metadata")
    if not os.path.exists(TRACKS_PATH) or not os.path.exists(FEATURES_PATH):
        print(f"[解压] → {extract_dir}")
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            members = [m for m in zf.namelist()
                       if os.path.basename(m) in ("tracks.csv", "features.csv")]
            zf.extractall(DATA_DIR, members=members)
        print("  解压完成。")
    else:
        print("[解压] CSV 文件已存在，跳过解压。")


# ─── 1. tracks.csv 审计 ──────────────────────────────────────────────────────

def audit_tracks():
    print("\n" + "=" * 60)
    print("【审计 1】tracks.csv")
    print("=" * 60)

    tracks = pd.read_csv(TRACKS_PATH, index_col=0, header=[0, 1])

    # subset 列位于 ('set', 'subset')
    subset_col = ("set", "subset")
    medium = tracks[tracks[subset_col] == "medium"]

    row_count = len(medium)
    pass_audit = row_count >= 10_000

    print(f"  总行数 (全集):          {len(tracks):,}")
    print(f"  subset == 'medium' 行数: {row_count:,}")
    print(f"  审计红线 ≥ 10,000:       {'✓ 通过' if pass_audit else '✗ 未达标'}")

    # 流派分布（用于 K-Fold 分析）
    genre_col = ("track", "genre_top")
    genre_dist = medium[genre_col].value_counts(dropna=False)
    print(f"\n  流派分布 (genre_top)，共 {genre_dist.shape[0]} 个类别:")
    print(genre_dist.to_string(dtype=False))

    return medium


# ─── 2. features.csv 审计 ────────────────────────────────────────────────────

def audit_features(medium):
    print("\n" + "=" * 60)
    print("【审计 2】features.csv")
    print("=" * 60)

    features = pd.read_csv(FEATURES_PATH, index_col=0, header=[0, 1, 2])

    col_count = features.shape[1]
    pass_audit = col_count >= 100

    print(f"  特征矩阵形状: {features.shape[0]:,} 行 × {col_count:,} 列")
    print(f"  审计红线 ≥ 100 列:  {'✓ 通过' if pass_audit else '✗ 未达标'}")

    # 顶层特征组
    top_groups = features.columns.get_level_values(0).value_counts()
    print(f"\n  顶层特征组:")
    print(top_groups.to_string())

    # 只保留 medium 子集
    medium_ids = medium.index
    feat_medium = features.loc[features.index.isin(medium_ids)]
    print(f"\n  medium 子集特征行数: {len(feat_medium):,}")

    return feat_medium


# ─── 3. 缺失值 & K-Fold 策略分析 ─────────────────────────────────────────────

def audit_kfold(medium, feat_medium):
    print("\n" + "=" * 60)
    print("【审计 3】缺失值 & K-Fold 策略")
    print("=" * 60)

    # --- NaN 检查 ---
    nan_feat = feat_medium.isna().sum().sum()
    nan_label = medium[("track", "genre_top")].isna().sum()

    print(f"  features（medium）NaN 总数: {nan_feat:,}")
    print(f"  genre_top（medium） NaN 数: {nan_label:,}")

    if nan_feat > 0:
        print("  ⚠ 特征含缺失值，K-Fold 前需填补（均值/中位数）或删除对应行。")
    else:
        print("  ✓ 特征无缺失值，无需预处理。")

    if nan_label > 0:
        print(f"  ⚠ {nan_label} 条记录无 genre_top 标签，建议在分层前剔除。")
    else:
        print("  ✓ genre_top 标签完整。")

    # --- 分层必要性分析 ---
    genre_col = ("track", "genre_top")
    genre_dist = medium[genre_col].value_counts(dropna=True)
    total = genre_dist.sum()
    imbalance_ratio = genre_dist.max() / genre_dist.min()

    print(f"\n  流派最大/最小比值（不平衡度）: {imbalance_ratio:.1f}x")

    print("""
  ┌──────────────────────────────────────────────────────┐
  │            K-Fold 策略推荐                           │
  ├──────────────────────────────────────────────────────┤
  │ FMA-Medium 有 16 个流派，分布不均（不平衡度见上）。   │
  │                                                      │
  │ ✗ 简单随机切分（Random K-Fold）：                    │
  │   小类流派可能在某些折中严重缺失，                   │
  │   导致训练/验证集标签分布偏差大，                    │
  │   评估结果不稳定。                                   │
  │                                                      │
  │ ✓ 分层 K-Fold（Stratified K-Fold）：                 │
  │   每折保持与整体相同的流派比例，                     │
  │   是多分类不平衡场景的标准做法。                     │
  │   实现要点：                                         │
  │     1. 删除 genre_top 为 NaN 的行                    │
  │     2. 将 genre_top 编码为整数标签                   │
  │     3. 手写分层逻辑：按标签分组后                    │
  │        在每组内随机抽 1/K 放入当前折                 │
  │     4. 或直接用 sklearn.StratifiedKFold              │
  └──────────────────────────────────────────────────────┘
""")

    # 简易手写分层示意（不依赖 sklearn）
    print("  --- 手写分层 K-Fold 示意（K=5）---")
    df_labeled = medium[[genre_col]].dropna()
    df_labeled = df_labeled.copy()
    df_labeled["label"] = df_labeled[genre_col].astype("category").cat.codes

    K = 5
    folds = [[] for _ in range(K)]
    for label, group in df_labeled.groupby("label"):
        idx = group.index.tolist()
        np.random.shuffle(idx)
        for i, track_id in enumerate(idx):
            folds[i % K].append(track_id)

    for k, fold in enumerate(folds):
        print(f"    Fold {k+1}: {len(fold):,} 条")

    print("\n  各折流派分布均衡性验证（每折中各流派占比 vs 全集）:")
    global_ratio = df_labeled["label"].value_counts(normalize=True).sort_index()
    for k, fold in enumerate(folds[:2]):  # 只展示前 2 折作示例
        fold_labels = df_labeled.loc[fold, "label"]
        fold_ratio  = fold_labels.value_counts(normalize=True).sort_index()
        max_dev = (fold_ratio - global_ratio).abs().max()
        print(f"    Fold {k+1} 最大偏差: {max_dev:.4f}  "
              f"{'✓ 均衡' if max_dev < 0.02 else '⚠ 偏差较大'}")


# ─── 主流程 ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    download_metadata()
    extract_metadata()

    medium      = audit_tracks()
    feat_medium = audit_features(medium)
    audit_kfold(medium, feat_medium)

    print("\n[完成] 所有审计项已执行。")
