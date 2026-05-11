"""
FT-Transformer Colab T4 GPU 时间估算
方法：用理论 FLOPS 计算，比 numpy 单线程计时更可靠。

运行: python benchmark_ft.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# FT-Transformer 在表格数据上的计算量分析
#
# 核心结构：
#   - 每个特征 = 一个 token  →  序列长度 = n_features = 639
#   - Multi-head Self-Attention: O(n_features^2 × d_token) per sample
#   - n_blocks = 2（固定），n_heads = 8，d_token ∈ {64, 128, 256}
#
# 嵌套 CV 参数：
#   - 外层 K=10，内层 K=3，超参数组合 3 个  →  90 次训练
#   - 每次训练：内层训练集约 10,200 条，100 epoch，batch_size=256
# ─────────────────────────────────────────────────────────────────────────────

n_features   = 639      # 特征数 = attention 序列长度
n_classes    = 16
n_blocks     = 2        # transformer blocks 数量
EPOCHS       = 100
batch_size   = 256
n_inner_train = 10_200  # 外层 9/10，内层 8/9 ≈ 10,200 条
total_runs   = 10 * 3 * 3   # = 90

T4_TFLOPS = 8.1   # NVIDIA T4 FP32 峰值算力

print("=" * 64)
print("FT-Transformer 计算量 & 时间估算")
print("=" * 64)

for d_token in [64, 128, 256]:
    n_batches = n_inner_train // batch_size  # ≈ 39

    # ── 每 batch 的 FLOPS（只计算主要矩阵乘法）────────────────────────────
    # 1. Feature tokenization: (B, F) × (F, d) × 2 (fwd+bwd近似)
    flops_embed = 2 * batch_size * n_features * d_token

    # 2. Q, K, V 投影（每 block）: 3 × (B × F, d) × (d, d)
    flops_qkv   = n_blocks * 3 * 2 * batch_size * n_features * d_token * d_token

    # 3. Attention scores (B × h × F × F × head_dim) × 2：
    #    = B × n_features^2 × d_token per block
    flops_attn  = n_blocks * 2 * batch_size * n_features * n_features * d_token

    # 4. Attention output projection per block
    flops_proj  = n_blocks * 2 * batch_size * n_features * d_token * d_token

    # 5. FFN (4× expansion, 2 layers) per block
    flops_ffn   = n_blocks * 2 * 2 * batch_size * n_features * d_token * (4 * d_token)

    # 6. Classification head
    flops_cls   = 2 * batch_size * d_token * n_classes

    flops_fwd_batch = (flops_embed + flops_qkv + flops_attn +
                       flops_proj + flops_ffn + flops_cls)

    # 反向传播 ≈ 2× 前向（链式法则）
    flops_full_batch = flops_fwd_batch * 3   # fwd + bwd

    # ── 时间估算 ─────────────────────────────────────────────────────────
    # T4 实际利用率约 40-60%（小 batch tabular，不如 NLP 高效）
    for util_label, utilization in [("保守(40%)", 0.40), ("乐观(60%)", 0.60)]:
        effective_tflops = T4_TFLOPS * utilization
        t_batch = flops_full_batch / (effective_tflops * 1e12)   # seconds
        t_epoch = t_batch * n_batches
        t_run   = t_epoch * EPOCHS
        t_total = t_run * total_runs / 3600   # hours

    # 直接展示区间
    t_batch_lo = flops_full_batch / (T4_TFLOPS * 0.60 * 1e12)
    t_batch_hi = flops_full_batch / (T4_TFLOPS * 0.40 * 1e12)
    t_total_lo = t_batch_lo * n_batches * EPOCHS * total_runs / 3600
    t_total_hi = t_batch_hi * n_batches * EPOCHS * total_runs / 3600

    # 注意：上面假设没有数据传输开销；加上 Python/PyTorch overhead，乘以 1.5×
    t_total_lo *= 1.5
    t_total_hi *= 1.5

    print(f"\nd_token = {d_token}")
    print(f"  Attention 矩阵大小 per batch: "
          f"({batch_size}, {n_features}, {n_features}) × {d_token} "
          f"= {flops_attn/1e9:.1f} GFLOPS")
    print(f"  每 batch 总 FLOPS (含反向):  {flops_full_batch/1e9:.1f} GFLOPS")
    print(f"  T4 GPU 每次训练 (100 ep):    "
          f"{t_batch_lo*n_batches*EPOCHS/60:.1f} ~ "
          f"{t_batch_hi*n_batches*EPOCHS/60:.1f} 分钟")
    print(f"  90 次总计 (含 overhead ×1.5):"
          f" {t_total_lo:.1f} ~ {t_total_hi:.1f} 小时")

    if t_total_hi <= 6:
        verdict = "✓ Colab 单次会话内可完成"
    elif t_total_hi <= 12:
        verdict = "⚠ 偏长，建议保存断点续跑"
    else:
        verdict = "✗ 太长，需缩减 epoch 或用早停"
    print(f"  结论: {verdict}")

print()
print("=" * 64)
print()
print("=" * 64)
print("[Problem] seq_len=639 makes Attention expensive; d_token=128/256 infeasible")
print()
print("[Note] Each d_token value is trained 30 times in nested CV (10*3),")
print("       NOT 90 times.  Total 90 = 30*d32 + 30*d64 + 30*d128.")
print()
print("[Option A] Change hyperparams: d_token in {32, 64, 128} + early stopping (avg 40 ep)")
runs_per_hparam = 10 * 3   # 30 per hyperparameter value
EPOCHS_ES = 40
total_A_lo = total_A_hi = 0.0
for d_token in [32, 64, 128]:
    n_batches = n_inner_train // batch_size
    fa = 2 * n_blocks * 2 * batch_size * n_features**2 * d_token
    ff = 2 * n_blocks * 4 * batch_size * n_features * d_token * (4*d_token)
    fo = 2 * n_blocks * (3+1) * 2 * batch_size * n_features * d_token**2
    fb = (fa + ff + fo) * 3
    t_lo = fb * n_batches * EPOCHS_ES * runs_per_hparam / (T4_TFLOPS*0.60*1e12)*1.5/3600
    t_hi = fb * n_batches * EPOCHS_ES * runs_per_hparam / (T4_TFLOPS*0.40*1e12)*1.5/3600
    total_A_lo += t_lo; total_A_hi += t_hi
    ok = "OK" if t_hi < 4 else ("borderline" if t_hi < 8 else "TOO LONG")
    print(f"  d_token={d_token:3d}: {t_lo:.1f}~{t_hi:.1f} h  [{ok}]")
verdict = "2 Colab sessions OK" if total_A_hi < 12 else "~3 sessions needed"
print(f"  TOTAL: {total_A_lo:.1f}~{total_A_hi:.1f} h  -> {verdict}")

print()
print("[Option B] F-test top-100 features -> FT-Transformer (seq_len 639->100, 40x faster)")
n_feat_B = 100
for d_token in [64, 128, 256]:
    n_batches_B = n_inner_train // batch_size
    fa = 2 * n_blocks * 2 * batch_size * n_feat_B**2 * d_token
    ff = 2 * n_blocks * 4 * batch_size * n_feat_B * d_token * (4*d_token)
    fo = 2 * n_blocks * (3+1) * 2 * batch_size * n_feat_B * d_token**2
    fb = (fa + ff + fo) * 3
    t = fb * n_batches_B * EPOCHS * total_runs / (T4_TFLOPS*0.50*1e12)*1.5/3600
    ok = "OK" if t < 6 else "borderline"
    print(f"  d_token={d_token:3d}: ~{t:.1f} h  [{ok}]")
print("  Trade-off: must justify feature selection in report (F-test is statistically sound)")

print()
print("[Recommendation] Option A")
print("  d_token in {32, 64, 128}, early stopping patience=5 (inner) / 10 (outer)")
print("  d_token=64 as middle value should win most folds (assignment requirement)")
print("=" * 64)
