"""
可视化 FOMO 因子特征
用法: python -m scripts.visualize_fomo [--data-path PATH]
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt
import numpy as np

from model_core.data_loader import CryptoDataLoader
from model_core.factors import MemeIndicators, KlinesFeatureEngineer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default='./data/ETHUSDT-futures_15m_2020-01-01-2026-02-20.parquet',
                        help='K线数据路径')
    parser.add_argument('--output', default='fomo_visualization.png', help='输出图片路径')
    parser.add_argument('--sample', type=int, default=None, help='仅绘制最近 N 个 bar（默认全部）')
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"数据文件不存在: {args.data_path}")
        print("请指定有效的 --data-path 或准备数据文件")
        return 1

    loader = CryptoDataLoader(args.data_path)
    loader.load_klines_data()

    v = loader.raw_data_cache['volume']
    # 原始 FOMO 加速度
    fomo_raw = MemeIndicators.fomo_acceleration(v)

    # 完整特征中的 FOMO（robust_norm 后），索引 3
    feat = loader.feat_tensor
    fomo_norm = feat[:, 3, :] if feat.dim() == 3 else feat[3]

    # 转为 numpy，取第一个品种
    fomo_raw_np = fomo_raw[0].cpu().numpy() if fomo_raw.dim() > 1 else fomo_raw.cpu().numpy()
    fomo_norm_np = fomo_norm[0].cpu().numpy() if fomo_norm.dim() > 1 else fomo_norm.cpu().numpy()

    if args.sample:
        fomo_raw_np = fomo_raw_np[-args.sample:]
        fomo_norm_np = fomo_norm_np[-args.sample:]

    # 时间轴
    T = len(fomo_raw_np)
    try:
        import pandas as pd
        if 'open_time' in loader.raw_data.columns:
            dates = loader.raw_data['open_time'].drop_duplicates().sort_values().reset_index(drop=True)
        else:
            dates = pd.Series(loader.raw_data.index).drop_duplicates()
        if args.sample:
            dates = dates.iloc[-args.sample:]
        x = pd.to_datetime(dates).values
        if len(x) != T:
            x = np.arange(T)
    except Exception:
        x = np.arange(T)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # 1. 原始 FOMO 加速度
    axes[0].plot(x, fomo_raw_np, color='#2ecc71', linewidth=0.8, alpha=0.9)
    axes[0].axhline(0, color='gray', linestyle='--', linewidth=0.5)
    axes[0].set_ylabel('FOMO (raw)')
    axes[0].set_title('FOMO 加速度因子 - 成交量变化二阶导')
    axes[0].grid(True, alpha=0.3)

    # 2. Robust 标准化后的 FOMO
    axes[1].plot(x, fomo_norm_np, color='#3498db', linewidth=0.8, alpha=0.9)
    axes[1].axhline(0, color='gray', linestyle='--', linewidth=0.5)
    axes[1].set_ylabel('FOMO (normalized)')
    axes[1].set_title('FOMO - Robust 标准化后（用于模型输入）')
    axes[1].grid(True, alpha=0.3)

    # 3. 成交量（参考）
    vol_np = v[0].cpu().numpy() if v.dim() > 1 else v.cpu().numpy()
    if args.sample:
        vol_np = vol_np[-args.sample:]
    axes[2].fill_between(x, 0, vol_np, alpha=0.4, color='#9b59b6')
    axes[2].plot(x, vol_np, color='#8e44ad', linewidth=0.6)
    axes[2].set_ylabel('Volume')
    axes[2].set_xlabel('Bar Index')
    axes[2].set_title('成交量（原始）')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"✅ 已保存: {args.output}")
    plt.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
