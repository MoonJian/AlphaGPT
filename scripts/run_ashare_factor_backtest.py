#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A股因子回测示例：结合 Alphalens 分析框架，对接 times.py 或本地因子/收益数据。

用法:
  # 需先安装依赖：numpy, pandas（可选：tushare 用于 times.py，alphalens 用于完整 tear sheet）
  # 使用合成数据快速跑通（无需 tushare）
  python scripts/run_ashare_factor_backtest.py

  # 若已配置 times.py 与 tushare，可在 times 中调用本模块接口
  from model_core.ashare_factor_backtest import evaluate_factor_ashare, run_single_asset_backtest
"""

import sys
import os
import numpy as np
import pandas as pd

# 允许从项目根目录运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def demo_with_synthetic():
    """无外部数据依赖：用合成因子与收益演示单标的回测流程。"""
    from model_core.ashare_factor_backtest import (
        run_single_asset_backtest,
        FactorBacktestResult,
        ic_single_asset,
        quantile_returns_single_asset,
    )
    np.random.seed(42)
    T = 504  # 约 2 年日频
    # 因子：带一定预测性的信号 + 噪声
    signal = np.cumsum(np.random.randn(T)) * 0.01
    noise = np.random.randn(T) * 0.5
    factor = signal + noise
    # 前向收益：与因子弱相关
    forward_returns = np.roll(factor, -1) * 0.1 + np.random.randn(T) * 0.01
    forward_returns[-1] = np.nan

    res = run_single_asset_backtest(
        factor,
        forward_returns,
        cost_rate=0.0005,
        quantiles=5,
        long_only=True,
    )
    print("===== 合成数据单标的回测 =====")
    print(f"  IC 均值:     {res.ic_mean:.4f}")
    print(f"  IC 标准差:   {res.ic_std:.4f}")
    print(f"  IC_IR:       {res.ic_ir:.4f}")
    print(f"  IC>0 比例:   {res.ic_positive_ratio:.2%}")
    print(f"  分位收益差:  {res.quantile_return_spread:.4f}")
    print(f"  换手均值:    {res.turnover_mean:.4f}")
    print(f"  净收益(累):  {res.net_cumret:.2%}")
    print(f"  净夏普:      {res.net_sharpe:.2f}")
    print(f"  观测数:      {res.n_obs}")
    return res


def demo_with_times_engine():
    """需要 times.py 与 tushare：从 DataEngine 取 target_oto_ret 做因子评估。"""
    try:
        from times import DataEngine
    except ImportError:
        print("未找到 times 模块或 tushare，跳过 DataEngine 示例。")
        return None
    from model_core.ashare_factor_backtest import from_times_engine, evaluate_factor_ashare

    engine = DataEngine()
    engine.load()
    # 用简单动量作为示例因子（与 times 特征一致时可换成公式因子）
    factor = engine.feat_data[0].cpu().numpy()  # RET
    res = evaluate_factor_ashare(factor, forward_returns=None, engine=engine, cost_rate=0.0005)
    print("===== times.DataEngine 因子回测 =====")
    print(f"  IC 均值:     {res.ic_mean:.4f}")
    print(f"  IC_IR:       {res.ic_ir:.4f}")
    print(f"  净夏普:      {res.net_sharpe:.2f}")
    print(f"  观测数:      {res.n_obs}")
    return res


def demo_alphalens_tear_sheet():
    """若已安装 alphalens，生成完整 tear sheet（可选）。"""
    from model_core.ashare_factor_backtest import create_tear_sheet_optional

    dates = pd.date_range("2020-01-01", periods=252, freq="B")
    factor_s = pd.Series(np.random.randn(252).cumsum() * 0.01, index=dates)
    prices = pd.DataFrame({"asset": 100 * (1 + np.random.randn(252).cumsum() * 0.01)}, index=dates)
    create_tear_sheet_optional(factor_s, prices, quantiles=5, periods=(1, 5, 10))
    print("若已安装 alphalens，上方会显示完整 tear sheet。")


if __name__ == "__main__":
    demo_with_synthetic()
    print()
    demo_with_times_engine()
    print()
    demo_alphalens_tear_sheet()
