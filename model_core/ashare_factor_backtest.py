# -*- coding: utf-8 -*-
"""
A股因子回测模块（Alphalens 风格，侧重运行效率）

结合 Alphalens 的因子分析框架，针对 A 股市场做向量化实现：
- IC / IC_IR / 分位数收益 / 换手率
- 支持多标的截面因子
- 可选与 alphalens 对接生成完整 tear sheet

参考：
- https://github.com/quantopian/alphalens
- https://github.com/JiaxuZhang-03/factor_backtest_system
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    import torch


# A 股常用年化交易日数
TRADING_DAYS_PER_YEAR = 252


@dataclass
class FactorBacktestResult:
    """单次因子回测结果（便于序列化与下游使用）"""
    ic_mean: float = np.nan
    ic_std: float = np.nan
    ic_ir: float = np.nan
    ic_positive_ratio: float = np.nan
    quantile_return_spread: float = np.nan   # 多空收益差 (Q5 - Q1)
    turnover_mean: float = np.nan
    turnover_std: float = np.nan
    long_only_cumret: float = np.nan
    long_only_ann_vol: float = np.nan
    long_only_sharpe: float = np.nan
    net_cumret: float = np.nan
    net_sharpe: float = np.nan
    n_obs: int = 0
    n_dates: int = 0
    quantile_returns: Optional[Dict[int, float]] = None
    ic_series: Optional[np.ndarray] = None
    # 最大分组（纯多头）常用指标
    long_only_avg_ret: float = np.nan        # 日均收益
    long_only_ann_ret: float = np.nan        # 年化收益
    long_only_max_dd: float = np.nan         # 最大回撤
    long_only_win_rate: float = np.nan       # 胜率（日收益>0 比例）
    long_only_best_quantile: Optional[int] = None  # 收益最高的分位编号


# --------------- 多标的（截面）向量化实现 ---------------

def get_clean_factor_and_forward_returns_ashare(
    factor: Union[pd.DataFrame, pd.Series],
    prices: Union[pd.DataFrame, pd.Series],
    *,
    periods: Tuple[int, ...] = (1, 5, 10),
) -> pd.DataFrame:
    """
    将因子与价格整理为 Alphalens 风格的 factor_data（MultiIndex: date, asset）。
    factor: index=date 的 Series，或 index=date/columns=asset 的 DataFrame;
    prices: index=date 的 Series 或 columns=asset 的 DataFrame.
    返回 MultiIndex DataFrame，含 factor 与 1/5/10 日前向收益。
    """
    if isinstance(prices, pd.Series):
        prices = prices.to_frame("close")
    prices = prices.ffill().bfill()
    if isinstance(factor, pd.Series):
        common_idx = factor.reindex(prices.index).dropna().index.intersection(prices.index).sort_values()
    else:
        common_idx = factor.index.intersection(prices.index).sort_values()
    if len(common_idx) < 2:
        raise ValueError("factor 与 prices 的日期交集不足")
    prices = prices.reindex(common_idx).ffill().bfill()
    if isinstance(factor, pd.Series):
        factor = factor.reindex(common_idx).ffill().bfill()
        assets = prices.columns.tolist() if hasattr(prices, "columns") else ["asset"]
        idx = pd.MultiIndex.from_product([common_idx, assets], names=["date", "asset"])
        factor_data = pd.DataFrame({"factor": np.tile(factor.values, len(assets))}, index=idx)
    else:
        factor = factor.reindex(common_idx).ffill().bfill()
        factor_stacked = factor.stack()
        factor_data = factor_stacked.to_frame("factor")
        factor_data.index.names = ["date", "asset"]
    prices = prices.reindex(common_idx).ffill().bfill()
    for p in periods:
        ret = prices.pct_change(p).shift(-p)
        if isinstance(ret, pd.DataFrame):
            ret = ret.stack()
        factor_data[f"{p}D"] = ret.reindex(factor_data.index).values
    factor_data = factor_data.dropna(subset=["factor"], how="all")
    return factor_data


def ic_cross_section(
    factor_data: pd.DataFrame,
    period: str = "1D",
) -> Tuple[float, float, float, float, pd.Series]:
    """
    截面 IC：按 date 分组，计算当日因子与 period 收益的相关系数，再汇总。
    返回 (ic_mean, ic_std, ic_ir, ic_positive_ratio, ic_series)。
    """
    if period not in factor_data.columns:
        return np.nan, np.nan, np.nan, np.nan, pd.Series(dtype=float)
    if "factor" not in factor_data.columns:
        return np.nan, np.nan, np.nan, np.nan, pd.Series(dtype=float)

    by_date = factor_data.groupby(level="date")
    ic_s = by_date.apply(
        lambda g: g["factor"].corr(g[period]) if g["factor"].notna().sum() > 5 else np.nan
    )
    ic_s = ic_s.dropna()
    if len(ic_s) < 2:
        return np.nan, np.nan, np.nan, np.nan, ic_s
    ic_mean = float(ic_s.mean())
    ic_std = float(ic_s.std())
    ic_ir = ic_mean / (ic_std + 1e-12)
    ic_pos = float((ic_s > 0).mean())

    return ic_mean, ic_std, ic_ir, ic_pos, ic_s


def _long_only_quantile_metrics(
    factor_data: pd.DataFrame,
    period: str,
    q_ret: Dict[int, float],
    quantiles: int,
    risk_free: float = 0.0,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> Tuple[
    float, float, float, float, float, float, float, Optional[int]
]:
    """
    按分位等权组合计算「最大分组」纯多头指标。
    最大分组 = 平均收益最高的那个分位（用于衡量 A 股纯多头表现）。
    返回: (日均收益, 年化收益, 累计收益, 年化波动, 夏普, 最大回撤, 胜率, 最佳分位编号)。
    """
    if not q_ret or period not in factor_data.columns:
        return (np.nan,) * 7 + (None,)
    best_q = max(q_ret, key=q_ret.get)
    # 每日：该分位内标的的 period 收益等权平均 → 组合日收益序列
    daily_ret = (
        factor_data.loc[factor_data["quantile"] == best_q]
        .groupby(level="date")[period]
        .mean()
    )
    daily_ret = daily_ret.dropna()
    if len(daily_ret) < 2:
        return (np.nan,) * 7 + (best_q,)
    dr = daily_ret.values.astype(np.float64)
    avg_ret = float(np.mean(dr))
    ann_ret = float((1 + avg_ret) ** trading_days - 1)
    cumret = float(np.prod(1 + dr) - 1)
    ann_vol = float(np.std(dr) * np.sqrt(trading_days))
    sharpe = (ann_ret - risk_free) / (ann_vol + 1e-12)
    # 最大回撤
    cum = np.cumprod(1 + dr)
    run_max = np.maximum.accumulate(cum)
    dd = (cum - run_max) / (run_max + 1e-12)
    max_dd = float(np.min(dd))
    win_rate = float(np.mean(dr > 0))
    return (
        avg_ret,
        ann_ret,
        cumret,
        ann_vol,
        float(sharpe),
        max_dd,
        win_rate,
        best_q,
    )


def run_panel_backtest(
    factor_data: pd.DataFrame, # MultiIndex DataFrame,含 factor 与 1D/5D/10D 收益列
    *,
    period: str = "1D",
    quantiles: int = 5,
    cost_rate: float = 0.0005,
) -> FactorBacktestResult:
    """
    多标的截面回测：基于 factor_data（含 factor 与 1D/5D/10D 收益列）计算 IC、分位收益、换手等。
    factor_data 须含列 "factor" 与 period（如 "1D"）。
    """
    if "factor" not in factor_data.columns or period not in factor_data.columns:
        return FactorBacktestResult(n_obs=0, n_dates=0)
    
    ic_mean, ic_std, ic_ir, ic_pos, ic_series = ic_cross_section(factor_data, period)
    
    # 分位数收益
    factor_data = factor_data.copy()
    factor_data["quantile"] = factor_data.groupby(level="date")["factor"].transform(
        lambda x: pd.qcut(x, quantiles, labels=False, duplicates="drop")
    )
    q_ret = factor_data.groupby(["date", "quantile"])[period].mean().groupby("quantile").mean().to_dict()
    # 多空收益差
    spread = (q_ret.get(max(q_ret, default=0), 0) - q_ret.get(min(q_ret, default=0), 0)) if q_ret else np.nan

    # 最大分组（纯多头）收益与风险指标：选收益最高的分位做等权组合
    (
        long_only_avg_ret,
        long_only_ann_ret,
        long_only_cumret_lo,
        long_only_ann_vol_lo,
        long_only_sharpe_lo,
        long_only_max_dd,
        long_only_win_rate,
        best_quantile,
    ) = _long_only_quantile_metrics(factor_data, period, q_ret, quantiles)

    # 换手：按日计算分位变动
    turnover_s = factor_data.groupby(level="date")["quantile"].apply(
        lambda x: x.diff().abs().sum() / (2 * (quantiles - 1)) if len(x) > 1 else 0
    )
    turn_mean = float(turnover_s.mean())
    turn_std = float(turnover_s.std())
    n_obs = factor_data["factor"].notna().sum()
    n_dates = factor_data.index.get_level_values("date").nunique()

    return FactorBacktestResult(
        ic_mean=ic_mean,
        ic_std=ic_std,
        ic_ir=ic_ir,
        ic_positive_ratio=ic_pos,
        quantile_return_spread=float(spread) if np.isfinite(spread) else np.nan,
        turnover_mean=turn_mean,
        turnover_std=turn_std,
        net_cumret=np.nan,
        net_sharpe=np.nan,
        n_obs=int(n_obs),
        n_dates=int(n_dates),
        quantile_returns=q_ret,
        ic_series=ic_series.values if hasattr(ic_series, "values") else ic_series,
        long_only_avg_ret=long_only_avg_ret,
        long_only_ann_ret=long_only_ann_ret,
        long_only_cumret=long_only_cumret_lo,
        long_only_ann_vol=long_only_ann_vol_lo,
        long_only_sharpe=long_only_sharpe_lo,
        long_only_max_dd=long_only_max_dd,
        long_only_win_rate=long_only_win_rate,
        long_only_best_quantile=best_quantile,
    )


# --------------- 可选：Alphalens 对接 ---------------


def create_tear_sheet_optional(
    factor: pd.Series,
    prices: pd.DataFrame,
    quantiles: int = 5,
    periods: Tuple[int, ...] = (1, 5, 10),
) -> None:
    """
    若已安装 alphalens，则生成完整 tear sheet；否则仅打印简要统计。
    factor/prices 为 Alphalens 标准格式（MultiIndex 或 date index）。
    """
    try:
        import alphalens
        from alphalens.utils import get_clean_factor_and_forward_returns
        from alphalens.tears import create_full_tear_sheet
    except ImportError:
        print("未安装 alphalens，跳过完整 tear sheet。可运行: pip install alphalens")
        return
    factor_data = get_clean_factor_and_forward_returns(
        factor, prices, quantiles=quantiles, periods=list(periods)
    )
    create_full_tear_sheet(factor_data)
