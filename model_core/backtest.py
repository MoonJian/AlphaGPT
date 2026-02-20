"""
中国期货市场回测模块
- 输出兼容聚宽格式：trade_date, symbol, factor_value, forward_return
- forward_return 含主力切换换仓成本
"""
import torch
import math
import pandas as pd
from typing import Tuple, Optional
from .config import ModelConfig


def to_joinquant_format(
    trade_dates: pd.DatetimeIndex,
    symbol: str,
    factor_values: torch.Tensor,
    forward_returns: torch.Tensor,
) -> pd.DataFrame:
    """
    输出聚宽兼容格式
    列: trade_date (Date), symbol (str), factor_value (Float64), forward_return (Float64)
    """
    if factor_values.dim() > 1:
        factor_values = factor_values.squeeze()
    if forward_returns.dim() > 1:
        forward_returns = forward_returns.squeeze()
    n = min(len(trade_dates), factor_values.numel(), forward_returns.numel())
    f = factor_values.cpu().numpy()[:n]
    r = forward_returns.cpu().numpy()[:n]
    return pd.DataFrame({
        'trade_date': trade_dates[:n],
        'symbol': symbol,
        'factor_value': f.astype('float64'),
        'forward_return': r.astype('float64'),
    })


class FuturesBacktest:
    """
    期货因子回测
    - 涨跌停日过滤（因子置 null 不参与信号）
    - 换仓成本计入 forward_return
    - 输出聚宽格式
    - 持仓规则：盈利目标/止损/最大持仓 K bar 平仓；信号反向后立即平仓并反向开仓
    """
    def __init__(
        self,
        base_fee: float = None,
        impact_slippage: float = None,
        zscore_long: float = 1.5,
        zscore_short: float = -1.5,
        min_activity_bars: int = 20,
        min_trades: int = 10,
        big_drawdown_threshold: float = -0.02,
        big_drawdown_penalty: float = 2.0,
        profit_target: float = None,
        stop_loss: float = None,
        max_hold_bars: int = None,
    ):
        self.base_fee = base_fee or ModelConfig.BASE_FEE
        self.impact_slippage = impact_slippage or ModelConfig.IMPACT_SLIPPAGE
        self.zscore_long = zscore_long
        self.zscore_short = zscore_short
        self.min_activity_bars = min_activity_bars
        self.min_trades = min_trades
        self.big_drawdown_threshold = big_drawdown_threshold
        self.big_drawdown_penalty = big_drawdown_penalty
        self.profit_target = profit_target if profit_target is not None else ModelConfig.PROFIT_TARGET
        self.stop_loss = stop_loss if stop_loss is not None else ModelConfig.STOP_LOSS
        self.max_hold_bars = max_hold_bars if max_hold_bars is not None else ModelConfig.MAX_HOLD_BARS

    def _evaluate_single(
        self,
        signal: torch.Tensor,
        target_ret: torch.Tensor,
        norm_type: str = 'ZSCORE_ROLL',
        threshold: Tuple[float, float] = (1.5, -1.5),
    ) -> Tuple[torch.Tensor, float, float]:
        """
        基于盈亏比与持仓时间的回测逻辑：
        - 信号 +1 做多，-1 做空
        - 信号 0：不立即平仓，按盈利目标/止损/最大持仓 K bar 决定是否平仓
        - 信号同向：不加仓，继续持有
        - 信号反向：立即平仓并反向开仓，然后按上述规则持仓
        """
        # 离散化信号：+1 多，-1 空，0 中性（不触发新开仓，但持仓时按规则处理）
        if signal.dim() == 1:
            signal = signal.unsqueeze(0)
        sig_long = (signal > threshold[0]).float()
        sig_short = (signal < threshold[1]).float()
        raw_signal = sig_long - sig_short  # +1, -1, 0

        B, T = raw_signal.shape
        tx_cost_one = self.base_fee + self.impact_slippage
        net_pnl = torch.zeros_like(raw_signal)
        trade_count = torch.zeros(B, device=raw_signal.device)

        def get_ret(b, t):
            if target_ret.dim() == 1:
                return target_ret[t].item()
            return target_ret[b, t].item()

        for b in range(B):
            pos = 0
            cum_ret = 0.0
            bars_held = 0
            entry_bar = -1

            for t in range(T):
                sig_t = int(raw_signal[b, t].item())

                if pos != 0:
                    # 持仓中：计入上一 bar 到本 bar 的收益（target_ret[t-1] = 从 t-1 到 t 的收益）
                    if t > 0:
                        prev_ret = get_ret(b, t - 1)
                        cum_ret += prev_ret * pos
                    bars_held += 1

                    # 检查退出条件
                    exit_now = False
                    if cum_ret >= self.profit_target:
                        exit_now = True
                    elif cum_ret <= self.stop_loss:
                        exit_now = True
                    elif bars_held >= self.max_hold_bars:
                        exit_now = True
                    elif sig_t == -pos:
                        exit_now = True  # 信号反向：平仓并反向开仓

                    if exit_now:
                        # 平仓：净 PnL = 累计收益 - 平仓手续费
                        net_pnl[b, t] = cum_ret - tx_cost_one
                        trade_count[b] += 1
                        pos = 0
                        cum_ret = 0.0
                        bars_held = 0

                        # 若信号反向，立即反向开仓（再扣一次开仓手续费）
                        if sig_t == -1:
                            pos = -1
                            entry_bar = t
                            cum_ret = 0.0
                            bars_held = 0
                            trade_count[b] += 1
                            net_pnl[b, t] -= tx_cost_one
                        elif sig_t == 1:
                            pos = 1
                            entry_bar = t
                            cum_ret = 0.0
                            bars_held = 0
                            trade_count[b] += 1
                            net_pnl[b, t] -= tx_cost_one
                    else:
                        net_pnl[b, t] = 0.0
                else:
                    # 空仓：检查新开仓
                    if sig_t == 1:
                        pos = 1
                        entry_bar = t
                        cum_ret = 0.0
                        bars_held = 0
                        trade_count[b] += 1
                        net_pnl[b, t] = -tx_cost_one
                    elif sig_t == -1:
                        pos = -1
                        entry_bar = t
                        cum_ret = 0.0
                        bars_held = 0
                        trade_count[b] += 1
                        net_pnl[b, t] = -tx_cost_one
                    else:
                        net_pnl[b, t] = 0.0

        cum_ret_total = net_pnl.sum(dim=-1)
        mean_ret = net_pnl.mean(dim=-1)
        std_ret = net_pnl.std(dim=-1) + 1e-8
        sharpe = mean_ret / std_ret * math.sqrt(252)
        activity = (net_pnl != 0).float().sum(dim=-1)
        big_drawdowns = (net_pnl < self.big_drawdown_threshold).float().sum(dim=-1)
        activity_penalty = 0.02 * torch.relu(self.min_activity_bars - activity)
        trade_penalty = 0.02 * torch.relu(self.min_trades - trade_count)
        score = sharpe - big_drawdowns * self.big_drawdown_penalty - activity_penalty - trade_penalty
        return score.mean(), cum_ret_total.mean().item(), trade_count.mean().item()

    def evaluate(
        self,
        factors: torch.Tensor,
        raw_data: dict,
        target_ret: torch.Tensor,
        norm_type: str = 'ZSCORE_ROLL',
    ) -> Tuple[float, float, float, float, Optional[Tuple[float, float]]]:
        """
        评估因子
        返回: (score, cum_ret, correlation, trade_count, best_threshold)
        """
        signal = factors
        if signal.dim() == 3:
            signal = signal.squeeze(0)
        # 计算因子与收益的相关系数
        f_flat = (factors.flatten(0, 1) if factors.dim() > 2 else factors).flatten()
        r_flat = target_ret.expand_as(factors).flatten()
        mask = ~(torch.isnan(f_flat) | torch.isnan(r_flat))
        try:
            if mask.sum() > 10:
                cf = f_flat[mask]
                cr = r_flat[mask]
                correlation = torch.corrcoef(torch.stack([cf, cr]))[0, 1].item()
            else:
                correlation = 0.0
        except Exception:
            correlation = 0.0
        correlation = 0.0 if (correlation != correlation) else correlation

        best_score = -float('inf')
        best_ret = 0.0
        best_trades = 0.0
        best_threshold = None
        for th in [(1.0, -1.0), (1.5, -1.5), (2.0, -2.0)]:
            score, ret_val, trade_count = self._evaluate_single(signal, target_ret, norm_type, th)
            if score.item() > best_score:
                best_score = score.item()
                best_ret = ret_val
                best_trades = trade_count
                best_threshold = th
        return best_score, best_ret, correlation, best_trades, best_threshold


# 兼容旧接口
MemeBacktest = FuturesBacktest
MainCoinBacktest = FuturesBacktest
