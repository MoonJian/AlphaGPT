from curses import raw
import torch
import math
from typing import Tuple, Optional

class MemeBacktest:
    def __init__(self):
        self.trade_size = 1000.0
        self.min_liq = 500000.0
        self.base_fee = 0.0060

    def evaluate(self, factors, raw_data, target_ret):
        liquidity = raw_data['liquidity']
        signal = torch.sigmoid(factors)
        is_safe = (liquidity > self.min_liq).float()
        position = (signal > 0.85).float() * is_safe
        impact_slippage = self.trade_size / (liquidity + 1e-9)
        impact_slippage = torch.clamp(impact_slippage, 0.0, 0.05)
        total_slippage_one_way = self.base_fee + impact_slippage
        prev_pos = torch.roll(position, 1, dims=1)
        prev_pos[:, 0] = 0
        turnover = torch.abs(position - prev_pos)
        tx_cost = turnover * total_slippage_one_way
        gross_pnl = position * target_ret
        net_pnl = gross_pnl - tx_cost
        cum_ret = net_pnl.sum(dim=1)
        big_drawdowns = (net_pnl < -0.05).float().sum(dim=1)
        score = cum_ret - (big_drawdowns * 2.0)
        activity = position.sum(dim=1)
        score = torch.where(activity < 5, torch.tensor(-10.0, device=score.device), score)
        final_fitness = torch.median(score)
        return final_fitness, cum_ret.mean().item()
    

class MainCoinBacktest:
    """
    ETH/主流币 1h 频率回测。阈值与交易次数相关参数已针对 1h 数据做了默认优化。
    """
    def __init__(
        self,
        trade_size=1000.0,
        base_fee=0.0005,
        impact_slippage=0.0001,
        # Z-score 下多空阈值：2.0 很保守、交易少；1.5~1.75 更适配 1h 提高交易次数
        zscore_long=1.5,
        zscore_short=-1.5,
        # 非 Z-score 时多空阈值（假设因子约在 [-1, 1]）
        quantile_long=0.85,
        quantile_short=-0.85,
        # 最少“在仓” bar 数，用于连续惩罚的参考线
        min_activity_bars=80,
        # 最少换手次数（发生仓位变化的 bar 数），用于连续惩罚的参考线
        min_trades=50,
        # 活跃度不足时的连续惩罚系数（惩罚 = scale * relu(min_activity - activity)）
        activity_penalty_scale=0.02,
        # 交易次数不足时的连续惩罚系数（惩罚 = scale * relu(min_trades - trade_count)）
        trade_penalty_scale=0.02,
        # 单 bar 净亏损超过该比例算一次“大回撤”，计入惩罚
        big_drawdown_threshold=-0.02,
        big_drawdown_penalty=2.0,
        take_profit_pct=0.005,
        stop_loss_pct=-0.003,
        max_hold_bars=10,
    ):
        self.trade_size = trade_size
        self.base_fee = base_fee
        self.impact_slippage = impact_slippage
        self.zscore_long = zscore_long
        self.zscore_short = zscore_short
        self.quantile_long = quantile_long
        self.quantile_short = quantile_short
        self.min_activity_bars = min_activity_bars
        self.min_trades = min_trades
        self.activity_penalty_scale = activity_penalty_scale
        self.trade_penalty_scale = trade_penalty_scale
        self.big_drawdown_threshold = big_drawdown_threshold
        self.big_drawdown_penalty = big_drawdown_penalty
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.max_hold_bars = max_hold_bars

    def _run_rule_based_backtest(
        self,
        raw_signal: torch.Tensor,
        target_ret: torch.Tensor,
        total_slippage: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        按实际交易逻辑逐 bar 模拟：
        - 信号 +1 开多，-1 开空；信号 0 不改变持仓
        - 信号反转为相反方向时：立即平仓并反手
        - 否则按 TP/SL/持仓时间退出：盈利止盈、亏损止损、K bar 后强制平仓
        - 退出后读取下一 bar 信号再决定是否开仓
        返回: position [B,T], net_pnl [B,T], trade_count [B]
        """
        device = raw_signal.device
        if raw_signal.dim() == 1:
            raw_signal = raw_signal.unsqueeze(0)
        if target_ret.dim() == 1:
            target_ret = target_ret.unsqueeze(0)
        B, T = raw_signal.shape
        if target_ret.shape[0] != B:
            target_ret = target_ret.expand(B, -1)
        position = torch.zeros_like(raw_signal)
        net_pnl = torch.zeros_like(raw_signal)
        trade_count = torch.zeros(B, device=device)

        for b in range(B):
            pos = 0  # 当前仓位：+1 多，-1 空，0 空仓
            cum_ret_since_entry = 0.0
            bars_held = 0

            for t in range(T):
                sig_val = raw_signal[b, t]
                sig = 0 if torch.isnan(sig_val).item() else int(torch.clamp(sig_val, -1, 1).item())
                if sig > 0: sig = 1
                elif sig < 0: sig = -1
                ret_val = target_ret[b, t]
                ret = 0.0 if torch.isnan(ret_val).item() else float(ret_val.item())

                if pos != 0:
                    # 有持仓：先结算本 bar 收益
                    bar_pnl = pos * ret
                    cum_ret_since_entry += bar_pnl
                    bars_held += 1

                    # 1. 信号反转：立即平仓并反手
                    if (pos == 1 and sig == -1) or (pos == -1 and sig == 1):
                        # 平仓成本 + 反手开仓成本
                        turnover = 2
                        tx = turnover * total_slippage
                        net_pnl[b, t] = bar_pnl - tx
                        pos = sig
                        cum_ret_since_entry = 0.0
                        bars_held = 0
                        trade_count[b] += 2
                        position[b, t] = pos
                        continue

                    # 2. 止盈
                    if cum_ret_since_entry >= self.take_profit_pct:
                        tx = total_slippage
                        net_pnl[b, t] = bar_pnl - tx
                        pos = 0
                        trade_count[b] += 1
                        position[b, t] = 0
                        continue

                    # 3. 止损
                    if cum_ret_since_entry <= self.stop_loss_pct:
                        tx = total_slippage
                        net_pnl[b, t] = bar_pnl - tx
                        pos = 0
                        trade_count[b] += 1
                        position[b, t] = 0
                        continue

                    # 4. 持仓超时
                    if bars_held >= self.max_hold_bars:
                        tx = total_slippage
                        net_pnl[b, t] = bar_pnl - tx
                        pos = 0
                        trade_count[b] += 1
                        position[b, t] = 0
                        continue

                    # 未触发退出，继续持仓
                    net_pnl[b, t] = bar_pnl
                    position[b, t] = pos
                else:
                    # 空仓：根据信号开仓
                    if sig == 1:
                        pos = 1
                        cum_ret_since_entry = ret
                        bars_held = 1
                        tx = total_slippage
                        net_pnl[b, t] = ret - tx
                        trade_count[b] += 1
                    elif sig == -1:
                        pos = -1
                        cum_ret_since_entry = -ret
                        bars_held = 1
                        tx = total_slippage
                        net_pnl[b, t] = -ret - tx
                        trade_count[b] += 1
                    else:
                        net_pnl[b, t] = 0.0
                    position[b, t] = pos

        return position, net_pnl, trade_count
    def _evaluate_single(self, signal, target_ret, norm_type='ZSCORE_ROLL', threshold=(0.85, -0.85)):
        # 将因子转为离散信号：+1 多，-1 空，0 观望
        if norm_type == 'ZSCORE_ROLL':
            position_long = (signal > threshold[0]).float()
            position_short = (signal < threshold[1]).float()
        else:
            position_long = (signal > threshold[0]).float()
            position_short = (signal < threshold[1]).float()
        raw_signal = position_long - position_short  # +1, -1, 0

        total_slippage = self.base_fee + self.impact_slippage
        position, net_pnl, trade_count = self._run_rule_based_backtest(
            raw_signal, target_ret, total_slippage
        )

        cum_ret = net_pnl.sum(dim=1)
        mean_ret = net_pnl.mean(dim=1)
        std_ret = net_pnl.std(dim=1) + 1e-8
        # 日频用 252，小时频用 24*365
        sharpe = mean_ret / std_ret * math.sqrt(252)

        big_drawdowns = (net_pnl < self.big_drawdown_threshold).float().sum(dim=1)
        activity = torch.abs(position).sum(dim=1)
        T = position.shape[1]
        min_activity = min(self.min_activity_bars, T // 10)
        min_trades = min(self.min_trades, T // 20)
        activity_penalty = self.activity_penalty_scale * torch.relu(min_activity - activity)
        trade_penalty = self.trade_penalty_scale * torch.relu(min_trades - trade_count)

        score = (
            sharpe
            # - big_drawdowns * self.big_drawdown_penalty
            # - activity_penalty
            # - trade_penalty
        )
        return score.mean(), cum_ret.mean().item(), trade_count.mean().item()        

    def evaluate(self, factors, raw_data, target_ret, norm_type='ZSCORE_ROLL'):
        # 1. 把映射也当作一个OP
        signal = factors
        combined = torch.cat([factors, target_ret], dim=0)
        # 计算相关系数矩阵（因子与收益）；最后一行是 target，取首行与 target 的相关系数
        corr_matrix = torch.corrcoef(combined)
        correlation = torch.nan_to_num(corr_matrix[0, -1], nan=0.0)

        best_threshold = None
        best_score = -float('inf')
        if norm_type == 'ZSCORE_ROLL':
            threshold_list = [(1.0, -1.0), (1.5, -1.5), (2.0, -2.0)]
        else:
            # threshold_list = [(0.1, -0.1), (0.5, -0.5), (0.75, -0.75)]
            threshold_list = [(0.75, -0.75)]
        
        for threshold in threshold_list:
            score, ret_val, trade_count = self._evaluate_single(signal, target_ret, norm_type, threshold)
            sc = score.item() if hasattr(score, 'item') else float(score)
            if sc > best_score:
                best_score = sc
                best_ret_val = ret_val
                best_trade_count = trade_count
                best_threshold = threshold
        corr_val = correlation.item() if hasattr(correlation, 'item') else float(correlation)
        return best_score, best_ret_val, corr_val, best_trade_count, best_threshold
