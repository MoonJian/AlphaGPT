import torch
import math

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

    def _calc_purified_ic(self, factor_raw, future_returns, market_returns=None, method='pearson'):
        """
        计算剔除了市场 Beta 影响后的纯净 IC
        
        Args:
            factor_raw: [B, T] or [T], 你的原始因子
            future_returns: [B, T] or [T], 未来收益率 (Label)
            market_returns: [B, T] or [T], 市场基准收益率 (用于剔除 Beta)
            method: 'pearson' or 'rank' (Spearman)
        """
        # 1. 预处理：确保无 NaN
        mask = ~torch.isnan(factor_raw) & ~torch.isnan(future_returns)
        if market_returns is not None:
            mask &= ~torch.isnan(market_returns)
            
        f = factor_raw[mask].float()
        r = future_returns[mask].float()
        
        # 2. 正交化：剔除市场 Beta (如果有基准)
        # 也就是计算：Returns 对 Market 的回归残差
        # 或者是：Factor 对 Market 的回归残差 (通常对 Factor 做正交化更稳健)
        if market_returns is not None:
            m = market_returns[mask].float()
            
            # 线性回归: F = beta * M + alpha
            # beta = Cov(F, M) / Var(M)
            # 简单的一元回归写法:
            m_centered = m - m.mean()
            f_centered = f - f.mean()
            
            beta = (m_centered * f_centered).sum() / (m_centered ** 2).sum()
            f_residual = f - beta * m
            
            # 使用这一步处理后的因子替代原始因子
            f = f_residual

        # 3. 计算 IC
        if method == 'rank':
            # PyTorch 的 argsort 两次可以得到 rank
            f_rank = f.argsort().argsort().float()
            r_rank = r.argsort().argsort().float()
            
            # 归一化 rank 到 [0, 1] 或 Z-score 也可以，直接算 Pearson 即可等价于 Spearman
            f = f_rank
            r = r_rank

        # 计算 Pearson Correlation
        vx = f - torch.mean(f)
        vy = r - torch.mean(r)
        
        ic = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        
        return ic

    def _evaluate_single(self, signal, target_ret, norm_type='ZSCORE_ROLL', threshold=(0.85, -0.85)):
        # 2. 建立多空头寸（阈值可配置，适配 1h 提高交易次数）
        if norm_type == 'ZSCORE_ROLL':
            position_long = (signal > threshold[0]).float()
            position_short = (signal < threshold[1]).float()
        else:
            position_long = (signal > threshold[0]).float()
            position_short = (signal < threshold[1]).float()            
        
        # 关键：空头应该是负权，代表方向
        position = position_long - position_short 
        
        # 3. 计算交易成本
        total_slippage_one_way = self.base_fee + self.impact_slippage
        
        # 计算换手率与交易次数：使用差分
        # position 形状 [B, T]
        prev_pos = torch.zeros_like(position)
        prev_pos[:, 1:] = position[:, :-1]

        turnover = torch.abs(position - prev_pos)
        tx_cost = turnover * total_slippage_one_way
        # 交易次数：发生仓位变化的 bar 数（用于连续惩罚）
        trade_count = (position != prev_pos).float().sum(dim=1)
        
        # 4. 计算盈亏
        gross_pnl = position * target_ret
        net_pnl = gross_pnl - tx_cost
        
        # 5. 评分：收益 - 大回撤惩罚 - 活跃度/交易次数不足的连续惩罚（利于 RL 训练）
        cum_ret = net_pnl.sum(dim=1)

        # 夏普比率 (年化，假设 T 为小时线：24*365)
        mean_ret = net_pnl.mean(dim=1)
        std_ret = net_pnl.std(dim=1) + 1e-8
        sharpe = mean_ret / std_ret * math.sqrt(24 * 365)

        big_drawdowns = (net_pnl < self.big_drawdown_threshold).float().sum(dim=1)
        activity = torch.abs(position).sum(dim=1)  # 在仓 bar 数

        # 连续惩罚：交易次数/活跃度越少惩罚越大，无阶跃，便于梯度传播；短序列时放宽
        T = position.shape[1]
        min_activity = min(self.min_activity_bars, T // 10)
        min_trades = min(self.min_trades, T // 20)
        activity_penalty = self.activity_penalty_scale * torch.relu(
            min_activity - activity
        )
        trade_penalty = self.trade_penalty_scale * torch.relu(
            min_trades - trade_count
        )

        score = (
            sharpe
            - big_drawdowns * self.big_drawdown_penalty
            - activity_penalty
            - trade_penalty
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
            threshold_list = [(0.1, -0.1), (0.5, -0.5), (0.85, -0.85)]
        
        for threshold in threshold_list:
            score, ret_val, trade_count = self._evaluate_single(signal, target_ret, norm_type, threshold)
            if score.item() > best_score:
                best_score = score
                best_ret_val = ret_val
                best_trade_count = trade_count
                best_threshold = threshold
        return best_score, best_ret_val, correlation.item(), best_trade_count, best_threshold
