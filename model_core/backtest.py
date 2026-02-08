import torch

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
    def __init__(self):
        self.trade_size = 1000.0
        self.base_fee = 0.0005

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

    def evaluate(self, factors, raw_data, target_ret, norm_type='ZSCORE_ROLL'):
        # 1. 把映射也当作一个OP
        signal = factors

        # 2. 建立多空头寸
        if norm_type == 'ZSCORE_ROLL':
            position_long = (signal > 2.0).float() 
            position_short = (signal < -2.0).float()
        else:
            position_long = (signal > 0.85).float() 
            position_short = (signal < -0.85).float()            
        
        # 关键：空头应该是负权，代表方向
        position = position_long - position_short 
        
        # 3. 计算交易成本
        impact_slippage = 0.0001
        total_slippage_one_way = self.base_fee + impact_slippage
        
        # 计算换手率：使用差分更安全
        # 假设 position 形状是 [B, T]
        prev_pos = torch.zeros_like(position)
        prev_pos[:, 1:] = position[:, :-1] 
        
        turnover = torch.abs(position - prev_pos)
        tx_cost = turnover * total_slippage_one_way
        
        # 4. 计算盈亏
        gross_pnl = position * target_ret
        net_pnl = gross_pnl - tx_cost

        combined = torch.cat([factors, target_ret], dim=0)
        # 计算相关系数矩阵
        corr_matrix = torch.corrcoef(combined)
        correlation = corr_matrix[0, 1].item()

        # 计算纯净 IC
        # correlation = self._calc_purified_ic(factors, target_ret, market_returns=target_ret).item()
        
        # 5. 评分系统优化
        cum_ret = net_pnl.sum(dim=1)
        # 这里的 -0.05 很大，如果是 15min 频率，建议关注更小的回撤
        big_drawdowns = (net_pnl < -0.01).float().sum(dim=1) 
        
        # 加上活跃度惩罚：防止模型通过“不交易”来保命
        activity = torch.abs(position).sum(dim=1)
        
        score = cum_ret - (big_drawdowns * 2.0)
        # 惩罚不活跃的公式
        score = torch.where(activity < 100, torch.tensor(-10.0, device=score.device), score)
        
        final_fitness = torch.median(score)
        return final_fitness, cum_ret.mean().item(), correlation
