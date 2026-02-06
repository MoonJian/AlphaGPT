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
        
        # 5. 评分系统优化
        cum_ret = net_pnl.sum(dim=1)
        # 这里的 -0.05 很大，如果是 15min 频率，建议关注更小的回撤
        big_drawdowns = (net_pnl < -0.01).float().sum(dim=1) 
        
        # 加上活跃度惩罚：防止模型通过“不交易”来保命
        activity = torch.abs(position).sum(dim=1)
        
        score = cum_ret - (big_drawdowns * 2.0)
        # 惩罚不活跃的公式
        score = torch.where(activity < 5, torch.tensor(-10.0, device=score.device), score)
        
        final_fitness = torch.median(score)
        return final_fitness, cum_ret.mean().item(), correlation
