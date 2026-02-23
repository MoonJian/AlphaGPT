import torch
import torch.nn as nn
from .config import ModelConfig
import math


class RMSNormFactor(nn.Module):
    """RMSNorm for factor normalization"""
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class MemeIndicators:
    @staticmethod
    def liquidity_health(liquidity, fdv):
        ratio = liquidity / (fdv + 1e-6)
        return torch.clamp(ratio * 4.0, 0.0, 1.0)

    @staticmethod
    def oc_in_hl_ratio(close, open_, high, low):
        range_hl = high - low + 1e-9
        body = close - open_
        strength = body / range_hl
        return torch.tanh(strength * 3.0)

    @staticmethod
    def fomo_acceleration(volume, window=5):
        vol_prev = torch.roll(volume, 1, dims=1)
        vol_chg = (volume - vol_prev) / (vol_prev + 1.0)
        acc = vol_chg - torch.roll(vol_chg, 1, dims=1)
        acc[:, :2] = 0
        return torch.clamp(acc, -5.0, 5.0)

    @staticmethod
    def pump_deviation(close, window=20):
        pad = torch.zeros((close.shape[0], window-1), device=close.device)
        c_pad = torch.cat([pad, close], dim=1)
        ma = c_pad.unfold(1, window, 1).mean(dim=-1)
        dev = (close - ma) / (ma + 1e-9)
        return dev

    @staticmethod
    def volatility_clustering(close, window=10):
        """Detect volatility clustering patterns"""
        ret = torch.log(close / (torch.roll(close, 1, dims=1) + 1e-9))
        ret_sq = ret ** 2
        
        pad = torch.zeros((ret_sq.shape[0], window-1), device=close.device)
        ret_sq_pad = torch.cat([pad, ret_sq], dim=1)
        vol_ma = ret_sq_pad.unfold(1, window, 1).mean(dim=-1)
        
        return torch.sqrt(vol_ma + 1e-9)

    @staticmethod
    def momentum_reversal(close, window=5):
        """Capture momentum reversal signals"""
        ret = torch.log(close / (torch.roll(close, 1, dims=1) + 1e-9))
        
        pad = torch.zeros((ret.shape[0], window-1), device=close.device)
        ret_pad = torch.cat([pad, ret], dim=1)
        mom = ret_pad.unfold(1, window, 1).sum(dim=-1)
        
        # Detect reversals
        mom_prev = torch.roll(mom, 1, dims=1)
        prod = mom * mom_prev
        reversal = (prod < 0).float()

        return reversal

    @staticmethod
    def relative_strength(close, high, low, window=14):
        """RSI-like indicator for strength detection"""
        ret = close - torch.roll(close, 1, dims=1)
        
        gains = torch.relu(ret)
        losses = torch.relu(-ret)
        
        pad = torch.zeros((gains.shape[0], window-1), device=close.device)
        gains_pad = torch.cat([pad, gains], dim=1)
        losses_pad = torch.cat([pad, losses], dim=1)
        
        avg_gain = gains_pad.unfold(1, window, 1).mean(dim=-1)
        avg_loss = losses_pad.unfold(1, window, 1).mean(dim=-1)
        
        rs = (avg_gain + 1e-9) / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        
        return (rsi - 50) / 50  # Normalize


class AdvancedFactorEngineer:
    """Advanced feature engineering with multiple factor types"""
    def __init__(self):
        self.rms_norm = RMSNormFactor(1)
    
    def robust_norm(self, t):
        """Robust normalization using median absolute deviation"""
        median = torch.nanmedian(t, dim=1, keepdim=True)[0]
        mad = torch.nanmedian(torch.abs(t - median), dim=1, keepdim=True)[0] + 1e-6
        norm = (t - median) / mad
        return torch.clamp(norm, -5.0, 5.0)
    
    def compute_advanced_features(self, raw_dict):
        """Compute 12-dimensional feature space with advanced factors"""
        c = raw_dict['close']
        o = raw_dict['open']
        h = raw_dict['high']
        l = raw_dict['low']
        v = raw_dict['volume']
        liq = raw_dict['liquidity']
        fdv = raw_dict['fdv']
        
        # Basic factors
        ret = torch.log(c / (torch.roll(c, 1, dims=1) + 1e-9))
        liq_score = MemeIndicators.liquidity_health(liq, fdv)
        oc_ratio = MemeIndicators.oc_in_hl_ratio(c, o, h, l)
        fomo = MemeIndicators.fomo_acceleration(v)
        dev = MemeIndicators.pump_deviation(c)
        log_vol = torch.log1p(v)
        
        # Advanced factors
        vol_cluster = MemeIndicators.volatility_clustering(c)
        momentum_rev = MemeIndicators.momentum_reversal(c)
        rel_strength = MemeIndicators.relative_strength(c, h, l)
        
        # High-low range
        hl_range = (h - l) / (c + 1e-9)
        
        # Close position in range
        close_pos = (c - l) / (h - l + 1e-9)
        
        # Volume trend
        vol_prev = torch.roll(v, 1, dims=1)
        vol_trend = (v - vol_prev) / (vol_prev + 1.0)
        
        features = torch.stack([
            self.robust_norm(ret),
            liq_score,
            oc_ratio,
            self.robust_norm(fomo),
            self.robust_norm(dev),
            self.robust_norm(log_vol),
            self.robust_norm(vol_cluster),
            momentum_rev,
            self.robust_norm(rel_strength),
            self.robust_norm(hl_range),
            close_pos,
            self.robust_norm(vol_trend)
        ], dim=1)
        
        return features


class FeatureEngineer:
    INPUT_DIM = 6

    @staticmethod
    def compute_features(raw_dict):
        c = raw_dict['close']
        o = raw_dict['open']
        h = raw_dict['high']
        l = raw_dict['low']
        # v = raw_dict['volume']
        # liq = raw_dict['liquidity']
        # fdv = raw_dict['fdv']
        
        ret = torch.log(c / (torch.roll(c, 1, dims=1) + 1e-9))
        # liq_score = MemeIndicators.liquidity_health(liq, fdv)
        oc_ratio = MemeIndicators.oc_in_hl_ratio(c, o, h, l)
        # fomo = MemeIndicators.fomo_acceleration(v)
        dev = MemeIndicators.pump_deviation(c)
        # log_vol = torch.log1p(v)
        
        def robust_norm(t):
            median = torch.nanmedian(t, dim=1, keepdim=True)[0]
            mad = torch.nanmedian(torch.abs(t - median), dim=1, keepdim=True)[0] + 1e-6
            norm = (t - median) / mad
            return torch.clamp(norm, -5.0, 5.0)

        features = torch.stack([
            robust_norm(ret),
            # liq_score,
            oc_ratio,
            # robust_norm(fomo),
            robust_norm(dev),
            # robust_norm(log_vol)
        ], dim=1)
        
        return features


class KlinesFeatureEngineer:
    INPUT_DIM = 26

    @staticmethod
    def compute_features(raw_dict):
        c = raw_dict['close']
        o = raw_dict['open']
        h = raw_dict['high']
        l = raw_dict['low']        
        v = raw_dict['volume']
        buy_v = raw_dict['taker_buy_volume']

        const_1 = torch.full(c.shape, 1, dtype=torch.float32).to(c.device)
        const_e = torch.full(c.shape, math.e, dtype=torch.float32).to(c.device)
        const_10 = torch.full(c.shape, 10, dtype=torch.float32).to(c.device)
        const_100 = torch.full(c.shape, 100, dtype=torch.float32).to(c.device)
        
        ret = torch.log(c / (torch.roll(c, 1, dims=1) + 1e-9))
        oc_ratio = MemeIndicators.oc_in_hl_ratio(c, o, h, l)
        
        fomo_5 = MemeIndicators.fomo_acceleration(v, window=5)
        fomo_15 = MemeIndicators.fomo_acceleration(v, window=15)
        fomo_30 = MemeIndicators.fomo_acceleration(v, window=30)
        
        dev_20 = MemeIndicators.pump_deviation(c)
        dev_50 = MemeIndicators.pump_deviation(c, window=50)
        dev_100 = MemeIndicators.pump_deviation(c, window=100)
        
        log_vol = torch.log1p(v)
        log_buy_vol = torch.log1p(buy_v)

        # Advanced factors
        # 按照默认的周期计算
        vol_cluster_10 = MemeIndicators.volatility_clustering(c, window=10)
        vol_cluster_30 = MemeIndicators.volatility_clustering(c, window=30)
        vol_cluster_60 = MemeIndicators.volatility_clustering(c, window=60)

        momentum_rev_5 = MemeIndicators.momentum_reversal(c, window=5)
        momentum_rev_15 = MemeIndicators.momentum_reversal(c, window=15)
        momentum_rev_30 = MemeIndicators.momentum_reversal(c, window=30)

        rel_strength_10 = MemeIndicators.relative_strength(c, h, l, window=10)
        rel_strength_30 = MemeIndicators.relative_strength(c, h, l, window=30)
        rel_strength_60 = MemeIndicators.relative_strength(c, h, l, window=60)

        # High-low range
        hl_range = (h - l) / (c + 1e-9)        
        # Close position in range
        close_pos = (c - l) / (h - l + 1e-9)
        
        # Volume trend
        vol_prev = torch.roll(v, 1, dims=1)
        vol_trend = (v - vol_prev) / (vol_prev + 1.0)

        # 替换原来的 robust_norm 函数
        @torch.jit.script
        def robust_norm_rolling(x: torch.Tensor, window: int = ModelConfig.OP_ROLL_WINDOW) -> torch.Tensor:
            """
            PyTorch版滚动Robust Normalization (针对 1D 或 2D Tensor)
            x: [T] 或 [B, T]
            window: 滚动窗口大小
            """
            # 统一处理成 [B, T] 格式，方便批量处理
            if x.dim() == 1:
                x = x.unsqueeze(0)
            
            B, T = x.shape
            device = x.device
            
            # 1. 为了保持输出长度一致，在左侧填充 (Padding)
            # 使用第一个值填充，或者填0。这里建议用第一个有效值填充，减少边缘效应
            pad = x[:, 0:1].repeat(1, window - 1)
            x_padded = torch.cat([pad, x], dim=1) # [B, T + window - 1]
            
            # 2. 利用 unfold 展开滑动窗口 -> [B, T, window]
            windows = x_padded.unfold(1, window, 1)
            
            # 3. 计算滚动中位数 (Rolling Median)
            # torch.median 返回一个 namedtuple (values, indices)
            roll_median = windows.median(dim=-1).values # [B, T]
            
            # 4. 计算滚动 MAD
            # MAD = Median(|x - Median|)
            abs_diff = torch.abs(windows - roll_median.unsqueeze(-1))
            roll_mad = abs_diff.median(dim=-1).values + 1e-6 # [B, T]
            
            # 5. 计算结果并截断 (Clip)
            res = (x - roll_median) / roll_mad
            res = torch.clamp(res, -5.0, 5.0)
            
            return res # 如果输入是1D，返回1D

        features = torch.stack([
            robust_norm_rolling(ret), 
            oc_ratio,
            robust_norm_rolling(fomo_5),
            robust_norm_rolling(fomo_15),
            robust_norm_rolling(fomo_30),
            robust_norm_rolling(dev_20),
            robust_norm_rolling(dev_50),
            robust_norm_rolling(dev_100),
            robust_norm_rolling(log_vol),
            robust_norm_rolling(log_buy_vol),
            robust_norm_rolling(vol_cluster_10),
            robust_norm_rolling(vol_cluster_30),
            robust_norm_rolling(vol_cluster_60),
            robust_norm_rolling(momentum_rev_5),
            robust_norm_rolling(momentum_rev_15),
            robust_norm_rolling(momentum_rev_30),
            robust_norm_rolling(rel_strength_10),
            robust_norm_rolling(rel_strength_30),
            robust_norm_rolling(rel_strength_60),
            robust_norm_rolling(hl_range),
            robust_norm_rolling(close_pos),
            robust_norm_rolling(vol_trend),          
            const_1,
            const_e,
            const_10,
            const_100
        ], dim=1)        

        return features