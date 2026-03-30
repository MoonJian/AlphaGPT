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


class Indicators:
    @staticmethod
    def _prev_on_time(x: torch.Tensor, t: int = 1) -> torch.Tensor:
        """Previous timestep along dim=1; first column is 0 (no circular wrap)."""
        prev = torch.zeros_like(x)
        prev[:, t:] = x[:, :-t]
        return prev

    @staticmethod
    def oc_vs_hl(close, open_, high, low):
        range_hl = high - low + 1e-9
        body = close - open_
        strength = body / range_hl
        return torch.tanh(strength * 3.0)

    @staticmethod
    def fomo_acceleration(volume):
        vol_prev = Indicators._prev_on_time(volume)
        vol_chg = (volume - vol_prev) / (vol_prev + 1.0)
        acc = vol_chg - Indicators._prev_on_time(vol_chg)
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
        prev = Indicators._prev_on_time(close)
        ret = torch.log1p(close / (prev + 1e-9))
        ret_sq = ret ** 2
        
        pad = torch.zeros((ret_sq.shape[0], window-1), device=close.device)
        ret_sq_pad = torch.cat([pad, ret_sq], dim=1)
        vol_ma = ret_sq_pad.unfold(1, window, 1).mean(dim=-1)
        
        return torch.sqrt(vol_ma + 1e-9)

    @staticmethod
    def momentum_reversal(close, window=5):
        """Capture momentum reversal signals"""
        prev = Indicators._prev_on_time(close)
        ret = torch.log1p(close / (prev + 1e-9))
        
        pad = torch.zeros((ret.shape[0], window-1), device=close.device)
        ret_pad = torch.cat([pad, ret], dim=1)
        mom = ret_pad.unfold(1, window, 1).sum(dim=-1)
        
        # Detect reversals
        mom_prev = Indicators._prev_on_time(mom)
        prod = mom * mom_prev
        reversal = (prod < 0).float()

        return reversal

    @staticmethod
    def relative_strength(close, window=14):
        """RSI-like indicator for strength detection"""
        ret = close - Indicators._prev_on_time(close)
        
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


class KlinesFeatureEngineer:
    INPUT_DIM = 27

    @staticmethod
    def compute_features(raw_dict, trade_mask):
        c = raw_dict['close']
        o = raw_dict['open']
        h = raw_dict['high']
        l = raw_dict['low']        
        v = raw_dict['vol']

        const_1 = torch.full(c.shape, 1, dtype=torch.float32).to(c.device)
        const_e = torch.full(c.shape, math.e, dtype=torch.float32).to(c.device)
        const_10 = torch.full(c.shape, 10, dtype=torch.float32).to(c.device)
        const_100 = torch.full(c.shape, 100, dtype=torch.float32).to(c.device)

        trade_mask_prev1 = Indicators._prev_on_time(trade_mask, t=1)
        trade_mask_prev2 = Indicators._prev_on_time(trade_mask, t=2)

        prev_c = Indicators._prev_on_time(c)
        ret = torch.log1p(c / (prev_c + 1e-9)) * Indicators._prev_on_time(trade_mask, t=1)

        oc_ratio = Indicators.oc_vs_hl(c, o, h, l) * trade_mask
        
        fomo = Indicators.fomo_acceleration(v) * Indicators._prev_on_time(trade_mask, t=2)
        
        dev_5 = Indicators.pump_deviation(c, window=5) * Indicators._prev_on_time(trade_mask, t=5)
        dev_15 = Indicators.pump_deviation(c, window=15) * Indicators._prev_on_time(trade_mask, t=15)
        dev_30 = Indicators.pump_deviation(c, window=30) * Indicators._prev_on_time(trade_mask, t=30)
        dev_90 = Indicators.pump_deviation(c, window=90) * Indicators._prev_on_time(trade_mask, t=90)
        
        log_vol = torch.log1p(v) * trade_mask

        # Advanced factors
        # 按照默认的周期计算
        vol_cluster_5 = Indicators.volatility_clustering(c, window=5) * Indicators._prev_on_time(trade_mask, t=5)
        vol_cluster_15 = Indicators.volatility_clustering(c, window=15) * Indicators._prev_on_time(trade_mask, t=15)
        vol_cluster_30 = Indicators.volatility_clustering(c, window=30) * Indicators._prev_on_time(trade_mask, t=30)
        vol_cluster_90 = Indicators.volatility_clustering(c, window=90) * Indicators._prev_on_time(trade_mask, t=90)

        momentum_rev_5 = Indicators.momentum_reversal(c, window=5) * Indicators._prev_on_time(trade_mask, t=5)
        momentum_rev_15 = Indicators.momentum_reversal(c, window=15) * Indicators._prev_on_time(trade_mask, t=15)
        momentum_rev_30 = Indicators.momentum_reversal(c, window=30) * Indicators._prev_on_time(trade_mask, t=30)
        momentum_rev_90 = Indicators.momentum_reversal(c, window=90) * Indicators._prev_on_time(trade_mask, t=90)

        rel_strength_5 = Indicators.relative_strength(c, window=5) * Indicators._prev_on_time(trade_mask, t=5)
        rel_strength_15 = Indicators.relative_strength(c, window=15) * Indicators._prev_on_time(trade_mask, t=15)
        rel_strength_30 = Indicators.relative_strength(c, window=30) * Indicators._prev_on_time(trade_mask, t=30)
        rel_strength_90 = Indicators.relative_strength(c, window=90) * Indicators._prev_on_time(trade_mask, t=90)

        hl_range = (h - l) / (c + 1e-9) * trade_mask
        close_pos = (c - l) / (h - l + 1e-9) * trade_mask
        
        vol_prev = Indicators._prev_on_time(v)
        vol_trend = (v - vol_prev) / (vol_prev + 1.0) * trade_mask

        features = torch.stack([
            ret,
            oc_ratio,
            fomo,
            dev_5,
            dev_15,
            dev_30,
            dev_90,
            log_vol,
            vol_cluster_5,
            vol_cluster_15,
            vol_cluster_30,
            vol_cluster_90,
            momentum_rev_5,
            momentum_rev_15,
            momentum_rev_30,   
            momentum_rev_90,
            rel_strength_5,
            rel_strength_15,
            rel_strength_30,
            rel_strength_90,
            hl_range,
            close_pos,
            vol_trend,
            const_1,
            const_e,
            const_10,
            const_100
        ], dim=1)

        return features