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
    def direction_convict(end, start, range_high, range_low):
        range_width = range_high - range_low + 1e-9
        body = end - start
        strength = body / range_width
        return torch.tanh(strength * 3.0)

    @staticmethod
    def momentum_log(x, t=1):
        x_prev = Indicators._prev_on_time(x, t)
        x_chg = torch.log(x / (x_prev + 1e-9))
        x_chg[:, :t] = 0.0
        return x_chg

    @staticmethod
    def momentum(x, t=1):
        x_prev = Indicators._prev_on_time(x, t)
        x_chg = (x - x_prev) / (x_prev + 1e-9)
        x_chg[:, :t] = 0.0
        return torch.clamp(x_chg, -5.0, 5.0)

    @staticmethod
    def momentum_acceleration(x, window=1):
        t = window
        x_prev = Indicators._prev_on_time(x, t)
        x_chg = (x - x_prev) / (x_prev + 1.0)
        x_chg[:, :t] = 0.0
        acc = x_chg - Indicators._prev_on_time(x_chg)
        acc[:, : t + 1] = 0.0
        return torch.clamp(acc, -5.0, 5.0)

    @staticmethod
    def relative_sma_deviation(x, window=20):
        pad = torch.zeros((x.shape[0], window-1), device=x.device)
        x_pad = torch.cat([pad, x], dim=1)
        ma = x_pad.unfold(1, window, 1).mean(dim=-1)
        dev = (x - ma) / (ma + 1e-9)
        dev[:, :window] = 0.0
        return dev

    @staticmethod
    def rolling_realized_volatility(x, window=10):
        """Detect volatility clustering patterns"""
        x_prev = Indicators._prev_on_time(x)
        x_sq = torch.log(x / (x_prev + 1e-9)) ** 2        
        pad = torch.zeros((x_sq.shape[0], window-1), device=x.device)
        x_sq_pad = torch.cat([pad, x_sq], dim=1)
        volat_ma = x_sq_pad.unfold(1, window, 1).mean(dim=-1)
        volat = torch.sqrt(volat_ma + 1e-9)
        volat[:, :window] = 0.0
        return volat

    @staticmethod
    def momentum_zero_cross(x, window=5):
        first_val = x[:, 0:1]
        pad = first_val.expand(-1, window)
        x_padded = torch.cat([pad, x], dim=1)
        x_shifted_N = x_padded[:, :-window]
        
        mom = torch.log(x / (x_shifted_N + 1e-9))
        mom_prev = Indicators._prev_on_time(mom)
        mom_prod = mom * mom_prev

        cross_signal = (mom_prod < 0).float()        
        cross_signal[:, :window] = 0.0
        
        return cross_signal

    @staticmethod
    def relative_strength(x, window=14):
        """RSI-like indicator for strength detection"""
        x_diff = x - Indicators._prev_on_time(x)
        
        gains = torch.relu(x_diff)
        losses = torch.relu(-x_diff)
        
        pad = torch.zeros((gains.shape[0], window-1), device=x.device)
        gains_pad = torch.cat([pad, gains], dim=1)
        losses_pad = torch.cat([pad, losses], dim=1)
        
        avg_gain = gains_pad.unfold(1, window, 1).mean(dim=-1)
        avg_loss = losses_pad.unfold(1, window, 1).mean(dim=-1)
        
        rs = (avg_gain + 1e-9) / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        rsi[:, :window] = 50.0
        return (rsi - 50) / 50 


class BasicFactorsEngineer:
    INPUT_DIM = 24

    @torch.jit.script
    def robust_norm_rolling(self, x: torch.Tensor, window: int = ModelConfig.OP_ROLL_WINDOW) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        B, T = x.shape

        pad = x[:, 0:1].repeat(1, window - 1)
        x_padded = torch.cat([pad, x], dim=1) 
        windows = x_padded.unfold(1, window, 1)

        roll_median = windows.median(dim=-1).values # [B, T]
        
        abs_diff = torch.abs(windows - roll_median.unsqueeze(-1))
        roll_mad = abs_diff.median(dim=-1).values + 1e-6 
        
        res = (x - roll_median) / roll_mad
        res = torch.clamp(res, -5.0, 5.0)
        
        return res

    @staticmethod
    def compute_features(raw_dict):
        c = raw_dict['close']
        o = raw_dict['open']
        h = raw_dict['high']
        l = raw_dict['low']        
        v = raw_dict['volume']

        const_1 = torch.full(c.shape, 1, dtype=torch.float32).to(c.device)
        const_e = torch.full(c.shape, math.e, dtype=torch.float32).to(c.device)
        const_10 = torch.full(c.shape, 10, dtype=torch.float32).to(c.device)
        const_100 = torch.full(c.shape, 100, dtype=torch.float32).to(c.device)
        
        ret = Indicators.momentum(c, t=1)
        hl_range = (h - l) / (c + 1e-9) 
        close_pos = (c - l) / (h - l + 1e-9)
        vol_mom = Indicators.momentum(v, t=1)

        close_dominance_ratio = Indicators.direction_convict(c, o, h, l)
        
        fomo_5 = Indicators.momentum_acceleration(v, window=5)
        fomo_15 = Indicators.momentum_acceleration(v, window=15)
        fomo_30 = Indicators.momentum_acceleration(v, window=30)
        
        dev_20 = Indicators.relative_sma_deviation(c, window=20)
        dev_50 = Indicators.relative_sma_deviation(c, window=50)
        dev_100 = Indicators.relative_sma_deviation(c, window=100)
        
        realized_volat_10 = Indicators.rolling_realized_volatility(c, window=10)
        realized_volat_30 = Indicators.rolling_realized_volatility(c, window=30)
        realized_volat_60 = Indicators.rolling_realized_volatility(c, window=60)

        momentum_zero_cross_5 = Indicators.momentum_zero_cross(c, window=5)
        momentum_zero_cross_15 = Indicators.momentum_zero_cross(c, window=15)
        momentum_zero_cross_30 = Indicators.momentum_zero_cross(c, window=30)

        rel_strength_10 = Indicators.relative_strength(c, window=10)
        rel_strength_30 = Indicators.relative_strength(c, window=30)
        rel_strength_60 = Indicators.relative_strength(c, window=60)

        @torch.jit.script       
        def robust_norm(x: torch.Tensor) -> torch.Tensor:
            median = torch.nanmedian(x, dim=1, keepdim=True)[0]
            mad = torch.nanmedian(torch.abs(x - median), dim=1, keepdim=True)[0] + 1e-6
            norm = (x - median) / mad            
            return torch.clamp(norm, -5.0, 5.0)

        features = torch.stack([
            ret,
            hl_range,
            close_pos,
            vol_mom,
            close_dominance_ratio,
            fomo_5,
            fomo_15,
            fomo_30,
            dev_20,
            dev_50,
            dev_100,
            realized_volat_10,
            realized_volat_30,
            realized_volat_60,
            momentum_zero_cross_5,
            momentum_zero_cross_15,
            momentum_zero_cross_30,
            rel_strength_10,
            rel_strength_30,
            rel_strength_60,
            const_1,
            const_e,
            const_10,
            const_100
        ], dim=1)

        return features