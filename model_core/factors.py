"""
中国期货市场因子引擎 - 因子层
移除所有 Crypto 专属因子，新增 5 个期货特有因子：
1. 持仓量动量 (OI_MOM)
2. 期限结构斜率 (TERM_SLOPE)
3. 夜盘溢价 (NIGHT_PREM)
4. 涨跌停触及强度 (LIMIT_HIT)
5. 基差动量 (BASIS_MOM)
"""
import torch
import numpy as np
from typing import Dict, Optional
from .config import ModelConfig


class RMSNormFactor(torch.nn.Module):
    """RMSNorm for factor normalization"""
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(d_model))
    
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


def _robust_norm(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """按品种/截面标准化"""
    median = torch.nanmedian(x, dim=dim, keepdim=True).values
    mad = torch.nanmedian(torch.abs(x - median), dim=dim, keepdim=True).values + 1e-6
    return torch.clamp((x - median) / mad, -5.0, 5.0)


class FuturesFactorEngineer:
    """
    期货特有因子计算
    输入: 多品种日频 DataFrame，列含 symbol, trade_date, settle, oi, open, close, limit_hit_mask 等
    输出: [B, T, 5] 特征张量
    """
    INPUT_DIM = 5
    FEATURE_NAMES = ['OI_MOM', 'TERM_SLOPE', 'NIGHT_PREM', 'LIMIT_HIT', 'BASIS_MOM']
    
    @staticmethod
    def oi_momentum(oi: torch.Tensor, window: int = 5) -> torch.Tensor:
        """
        持仓量动量：过去 5 日持仓量变化率，按品种标准化
        oi: [B, T]
        """
        if oi.dim() == 1:
            oi = oi.unsqueeze(0)
        pad = torch.zeros((oi.shape[0], window), device=oi.device)
        oi_pad = torch.cat([pad, oi], dim=1)
        oi_lag = oi_pad[:, :-window]
        oi_curr = oi_pad[:, window:]
        chg = (oi_curr - oi_lag) / (oi_lag + 1e-9)
        return _robust_norm(torch.nan_to_num(chg, nan=0.0))
    
    @staticmethod
    def term_structure_slope(main_settle: torch.Tensor, second_settle: torch.Tensor) -> torch.Tensor:
        """
        期限结构斜率：(主力结算价 - 次主力结算价) / 主力结算价
        main_settle, second_settle: [B, T]
        """
        slope = (main_settle - second_settle) / (main_settle + 1e-9)
        return _robust_norm(torch.nan_to_num(slope, nan=0.0))
    
    @staticmethod
    def night_premium(night_close: torch.Tensor, next_day_open: torch.Tensor) -> torch.Tensor:
        """
        夜盘溢价：夜盘收盘价 / 次日日盘开盘价 - 1
        仅适用于有夜盘品种，无夜盘时返回 0
        night_close, next_day_open: [B, T]
        """
        prem = night_close / (torch.roll(next_day_open, -1, dims=1) + 1e-9) - 1.0
        prem[:, -1] = 0.0  # 最后一日无次日
        return _robust_norm(torch.nan_to_num(prem, nan=0.0))
    
    @staticmethod
    def limit_hit_intensity(limit_hit_count: torch.Tensor, limit_day_mask: torch.Tensor, window: int = 10) -> torch.Tensor:
        """
        涨跌停触及强度：过去 10 日加权触及次数
        触及当日因子值置为 null（通过 mask 传入，外部将对应位置填 0 或 nan）
        limit_hit_count: [B, T] 每日是否触及（0/1）或加权计数
        limit_day_mask: [B, T] True 表示当日触及，应置 null
        """
        B, T = limit_hit_count.shape
        pad = torch.zeros((B, window - 1), device=limit_hit_count.device)
        cnt_pad = torch.cat([pad, limit_hit_count], dim=1)
        # 线性衰减权重
        w = torch.arange(1, window + 1, device=limit_hit_count.device, dtype=limit_hit_count.dtype)
        w = w / w.sum()
        windows = cnt_pad.unfold(1, window, 1)
        intensity = (windows * w.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        # 触及当日置 nan，下游回测时过滤
        intensity = torch.where(limit_day_mask, torch.tensor(float('nan'), device=intensity.device, dtype=intensity.dtype), intensity)
        # robust_norm 时对 nan 位置保持 nan
        normed = _robust_norm(torch.nan_to_num(intensity, nan=0.0))
        normed = torch.where(limit_day_mask, torch.tensor(float('nan'), device=normed.device, dtype=normed.dtype), normed)
        return normed
    
    @staticmethod
    def basis_momentum(basis: torch.Tensor, window: int = 5) -> torch.Tensor:
        """
        基差动量：商品期货用 (期货-现货) 价差变化，股指期货用 (期货-指数) 价差变化
        basis: [B, T] 基差序列
        """
        pad = torch.zeros((basis.shape[0], window), device=basis.device)
        basis_pad = torch.cat([pad, basis], dim=1)
        mom = basis_pad[:, window:] - basis_pad[:, :-window]
        return _robust_norm(torch.nan_to_num(mom, nan=0.0))
    
    @classmethod
    def compute_features(cls, raw_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算 5 维期货因子
        raw_dict 需包含:
          - oi: 持仓量
          - settle: 主力结算价
          - second_settle: 次主力结算价（可选，无则 TERM_SLOPE 填 0）
          - night_close: 夜盘收盘（可选，无夜盘填日收盘）
          - next_open: 次日开盘
          - limit_hit_count: 涨跌停计数
          - limit_day_mask: 当日是否涨跌停
          - basis: 基差（可选，无则填 0）
        """
        oi = raw_dict.get('oi')
        settle = raw_dict.get('settle')
        second_settle = raw_dict.get('second_settle', settle)  # 无次主力则用主力
        night_close = raw_dict.get('night_close', raw_dict.get('close', settle))
        next_open = raw_dict.get('next_open', raw_dict.get('open', settle))
        limit_hit = raw_dict.get('limit_hit_count', torch.zeros_like(settle))
        limit_mask = raw_dict.get('limit_day_mask', torch.zeros_like(settle, dtype=torch.bool))
        basis = raw_dict.get('basis', torch.zeros_like(settle))
        
        f1 = cls.oi_momentum(oi)
        f2 = cls.term_structure_slope(settle, second_settle)
        f3 = cls.night_premium(night_close, next_open)
        f4 = cls.limit_hit_intensity(limit_hit, limit_mask)
        f5 = cls.basis_momentum(basis)
        
        # 输出 [B, 5, T] 以兼容 formula 的 feat_tensor[:, feat_idx]
        return torch.stack([f1, f2, f3, f4, f5], dim=1)


# 兼容旧接口：FeatureEngineer 指向期货版本
FeatureEngineer = FuturesFactorEngineer
KlinesFeatureEngineer = FuturesFactorEngineer
