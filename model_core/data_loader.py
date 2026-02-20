"""
中国期货市场数据加载器
基于 Tushare Pro 唯一数据源，处理主力切换、涨跌停过滤、交割月规避
"""
import pandas as pd
import numpy as np
import torch
from typing import List, Optional
from .config import ModelConfig
from .factors import FuturesFactorEngineer
from .tushare_provider import (
    TushareFuturesProvider,
    ContinuousContractBuilder,
    LimitFilter,
    NIGHT_SESSION_SYMBOLS,
)


class FuturesDataLoader:
    """
    期货数据加载器
    - 通过 Tushare fut_mapping 获取主力映射
    - 构建持仓量加权平滑的连续合约
    - 涨跌停日因子置 null
    - 交割月前 5 日切换次主力
    """
    
    def __init__(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
        token: Optional[str] = None,
        cache_path: Optional[str] = None,
    ):
        """
        Args:
            ts_code: 主力/连续合约代码，如 RB.SHF（螺纹钢）、IF.CFX（沪深300）
            start_date: 开始日期 YYYYMMDD
            end_date: 结束日期 YYYYMMDD
            token: Tushare token，默认从 ModelConfig.TUSHARE_TOKEN 读取
            cache_path: 缓存路径，存在则优先加载
        """
        self.ts_code = ts_code
        self.start_date = start_date
        self.end_date = end_date
        self.token = token or ModelConfig.TUSHARE_TOKEN
        self.cache_path = cache_path
        
        self.raw_data: Optional[pd.DataFrame] = None
        self.feat_tensor: Optional[torch.Tensor] = None
        self.raw_data_cache: Optional[dict] = None
        self.target_ret: Optional[torch.Tensor] = None
        self.dates: Optional[pd.DatetimeIndex] = None
        self.symbol: str = ts_code.split('.')[0]
        self._provider: Optional[TushareFuturesProvider] = None
        self._builder: Optional[ContinuousContractBuilder] = None
    
    def _get_provider(self) -> TushareFuturesProvider:
        if self._provider is None:
            if not self.token:
                raise ValueError("请设置 TUSHARE_TOKEN 或传入 token 参数")
            self._provider = TushareFuturesProvider(self.token)
        return self._provider
    
    def _get_builder(self) -> ContinuousContractBuilder:
        if self._builder is None:
            self._builder = ContinuousContractBuilder(
                self._get_provider(),
                smooth_days=ModelConfig.SMOOTH_TRANSITION_DAYS,
                delivery_avoid_days=ModelConfig.DELIVERY_MONTH_AVOID_DAYS,
            )
        return self._builder
    
    def load(self) -> "FuturesDataLoader":
        """加载并构建连续合约数据及因子"""
        if self.cache_path:
            try:
                df = pd.read_parquet(self.cache_path)
                if not df.empty and 'trade_date' in df.columns:
                    self.raw_data = df.sort_values('trade_date').reset_index(drop=True)
                    self._build_from_df()
                    return self
            except Exception:
                pass
        
        provider = self._get_provider()
        builder = self._get_builder()
        
        # 构建连续合约（结算价序列）
        cont = builder.build_continuous(
            self.ts_code,
            self.start_date,
            self.end_date,
            price_col='settle',
            use_smooth=True,
        )
        if cont.empty:
            raise ValueError(f"未获取到 {self.ts_code} 的连续合约数据")
        
        # 次主力用于期限结构（简化：取主力映射的次月合约日线）
        second_main = provider.get_second_main_mapping(self.ts_code, self.start_date, self.end_date)
        
        # 合并次主力结算价（用于期限结构斜率）
        if not second_main.empty and 'second_main_ts_code' in second_main.columns:
            second_daily = []
            for sc in second_main['second_main_ts_code'].dropna().unique():
                dd = provider.get_fut_daily_single(sc, self.start_date, self.end_date)
                if not dd.empty:
                    dd = dd[['trade_date', 'settle']].rename(columns={'settle': 'second_settle'})
                    dd['second_main_ts_code'] = sc
                    second_daily.append(dd)
            if second_daily:
                second_df = pd.concat(second_daily)
                # 按 trade_date + second_main_ts_code 与 mapping 对齐
                merge_df = second_main[['trade_date', 'second_main_ts_code']].merge(
                    second_df, on=['trade_date', 'second_main_ts_code'], how='left'
                )[['trade_date', 'second_settle']].drop_duplicates('trade_date', keep='last')
                cont = cont.merge(merge_df, on='trade_date', how='left')
                cont['second_settle'] = cont['second_settle'].fillna(cont['settle'])
            else:
                cont['second_settle'] = cont['settle']
        else:
            cont['second_settle'] = cont['settle']
        
        # 涨跌停检测
        limit_mask = LimitFilter.apply_limit_mask(cont, limit_pct=0.04)
        cont['limit_day'] = limit_mask.values if hasattr(limit_mask, 'values') else limit_mask
        
        # 涨跌停触及强度：过去 10 日触及次数（不含当日，当日触及则因子置 null）
        cont['limit_hit_count'] = 0
        for i in range(1, 11):
            cont['limit_hit_count'] += limit_mask.shift(i).fillna(0).astype(int)
        
        # 基差：商品期货需现货指数，此处简化填 0；股指期货可用 index_daily
        cont['basis'] = 0.0
        if self.symbol in ('IF', 'IC', 'IH', 'IM'):
            # 获取对应指数日线作为基差参考
            index_map = {'IF': '000300.SH', 'IC': '000905.SH', 'IH': '000016.SH', 'IM': '000852.SH'}
            idx_code = index_map.get(self.symbol)
            if idx_code:
                try:
                    idx_df = provider.pro.index_daily(ts_code=idx_code, start_date=self.start_date, end_date=self.end_date)
                    if idx_df is not None and not idx_df.empty:
                        idx_df = idx_df[['trade_date', 'close']].rename(columns={'close': 'index_close'})
                        cont = cont.merge(idx_df, on='trade_date', how='left')
                        cont['basis'] = (cont['settle'] - cont['index_close'].ffill()).fillna(0)
                except Exception:
                    pass
        
        # 夜盘：日频数据无分钟，夜盘收盘用当日 close 近似（日盘 close 即日收盘）
        cont['night_close'] = cont['close']
        cont['next_open'] = cont['open'].shift(-1)
        cont.loc[cont.index[-1], 'next_open'] = cont.iloc[-1]['close']
        
        self.raw_data = cont
        if self.cache_path:
            self.raw_data.to_parquet(self.cache_path)
        
        self._build_from_df()
        return self
    
    def _build_from_df(self):
        """从 raw_data 构建特征张量和目标收益"""
        df = self.raw_data
        self.dates = pd.to_datetime(df['trade_date'])
        
        # 数值列
        for col in ['settle', 'oi', 'open', 'high', 'low', 'close', 'second_settle', 'basis', 'limit_hit_count', 'next_open', 'night_close']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').ffill().bfill()
        
        def to_tensor(arr, dtype=np.float32):
            t = torch.tensor(arr.astype(dtype), device=ModelConfig.DEVICE)
            return t.unsqueeze(0) if t.dim() == 1 else t
        
        settle = to_tensor(df['settle'].values)
        oi = to_tensor(df['oi'].fillna(0).values)
        second_settle = to_tensor(df['second_settle'].values)
        open_ = to_tensor(df['open'].values)
        close = to_tensor(df['close'].values)
        next_open = to_tensor(df['next_open'].values)
        night_close = to_tensor(df['night_close'].values)
        limit_hit = to_tensor(df['limit_hit_count'].fillna(0).values)
        limit_mask = torch.tensor(df['limit_day'].fillna(False).values, device=ModelConfig.DEVICE)
        if limit_mask.dim() == 1:
            limit_mask = limit_mask.unsqueeze(0)
        basis = to_tensor(df['basis'].fillna(0).values)
        
        raw_dict = {
            'settle': settle,
            'oi': oi,
            'second_settle': second_settle,
            'open': open_,
            'close': close,
            'next_open': next_open,
            'night_close': night_close,
            'limit_hit_count': limit_hit,
            'limit_day_mask': limit_mask,
            'basis': basis,
        }
        
        self.raw_data_cache = raw_dict
        self.feat_tensor = FuturesFactorEngineer.compute_features(raw_dict)
        
        # 目标收益：次日复权收益率，考虑主力切换换仓成本
        # forward_return[t] = (settle[t+1] - settle[t]) / settle[t] - rollover_cost_if_switch
        settle_np = df['settle'].values.astype(np.float64)
        mapping = df['mapping_ts_code'].values if 'mapping_ts_code' in df.columns else np.zeros(len(df), dtype=object)
        ret = np.zeros(len(settle_np))
        ret[:-1] = (settle_np[1:] - settle_np[:-1]) / (settle_np[:-1] + 1e-9)
        # 主力切换日加换仓成本（0.5 跳 ≈ 0.5 * tick/price）
        tick_approx = settle_np * 0.0001  # 粗略
        rollover_cost = ModelConfig.ROLLOVER_COST_TICKS * tick_approx
        for i in range(1, len(mapping) - 1):
            if mapping[i] != mapping[i - 1]:
                ret[i] -= rollover_cost[i] / (settle_np[i] + 1e-9)
        ret[-1] = 0.0
        self.target_ret = torch.tensor(ret.astype(np.float32), device=ModelConfig.DEVICE)
        if self.target_ret.dim() == 1:
            self.target_ret = self.target_ret.unsqueeze(0)
        
        self.split_idx = int(len(df) * 0.8)
        print(f"✅ {self.ts_code} 期货数据就绪. 样本数={len(df)}, 特征维度={self.feat_tensor.shape}")
        return self


# 兼容旧名称（注意：接口已变更，CryptoDataLoader 现指向 FuturesDataLoader）
CryptoDataLoader = FuturesDataLoader
