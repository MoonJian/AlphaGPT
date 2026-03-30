import numpy as np
import pandas as pd
import torch
from .config import ModelConfig
from .factors import KlinesFeatureEngineer

class TushareDataLoader:
    def __init__(self, data_path, data_type='parquet'):
        if data_type == 'parquet':
            self.raw_data = pd.read_parquet(data_path)
        elif data_type == 'csv':
            self.raw_data = pd.read_csv(data_path)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

        if 'adj_factor' in self.raw_data.columns:
            self.raw_data['adj_factor'] = self.raw_data['adj_factor'].fillna(1.0)
            self.raw_data['close'] = self.raw_data['close'] * self.raw_data['adj_factor']
            self.raw_data['open'] = self.raw_data['open'] * self.raw_data['adj_factor']
            self.raw_data['high'] = self.raw_data['high'] * self.raw_data['adj_factor']
            self.raw_data['low'] = self.raw_data['low'] * self.raw_data['adj_factor']

        self.feat_tensor = None
        self.raw_data_cache = None
        self.target_ret = None
        self.trade_mask = None
        self.ts_codes = None
        self.trade_dates = None

    def build_trade_mask(self, cols=None, open_col: str = 'open'):
        """
        Build a 0/1 tradability mask from raw kline data.

        mask[s, t] = 1 if stock s has a valid open price on day t, else 0.
        The returned tensor shape matches the other tensors built from pivot.values.T:
        (num_stocks, num_days).
        """
        if cols is None:
            cols = sorted(self.raw_data['ts_code'].unique().tolist())

        pivot_open = self.raw_data.pivot(index='trade_date', columns='ts_code', values=open_col)
        pivot_open = pivot_open.reindex(columns=cols)
        mask = (~pivot_open.isna()).astype('float32').fillna(0.0)

        # Cache alignment metadata for downstream usage/debugging.
        self.ts_codes = cols
        self.trade_dates = pivot_open.index.tolist()

        return torch.tensor(mask.values.T, dtype=torch.float32, device=ModelConfig.DEVICE)

    def build_valid_mask_after_listing(
        self,
        cols=None,
        open_col: str = 'open',
        min_valid_open_days: int = 60,
    ):
        """
        Same layout as build_trade_mask after transpose: (num_stocks, num_days).

        mask[s, t] = 1 only if stock s has a valid open on day t and that day is
        at least the min_valid_open_days-th such day for s (counting only rows
        where open is non-NaN along time). Days without open do not advance the
        count—e.g. suspension gaps are excluded.
        """
        if cols is None:
            cols = sorted(self.raw_data['ts_code'].unique().tolist())

        pivot_open = self.raw_data.pivot(index='trade_date', columns='ts_code', values=open_col)
        pivot_open = pivot_open.reindex(columns=cols)

        has_open = pivot_open.notna().to_numpy()
        cum_open_days = np.cumsum(has_open, axis=0)
        valid = has_open & (cum_open_days >= min_valid_open_days)
        mask = valid.astype(np.float32)

        self.ts_codes = cols
        self.trade_dates = pivot_open.index.tolist()

        return torch.tensor(mask.T, dtype=torch.float32, device=ModelConfig.DEVICE)

    def load_klines_data(self):
        cols = sorted(self.raw_data['ts_code'].unique().tolist())
        self.trade_mask = self.build_trade_mask(cols=cols, open_col='open') # 当天有交易
        self.valid_mask = self.build_valid_mask_after_listing(cols=cols, open_col='open', min_valid_open_days=60)

        def to_tensor(col):
            pivot = self.raw_data.pivot(index='trade_date', columns='ts_code', values=col)
            pivot = pivot.reindex(columns=cols)
            pivot = pivot.ffill().fillna(0.0)
            return torch.tensor(pivot.values.T, dtype=torch.float32, device=ModelConfig.DEVICE)
        
        self.raw_data_cache = {
            'open': to_tensor('open'),
            'high': to_tensor('high'),
            'low': to_tensor('low'),
            'close': to_tensor('close'),
            'vol': to_tensor('vol'),
            'is_limit_up': to_tensor('is_limit_up'),
            'is_limit_down': to_tensor('is_limit_down'),
        }

        self.feat_tensor = KlinesFeatureEngineer.compute_features(self.raw_data_cache, self.trade_mask)

        op = self.raw_data_cache['open']
        t1 = torch.roll(op, -1, dims=1)
        t2 = torch.roll(op, -2, dims=1)
        self.target_ret = torch.log(t2 / (t1 + 1e-9)) # O2O计算目标收益率
        self.target_ret[:, -2:] = 0.0
        print(f"Data Ready. Shape: {self.feat_tensor.shape}")


if __name__ == "__main__":
    data_loader = TushareDataLoader(data_path='/mnt/h/data/ashare/tushare_processed/ashare_daily.parquet')
    data_loader.load_klines_data()
