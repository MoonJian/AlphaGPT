import pandas as pd
import torch
from .config import ModelConfig
from .factors import FeatureEngineer

class CryptoDataLoader:
    def __init__(self, data_path, symbol='ETHUSDT', data_type='parquet'):
        if data_type == 'parquet':
            self.raw_data = pd.read_parquet(data_path)
        elif data_type == 'csv':
            self.raw_data = pd.read_csv(data_path)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

        self.raw_data["symbol"] = symbol
        self.feat_tensor = None
        self.raw_data_cache = None
        self.target_ret = None
        
    def load_data(self):
        def to_tensor(col):
            pivot = self.raw_data.pivot(index='global_bar_index', columns='symbol', values=col)
            pivot = pivot.fillna(method='ffill').fillna(0.0)
            return torch.tensor(pivot.values.T, dtype=torch.float32, device=ModelConfig.DEVICE)
        
        self.raw_data_cache = {
            'open': to_tensor('open'),
            'high': to_tensor('high'),
            'low': to_tensor('low'),
            'close': to_tensor('close'),
            # 'volume': to_tensor('volume'),
            # 'liquidity': to_tensor('liquidity'),
            # 'fdv': to_tensor('fdv')
        }

        self.feat_tensor = FeatureEngineer.compute_features(self.raw_data_cache)
        op = self.raw_data_cache['open']
        t1 = torch.roll(op, -1, dims=1)
        t2 = torch.roll(op, -2, dims=1)
        self.target_ret = torch.log(t2 / (t1 + 1e-9))
        self.target_ret[:, -2:] = 0.0
        print(f"Data Ready. Shape: {self.feat_tensor.shape}")