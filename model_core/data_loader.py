import pandas as pd
import torch
from .config import ModelConfig
from .factors import BasicFactorsEngineer

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

    def _pivot_to_tensor(self, col, index_col='open_time'):
        pivot = self.raw_data.pivot(index=index_col, columns='symbol', values=col)
        pivot = pivot.ffill().fillna(0.0)
        return torch.tensor(pivot.values.T, dtype=torch.float32, device=ModelConfig.DEVICE)

    def load_data(self, 
        index_col='open_time', 
        basic_columns=['open', 'high', 'low', 'close', 'volume'],
        advanced_columns=None):

        columns = basic_columns
        if advanced_columns:
            columns.extend(advanced_columns)
        
        self.raw_data_cache = {
            col: self._pivot_to_tensor(col, index_col) for col in columns
        }

        self.feat_tensor = BasicFactorsEngineer.compute_features(self.raw_data_cache)

        op = self.raw_data_cache['open']
        t1 = torch.roll(op, -1, dims=1)
        t2 = torch.roll(op, -2, dims=1)
        self.target_ret = torch.log(t2 / (t1 + 1e-9))
        self.target_ret[:, -2:] = 0.0


if __name__ == "__main__":
    loader = CryptoDataLoader('/mnt/h/data/crypto/um_klines_data/BTCUSDT-futures_15m_2020-01-01-2026-02-01.parquet')
    loader.load_data()
    print(loader.feat_tensor.shape)
    print(loader.target_ret.shape)
    for i in range(loader.feat_tensor.shape[1]):
        print(loader.feat_tensor[0][i].max(), loader.feat_tensor[0][i].min())