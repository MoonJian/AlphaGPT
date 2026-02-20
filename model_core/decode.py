import torch
from torch.distributions import Categorical
from tqdm import tqdm
import json

from .config import ModelConfig
from .data_loader import FuturesDataLoader
from .alphagpt import AlphaGPT, NewtonSchulzLowRankDecay, StableRankMonitor
from .vm import StackVM
from .formula import JITFormulaCompiler
from .backtest import FuturesBacktest
from .utils import check_tensor_nan


class AlphaEngine:
    def __init__(self, ts_code='RB.SHF', start_date='20200101', end_date='20241231', token=None, cache_path=None, use_lord_regularization=True, lord_decay_rate=1e-3, lord_num_iterations=5):
        """
        Initialize AlphaGPT training engine.
        
        Args:
            use_lord_regularization: Enable Low-Rank Decay (LoRD) regularization
            lord_decay_rate: Strength of LoRD regularization
            lord_num_iterations: Number of Newton-Schulz iterations per step
        """
        self.loader = FuturesDataLoader(ts_code=ts_code, start_date=start_date, end_date=end_date, token=token, cache_path=cache_path)
        self.loader.load()
        
        self.model = AlphaGPT().to(ModelConfig.DEVICE)
        self.compiler = JITFormulaCompiler()
        self.bt = FuturesBacktest()

    def decode(self, rpn_ids):
        return self.model.translate_to_exprs(rpn_id_list=rpn_ids)

    def backtest(self, rpn_ids):
        formula = rpn_ids     

        fast_factor_func = self.compiler.compile(formula)
        res = fast_factor_func(self.loader.feat_tensor)
        print(f'res max: {res.max()}, min: {res.min()}')
        norm_type = self.compiler.get_op_name(formula[-1])
        score, ret_val, corr, trade_count, _ = self.bt.evaluate(res, self.loader.raw_data_cache, self.loader.target_ret, norm_type)
        print(score, ret_val, corr, trade_count)

if __name__ == "__main__":
    eng = AlphaEngine(ts_code='RB.SHF', start_date='20200101', end_date='20241231', use_lord_regularization=True)
    tokens = [0, 0, 25, 4, 13, 27, 1, 23, 18, 26, 18, 21, 28]
    exprs = eng.decode(tokens)
    print(exprs)

    eng.backtest(tokens)