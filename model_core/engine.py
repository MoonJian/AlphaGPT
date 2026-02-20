"""
中国期货市场 AlphaGPT 训练引擎
基于 Tushare Pro 数据源，期货因子挖掘
"""
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm
import json
import numpy as np

from .config import ModelConfig
from .data_loader import FuturesDataLoader
from .alphagpt import AlphaGPT, NewtonSchulzLowRankDecay, StableRankMonitor
from .vm import StackVM
from .formula import JITFormulaCompiler
from .backtest import FuturesBacktest
from .utils import check_tensor_nan

from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

log_dir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
writer = SummaryWriter(log_dir=log_dir)


class AlphaEngine:
    def __init__(
        self,
        ts_code: str = 'RB.SHF',
        start_date: str = '20200101',
        end_date: str = '20241231',
        token: str = None,
        cache_path: str = None,
        use_lord_regularization: bool = True,
        lord_decay_rate: float = 1e-3,
        lord_num_iterations: int = 5,
    ):
        """
        期货因子挖掘引擎
        
        Args:
            ts_code: 主力合约代码，如 RB.SHF（螺纹钢）、IF.CFX（沪深300）
            start_date: 开始日期 YYYYMMDD
            end_date: 结束日期 YYYYMMDD
            token: Tushare token
            cache_path: 数据缓存路径
            use_lord_regularization: 是否启用 LoRD 正则
            lord_decay_rate: LoRD 衰减率
            lord_num_iterations: LoRD 迭代次数
        """
        self.loader = FuturesDataLoader(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            token=token or ModelConfig.TUSHARE_TOKEN,
            cache_path=cache_path,
        )
        self.loader.load()
        
        self.model = AlphaGPT().to(ModelConfig.DEVICE)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        
        self.use_lord = use_lord_regularization
        if self.use_lord:
            self.lord_opt = NewtonSchulzLowRankDecay(
                self.model.named_parameters(),
                decay_rate=lord_decay_rate,
                num_iterations=lord_num_iterations,
                target_keywords=["q_proj", "k_proj", "attention", "qk_norm"]
            )
            self.rank_monitor = StableRankMonitor(
                self.model,
                target_keywords=["q_proj", "k_proj"]
            )
        else:
            self.lord_opt = None
            self.rank_monitor = None
        
        self.vm = StackVM()
        self.compiler = JITFormulaCompiler()
        self.bt = FuturesBacktest()
        
        self.best_score = -float('inf')
        self.best_corr = -float('inf')
        self.best_threshold = None
        self.best_formula = None
        self.training_history = {
            'step': [],
            'avg_reward': [],
            'best_score': [],
            'best_corr': [],
            'stable_rank': []
        }

    def train(self):
        print("🚀 期货因子挖掘 (AlphaGPT + PPO)" + (" + LoRD" if self.use_lord else "") + " ...")
        print(f"   品种: {self.loader.ts_code}")
        print(f"   PPO: clip_eps={ModelConfig.PPO_CLIP_EPS}, value_coef={ModelConfig.PPO_VALUE_COEF}, epochs={ModelConfig.PPO_EPOCHS}")

        pbar = tqdm(range(ModelConfig.TRAIN_STEPS))
        L = ModelConfig.MAX_FORMULA_LEN + 1

        for step in pbar:
            bs = ModelConfig.BATCH_SIZE
            inp = torch.zeros((bs, 1), dtype=torch.long, device=ModelConfig.DEVICE)
            log_probs_old = []
            values_old = []
            tokens_list = []
            stack_sizes = torch.zeros(bs, dtype=torch.int32).to(inp.device)

            for _ in range(L):
                logits, value, _ = self.model(inp, stack_sizes)
                dist = Categorical(logits=logits)
                action = dist.sample()
                log_probs_old.append(dist.log_prob(action))
                values_old.append(value.squeeze(-1))
                tokens_list.append(action)
                inp = torch.cat([inp, action.unsqueeze(1)], dim=1)
                stack_sizes = self.model.compute_stack_size(stack_sizes, action)

            seqs = torch.stack(tokens_list, dim=1)
            rewards = torch.zeros(bs, device=ModelConfig.DEVICE)
            legal_cnt = 0
            score_list, corr_list, trade_count_list, threshold_list = [], [], [], []

            for i in range(bs):
                formula = seqs[i].tolist()
                fast_factor_func = self.compiler.compile(formula)
                if fast_factor_func is None:
                    rewards[i] = -5.0
                    continue
                res = fast_factor_func(self.loader.feat_tensor)
                if res is None:
                    rewards[i] = -5.0
                    continue
                if res.std() < 1e-4:
                    rewards[i] = -10.0
                    continue
                if check_tensor_nan(res, f'res-{i}'):
                    rewards[i] = -5.0
                    continue

                legal_cnt += 1
                norm_type = self.compiler.get_op_name(formula[-1])
                score, ret_val, corr, trade_count, best_threshold = self.bt.evaluate(
                    res, self.loader.raw_data_cache, self.loader.target_ret, norm_type
                )
                score_list.append(score.item())
                corr_list.append(corr)
                trade_count_list.append(trade_count)
                threshold_list.append(best_threshold)
                rewards[i] = (
                    ModelConfig.REWARD_SCORE_WEIGHT * score.item()
                    + ModelConfig.REWARD_CORR_WEIGHT * abs(corr)
                )

                if score.item() > self.best_score:
                    self.best_score = score.item()
                    self.best_formula = formula
                    self.best_threshold = best_threshold
                    tqdm.write(f"[!] 新最优: Score {score:.2f} | Ret {ret_val:.2%} | Corr {corr} | Trades {trade_count} | Threshold {best_threshold}")
                if abs(corr) > self.best_corr:
                    self.best_corr = abs(corr)
                    tqdm.write(f"[!] 新相关: Score {score:.2f} | Corr {corr} | Trades {trade_count}")

            rewards = torch.nan_to_num(rewards, nan=-5.0)

            old_log_probs = torch.stack(log_probs_old, dim=1)
            old_values = torch.stack(values_old, dim=1)
            returns_ppo = rewards.unsqueeze(1).expand(-1, L)
            advantages = returns_ppo - old_values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            full_idx = torch.cat([torch.zeros(bs, 1, dtype=torch.long, device=ModelConfig.DEVICE), seqs], dim=1)

            total_policy_loss = 0.0
            total_value_loss = 0.0
            total_entropy = 0.0
            n_ppo = 0
            for _ in range(ModelConfig.PPO_EPOCHS):
                new_logits, new_values = self.model.forward_sequence(full_idx)
                new_log_probs = torch.log_softmax(new_logits, dim=-1).gather(2, seqs.unsqueeze(-1)).squeeze(-1)
                dist_new = torch.distributions.Categorical(logits=new_logits)
                entropy = dist_new.entropy().mean()

                ratio = torch.exp(new_log_probs - old_log_probs.detach())
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - ModelConfig.PPO_CLIP_EPS, 1.0 + ModelConfig.PPO_CLIP_EPS) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(new_values, returns_ppo)

                loss = (
                    policy_loss
                    + ModelConfig.PPO_VALUE_COEF * value_loss
                    - ModelConfig.PPO_ENTROPY_COEF * entropy
                )
                self.opt.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3.0)
                self.opt.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_ppo += 1

            if self.use_lord:
                self.lord_opt.step()

            if legal_cnt > 0:
                print(f'legal cnt/bs: {legal_cnt}/{bs}, legal ratio: {legal_cnt/bs:.4f}')
            avg_reward = rewards.mean().item()
            postfix_dict = {'AvgRew': f"{avg_reward:.3f}", 'BestScore': f"{self.best_score:.3f}", 'BestCorr': f"{self.best_corr:.4f}", 'BestThreshold': f"{self.best_threshold}"}
            if self.use_lord and step % 100 == 0:
                stable_rank = self.rank_monitor.compute()
                postfix_dict['Rank'] = f"{stable_rank:.2f}"
                self.training_history['stable_rank'].append(stable_rank)
            self.training_history['step'].append(step)
            self.training_history['avg_reward'].append(avg_reward)
            self.training_history['best_score'].append(self.best_score)
            pbar.set_postfix(postfix_dict)

            writer.add_scalar('Train/loss_policy', total_policy_loss / n_ppo, step)
            writer.add_scalar('Train/loss_value', total_value_loss / n_ppo, step)
            writer.add_scalar('Train/entropy', total_entropy / n_ppo, step)
            writer.add_scalar('Train/grad_norm', grad_norm.item(), step)
            writer.add_scalar('Train/avg_reward', avg_reward, step)
            writer.add_scalar('Train/best_score', self.best_score, step)
            if score_list:
                writer.add_scalar('Train/score', np.mean(score_list), step)
                writer.add_scalar('Train/corr', np.mean(corr_list), step)
                writer.add_scalar('Train/trade_count', np.mean(trade_count_list), step)
            writer.add_text('Train/best_formula', str(self.best_formula), step)
            writer.add_text('Train/best_formula_exprs', self.model.translate_to_exprs(self.best_formula), step)
            writer.add_text('Train/best_threshold', str(self.best_threshold), step)

        with open("best_futures_strategy.json", "w") as f:
            json.dump(self.best_formula, f)
        
        import json as js
        with open("training_history.json", "w") as f:
            js.dump(self.training_history, f)
        
        print(f"\n✓ 训练完成!")
        print(f"  最优得分: {self.best_score:.4f}")
        print(f"  最优公式: {self.best_formula}")

        writer.close()

    def export_joinquant(self, formula_tokens: list = None, output_path: str = None) -> pd.DataFrame:
        """
        导出聚宽兼容格式因子表
        列: trade_date, symbol, factor_value, forward_return
        """
        import pandas as pd
        from .backtest import to_joinquant_format
        
        tokens = formula_tokens or self.best_formula
        if tokens is None:
            raise ValueError("无可用公式，请先训练或传入 formula_tokens")
        
        fast_factor_func = self.compiler.compile(tokens)
        if fast_factor_func is None:
            raise ValueError("公式编译失败")
        
        factors = fast_factor_func(self.loader.feat_tensor)
        if factors is None:
            raise ValueError("因子计算失败")
        
        df = to_joinquant_format(
            self.loader.dates,
            self.loader.symbol,
            factors,
            self.loader.target_ret,
        )
        if output_path:
            df.to_parquet(output_path)
            print(f"📤 聚宽格式因子已保存: {output_path}")
        return df


if __name__ == "__main__":
    eng = AlphaEngine(
        ts_code='RB.SHF',
        start_date='20200101',
        end_date='20241231',
        use_lord_regularization=True,
    )
    eng.train()
