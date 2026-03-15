import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm
import json
import numpy as np

from .config import ModelConfig
from .data_loader import CryptoDataLoader
from .alphagpt import AlphaGPT, NewtonSchulzLowRankDecay, StableRankMonitor
from .vm import StackVM
from .formula import JITFormulaCompiler
from .backtest import MemeBacktest, MainCoinBacktest
from .utils import check_tensor_nan

from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

import signal
import sys

def signal_handler(sig, frame):
    print(f"Received signal {sig}. Exiting gracefully...")
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# 建议使用带时间戳的路径，避免多次实验的数据混在一起
log_dir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
writer = SummaryWriter(log_dir=log_dir)

class AlphaEngine:
    def __init__(self, data_path='./data/AWSData_15m_2020-01-01-2026-02-01.parquet', use_lord_regularization=True, lord_decay_rate=1e-3, lord_num_iterations=5):
        """
        Initialize AlphaGPT training engine.
        
        Args:
            use_lord_regularization: Enable Low-Rank Decay (LoRD) regularization
            lord_decay_rate: Strength of LoRD regularization
            lord_num_iterations: Number of Newton-Schulz iterations per step
        """
        self.loader = CryptoDataLoader(data_path)        
        self.loader.load_klines_data()    
        
        self.model = AlphaGPT().to(ModelConfig.DEVICE)
        
        # Standard optimizer
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        
        # Low-Rank Decay regularizer
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
        self.bt = MainCoinBacktest()
        
        self.best_score = -float('inf')
        self.best_corr = -float('inf')
        self.best_threshold = None
        self.best_formula = None
        # Reward 归一化：EMA 维护 running mean/std，解决 value loss 尺度爆炸
        self._reward_mean = 0.0
        self._reward_var = 1.0
        self._reward_count = 0
        self.training_history = {
            'step': [],
            'avg_reward': [],
            'best_score': [],
            'best_corr': [],
            'stable_rank': []
        }

    def _check_formula(self, formula):
        from .ops import _op_tanh, _op_gate, _ts_delay, _op_jump, _op_decay, _op_ts_zscore_rolling, _op_rolling_mean
        context = {
            'torch': torch,
            '_op_gate': _op_gate,
            '_op_jump': _op_jump,
            '_op_decay': _op_decay,
            '_ts_delay': _ts_delay,
            '_op_tanh': _op_tanh,
            '_op_ts_zscore_rolling': _op_ts_zscore_rolling,
            '_op_rolling_mean': _op_rolling_mean
        }

        fast_factor_func, source_code = self.compiler.compile(formula)
        if fast_factor_func is None:
            print("_check_formula: compile failed")
            return
        res = fast_factor_func(self.loader.feat_tensor)
        check_tensor_nan(res)
        return res

    def train(self):
        print("🚀 Starting Alpha Mining with REINFORCE+Baseline" + (" + LoRD" if self.use_lord else "") + " ...")
        if self.use_lord:
            print(f"   LoRD Regularization enabled")
            print(f"   Target keywords: ['q_proj', 'k_proj', 'attention', 'qk_norm']")
        print(f"   Policy: value_coef={ModelConfig.VALUE_COEF}, entropy_coef={ModelConfig.ENTROPY_COEF}")
        print(f"   Value: gamma={ModelConfig.GAMMA}, reward_norm=True, loss={ModelConfig.VALUE_LOSS_TYPE}")

        pbar = tqdm(range(ModelConfig.TRAIN_STEPS))
        L = ModelConfig.MAX_FORMULA_LEN + 1  # 动作数

        for step in pbar:
            bs = ModelConfig.BATCH_SIZE
            inp = torch.zeros((bs, 1), dtype=torch.long, device=ModelConfig.DEVICE)
            log_probs_old = []
            values_old = []
            tokens_list = []
            stack_sizes = torch.zeros(bs, dtype=torch.int32).to(inp.device)

            # 采样轨迹，并记录每步的 log_prob 与 value（供 REINFORCE+baseline 用）
            # 采样温度 >1 时分布更平坦，减轻前期 Categorical 过早尖锐化、陷入局部最优
            temperature = getattr(ModelConfig, 'SAMPLING_TEMPERATURE', 1.0)
            for _ in range(L):
                logits, value, _ = self.model(inp, stack_sizes)
                tempered_logits = logits / temperature if temperature != 1.0 else logits
                dist = Categorical(logits=tempered_logits)
                action = dist.sample()
                log_probs_old.append(dist.log_prob(action))
                values_old.append(value.squeeze(-1))
                tokens_list.append(action)
                inp = torch.cat([inp, action.unsqueeze(1)], dim=1)
                stack_sizes = self.model.compute_stack_size(stack_sizes, action)

            seqs = torch.stack(tokens_list, dim=1)  # [B, L]
            rewards = torch.zeros(bs, device=ModelConfig.DEVICE)
            legal_cnt = 0
            score_list, corr_list, trade_count_list, threshold_list = [], [], [], []

            for i in range(bs):
                formula = seqs[i].tolist()
                fast_factor_func, source_code = self.compiler.compile(formula)       
                if fast_factor_func is None:
                    rewards[i] = -50.0
                    continue
                res = fast_factor_func(self.loader.feat_tensor)
                if res is None:
                    rewards[i] = -50.0
                    continue
                if res.std() < 1e-4:
                    rewards[i] = -100.0
                    continue

                if check_tensor_nan(res, f'res-{i}'):
                    rewards[i] = -50.0
                    continue

                legal_cnt += 1
                norm_type = self.compiler.get_op_name(formula[-1])
                score, ret_val, corr, trade_count, best_threshold = self.bt.evaluate(
                    res, self.loader.raw_data_cache, self.loader.target_ret, norm_type,
                    position_mode=ModelConfig.POSITION_MODE
                )
                score_list.append(score)
                corr_list.append(corr)
                trade_count_list.append(trade_count)
                threshold_list.append(best_threshold)
                rewards[i] = (
                    ModelConfig.REWARD_SCORE_WEIGHT * score
                    # + ModelConfig.REWARD_CORR_WEIGHT * abs(corr)
                )

                if score > self.best_score:
                    self.best_score = score
                    self.best_formula = formula
                    self.best_threshold = best_threshold
                    tqdm.write(f"[!] New King: Score {score:.2f} | Ret {ret_val:.2%} | Formula {formula} | Corr {corr} | Trades {trade_count} | Threshold {best_threshold}")
                if abs(corr) > self.best_corr:
                    self.best_corr = abs(corr)
                    tqdm.write(f"[!] New Corr King: Score {score:.2f} | Ret {ret_val:.2%} | Formula {formula} | Corr {corr} | Trades {trade_count} | Threshold {best_threshold}")

            rewards = torch.nan_to_num(rewards, nan=-50.0)

            # 1. Reward 归一化：用 EMA 维护 running mean/std，将 reward 缩放到合理范围
            with torch.no_grad():
                batch_mean = rewards.mean().item()
                batch_var = rewards.var().item()
                m = ModelConfig.REWARD_NORM_MOMENTUM
                if self._reward_count == 0:
                    self._reward_mean = batch_mean
                    self._reward_var = max(batch_var, 1e-4)
                    self._reward_count = 1
                else:
                    self._reward_mean = m * self._reward_mean + (1 - m) * batch_mean
                    self._reward_var = m * self._reward_var + (1 - m) * batch_var
                    self._reward_var = max(self._reward_var, 1e-4)
                reward_std = (self._reward_var ** 0.5) + 1e-8
                rewards_norm = (rewards - self._reward_mean) / reward_std

            # 2. Gamma 折扣回报：G_t = γ^(L-1-t) * R，早期步 target 更小，合理时间信用分配
            #    标准 RL：只有终端 reward 时，G_t = γ^(L-1-t) * R
            gamma = ModelConfig.GAMMA
            gamma_powers = torch.pow(
                gamma,
                torch.arange(L - 1, -1, -1, dtype=torch.float32, device=rewards.device)
            )  # [L]: γ^(L-1), γ^(L-2), ..., γ^0
            returns = rewards_norm.unsqueeze(1) * gamma_powers.unsqueeze(0)  # [B, L]

            # 构造 REINFORCE + baseline 所需张量
            old_log_probs = torch.stack(log_probs_old, dim=1)   # [B, L]
            old_values = torch.stack(values_old, dim=1)         # [B, L]
            advantages = returns - old_values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            full_idx = torch.cat([torch.zeros(bs, 1, dtype=torch.long, device=ModelConfig.DEVICE), seqs], dim=1)  # [B, L+1]

            # 单次前向：REINFORCE + baseline（无 PPO 多轮更新）
            new_logits, new_values = self.model.forward_sequence(full_idx)  # [B, L, V], [B, L]
            dist_new = torch.distributions.Categorical(logits=new_logits)
            entropy = dist_new.entropy().mean()

            # REINFORCE: policy_loss = -E[log π(a|s) * A]，无 clip 与 ratio
            policy_loss = -(old_log_probs * advantages).mean()
            # 3. Value loss：Huber 对异常值更鲁棒，或 MSE
            if ModelConfig.VALUE_LOSS_TYPE == 'huber':
                value_loss = F.huber_loss(new_values, returns, delta=ModelConfig.HUBER_DELTA)
            else:
                value_loss = F.mse_loss(new_values, returns)

            loss = (
                policy_loss
                + ModelConfig.VALUE_COEF * value_loss
                - ModelConfig.ENTROPY_COEF * entropy
            )

            self.opt.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3.0)
            self.opt.step()

            total_policy_loss = policy_loss.item()
            total_value_loss = value_loss.item()
            total_entropy = entropy.item()

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

            writer.add_scalar('Train/loss_policy', total_policy_loss, step)
            writer.add_scalar('Train/loss_value', total_value_loss, step)
            writer.add_scalar('Train/entropy', total_entropy, step)
            writer.add_scalar('Train/grad_norm', grad_norm.item(), step)
            writer.add_scalar('Train/avg_reward', avg_reward, step)
            writer.add_scalar('Train/best_score', self.best_score, step)
            if score_list:
                writer.add_scalar('Train/score', np.mean(score_list), step)
                writer.add_scalar('Train/corr', np.mean(corr_list), step)
                writer.add_scalar('Train/trade_count', np.mean(trade_count_list), step)
            if self.best_formula is not None:
                writer.add_text('Train/best_formula', str(self.best_formula), step)            
                writer.add_text('Train/best_formula_exprs', self.model.translate_to_exprs(self.best_formula), step)
                writer.add_text('Train/best_threshold', str(self.best_threshold), step)

        # Save best formula
        with open("best_meme_strategy.json", "w") as f:
            json.dump(self.best_formula, f)
        
        # Save training history
        import json as js
        with open("training_history.json", "w") as f:
            js.dump(self.training_history, f)
        
        print(f"\n✓ Training completed!")
        print(f"  Best score: {self.best_score:.4f}")
        print(f"  Best formula: {self.best_formula}")

        writer.close()


if __name__ == "__main__":
    eng = AlphaEngine(use_lord_regularization=True)
    eng.train()