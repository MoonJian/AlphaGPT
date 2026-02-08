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

# 建议使用带时间戳的路径，避免多次实验的数据混在一起
log_dir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
writer = SummaryWriter(log_dir=log_dir)

class AlphaEngine:
    def __init__(self, data_path='./data/ETHUSDT-futures_1h_2020-01-01-2026-02-02.parquet', use_lord_regularization=True, lord_decay_rate=1e-3, lord_num_iterations=5):
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
        self.best_formula = None
        self.training_history = {
            'step': [],
            'avg_reward': [],
            'best_score': [],
            'best_corr': [],
            'stable_rank': []
        }

    def train(self):
        print("🚀 Starting Alpha Mining with PPO" + (" + LoRD" if self.use_lord else "") + " ...")
        if self.use_lord:
            print(f"   LoRD Regularization enabled")
            print(f"   Target keywords: ['q_proj', 'k_proj', 'attention', 'qk_norm']")
        print(f"   PPO: clip_eps={ModelConfig.PPO_CLIP_EPS}, value_coef={ModelConfig.PPO_VALUE_COEF}, epochs={ModelConfig.PPO_EPOCHS}")

        pbar = tqdm(range(ModelConfig.TRAIN_STEPS))
        L = ModelConfig.MAX_FORMULA_LEN + 1  # 动作数

        for step in pbar:
            bs = ModelConfig.BATCH_SIZE
            inp = torch.zeros((bs, 1), dtype=torch.long, device=ModelConfig.DEVICE)
            log_probs_old = []
            values_old = []
            tokens_list = []
            stack_sizes = torch.zeros(bs, dtype=torch.int32).to(inp.device)

            # 采样轨迹，并记录每步的 log_prob 与 value（供 PPO 用）
            for _ in range(L):
                logits, value, _ = self.model(inp, stack_sizes)
                dist = Categorical(logits=logits)
                action = dist.sample()
                log_probs_old.append(dist.log_prob(action))
                values_old.append(value.squeeze(-1))
                tokens_list.append(action)
                inp = torch.cat([inp, action.unsqueeze(1)], dim=1)
                stack_sizes = self.model.compute_stack_size(stack_sizes, action)

            seqs = torch.stack(tokens_list, dim=1)  # [B, L]
            rewards = torch.zeros(bs, device=ModelConfig.DEVICE)
            legal_cnt = 0
            score_list, corr_list, trade_count_list = [], [], []

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
                score, ret_val, corr, trade_count = self.bt.evaluate(
                    res, self.loader.raw_data_cache, self.loader.target_ret, norm_type
                )
                score_list.append(score.item())
                corr_list.append(corr)
                trade_count_list.append(trade_count)
                rewards[i] = (
                    ModelConfig.REWARD_SCORE_WEIGHT * score.item()
                    + ModelConfig.REWARD_CORR_WEIGHT * abs(corr)
                )

                if score.item() > self.best_score:
                    self.best_score = score.item()
                    self.best_formula = formula
                    tqdm.write(f"[!] New King: Score {score:.2f} | Ret {ret_val:.2%} | Formula {formula} | Corr {corr} | Trades {trade_count}")
                if abs(corr) > self.best_corr:
                    self.best_corr = abs(corr)
                    tqdm.write(f"[!] New Corr King: Score {score:.2f} | Ret {ret_val:.2%} | Formula {formula} | Corr {corr} | Trades {trade_count}")

            rewards = torch.nan_to_num(rewards, nan=-5.0)

            # 构造 PPO 所需张量
            old_log_probs = torch.stack(log_probs_old, dim=1)   # [B, L]
            old_values = torch.stack(values_old, dim=1)         # [B, L]
            returns_ppo = rewards.unsqueeze(1).expand(-1, L)     # 终端 reward，每步相同
            advantages = returns_ppo - old_values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            full_idx = torch.cat([torch.zeros(bs, 1, dtype=torch.long, device=ModelConfig.DEVICE), seqs], dim=1)  # [B, L+1]

            # PPO 多轮更新
            total_policy_loss = 0.0
            total_value_loss = 0.0
            total_entropy = 0.0
            n_ppo = 0
            for _ in range(ModelConfig.PPO_EPOCHS):
                new_logits, new_values = self.model.forward_sequence(full_idx)  # [B, L, V], [B, L]
                new_log_probs = torch.log_softmax(new_logits, dim=-1).gather(2, seqs.unsqueeze(-1)).squeeze(-1)  # [B, L]
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
            postfix_dict = {'AvgRew': f"{avg_reward:.3f}", 'BestScore': f"{self.best_score:.3f}", 'BestCorr': f"{self.best_corr:.4f}"}
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