import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import ModelConfig
from .ops import OPS_CONFIG, OPS_NORM_CONFIG


class NewtonSchulzLowRankDecay:
    """
    Low-Rank Decay (LoRD) using Newton-Schulz iteration.
    
    A more efficient regularization method that targets low-rank structure
    in attention and key parameters. Uses Newton-Schulz iteration to compute
    the minimum singular vectors without explicit SVD.
    
    Args:
        named_parameters: Model's named parameters
        decay_rate: Strength of low-rank decay
        num_iterations: Number of Newton-Schulz iterations (default: 5)
        target_keywords: If specified, only decay parameters matching these keywords
    """
    def __init__(self, named_parameters, decay_rate=1e-3, num_iterations=5, target_keywords=None):
        self.decay_rate = decay_rate
        self.num_iterations = num_iterations
        self.target_keywords = target_keywords or ["qk_norm", "attention"]
        self.params_to_decay = []
        
        for name, param in named_parameters:
            if not param.requires_grad or param.ndim != 2:
                continue
            if not any(k in name for k in self.target_keywords):
                continue
            self.params_to_decay.append((name, param))
    
    @torch.no_grad()
    def step(self):
        """Apply Newton-Schulz low-rank decay to attention parameters."""
        for name, W in self.params_to_decay:
            orig_dtype = W.dtype
            X = W.float()
            r, c = X.shape
            
            # Transpose if needed for efficiency
            transposed = False
            if r > c:
                X = X.T
                transposed = True
            
            # Normalize by spectral norm
            norm = X.norm() + 1e-8
            X = X / norm
            
            # Initialize Y for Newton-Schulz iteration
            Y = X
            I = torch.eye(X.shape[-1], device=X.device, dtype=X.dtype)
            
            # Newton-Schulz iteration: Y_{k+1} = 0.5 * Y_k * (3*I - Y_k^T * Y_k)
            # This converges to the orthogonal matrix with same singular vectors
            for _ in range(self.num_iterations):
                A = Y.T @ Y
                Y = 0.5 * Y @ (3.0 * I - A)
            
            if transposed:
                Y = Y.T
            
            # Apply low-rank decay
            W.sub_(self.decay_rate * Y.to(orig_dtype))


class StableRankMonitor:
    """Monitor the effective rank (stable rank) of model parameters."""
    def __init__(self, model, target_keywords=None):
        self.model = model
        self.target_keywords = target_keywords or ["q_proj", "k_proj", "attention"]
        self.history = []
    
    @torch.no_grad()
    def compute(self):
        """Compute average stable rank of target parameters."""
        ranks = []
        for name, param in self.model.named_parameters():
            if param.ndim != 2:
                continue
            if not any(k in name for k in self.target_keywords):
                continue
            
            W = param.detach().float()
            S = torch.linalg.svdvals(W)
            # Stable Rank = ||W||_F^2 / ||W||_2^2
            stable_rank = (S.norm() ** 2) / (S[0] ** 2 + 1e-9)
            ranks.append(stable_rank.item())
        
        avg_rank = sum(ranks) / len(ranks) if ranks else 0.0
        self.history.append(avg_rank)
        return avg_rank


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class QKNorm(nn.Module):
    """Query-Key Normalization for Attention"""
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(1, 1, 1, d_model) * (d_model ** -0.5))
    
    def forward(self, q, k):
        # Normalize Q and K independently
        q_norm = F.normalize(q, p=2, dim=-1)
        k_norm = F.normalize(k, p=2, dim=-1)
        return q_norm * self.scale, k_norm * self.scale


class SwiGLU(nn.Module):
    """Swish GLU activation function"""
    def __init__(self, d_in, d_ff):
        super().__init__()
        self.w = nn.Linear(d_in, d_ff * 2)
        self.fc = nn.Linear(d_ff, d_in)
    
    def forward(self, x):
        x_glu = self.w(x)
        x, gate = x_glu.chunk(2, dim=-1)
        x = x * F.silu(gate)  # Swish activation
        return self.fc(x)


class MTPHead(nn.Module):
    """Multi-Task Pooling Head for multi-objective learning"""
    def __init__(self, d_model, vocab_size, num_tasks=3):
        super().__init__()
        self.num_tasks = num_tasks
        self.task_heads = nn.ModuleList([
            nn.Linear(d_model, vocab_size) for _ in range(num_tasks)
        ])
        self.task_weights = nn.Parameter(torch.ones(num_tasks) / num_tasks)
        self.task_router = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_tasks)
        )
    
    def forward(self, x):
        # Route to appropriate task heads
        task_logits = self.task_router(x)
        task_probs = F.softmax(task_logits, dim=-1)
        
        # Compute all task outputs
        task_outputs = [head(x) for head in self.task_heads]
        task_outputs = torch.stack(task_outputs, dim=1)  # [B, num_tasks, vocab_size]
        
        # Weighted combination
        weighted = (task_probs.unsqueeze(-1) * task_outputs).sum(dim=1)
        return weighted, task_probs


class LoopedTransformerLayer(nn.Module):
    """Looped Transformer Layer - recurrent processing within a layer"""
    def __init__(self, d_model, nhead, dim_feedforward, num_loops=3, dropout=0.1):
        super().__init__()
        self.num_loops = num_loops
        self.d_model = d_model
        self.nhead = nhead
        
        # QK-Norm attention
        self.qk_norm = QKNorm(d_model // nhead)
        
        # Standard attention components
        self.attention = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        
        # RMSNorm instead of LayerNorm
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        
        # SwiGLU FFN instead of standard FFN
        self.ffn = SwiGLU(d_model, dim_feedforward)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None, is_causal=False):
        # Looped processing - recurrent refinement
        for _ in range(self.num_loops):
            # Self-attention with residual
            x_norm = self.norm1(x)
            attn_out, _ = self.attention(x_norm, x_norm, x_norm, attn_mask=mask, is_causal=is_causal)
            x = x + self.dropout(attn_out)
            
            # FFN with residual
            x_norm = self.norm2(x)
            ffn_out = self.ffn(x_norm)
            x = x + self.dropout(ffn_out)
        
        return x


class LoopedTransformer(nn.Module):
    """Looped Transformer Encoder with multiple loop iterations"""
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, num_loops=3, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            LoopedTransformerLayer(d_model, nhead, dim_feedforward, num_loops, dropout)
            for _ in range(num_layers)
        ])
    
    def forward(self, x, mask=None, is_causal=False):
        for layer in self.layers:
            x = layer(x, mask=mask, is_causal=is_causal)
        return x


class AlphaGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = 64
        # self.features_list = ['RET', 'VOL', 'V_CHG', 'PV', 'TREND']
        self.features_list = ['RET', 'PRESS', 'FOMO', 'DEV', 'VOL', 'BUY_VOL', 'CONST_1', 'CONST_E', 'CONST_10', 'CONST_100']
        self.ops_list = [cfg[0] for cfg in OPS_CONFIG] 
        self.ops_norm_list = [cfg[0] for cfg in OPS_NORM_CONFIG]
        self.ops_arities = [cfg[2] for cfg in OPS_CONFIG] 
        self.ops_norm_arities = [cfg[2] for cfg in OPS_NORM_CONFIG]
        self.ops_all_list = self.ops_list + self.ops_norm_list
        self.ops_all_arities = self.ops_arities + self.ops_norm_arities
        
        self.vocab = self.features_list + self.ops_list + self.ops_norm_list
        self.vocab_size = len(self.vocab)
        self.feat_offset = len(self.features_list)
        self.feat_ops_size = len(self.features_list + self.ops_list)
        self.ops_norm_size = len(self.ops_norm_list)
        
        # Embedding
        self.token_emb = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, ModelConfig.MAX_FORMULA_LEN + 1, self.d_model))
        
        # Enhanced Transformer with Looped Transformer
        self.blocks = LoopedTransformer(
            d_model=self.d_model,
            nhead=4,
            num_layers=2,
            dim_feedforward=128,
            num_loops=3,
            dropout=0.1
        )
        
        # RMSNorm instead of LayerNorm
        self.ln_f = RMSNorm(self.d_model)
        
        # MTPHead for multi-task output
        self.mtp_head = MTPHead(self.d_model, self.vocab_size, num_tasks=3)
        self.head_critic = nn.Linear(self.d_model, 1)

        # self._build_valid_masks()
        
        self.max_len = ModelConfig.MAX_FORMULA_LEN
        self.max_stack = ModelConfig.MAX_FORMULA_LEN
        self.max_arity = max(self.ops_arities)
        self._precompute_valid_mask()

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def translate_to_exprs(self, rpn_id_list):
        """
        高级中序转换器：支持优先级优化与金融语义映射
        """
        # 1. 金融语义映射
        semantic_map = {
            'CONST_1': '1', 'CONST_E': 'e', 'CONST_10': '10', 'CONST_100': '100',
            'ADD': '+', 'SUB': '-', 'MUL': '*', 'DIV': '/'
        }

        # 2. 运算符优先级（仅用于中缀操作符）
        precedence = {'+': 1, '-': 1, '*': 2, '/': 2}
        
        # 3. 左结合操作符（减法/除法需特殊处理）
        left_associative = {'-', '/'}

        stack = []  # (expression_string, current_precedence)
        
        if isinstance(rpn_id_list, torch.Tensor):
            rpn_id_list = rpn_id_list.tolist()

        for idx in rpn_id_list:
            if idx < self.feat_offset:
                # --- 处理特征/常量 ---
                raw_symbol = self.features_list[idx]
                symbol = semantic_map.get(raw_symbol, raw_symbol)
                stack.append((symbol, 10))  # 原子优先级设为10（足够高）
                
            else:
                # --- 处理操作符 ---
                op_idx = idx - self.feat_offset
                op_name = self.ops_all_list[op_idx]
                symbol = semantic_map.get(op_name, op_name)
                arity = self.ops_all_arities[op_idx]
                
                try:
                    if arity == 1:
                        arg_str, arg_prec = stack.pop()
                        
                        # 特殊处理负号（前缀操作符）
                        if symbol == '-':
                            # 负号优先级=3，高于+/-，低于*/
                            current_prec = 3
                            # 如果参数优先级 <= 负号，需要括号: -(a+b)
                            expr = f"-({arg_str})" if arg_prec <= current_prec else f"-{arg_str}"
                            stack.append((expr, current_prec))
                        else:
                            # 其他一元操作符：函数调用格式
                            stack.append((f"{symbol}({arg_str})", 10))
                    
                    elif arity == 2:
                        right_str, right_prec = stack.pop()
                        left_str, left_prec = stack.pop()
                        
                        if symbol in precedence:
                            current_prec = precedence[symbol]
                            
                            # 左操作数：如果优先级 < 当前，加括号
                            l_out = f"({left_str})" if left_prec < current_prec else left_str
                            
                            # 右操作数：更复杂！
                            # 情况1: 优先级 < 当前 → 必须加括号
                            # 情况2: 优先级 == 当前 且 左结合 → 加括号（如 a - (b - c) 是错的，但 a - b - c 正确）
                            if (right_prec < current_prec or 
                                (right_prec == current_prec and symbol in left_associative)):
                                r_out = f"({right_str})"
                            else:
                                r_out = right_str
                            
                            stack.append((f"{l_out} {symbol} {r_out}", current_prec))
                        else:
                            # 非中缀操作符（如 POW）→ 函数调用
                            stack.append((f"{symbol}({left_str}, {right_str})", 10))
                    
                    elif arity == 3:
                        # 三元操作符：如 GATE(condition, x, y)
                        third_str, _ = stack.pop()  # y
                        second_str, _ = stack.pop() # x  
                        first_str, _ = stack.pop()  # condition
                        stack.append((f"{symbol}({first_str}, {second_str}, {third_str})", 10))
                    
                    else:
                        # 通用多参数处理（安全兜底）
                        args = []
                        for _ in range(arity):
                            arg, _ = stack.pop()
                            args.append(arg)
                        args.reverse()  # RPN弹出顺序是反的
                        stack.append((f"{symbol}({', '.join(args)})", 10))
                        
                except IndexError:
                    return "Malformed RPN"

        if len(stack) != 1:
            return "Malformed RPN"
        
        result_str, _ = stack[0]
        return result_str

    def _precompute_valid_mask(self):
        """
        预计算所有(stack_size, remaining_steps, token)组合的合法性
        形状: [max_stack+1, max_len+1, vocab_size]
        """
        mask_3d = torch.zeros((self.max_stack + 1, self.max_len + 1, self.vocab_size), dtype=torch.bool)
        for s in range(self.max_stack + 1):          # 当前栈深度
            for r in range(1, self.max_len + 1):     # 剩余步数 (≥1)
                for token_id in range(self.vocab_size):
                    valid = False
                    
                    if token_id < self.feat_offset:  # 特征加载
                        # 最后一步不能是特征（否则栈深度>1）
                        if r == 1:
                            valid = False
                        else:
                            # 条件: s ≤ (r-1) × (max_arity - 1)
                            max_reducible = (r - 1) * (self.max_arity - 1)
                            valid = (s <= max_reducible)
                    
                    elif token_id < self.feat_ops_size:  # 操作符
                        op_idx = token_id - self.feat_offset
                        a = self.ops_arities[op_idx]
                        
                        # 基本合法性: 栈深度 ≥ 操作数
                        if s < a:
                            valid = False
                        else:
                            new_s = s - a + 1  # 操作后栈深度
                            
                            if r == 1:  # 最后一步
                                valid = (new_s == 1)  # 必须恰好剩1个元素
                            else:
                                # 条件: new_s - 1 ≤ (r-1) × (max_arity - 1)
                                max_reducible = (r - 1) * (self.max_arity - 1)
                                valid = (new_s - 1 <= max_reducible)
                    
                    mask_3d[s, r, token_id] = valid
        
        mask_3d[1, 0, -self.ops_norm_size:] = True # 最后ops norm的范围
        self.register_buffer('valid_mask_3d', mask_3d)  # 注册为buffer，随模型移动设备        

    def compute_stack_size(self, stack_sizes, actions):        
        feat_indices = torch.where(actions < len(self.features_list))
        stack_sizes[feat_indices] += 1
        for i in range(len(self.features_list), self.vocab_size):            
            indices = torch.where(actions == i)            
            stack_sizes[indices] -= self.ops_all_arities[i-len(self.features_list)] - 1
        return stack_sizes

    def forward(self, idx, stack_sizes):
        # idx: [Batch, SeqLen]
        B, T = idx.size()
        r = ModelConfig.MAX_FORMULA_LEN - T + 1
        
        x = self.token_emb(idx) + self.pos_emb[:, :T, :]
        
        # Causal Mask
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(idx.device)
        
        # Process through looped transformer
        x = self.blocks(x, mask=mask, is_causal=True)
        x = self.ln_f(x)
        
        last_emb = x[:, -1, :]
        
        # Multi-task pooling head for logits
        logits, task_probs = self.mtp_head(last_emb)
        value = self.head_critic(last_emb)

        '''
        masked_logits = torch.zeros_like(logits)
        for i in range(B):
            stack = stack_sizes[i].item()
            if stack not in self.valid_masks:
                raise Exception(f"Stack size {stack} not found in valid stack mask!")
            mask = self.valid_masks[stack].to(logits.device)
            masked_logits[i] = logits[i].masked_fill(mask==0, -1e9)
        '''

        batch_mask = self.valid_mask_3d[stack_sizes, r, :]
        masked_logits = logits.masked_fill(~batch_mask, -1e9)

        if torch.isnan(logits).any():
            print('idx0: ', idx[0])
            print(f'stack sizes: ', stack_sizes[0])
            print('token 0 embed: ', self.token_emb.weight[0])
            print('pos emb: ', self.token_emb.weight[0])
            print('token embed: ', self.token_emb(idx)[0])
            print('pos embed: ', self.pos_emb[:, :T, :][0])
            print('last embs: ', last_emb)
            print(logits)
            assert 0

        return masked_logits, value, task_probs