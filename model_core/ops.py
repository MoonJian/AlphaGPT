import torch
from .config import ModelConfig

@torch.jit.script
def _ts_delay(x: torch.Tensor, d: int) -> torch.Tensor:
    if d == 0: return x
    pad = torch.zeros((x.shape[0], d), device=x.device)
    return torch.cat([pad, x[:, :-d]], dim=1)

@torch.jit.script
def _op_gate(condition: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mask = (condition > 0).float()
    return mask * x + (1.0 - mask) * y

# @torch.jit.script
# def _op_jump(x: torch.Tensor) -> torch.Tensor:
#     mean = x.mean(dim=1, keepdim=True)
#     std = x.std(dim=1, keepdim=True) + 1e-6
#     z = (x - mean) / std
#     return torch.relu(z - 3.0)

@torch.jit.script
def _op_jump(x: torch.Tensor, window: int = ModelConfig.OP_ROLL_WINDOW) -> torch.Tensor:
# 强制转换类型确保精度
    x = x.to(torch.float32)
    B, T = x.size()

    # 1. 准备填充后的序列
    x_padded = torch.nn.functional.pad(x, (1, 0), mode='constant', value=0.0)
    
    # 2. 计算前缀和 (Prefix Sum)
    csum = torch.cumsum(x_padded, dim=1)
    csum_sq = torch.cumsum(x_padded * x_padded, dim=1)

    # 3. 计算滑动窗口内的和 (利用 S = csum[t] - csum[t-window])
    # 处理起始阶段：对于 t < window，使用 expanding window 或直接补零
    # 为简化逻辑且对齐原代码，我们使用 padding 后的差分
    
    # 构造 shift 后的前缀和
    s_csum = torch.nn.functional.pad(csum, (window, 0), mode='constant', value=0.0)
    s_csum_sq = torch.nn.functional.pad(csum_sq, (window, 0), mode='constant', value=0.0)

    # 计算窗口内的 Sum 和 Sum_Squares
    # 截取对应长度使其对齐 x
    win_sum = csum[:, 1:] - s_csum[:, :T]
    win_sum_sq = csum_sq[:, 1:] - s_csum_sq[:, :T]

    # 4. 计算滚动均值和标准差
    rolling_mean = win_sum / window
    # 方差公式: E[X^2] - (E[X])^2
    rolling_var = (win_sum_sq / window) - (rolling_mean ** 2)
    rolling_std = torch.sqrt(torch.clamp(rolling_var, min=1e-8)) + 1e-6

    # 5. 计算 Z-Score 和 Jump
    z = (x - rolling_mean) / rolling_std
    
    # 双向跳变识别 (正向 3 sigma 或 负向 3 sigma)
    res = torch.relu(z - 3.0) - torch.relu(-z - 3.0)
    
    # 6. Mask 掉初始不完整窗口
    # 创建一个 mask：前 window-1 个元素设为 0
    mask = torch.ones_like(x)
    mask[:, :window-1] = 0.0
    
    return res * mask

@torch.jit.script
def _op_decay(x: torch.Tensor) -> torch.Tensor:
    return x + 0.8 * _ts_delay(x, 1) + 0.6 * _ts_delay(x, 2)

@torch.jit.script
def _op_tanh(x: torch.Tensor, window: int=ModelConfig.OP_ROLL_WINDOW) -> torch.Tensor:
    return torch.tanh(x)

@torch.jit.script
def _op_ts_zscore_rolling(x: torch.Tensor, window: int = ModelConfig.OP_ROLL_WINDOW) -> torch.Tensor:
    """
    x: [Batch, TimeSeries]
    window: 滚动的窗口大小
    """
    B, T = x.size()
    
    # 1. 对序列进行填充，保证输出长度与输入一致 (Causal Padding)
    # 在左侧填充 window-1 个值，这样第一个窗口的末尾正好对齐序列的第一个元素
    x_padded = torch.nn.functional.pad(x, (window - 1, 0), mode='constant', value=0.0)
    
    # 2. 创建滑动窗口 [B, T, window]
    # 使用 .contiguous() 预防你之前遇到的 CUDA illegal memory access
    windows = x_padded.contiguous().unfold(1, window, 1)
    
    # 3. 在窗口维度（最后一个维度）计算均值和标准差
    # 注意：这些指标只包含当前时刻及之前的信息
    rolling_mean = windows.mean(dim=-1)
    rolling_std = windows.std(dim=-1) + 1e-6
    
    # 4. 计算 Z-Score
    z = (x - rolling_mean) / rolling_std

    # 5. 考虑两个跳变范围
    res = torch.relu(z - 3.0) + torch.relu(-z - 3.0) 
    
    return res


OPS_CONFIG = [
    # ('INDENTIFY', lambda x: x, 1),
    ('ADD', lambda x, y: x + y, 2),
    ('SUB', lambda x, y: x - y, 2),
    ('MUL', lambda x, y: x * y, 2),
    ('DIV', lambda x, y: x / (y + 1e-6), 2),
    ('NEG', lambda x: -x, 1),
    ('ABS', torch.abs, 1),
    ('SIGN', torch.sign, 1),
    ('GATE', _op_gate, 3),
    ('JUMP', _op_jump, 1),
    ('DECAY', _op_decay, 1),
    ('DELAY1', lambda x: _ts_delay(x, 1), 1),
    ('MAX3', lambda x: torch.max(x, torch.max(_ts_delay(x,1), _ts_delay(x,2))), 1)
]

OPS_NORM_CONFIG = [
    ('TANH', _op_tanh, 1),
    ('ZSCORE_ROLL', _op_ts_zscore_rolling, 1)
]
