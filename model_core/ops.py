import torch

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
def _op_jump(x: torch.Tensor, window: int = 960) -> torch.Tensor:
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

    # 5. 考虑两个跳变范围并将前 window-1 个结果设为0
    res = torch.relu(z - 3.0) + torch.relu(-z - 3.0) 
    res[:, :window-1] = 0.0
    
    return res

@torch.jit.script
def _op_decay(x: torch.Tensor) -> torch.Tensor:
    return x + 0.8 * _ts_delay(x, 1) + 0.6 * _ts_delay(x, 2)

@torch.jit.script
def _op_tanh(x: torch.Tensor, window: int=960) -> torch.Tensor:
    return torch.tanh(x)

@torch.jit.script
def _op_ts_zscore_rolling(x: torch.Tensor, window: int = 960) -> torch.Tensor:
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
    # ('TANH', _op_tanh, 1),
    ('ZSCORE_ROLL', _op_ts_zscore_rolling, 1)
]
