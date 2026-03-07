import torch

def check_tensor_nan(x, name="Tensor", verbose=True, raise_error=False):
    """
    检查 PyTorch Tensor 中的 NaN、Inf 以及数值状态。
    
    Args:
        x (torch.Tensor): 待检查的张量
        name (str): 张量的名称，便于调试输出
        verbose (bool): 是否打印详细统计信息
        raise_error (bool): 发现 NaN 时是否直接抛出 RuntimeError
    
    Returns:
        bool: 如果包含 NaN 返回 True，否则返回 False
    """
    if not isinstance(x, torch.Tensor):
        return False
    
    # 1. 基础布尔检查
    has_nan = torch.isnan(x).any().item()
    has_inf = torch.isinf(x).any().item()
    
    if has_nan or has_inf:
        # 2. 统计详细数值
        nan_count = torch.isnan(x).sum().item()
        inf_count = torch.isinf(x).sum().item()
        total_elements = x.numel()
        
        # 3. 定位第一个错误位置 (以 RPN 序列为例，通常能看出是哪个 Token 出错)
        first_nan_idx = None
        if has_nan:
            indices = torch.isnan(x).nonzero(as_tuple=False)
            first_nan_idx = indices[0].tolist() if len(indices) > 0 else "N/A"

        msg = f"\n[!!! {name} 数值异常报告 !!!]"
        msg += f"\n- NaN 数量: {nan_count} ({nan_count/total_elements:.2%})"
        msg += f"\n- Inf 数量: {inf_count} ({inf_count/total_elements:.2%})"
        msg += f"\n- 首次出现 NaN 索引: {first_nan_idx}"
        msg += f"\n- 张量统计量: Mean={x.mean().item():.4f}, Std={x.std().item():.4f}"
        
        if verbose:
            print(msg)
            pass
            
        if raise_error:
            raise RuntimeError(f"数值崩溃: {name} 包含 NaN/Inf")
            
        return True
    
    return False

if __name__ == "__main__":
    # --- 使用示例 ---
    # 模拟一个 RPN Logits 张量
    logits = torch.randn(2, 10)
    logits[0, 5] = float('nan')

    check_tensor_nan(logits, "RPN_Logits_Head")