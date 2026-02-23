import torch
from .ops import _op_gate, _op_jump, _op_decay, _ts_delay, _op_tanh, _op_ts_zscore_rolling, _op_rolling_mean, OPS_CONFIG, OPS_NORM_CONFIG
from .factors import FeatureEngineer, KlinesFeatureEngineer

class JITFormulaCompiler:
    def __init__(self):
        self.feat_offset = KlinesFeatureEngineer.INPUT_DIM
        # 建立索引与算子名称、参数数量的映射
        self.op_names = [cfg[0] for cfg in OPS_CONFIG] + [cfg[0] for cfg in OPS_NORM_CONFIG]
        self.arities = [cfg[2] for cfg in OPS_CONFIG] + [cfg[2] for cfg in OPS_NORM_CONFIG]
        
        # 建立命名空间，让编译后的代码能找到这些自定义算子
        self.context = {
            'torch': torch,
            '_op_gate': _op_gate,
            '_op_jump': _op_jump,
            '_op_decay': _op_decay,
            '_ts_delay': _ts_delay,
            '_op_tanh': _op_tanh,
            '_op_ts_zscore_rolling': _op_ts_zscore_rolling,
            '_op_rolling_mean': _op_rolling_mean
        }
        
        # 将 OPS_CONFIG 中的 lambda 和函数映射到字符串表达式
        self.op_to_str = {
            'INDENTIFY': "x0",
            'ADD': "(x0 + x1)",
            'SUB': "(x0 - x1)",
            'MUL': "(x0 * x1)",
            'DIV': "(x0 / (x1 + 1e-6))",
            'NEG': "(-x0)",
            'ABS': "torch.abs(x0)",
            'SIGN': "torch.sign(x0)",
            'GATE': "_op_gate(x0, x1, x2)",
            'JUMP': "_op_jump(x0)",
            'DECAY3': "_op_decay(x0, 3)",
            'DECAY5': "_op_decay(x0, 5)",
            'DECAY10': "_op_decay(x0, 10)",
            'DELAY1': "_ts_delay(x0, 1)",
            'DELAY3': "_ts_delay(x0, 3)",
            'DELAY5': "_ts_delay(x0, 5)",
            'MAX3': "torch.max(torch.stack([x0, _ts_delay(x0, 1), _ts_delay(x0, 2)]), dim=0)[0]",
            'MAX5': "torch.max(torch.stack([x0, _ts_delay(x0, 1), _ts_delay(x0, 2), _ts_delay(x0, 3), _ts_delay(x0, 4)]), dim=0)[0]",
            'MAX10': "torch.max(torch.stack([x0, _ts_delay(x0, 1), _ts_delay(x0, 2), _ts_delay(x0, 3), _ts_delay(x0, 4), _ts_delay(x0, 5), _ts_delay(x0, 6), _ts_delay(x0, 7), _ts_delay(x0, 8), _ts_delay(x0, 9), _ts_delay(x0, 10)]), dim=0)[0]",
            'MIN3': "torch.min(torch.stack([x0, _ts_delay(x0, 1), _ts_delay(x0, 2)]), dim=0)[0]",
            'MIN5': "torch.min(torch.stack([x0, _ts_delay(x0, 1), _ts_delay(x0, 2), _ts_delay(x0, 3), _ts_delay(x0, 4)]), dim=0)[0]",
            'MIN10': "torch.min(torch.stack([x0, _ts_delay(x0, 1), _ts_delay(x0, 2), _ts_delay(x0, 3), _ts_delay(x0, 4), _ts_delay(x0, 5), _ts_delay(x0, 6), _ts_delay(x0, 7), _ts_delay(x0, 8), _ts_delay(x0, 9), _ts_delay(x0, 10)]), dim=0)[0]",
            'ROLL_MEAN_5': "_op_rolling_mean(x0, 5)",
            'ROLL_MEAN_15': "_op_rolling_mean(x0, 15)",
            'ROLL_MEAN_30': "_op_rolling_mean(x0, 30)",
            'TANH': "_op_tanh(x0)",
            # 'ZSCORE_ROLL': "_op_ts_zscore_rolling(x0)"
        }

    def get_op_name(self, token):
        return self.op_names[token-self.feat_offset]

    def compile(self, formula_tokens):
        stack = []
        tokens = formula_tokens.tolist() if isinstance(formula_tokens, torch.Tensor) else formula_tokens
        try:
            for t in tokens:
                t = int(t)
                if t < self.feat_offset:
                    # 引用特征矩阵的特定列
                    stack.append(f"feat_tensor[:, {t}]")
                else:
                    op_idx = t - self.feat_offset
                    op_name = self.op_names[op_idx]
                    arity = self.arities[op_idx]
                    
                    # 弹出对应数量的参数
                    args = [stack.pop() for _ in range(arity)][::-1]
                    
                    # 取出对应的表达式模板并替换参数
                    expr = self.op_to_str[op_name]
                    for i in range(arity):
                        expr = expr.replace(f"x{i}", args[i])
                    
                    # 将合并后的表达式压回栈
                    stack.append(f"({expr})")
            
            if len(stack) != 1: return None
            
            # 构建完整的 Python 函数源码
            func_name = f"factor_gen_{id(formula_tokens)}"
            source_code = f"def {func_name}(feat_tensor):\n    return {stack[0]}"
            
            # 动态执行定义
            local_vars = {}
            exec(source_code, self.context, local_vars)
            py_func = local_vars[func_name]
            
            # 关键步骤：使用 TorchScript 编译，实现算子融合（Operator Fusion）
            # 这会将冗长的表达式转换成高效的计算图
            try:
                jit_func = torch.jit.script(py_func)
            except:
                # print('JIT compile failed')
                return py_func
            
            return jit_func

        except Exception as e:
            print(f"Compile Error: {e}")
            return None