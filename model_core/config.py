import torch
import os

class ModelConfig:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DB_URL = f"postgresql://{os.getenv('DB_USER','postgres')}:{os.getenv('DB_PASSWORD','password')}@{os.getenv('DB_HOST','localhost')}:5432/{os.getenv('DB_NAME','crypto_quant')}"
    BATCH_SIZE = 1280
    TRAIN_STEPS = 10000
    MAX_FORMULA_LEN = 10
    TRADE_SIZE_USD = 1000.0 
    MIN_LIQUIDITY = 5000.0 # 低于此流动性视为归零/无法交易
    BASE_FEE = 0.0005 # 基础费率 0.05% (Swap + Gas + Jito Tip)
    INPUT_DIM = 26
    OP_ROLL_WINDOW = 360

    # Policy Gradient (REINFORCE + baseline，替代 PPO)
    VALUE_COEF = 0.5      # value loss 权重
    ENTROPY_COEF = 0.01   # 熵正则，鼓励探索
    REWARD_SCORE_WEIGHT = 1.0
    REWARD_CORR_WEIGHT = 10.0  # reward = score * w1 + |corr| * w2

    # Value Loss 与 Return 优化（解决尺度失衡与时间信用分配）
    GAMMA = 0.99          # 折扣因子：G_t = γ^(L-1-t) * R，早期步的 target 更小，合理分配信用
    REWARD_NORM_MOMENTUM = 0.99  # reward 归一化的 EMA 动量
    VALUE_LOSS_TYPE = 'huber'  # 'mse' | 'huber'，Huber 对异常值更鲁棒
    HUBER_DELTA = 1.0     # Huber loss 的 delta 参数

    # 回测持仓模式: 'continuous' 连续持仓(TP/SL/反转/超时), 'discrete' 不连续持仓(每bar独立)
    POSITION_MODE = 'discrete'