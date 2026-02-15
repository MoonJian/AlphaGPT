import torch
import os

class ModelConfig:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DB_URL = f"postgresql://{os.getenv('DB_USER','postgres')}:{os.getenv('DB_PASSWORD','password')}@{os.getenv('DB_HOST','localhost')}:5432/{os.getenv('DB_NAME','crypto_quant')}"
    BATCH_SIZE = 3072
    TRAIN_STEPS = 10000
    MAX_FORMULA_LEN = 12
    TRADE_SIZE_USD = 1000.0
    MIN_LIQUIDITY = 5000.0 # 低于此流动性视为归零/无法交易
    BASE_FEE = 0.0005 # 基础费率 0.5% (Swap + Gas + Jito Tip)
    INPUT_DIM = 16
    OP_ROLL_WINDOW = 360

    # PPO
    PPO_CLIP_EPS = 0.2
    PPO_VALUE_COEF = 0.5
    PPO_ENTROPY_COEF = 0.01
    PPO_EPOCHS = 4
    GAMMA = 1.0          # 仅终端 reward 时可用 1.0
    GAE_LAM = 0.95      # GAE lambda（终端 reward 时影响小）
    REWARD_SCORE_WEIGHT = 1.0
    REWARD_CORR_WEIGHT = 10.0  # reward = score * w1 + |corr| * w2