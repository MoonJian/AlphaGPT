"""
中国期货市场因子引擎配置
基于 Tushare Pro 数据源，适配国内期货交易特性
"""
import torch
import os

class ModelConfig:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 数据源：仅 Tushare Pro
    TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN", "")
    
    # 训练参数
    BATCH_SIZE = 3072
    TRAIN_STEPS = 10000
    MAX_FORMULA_LEN = 12
    OP_ROLL_WINDOW = 20  # 期货日频，窗口适配日线
    
    # 期货交易参数
    ROLLOVER_COST_TICKS = 0.5  # 主力切换换仓成本（跳）
    DELIVERY_MONTH_AVOID_DAYS = 5  # 交割月前N日切换至次主力
    SMOOTH_TRANSITION_DAYS = 3  # 切换前N日启动平滑过渡
    
    # 中国期货交易时段（分钟，用于识别跨夜跳空）
    DAY_SESSION_START = 9 * 60  # 09:00
    DAY_SESSION_END = 15 * 60   # 15:00
    NIGHT_SESSION_START = 21 * 60  # 21:00
    NIGHT_SESSION_END = 23 * 60   # 23:00 (部分品种至23:30或02:30)
    
    # 回测
    BASE_FEE = 0.0003  # 期货手续费率
    IMPACT_SLIPPAGE = 0.0001
    PROFIT_TARGET = 0.005   # 盈利目标 0.5% 平仓
    STOP_LOSS = -0.003      # 最大亏损 -0.3% 平仓
    MAX_HOLD_BARS = 5       # 未达盈亏目标时最大持仓 K bar
    
    # PPO
    PPO_CLIP_EPS = 0.2
    PPO_VALUE_COEF = 0.5
    PPO_ENTROPY_COEF = 0.01
    PPO_EPOCHS = 4
    GAMMA = 1.0
    GAE_LAM = 0.95
    REWARD_SCORE_WEIGHT = 1.0
    REWARD_CORR_WEIGHT = 10.0
    
    # 因子特征维度（5个期货特有因子）
    INPUT_DIM = 5
