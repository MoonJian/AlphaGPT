# AlphaGPT 中国期货市场因子引擎

model_core 模块已从 Crypto 因子生成完整重构为适配中国期货市场的因子引擎。

## 核心改造

### 1. 数据层 (Tushare Pro 唯一数据源)

- **`tushare_provider.py`**: Tushare 期货数据提供层
  - `fut_mapping`: 每日主力合约映射
  - `fut_daily`: 日线行情（含持仓量 oi、结算价 settle）
  - `fut_basic`: 合约信息（交割日等）
  - `ContinuousContractBuilder`: 连续合约构建，切换前 3 日持仓量加权平滑
  - `LimitFilter`: 涨跌停检测

- **`data_loader.py`**: `FuturesDataLoader`
  - 参数: `ts_code`（如 RB.SHF）、`start_date`、`end_date`、`token`、`cache_path`
  - 交割月前 5 日自动切换次主力（通过 fut_mapping）
  - 涨跌停日因子置 null
  - 股指期货基差用 index_daily 计算

### 2. 因子层 (5 个期货特有因子)

| 因子 | 说明 |
|------|------|
| OI_MOM | 持仓量动量：过去 5 日持仓量变化率，按品种标准化 |
| TERM_SLOPE | 期限结构斜率：(主力-次主力)结算价差/主力结算价 |
| NIGHT_PREM | 夜盘溢价：夜盘收盘/次日开盘-1（无夜盘品种用日收盘近似） |
| LIMIT_HIT | 涨跌停触及强度：过去 10 日加权触及次数，触及当日置 null |
| BASIS_MOM | 基差动量：商品期货用现货价差，股指期货用指数价差 |

### 3. 回测接口 (聚宽兼容)

- 输出列: `trade_date`, `symbol`, `factor_value`, `forward_return`
- `forward_return` 含主力切换换仓成本（默认 0.5 跳）
- `AlphaEngine.export_joinquant()` 导出 parquet

### 4. 边界处理

- **涨跌停过滤**: 当日涨跌幅=±涨跌停幅度时，因子值置 null
- **交割月规避**: 主力进入交割月前 5 日切换至次主力
- **夜盘**: 日频数据用日收盘近似；分钟级需拼接 21:00-23:00 与 09:00-15:00

## 使用示例

```python
import os
os.environ["TUSHARE_TOKEN"] = "your_token"

from model_core import AlphaEngine

eng = AlphaEngine(
    ts_code='RB.SHF',      # 螺纹钢主力
    start_date='20200101',
    end_date='20241231',
    token=os.environ.get("TUSHARE_TOKEN"),
    cache_path='data_cache_rb.parquet',
)
eng.train()
df = eng.export_joinquant(output_path='factors_joinquant.parquet')
```

## 配置

- `ModelConfig.TUSHARE_TOKEN`: Tushare Pro token
- `ModelConfig.ROLLOVER_COST_TICKS`: 换仓成本（跳）
- `ModelConfig.DELIVERY_MONTH_AVOID_DAYS`: 交割月规避天数
- `ModelConfig.SMOOTH_TRANSITION_DAYS`: 平滑过渡天数

## 依赖

- tushare
- torch
- pandas
- numpy
