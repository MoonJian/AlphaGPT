"""
Tushare Pro 期货数据提供层
唯一数据源，处理主力合约映射、连续合约构建、涨跌停过滤、交割月规避
参考: https://tushare.pro/document/2
"""
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta
import warnings

try:
    import tushare as ts
except ImportError:
    ts = None


# 有夜盘品种（简化列表，实际可扩展）
NIGHT_SESSION_SYMBOLS = {
    'RB', 'HC', 'I', 'J', 'JM', 'FG', 'MA', 'TA', 'SA', 'UR',  # 黑色/化工
    'CU', 'AL', 'ZN', 'PB', 'NI', 'SN', 'AU', 'AG',  # 有色/贵金属
    'RU', 'BU', 'SP', 'SC', 'LU', 'NR',  # 能源化工
    'P', 'Y', 'OI', 'M', 'A', 'C', 'CS', 'JD', 'RR', 'L', 'V', 'PP', 'EB', 'EG', 'PF',  # 农产品/化工
    'IF', 'IC', 'IH', 'IM',  # 股指期货
}


def _chunk_dates(start: str, end: str, chunk_days: int = 180) -> List[Tuple[str, str]]:
    """按日期分块，避免单次请求超限"""
    from datetime import datetime
    s = datetime.strptime(start, '%Y%m%d')
    e = datetime.strptime(end, '%Y%m%d')
    chunks = []
    while s < e:
        end_chunk = min(s + timedelta(days=chunk_days), e)
        chunks.append((s.strftime('%Y%m%d'), end_chunk.strftime('%Y%m%d')))
        s = end_chunk + timedelta(days=1)
    return chunks


class TushareFuturesProvider:
    """
    Tushare Pro 期货数据提供者
    - fut_mapping: 主力合约映射
    - fut_daily: 日线行情（含持仓量 oi）
    - fut_basic: 合约信息（交割日等）
    - fut_settle: 结算参数（可选，用于涨跌停幅度）
    """
    
    def __init__(self, token: str):
        if ts is None:
            raise ImportError("请安装 tushare: pip install tushare")
        self.pro = ts.pro_api(token)
    
    def get_trade_calendar(self, start_date: str, end_date: str) -> pd.DataFrame:
        """获取期货交易日历"""
        df = self.pro.trade_cal(
            exchange='SSE',  # 用股票日历近似，期货有独立接口可换
            start_date=start_date,
            end_date=end_date,
            is_open=1
        )
        if df is None or df.empty:
            return pd.DataFrame(columns=['cal_date'])
        return df[['cal_date']].rename(columns={'cal_date': 'trade_date'})
    
    def get_fut_mapping(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取主力/连续合约与月合约映射
        ts_code: 如 RB.SHF, IF.CFX
        返回: trade_date, mapping_ts_code
        """
        all_dfs = []
        for s, e in _chunk_dates(start_date, end_date, 180):
            df = self.pro.fut_mapping(ts_code=ts_code, start_date=s, end_date=e)
            if df is not None and not df.empty:
                all_dfs.append(df)
        if not all_dfs:
            return pd.DataFrame(columns=['trade_date', 'mapping_ts_code'])
        out = pd.concat(all_dfs, ignore_index=True).drop_duplicates()
        out = out.sort_values('trade_date').reset_index(drop=True)
        return out[['trade_date', 'mapping_ts_code']]
    
    def get_fut_daily_batch(self, ts_codes: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """批量获取期货日线（按日期分片）"""
        all_dfs = []
        for s, e in _chunk_dates(start_date, end_date, 60):
            df = self.pro.fut_daily(ts_code=','.join(ts_codes[:50]), start_date=s, end_date=e)
            if df is not None and not df.empty:
                all_dfs.append(df)
        if not all_dfs:
            return pd.DataFrame()
        return pd.concat(all_dfs, ignore_index=True).drop_duplicates(
            subset=['ts_code', 'trade_date']
        ).sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
    
    def get_fut_daily_single(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """单合约日线"""
        all_dfs = []
        for s, e in _chunk_dates(start_date, end_date, 2000):
            df = self.pro.fut_daily(ts_code=ts_code, start_date=s, end_date=e)
            if df is not None and not df.empty:
                all_dfs.append(df)
        if not all_dfs:
            return pd.DataFrame()
        return pd.concat(all_dfs, ignore_index=True).drop_duplicates().sort_values('trade_date').reset_index(drop=True)
    
    def get_fut_basic(self, exchange: str = None) -> pd.DataFrame:
        """获取合约基本信息（含 list_date, delist_date, d_month）"""
        exchanges = [exchange] if exchange else ['DCE', 'SHFE', 'CZCE', 'CFFEX', 'INE', 'GFEX']
        all_dfs = []
        for ex in exchanges:
            df = self.pro.fut_basic(exchange=ex, fut_type='1')
            if df is not None and not df.empty:
                all_dfs.append(df)
        if not all_dfs:
            return pd.DataFrame()
        return pd.concat(all_dfs, ignore_index=True)
    
    def get_second_main_mapping(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取次主力合约映射（用于期限结构、交割月规避）
        通过同品种所有合约的持仓量排序，取第二大
        """
        # 解析品种代码
        sym = ts_code.split('.')[0]
        exch = ts_code.split('.')[1] if '.' in ts_code else ''
        # 获取该品种所有月合约的日线，按 oi 排序取次高
        basic = self.get_fut_basic()
        if basic.empty:
            return pd.DataFrame()
        contracts = basic[basic['ts_code'].str.startswith(sym) & (basic['fut_type'] == '1')]['ts_code'].tolist()
        if len(contracts) < 2:
            return pd.DataFrame(columns=['trade_date', 'mapping_ts_code', 'second_main_ts_code'])
        
        # 简化：使用 fut_mapping 的主力，次主力用同品种另一活跃合约近似
        # 完整实现需每日按 oi 排序，这里用主力映射 + 次月合约逻辑
        main_df = self.get_fut_mapping(ts_code, start_date, end_date)
        if main_df.empty:
            return pd.DataFrame()
        # 次主力：取主力合约的次月（根据合约代码规则推断）
        def get_next_month(contract: str) -> str:
            import re
            parts = contract.split('.')
            code = parts[0]
            suffix = parts[1] if len(parts) > 1 else exch
            m = re.match(r'([A-Za-z]+)(\d{4})(\d{2})', code)
            if m:
                pre, yy, mm = m.group(1), int(m.group(2)), int(m.group(3))
                mm += 1
                if mm > 12:
                    mm = 1
                    yy += 1
                return f"{pre}{yy:04d}{mm:02d}.{suffix}"
            return contract
        
        main_df = main_df.copy()
        main_df['second_main_ts_code'] = main_df['mapping_ts_code'].apply(get_next_month)
        return main_df


class ContinuousContractBuilder:
    """
    连续合约构建器
    - 基于 fut_mapping 主力映射
    - 切换前 3 日持仓量加权平滑
    - 交割月前 5 日切换至次主力
    """
    
    def __init__(self, provider: TushareFuturesProvider, smooth_days: int = 3, delivery_avoid_days: int = 5):
        self.provider = provider
        self.smooth_days = smooth_days
        self.delivery_avoid_days = delivery_avoid_days
    
    def _is_near_delivery(self, ts_code: str, trade_date: str, basic_df: pd.DataFrame) -> bool:
        """是否临近交割月（前 N 日）"""
        if basic_df is None or basic_df.empty:
            return False
        row = basic_df[basic_df['ts_code'] == ts_code]
        if row.empty:
            return False
        delist = row.iloc[0].get('delist_date', '')
        if pd.isna(delist) or delist == '':
            return False
        try:
            d = datetime.strptime(str(delist)[:8], '%Y%m%d')
            t = datetime.strptime(str(trade_date)[:8], '%Y%m%d')
            return (d - t).days <= self.delivery_avoid_days
        except Exception:
            return False
    
    def build_continuous(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
        price_col: str = 'settle',
        use_smooth: bool = True
    ) -> pd.DataFrame:
        """
        构建复权连续价格序列
        ts_code: 主力/连续合约代码，如 RB.SHF
        price_col: 价格列，优先 settle（结算价）
        """
        mapping = self.provider.get_fut_mapping(ts_code, start_date, end_date)
        if mapping.empty:
            return pd.DataFrame()
        
        basic = self.provider.get_fut_basic()
        unique_contracts = mapping['mapping_ts_code'].unique().tolist()
        
        # 获取所有涉及合约的日线
        daily_list = []
        for mc in unique_contracts:
            dd = self.provider.get_fut_daily_single(mc, start_date, end_date)
            if not dd.empty:
                daily_list.append(dd)
        if not daily_list:
            return pd.DataFrame()
        
        daily = pd.concat(daily_list, ignore_index=True)
        
        # 按 (trade_date, mapping_ts_code) 对齐：mapping 的 mapping_ts_code 对应 daily 的 ts_code
        daily_renamed = daily.rename(columns={'ts_code': 'mapping_ts_code'})
        mapping = mapping.merge(
            daily_renamed[['mapping_ts_code', 'trade_date', price_col, 'oi', 'pre_settle', 'open', 'high', 'low', 'close', 'vol']],
            on=['trade_date', 'mapping_ts_code'],
            how='left'
        )
        mapping = mapping.drop_duplicates(subset=['trade_date']).sort_values('trade_date').reset_index(drop=True)
        
        # 平滑过渡：切换日前 N 日，用新旧主力持仓量加权平均价格
        mapping['contract_changed'] = mapping['mapping_ts_code'] != mapping['mapping_ts_code'].shift(1)
        
        if use_smooth and self.smooth_days > 0:
            for i in range(1, len(mapping)):
                if not mapping.iloc[i]['contract_changed']:
                    continue
                curr_main = mapping.iloc[i]['mapping_ts_code']
                prev_main = mapping.iloc[i - 1]['mapping_ts_code']
                if curr_main == prev_main:
                    continue
                # 切换发生在 i 日，对 [i-smooth_days, i-1] 做平滑（即切换日前 smooth_days 日）
                start_i = max(0, i - self.smooth_days)
                for j in range(start_i, i):
                    d = mapping.iloc[j]['trade_date']
                    old_row = daily[(daily['ts_code'] == prev_main) & (daily['trade_date'] == d)]
                    new_row = daily[(daily['ts_code'] == curr_main) & (daily['trade_date'] == d)]
                    if not old_row.empty and not new_row.empty:
                        oi_old = float(old_row['oi'].iloc[0] or 0)
                        oi_new = float(new_row['oi'].iloc[0] or 0)
                        p_old = float(old_row[price_col].iloc[0])
                        p_new = float(new_row[price_col].iloc[0])
                        total_oi = oi_old + oi_new
                        if total_oi > 0:
                            smooth_price = (p_old * oi_old + p_new * oi_new) / total_oi
                            mapping.iloc[j, mapping.columns.get_loc(price_col)] = smooth_price
        
        # 提取品种代码（如 RB）
        symbol = ts_code.split('.')[0]
        mapping['symbol'] = symbol
        return mapping.sort_values('trade_date').reset_index(drop=True)


class LimitFilter:
    """涨跌停过滤：当日涨跌幅=±涨跌停幅度时，因子值置为 null"""
    
    @staticmethod
    def detect_limit_row(row: pd.Series, limit_pct: float = 0.04) -> bool:
        """
        检测是否触及涨跌停
        期货涨跌停幅度因品种而异，默认 4%，实际应从 fut_settle 或交易所规则获取
        """
        if pd.isna(row.get('pre_settle')) or row['pre_settle'] <= 0:
            return False
        pct = (row.get('settle', row.get('close', 0)) - row['pre_settle']) / row['pre_settle']
        return abs(pct) >= (limit_pct - 1e-6)
    
    @staticmethod
    def apply_limit_mask(df: pd.DataFrame, limit_pct: float = 0.04) -> pd.Series:
        """返回布尔 Series：True 表示该日触及涨跌停应过滤"""
        if df.empty:
            return pd.Series(dtype=bool)
        return df.apply(lambda r: LimitFilter.detect_limit_row(r, limit_pct), axis=1)
