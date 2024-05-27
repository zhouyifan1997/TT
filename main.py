import pandas as pd
from datetime import datetime
from typing import (
    NamedTuple,
    Optional,
    List,
)
from technical_utils import *

from technical_utils import *

class Trade(NamedTuple):
    time: datetime
    action: str
    price: float
    quantity: int
    max_profit: Optional[float]=None
    actual_profit: Optional[float]=None

class PerformanceStats(NamedTuple):
    start_year: int
    end_year: int
    total_trade: int
    win_rate: float
    risk_reward_ratio: float
    max_drawdown: float
    net_profit: float
    average_profit_yearly: Optional[float]=None
    average_drawdown_yearly: Optional[float]=None
    max_profit_yearly: Optional[float]=None
    max_drawdown_yearly: Optional[float]=None

    @classmethod
    def create_empty(self, start_year: int, end_year: int) -> 'PerformanceStats':
        return PerformanceStats(start_year, end_year, 0, 0, 0, 0, 0)

    def show_performance_summary(self) -> None:
        if self.start_year == self.end_year:
            print(
            """
            Range: {}/01/01 - {}/12/31\n\
            Total trade: {}\n\
            Win rate: {}%\n\
            Risk reward ratio: {}\n\
            Max drawdown: {}%\n\
            Net profit: {}%\n
            """.format(
                self.start_year,
                self.end_year,
                self.total_trade,
                round(self.win_rate * 100, 2),
                round(self.risk_reward_ratio, 2),
                round(self.max_drawdown * 100, 2),
                round(self.net_profit * 100, 2),
            ))
        else:
            assert self.average_profit_yearly
            assert self.average_drawdown_yearly
            assert self.max_profit_yearly
            assert self.max_drawdown_yearly
            print(
            """
            Range: {}/01/01 - {}/12/31\n\
            Total trade: {}\n\
            Win rate: {}%\n\
            Risk reward ratio: {}\n\
            Max drawdown: {}%\n\
            Net profit: {}%\n\
            Average profit yearly: {}%\n\
            Average drawdown yearly: {}%\n\
            Max profit yearly: {}%\n\
            Max drawdown yearly: {}%\n\
            """.format(
                self.start_year,
                self.end_year,
                self.total_trade,
                round(self.win_rate * 100, 2),
                round(self.risk_reward_ratio, 2),
                round(self.max_drawdown * 100, 2),
                round(self.net_profit * 100, 2),
                round(self.average_profit_yearly * 100, 2),
                round(self.average_drawdown_yearly * 100, 2),
                round(self.max_profit_yearly * 100, 2),
                round(self.max_drawdown_yearly * 100, 2),
            ))

class Account:
    def __init__(self) -> None:
        self.initial_balance: float = 50000
        self.realized_profit: float = 0
        self.open_profit: float = 0

        self.available_balance: float = 50000

        # Position data
        self.state = State()

        # Summary data
        self.values: List[float] = []

    @property
    def account_value(self) -> float:
        return self.initial_balance + self.realized_profit + self.open_profit
    
    @property
    def position(self) -> int:
        return self.state.position
    
    @property
    def trades(self) -> List[Trade]:
        return self.state.get_trades()
    
    def get_buying_power(self) -> float:
        return self.available_balance
    
    def process_bar(self, price: float) -> None:
        self.open_profit = self.state.get_pnl(price)
        self.values.append(self.account_value)

    def long(self, time: datetime, cost: float, quantity: int) -> None:
        self.state.long(time, cost, quantity)

    def short(self, time: datetime, cost: float, quantity: int) -> None:
        self.state.short(time, cost, quantity)

    def close(self, time: datetime, price: float) -> float:
        self.state.close(time, price)

        pnl = self.open_profit
        self.open_profit = 0
        self.realized_profit += pnl

        if self.account_value > self.available_balance * 1.2:
            self.available_balance *= 1.2
        elif self.account_value < self.available_balance * 0.9:
            self.available_balance *= 0.9
        return pnl


class State:
    def __init__(self):
        self.position: int = 0
        self.cost: float = 0
        self.max_profit: float = 0
        self.trades: List[Trade] = []
        
    def long(self, time: datetime, cost: float, count=1) -> None:
        self.position = count
        self.cost = cost
        self.trades.append(Trade(time, 'Long', cost, count))
        
    def short(self, time: datetime, cost: float, count=1) -> None:
        self.position = -count
        self.cost = cost
        self.trades.append(Trade(time, 'Short', cost, count))
        
    def close(self, time: datetime, price: float) -> float:
        # Returns pnl
        self.trades.append(
            Trade(
                time,
                'Close Long' if self.position > 0 else 'Close Short',
                price,
                self.position,
                self.max_profit,
                self.get_pnl(price),
            )
        )
        pnl = self.get_pnl(price)
        self.position = 0
        self.cost = 0
        self.max_profit = 0
        return pnl
    
    def get_pnl(self, price: float) -> float:
        if self.position > 0:
            return (price - self.cost) * self.position
        elif self.position < 0:
            return (price - self.cost) * self.position
        return 0
    
    def get_trades(self) -> List[Trade]:
        return self.trades

    def update_max_profit(self, price: float) -> None:
        pnl = self.get_pnl(price)
        self.max_profit = max(self.max_profit, pnl)

    def get_max_profit(self) -> float:
        return self.max_profit
    

def calculate_metrics(pnls: List[float], account_values: List[float], start: int, end: int) -> PerformanceStats:
    if len(pnls) == 0:
        return PerformanceStats.create_empty(start, end)
    total_trades = len(pnls)
    winning_trades = sum(1 for pnl in pnls if pnl > 0)
    losing_trades = sum(1 for pnl in pnls if pnl < 0)

    win_rate = winning_trades / total_trades

    net_profit = sum(pnl for pnl in pnls if pnl > 0)
    total_loss = sum(abs(pnl) for pnl in pnls if pnl < 0)

    if losing_trades == 0:
        risk_reward_ratio = float('inf')
    else:
        average_profit = net_profit / winning_trades
        average_loss = total_loss / losing_trades
        risk_reward_ratio = average_profit / average_loss

    max_drawdown = 0
    net_profit = (account_values[-1] - account_values[0]) / account_values[0]

    peak = account_values[0]
    for pnl in account_values:
        if pnl > peak:
            peak = pnl
        drawdown = (peak - pnl) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    return PerformanceStats(
        start_year=start,
        end_year=end,
        total_trade=len(pnls),
        win_rate=win_rate,
        risk_reward_ratio=risk_reward_ratio,
        max_drawdown=max_drawdown,
        net_profit=net_profit,
    )

def initialize_data(ticker: str, leveraged_ticker=None, use_leveraged_etf=False) -> pd.DataFrame:
    leveraged_ticker_filename_30min = 'data/{}_30min.csv'.format(leveraged_ticker)
    leveraged_ticker_df_30min = pd.read_csv(leveraged_ticker_filename_30min)
    leveraged_ticker_df_30min['timestamp'] = pd.to_datetime(leveraged_ticker_df_30min['timestamp'])
    leveraged_ticker_df_30min['day'] = leveraged_ticker_df_30min['timestamp'].dt.strftime('%Y-%m-%d')

    filename_30min = 'data/{}_30min.csv'.format(ticker)
    filename_daily = 'data/{}_daily_adjusted.csv'.format(ticker)

    ticker_df_30min = pd.read_csv(filename_30min)
    ticker_df_daily = pd.read_csv(filename_daily)

    ticker_df_30min['timestamp'] = pd.to_datetime(ticker_df_30min['timestamp'])
    ticker_df_30min['day'] = ticker_df_30min['timestamp'].dt.strftime('%Y-%m-%d')
    ticker_df_30min['year'] = ticker_df_30min['timestamp'].dt.strftime('%Y').astype('int')

    ticker_df_daily['day'] = ticker_df_daily['date']
    ticker_df_daily['previous_close'] = ticker_df_daily['adjustedClose'].shift(1)
    ticker_df_daily['previous_ema5_daily'] = get_ema(ticker_df_daily, 'previous_close', 5)
    ticker_df_daily['previous_ema13_daily'] = get_ema(ticker_df_daily, 'previous_close', 13)
    ticker_df_daily['previous_ema12_daily'] = get_ema(ticker_df_daily, 'previous_close', 12)
    ticker_df_daily['previous_ema26_daily'] = get_ema(ticker_df_daily, 'previous_close', 26)
    ticker_df_daily['previous_ema233_daily'] = get_ema(ticker_df_daily, 'previous_close', 233)
    ticker_df_daily['previous_macd_daily'], ticker_df_daily['previous_macd_signal_daily'], ticker_df_daily['previous_macd_histogram_daily'] = get_macd(ticker_df_daily, 'previous_close')
    ticker_df_daily['previous_is_top_divergence_daily'] = calculate_top_divergence(ticker_df_daily, 'previous_close', 'previous_macd_daily', 'previous_macd_signal_daily')

    ticker_df_30min['ema2'] = get_ema(ticker_df_30min, 'close', 2)
    ticker_df_30min['ema4'] = get_ema(ticker_df_30min, 'close', 4)
    ticker_df_30min['ema10'] = get_ema(ticker_df_30min, 'close', 10)
    ticker_df_30min['ema20'] = get_ema(ticker_df_30min, 'close', 20)
    ticker_df_30min['macd'], ticker_df_30min['macd_signal'], ticker_df_30min['macd_histogram'] = get_macd(ticker_df_30min, 'close')

    if leveraged_ticker is not None:
        ticker_df_30min = pd.merge(ticker_df_30min, leveraged_ticker_df_30min[['timestamp', 'close', 'open']], on='timestamp', suffixes=('', '_leveraged'), how='left')

    ticker_df_30min = pd.merge(ticker_df_30min, ticker_df_daily[[
        'day',
        'previous_close',
        'previous_ema5_daily',
        'previous_ema12_daily',
        'previous_ema13_daily',
        'previous_ema26_daily',
        'previous_ema233_daily',
        'previous_macd_daily',
        'previous_macd_signal_daily',
        'previous_macd_histogram_daily',
        'previous_is_top_divergence_daily',
    ]], on='day', how='left')

    if use_leveraged_etf:
        ticker_df_30min = ticker_df_30min.dropna(subset=['close_leveraged', 'open_leveraged'])
    return ticker_df_30min

def backtest_range(ticker_df_30min, start: int, end: int, use_leveraged_etf=False) -> PerformanceStats:
    df = ticker_df_30min[(ticker_df_30min['year']>=start) & (ticker_df_30min['year']<=end)]
    account = Account()
    stop_loss = 0
    pnls = []

    for i in range(1, len(df) - 1):
        prev_row = df.iloc[i-1]
        cur_row = df.iloc[i]
        next_row = df.iloc[i+1]

        # close price we use to determine entry/exit
        underlying_close = cur_row['close']
        next_underlying_open = next_row['open']

        # open price we use to trade, we trade on next open price upon entry signal
        next_open = next_row['open_leveraged'] if use_leveraged_etf else next_underlying_open

        # update account summary based on next available price
        account.process_bar(next_open)

        cur_ema5_daily = get_current_ema(cur_row['previous_ema5_daily'], underlying_close, 5)
        cur_ema13_daily = get_current_ema(cur_row['previous_ema13_daily'], underlying_close, 13)
        cur_ema233_daily = get_current_ema(cur_row['previous_ema233_daily'], underlying_close, 233)

        is_bearish = (cur_ema5_daily < cur_ema13_daily and cur_row['ema2'] < cur_row['ema4']) \
                        or (underlying_close < cur_ema233_daily and cur_row['ema2'] < cur_row['ema4']) \
                        or underlying_close < cur_ema13_daily * 0.98
        multiplier = 1
        if cur_row['previous_is_top_divergence_daily']:
            multiplier = 0
        should_long = False

        if (not is_bearish
                and cur_row['ema10'] < cur_row['ema20']
                and cur_row['macd_histogram'] > prev_row['macd_histogram']):
            if account.position == 0:
                buying_power = account.get_buying_power()
                account.long(cur_row['timestamp'], next_open, buying_power // next_open * multiplier)
            stop_loss = underlying_close * 0.995
            should_long = True

        if not should_long and account.position > 0:
            if underlying_close < stop_loss or crossunder(df, i, 'ema10', 'ema20'):
                pnl = account.close(cur_row['timestamp'], next_open)
                pnls.append(pnl)

    performance_stats = calculate_metrics(pnls, account.values, start, end)
    pd.DataFrame(account.trades).to_csv('trade_log/trades.csv'.format(), index=False)
    return performance_stats


def backtest_range_and_summarize(ticker_df_30min, start: int, end: int, use_leveraged_etf=False) -> None:
    performances: List[PerformanceStats] = []
    for year in range(start, end + 1):
        performances.append(backtest_range(ticker_df_30min, year, year, use_leveraged_etf))
    profits: List[float] = [performance.net_profit for performance in performances]
    drawdowns: List[float] = [performance.max_drawdown for performance in performances]

    overall_stats = backtest_range(ticker_df_30min, start, end, use_leveraged_etf)
    overall_stats = overall_stats._replace(
        average_profit_yearly=sum(profits) / len(profits),
        average_drawdown_yearly=sum(drawdowns) / len(drawdowns),
        max_profit_yearly=max(profits),
        max_drawdown_yearly=max(drawdowns),
    )
    overall_stats.show_performance_summary()
 
if __name__ == '__main__':
    TICKER = 'QQQ'
    LEVERAGED_TICKER = 'TQQQ'
    START_YEAR = 2016
    END_YEAR = 2023
    USE_LEVERAGED_ETF = True
    ticker_df_30min = initialize_data(TICKER, LEVERAGED_TICKER, USE_LEVERAGED_ETF)
    pd.DataFrame(ticker_df_30min[ticker_df_30min['previous_is_top_divergence_daily']]['day'].unique()).to_csv('output.csv', index=False)
    backtest_range_and_summarize(ticker_df_30min, START_YEAR, END_YEAR, USE_LEVERAGED_ETF)
