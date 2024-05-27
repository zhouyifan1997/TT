def get_current_ema(prev_ema, close, preiod):
    multiplier = 2 / (preiod + 1)
    return (close * multiplier) + (prev_ema * (1 - multiplier))

def get_current_macd(prev_fast, prev_low, cur_close, prev_signal, fast_period, slow_period, smooth_period):
    cur_fast = get_current_ema(prev_fast, cur_close, fast_period)
    cur_slow = get_current_ema(prev_low, cur_close, slow_period)
    macd = cur_fast - cur_slow
    signal = get_current_ema(prev_signal, macd, smooth_period)
    return macd, signal, macd - signal

def get_macd(df, column):
    short_ema = df[column].ewm(span=12, adjust=False).mean()
    long_ema = df[column].ewm(span=26, adjust=False).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_histogram = macd_line - signal_line
    return macd_line, signal_line, macd_histogram

def get_ema(df, column, period):
    return df[column].ewm(span=period, adjust=False).mean()

def calculate_top_divergence(df, close: str, macd: str, macd_signal: str):
    # Top Divergence => close at last macd cross < close at current macd cross and last macd > current macd
    # Reset => macd < 0 or macd > last macd

    MACD_PERIOD = 5

    is_top_divergence_list = []

    macd_cross_index_list = []
    top_divergence_started = False
    max_macd_value = 0
    prev_crossover = None

    for i in range(len(df)):
        row = df.iloc[i]
        prev_crossunder = -1 if len(macd_cross_index_list) == 0 else macd_cross_index_list[-1]
        prev_row = None if len(macd_cross_index_list) == 0 else df.iloc[macd_cross_index_list[-1]]

        if crossover(df, i, macd, macd_signal):
            if row[macd] < 0:
                macd_cross_index_list = []
                top_divergence_started = False
                max_macd_value = 0
            prev_crossover = i

        if prev_row is None or row[macd] > max_macd_value:
            top_divergence_started = False

        # top divergence only happens when macd crossunder macd signal
        if crossunder(df, i, macd, macd_signal):
            macd_cross_index_list.append(i)
            if i >= prev_crossover + MACD_PERIOD and prev_crossover >= prev_crossunder + MACD_PERIOD and (prev_row is not None and prev_row[close] < row[close] and prev_row[macd] > row[macd]):
                top_divergence_started = True
        max_macd_value = max(row[macd], max_macd_value)
        is_top_divergence_list.append(top_divergence_started)

    return is_top_divergence_list

def crossover(df, row, col1, col2):
    if row == 0:
        return False
    prev_row = df.iloc[row-1]
    cur_row = df.iloc[row]
    return prev_row[col1] <= prev_row[col2] and cur_row[col1] > cur_row[col2]
    
def crossunder(df, row, col1, col2):
    if row == 0:
        return False
    prev_row = df.iloc[row-1]
    cur_row = df.iloc[row]
    return prev_row[col1] >= prev_row[col2] and cur_row[col1] < cur_row[col2]