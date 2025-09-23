    # 4. Price relationship checks in order expected by legacy tests
    # Check high >= low first (for test_high_less_than_low_fails)
    if {'high', 'low'}.issubset(df.columns):
        invalid = df['high'] < df['low']
        if invalid.any():
            bad_indices = df[invalid].index[:3]
            if msg_compat:
                raise AssertionError("High price is less than low price")
            raise HighLessThanLowError(f"High < low at indices: {list(bad_indices)}")

    # Check high >= open (for test_high_less_than_open_fails)
    if {'high', 'open'}.issubset(df.columns):
        invalid = df['high'] < df['open']
        if invalid.any():
            bad_indices = df[invalid].index[:3]
            if msg_compat:
                raise AssertionError("High price is less than open price")
            raise HighLessThanOpenError(f"High < open at indices: {list(bad_indices)}")

    # Check low <= close (for test_low_greater_than_close_fails)
    if {'low', 'close'}.issubset(df.columns):
        invalid = df['low'] > df['close']
        if invalid.any():
            bad_indices = df[invalid].index[:3]
            if msg_compat:
                raise AssertionError("Low price is greater than close price")
            raise LowGreaterThanCloseError(f"Low > close at indices: {list(bad_indices)}")

    # Check remaining range constraints
    if {'open', 'high', 'low'}.issubset(df.columns):
        invalid = (df['open'] < df['low']) | (df['open'] > df['high'])
        if invalid.any():
            bad_indices = df[invalid].index[:3]
            if msg_compat:
                raise AssertionError("High price is less than open price")  # fallback
            raise OpenOutsideRangeError(f"Open outside [low, high] at indices: {list(bad_indices)}")

    if {'close', 'high', 'low'}.issubset(df.columns):
        invalid = (df['close'] < df['low']) | (df['close'] > df['high'])
        if invalid.any():
            bad_indices = df[invalid].index[:3]
            if msg_compat:
                raise AssertionError("Low price is greater than close price")  # fallback
            raise CloseOutsideRangeError(f"Close outside [low, high] at indices: {list(bad_indices)}")