import yfinance as yf
import pandas as pd
import backtrader as bt
import os
from datetime import datetime, timedelta

# Attempt optional parquet dependency; fallback to csv transparently
_HAVE_PARQUET = True
try:
    import pyarrow  # noqa: F401
except Exception:  # broad: if pyarrow or engine not present
    _HAVE_PARQUET = False


def _sanitize_date(d: str | datetime) -> str:
    if isinstance(d, datetime):
        return d.strftime('%Y%m%d')
    return str(d).replace('-', '')

class DataAcquisition:
    @staticmethod
    def get_stock_data(
        tickers,
        start,
        end,
        interval: str = '1d',
        auto_adjust: bool = True,
        enable_cache: bool = True,
        cache_dir: str = 'data/cache',
        force_refresh: bool = False,
        cache_format: str = 'parquet',
        ttl_days: int | None = None
    ):
        """Get stock data and convert to backtrader-compatible format with optional caching.

        Caching Strategy:
          - One file per (ticker, start, end, interval, auto_adjust)
          - If file exists and not forced, load from disk instead of downloading
          - Supports parquet (default) or csv via cache_format
          - TTL: if ttl_days specified and cache file older than TTL -> refresh
          - Env override: DATA_CACHE_FORCE_REFRESH=1 forces redownload regardless of config
        """
        if isinstance(tickers, str):
            tickers = [tickers]

        # Normalize requested format based on availability
        requested_format = cache_format.lower()
        if requested_format == 'parquet' and not _HAVE_PARQUET:
            print("Parquet dependency (pyarrow) not available; falling back to CSV caching.")
            requested_format = 'csv'

        if enable_cache and not os.path.isdir(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)

        print("\nFetching data for tickers:", tickers)
        data_feeds = {}
        s_tag = _sanitize_date(start)
        e_tag = _sanitize_date(end)
        # Environment-based force refresh override
        env_force = os.getenv('DATA_CACHE_FORCE_REFRESH', '').lower() in ('1','true','yes')
        eff_force_refresh = force_refresh or env_force

        verbose_env = os.getenv('VERBOSE_SIGNALS', '').lower() in ('1','true','yes')
        for ticker in tickers:
            ext = 'parquet' if requested_format == 'parquet' else 'csv'
            cache_filename = f"{ticker}_{s_tag}_{e_tag}_{interval}_{'adj' if auto_adjust else 'raw'}.{ext}"
            cache_path = os.path.join(cache_dir, cache_filename)
            df = None
            loaded_from_cache = False
            if enable_cache and not eff_force_refresh and os.path.isfile(cache_path):
                try:
                    if ext == 'parquet':
                        df = pd.read_parquet(cache_path)
                        # ensure index is DatetimeIndex
                        if not isinstance(df.index, pd.DatetimeIndex) and 'Date' in df.columns:
                            df['Date'] = pd.to_datetime(df['Date'])
                            df.set_index('Date', inplace=True)
                    else:
                        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                    # Legacy multi-header CSV detection: if first data row contains non-numeric in Open/Close
                    sample_cols = [c for c in df.columns if c.lower() in ('open','high','low','close','volume')]
                    if sample_cols:
                        first_row = df.iloc[0]
                        if any(isinstance(first_row[c], str) and not first_row[c].replace('.','',1).isdigit() for c in sample_cols):
                            # Drop up to first 2 rows that look like headers
                            df = df.iloc[2:]
                    # Coerce numeric columns
                    for c in list(df.columns):
                        if c.lower() in ('open','high','low','close','volume','adj close'):
                            df[c] = pd.to_numeric(df[c], errors='coerce')
                    df.dropna(how='any', subset=[c for c in df.columns if c.lower() in ('open','high','low','close')], inplace=True)
                    # TTL check AFTER successful load so we can inspect mtime
                    if ttl_days is not None and ttl_days > 0:
                        mtime = datetime.fromtimestamp(os.path.getmtime(cache_path))
                        age_days = (datetime.now() - mtime).days
                        if age_days > ttl_days:
                            print(f"[CACHE EXPIRED] {ticker} age={age_days}d > TTL {ttl_days}d -> refresh")
                            df = None  # force re-download below
                        else:
                            loaded_from_cache = True
                            print(f"[CACHE HIT] {ticker} path={cache_path} age={age_days}d format={ext}")
                    else:
                        loaded_from_cache = True
                        print(f"[CACHE HIT] {ticker} path={cache_path} format={ext}")
                except Exception as ce:
                    print(f"Cache load failed for {ticker} ({ce}), redownloading...")
                    df = None

            if df is None:
                print(f"[CACHE MISS] Downloading {ticker} from Yahoo Finance ...")
                df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=auto_adjust, progress=False)
                if df is None or df.empty:
                    print(f"WARNING: No data returned for {ticker}; skipping.")
                    continue
                # Flatten potential multi-index columns (yfinance with auto_adjust=True returns single level already but be safe)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                # Keep only standard OHLCV columns if present
                keep_cols = [c for c in ['Open','High','Low','Close','Volume','Adj Close'] if c in df.columns]
                df = df[keep_cols]
                # Persist to cache
                if enable_cache:
                    try:
                        if ext == 'parquet':
                            df.to_parquet(cache_path, index=True)
                        else:
                            df.to_csv(cache_path, index=True)
                        print(f"[CACHE STORE] {ticker} -> {cache_path}")
                    except Exception as se:
                        print(f"Could not write cache for {ticker}: {se}")

            # Ensure column names are correct format for backtrader
            ticker_data = df.copy()
            # Standardize column names
            col_map = {c: c.lower() for c in ticker_data.columns}
            ticker_data.rename(columns=col_map, inplace=True)
            for must in ['open','high','low','close','volume']:
                if must not in ticker_data.columns:
                    raise ValueError(f"{ticker}: required column '{must}' missing after normalization")
            if 'openinterest' not in ticker_data.columns:
                ticker_data['openinterest'] = 0

            if verbose_env:
                print(f"\nData shape for {ticker} ({'cache' if loaded_from_cache else 'download'}): {ticker_data.shape}")
                if not loaded_from_cache:
                    print(f"First few rows of {ticker} data:")
                    print(ticker_data.head())

            data_feeds[ticker] = bt.feeds.PandasData(
                dataname=ticker_data,
                datetime=None,  # Use df index
                open='open',
                high='high',
                low='low',
                close='close',
                volume='volume',
                openinterest='openinterest'
            )

        return data_feeds

    @staticmethod
    def get_options_chain(ticker, date=None):
        stock = yf.Ticker(ticker)
        if date:
            return stock.option_chain(date)
        else:
            return stock.options
