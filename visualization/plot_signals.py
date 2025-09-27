import os
import pandas as pd
from typing import Dict, Any, Optional


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def build_signals_dataframe(price_df: pd.DataFrame, raw_signals: pd.Series, score_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Build a signals dataframe from raw signals and scores."""
    df = pd.DataFrame(index=price_df.index)
    if raw_signals is not None:
        df['signal'] = raw_signals.reindex(df.index).fillna(0)
    if score_df is not None:
        for col in ['bull_score','bear_score','net_score']:
            if col in score_df.columns:
                df[col] = score_df[col].reindex(df.index)
    # Guarantee a signal column exists so downstream plotting never fails even if no signals produced
    if 'signal' not in df.columns:
        df['signal'] = 0
    return df
