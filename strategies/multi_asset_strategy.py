import backtrader as bt
import pandas as pd
from typing import Dict, Any, Optional
from types import SimpleNamespace
from utils.trading_logger import TradingLogger
from utils.strategy_performance import StrategyPerformance, TradeInfo


class MultiAssetStrategy(bt.Strategy):
    """Multi-strategy coordinator with real signal execution & order tagging (Phase 1).

    Capabilities now:
        - Instantiates supplied strategy logic classes (expected to follow TradingStrategy interface).
        - Per-bar builds lightweight OHLCV DataFrames for each data feed and passes to each strategy.
        - Generates latest signal per strategy & data, places orders sized by that strategy's
          allocated capital slice, and tags orders for attribution.
        - Tracks per-strategy performance using actual trade PnL rather than proportional split
          (if a strategy never trades, it accrues no trades). Daily value attribution still
          proportional for now (can be improved later with virtual sub-portfolio accounting).

    Phase 1 Limitations / TODO:
        - All strategies may trade the same ticker independently; no conflict resolution.
        - Daily valuation still proportional to allocation (does not reflect uninvested cash nuances).
        - Position sizing translates strategy-returned dollar size to integer shares (floor logic).
        - No stop-loss / exit logic beyond opposite signal (can integrate per-strategy stops later).
        - DataFrame construction each bar can be optimized (future: rolling buffers / numpy views).
    """
    params = (
        ('strategies', None),  # { name: { 'class': StrategyClass, 'params': {...} } }
        ('allocation', None),  # { name: percent, ... } must sum to 100
    )

    def __init__(self):
        if not self.p.strategies or not self.p.allocation:
            raise ValueError('Both strategies and allocation params required')
        alloc_sum = sum(self.p.allocation.values())
        if round(alloc_sum, 6) != 100.0:
            raise ValueError(f'Allocation must sum to 100, got {alloc_sum}')

        self.meta: Dict[str, Dict[str, Any]] = {}
        self.perf: Dict[str, StrategyPerformance] = {}
        self.loggers: Dict[str, TradingLogger] = {}
        self.logic: Dict[str, Any] = {}              # instantiated strategy logic objects
        self._order_strategy: Dict[int, str] = {}     # order ref -> strategy name
        self._data_last_strategy: Dict[str, str] = {} # data name -> last strategy that acted
        self._last_df_cache: Dict[str, pd.DataFrame] = {}
        self._value_history: list[float] = []  # portfolio value per bar for performance analysis
        # For visualization: store per-ticker per-strategy signal history & executed orders
        self._signal_history: Dict[str, Dict[str, list]] = {}  # {ticker: {strategy: [ (dt, sig, net,bull,bear) ]}}
        self._orders_log: list[Dict[str, Any]] = []  # executed order records

        # Capture broker cash at init (may be default if user sets cash AFTER addstrategy)
        self._init_broker_cash = self.broker.get_cash()

        for name, conf in self.p.strategies.items():
            if name not in self.p.allocation:
                raise ValueError(f'Missing allocation for {name}')
            strat_class = conf.get('class') or conf.get('strategy')
            params = conf.get('params') or conf.get('config') or {}
            pct = self.p.allocation[name]
            capital = self._init_broker_cash * (pct / 100.0)
            lg = TradingLogger(name)
            tracker = StrategyPerformance(name, capital)
            self.meta[name] = {
                'class': strat_class,
                'params': params,
                'pct': pct,
                'capital': capital,
            }
            self.perf[name] = tracker
            self.loggers[name] = lg
            lg.strategy(f"Init {name} pct={pct}% capital=${capital:,.2f} params={params}")

            # Instantiate strategy logic if class provided
            if strat_class is not None:
                try:
                    self.logic[name] = strat_class()
                except Exception as e:
                    lg.error(f"Failed to instantiate strategy class: {e}")
                    self.logic[name] = None

            # Build a lightweight config wrapper expected by strategy code (.strategy attribute)
            user_cfg = params.get('config') if isinstance(params, dict) else None
            cfg_attr_name = f"{name}_strategy"
            if user_cfg is not None and hasattr(user_cfg, cfg_attr_name):
                strat_cfg = getattr(user_cfg, cfg_attr_name)
                self.meta[name]['config_obj'] = SimpleNamespace(strategy=strat_cfg)
            elif user_cfg is not None and hasattr(user_cfg, 'long_term_strategy') and name == 'long_term':
                self.meta[name]['config_obj'] = SimpleNamespace(strategy=getattr(user_cfg, 'long_term_strategy'))
            elif user_cfg is not None and hasattr(user_cfg, 'short_term_strategy') and name == 'short_term':
                self.meta[name]['config_obj'] = SimpleNamespace(strategy=getattr(user_cfg, 'short_term_strategy'))
            elif user_cfg is not None and hasattr(user_cfg, 'options_strategy') and name == 'options':
                self.meta[name]['config_obj'] = SimpleNamespace(strategy=getattr(user_cfg, 'options_strategy'))
            else:
                # Fallback: use raw params wrapped so access doesn't crash
                self.meta[name]['config_obj'] = SimpleNamespace(strategy=params)

    def start(self):
        """Adjust initial capital attribution if broker cash was set AFTER strategy creation."""
        current_cash = self.broker.get_cash()
        if abs(current_cash - self._init_broker_cash) > 1e-6 and self._init_broker_cash > 0:
            scale = current_cash / self._init_broker_cash
            for name, meta in self.meta.items():
                old_cap = meta['capital']
                new_cap = old_cap * scale
                meta['capital'] = new_cap
                # Re-seed performance tracker base capital (simple approach: overwrite starting value history)
                self.perf[name].starting_capital = new_cap  # assuming attribute exists; else add if needed
                self.loggers[name].strategy(
                    f"Adjusted starting capital after broker cash set: {old_cap:,.2f} -> {new_cap:,.2f}")
            self._init_broker_cash = current_cash

    def notify_order(self, order):
        """Log order lifecycle with strategy attribution."""
        strat_name = self._order_strategy.get(order.ref)
        if strat_name is None:
            return
        lg = self.loggers.get(strat_name)
        if not lg:
            return
        status_map = {
            order.Submitted: 'Submitted',
            order.Accepted: 'Accepted',
            order.Partial: 'Partial',
            order.Completed: 'Completed',
            order.Canceled: 'Canceled',
            order.Margin: 'Margin',
            order.Rejected: 'Rejected'
        }
        lg.strategy(f"Order {order.ref} {order.data._name} -> {status_map.get(order.status, order.status)}")
        if order.status == order.Completed:
            lg.trade(ticker=order.data._name,
                     action='BUY' if order.isbuy() else 'SELL',
                     quantity=order.executed.size,
                     price=order.executed.price,
                     value=order.executed.size * order.executed.price)
            # store order for plotting
            self._orders_log.append({
                'timestamp': self.datetime.datetime(),
                'ticker': order.data._name,
                'strategy': strat_name,
                'action': 'BUY' if order.isbuy() else 'SELL',
                'quantity': order.executed.size,
                'price': order.executed.price,
                'value': order.executed.size * order.executed.price,
                'order_ref': order.ref
            })

    def notify_trade(self, trade):
        if trade.status != trade.Closed:
            return
        data_name = trade.data._name
        # Prefer last acting strategy on this data; fallback proportional if unknown
        strat_name: Optional[str] = self._data_last_strategy.get(data_name)
        if strat_name and strat_name in self.perf:
            lg = self.loggers[strat_name]
            ti = TradeInfo(
                timestamp=self.datetime.datetime(),
                ticker=data_name,
                action='SELL' if trade.size < 0 else 'BUY',
                quantity=abs(trade.size),
                price=trade.price,
                value=abs(trade.size * trade.price),
                pnl=trade.pnl or 0.0
            )
            self.perf[strat_name].record_trade(ti)
            lg.trade(ticker=data_name, action=ti.action, quantity=ti.quantity,
                     price=ti.price, value=ti.value, pnl=ti.pnl)
        else:
            # Fallback: proportional attribution (should be rare once tagging works)
            total_pct = sum(m['pct'] for m in self.meta.values()) or 1.0
            for name, m in self.meta.items():
                ratio = m['pct'] / total_pct
                ti = TradeInfo(
                    timestamp=self.datetime.datetime(),
                    ticker=data_name,
                    action='SELL' if trade.size < 0 else 'BUY',
                    quantity=int(abs(trade.size) * ratio),
                    price=trade.price,
                    value=abs(trade.size * trade.price) * ratio,
                    pnl=(trade.pnl or 0.0) * ratio
                )
                self.perf[name].record_trade(ti)
                self.loggers[name].trade(ticker=data_name, action=ti.action, quantity=ti.quantity,
                                          price=ti.price, value=ti.value, pnl=ti.pnl)

    def _build_dataframe(self, data) -> pd.DataFrame:
        """Return (and incrementally maintain) OHLCV DataFrame for a data feed without re-index errors.

        We append only the current bar each call instead of rebuilding the full history using
        positive integer indexing (which can trigger 'array index out of range' in some backtrader
        execution modes / memory optimizations). This keeps indicator consumers supplied with a
        growing history while avoiding O(n^2) rebuild cost and index errors.
        """
        name = data._name
        if name not in self._last_df_cache:
            self._last_df_cache[name] = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])

        df = self._last_df_cache[name]
        # Current bar timestamp
        try:
            current_dt = bt.num2date(data.datetime[0])
            current_ts = pd.Timestamp(current_dt)
        except Exception:
            current_ts = None

        # Append only if new timestamp
        if current_ts is not None and (len(df) == 0 or df.index[-1] != current_ts):
            try:
                new_row = pd.DataFrame({
                    'Open': [data.open[0]],
                    'High': [data.high[0]],
                    'Low': [data.low[0]],
                    'Close': [data.close[0]],
                    'Volume': [getattr(data, 'volume', [0])[0] if len(data) > 0 else 0]
                }, index=[current_ts])
                # Avoid pandas concat deprecation warning with empty frames
                if df.empty:
                    self._last_df_cache[name] = new_row
                else:
                    self._last_df_cache[name] = pd.concat([df, new_row])
                df = self._last_df_cache[name]
            except IndexError:
                # If for some reason current bar not accessible yet, just return existing history
                pass
        # Attach ticker name for downstream strategy logic (e.g., fundamentals lookup)
        try:
            df.attrs['ticker'] = name
        except Exception:
            pass
        return df

    def next(self):
        dt = self.datetime.datetime()
        total_val = self.broker.getvalue()
        cash = self.broker.get_cash()
        # Record portfolio value
        self._value_history.append(total_val)

        # 1. Strategy signal processing & order placement
        for strat_name, meta in self.meta.items():
            logic_obj = self.logic.get(strat_name)
            if logic_obj is None:
                continue
            # Temporary instrumentation for long_term debug
            try:
                import os
                if strat_name == 'long_term' and os.getenv('LONGTERM_DEBUG', '').lower() in ('1','true','yes'):
                    print(f"[LONGTERM_DEBUG] logic_obj type={type(logic_obj)} has_generate={hasattr(logic_obj,'generate_signals')}")
            except Exception:
                pass
            pct = meta['pct']
            alloc_capital = total_val * (pct / 100.0)
            config_wrapper = meta.get('config_obj')  # SimpleNamespace(strategy=...)
            config_obj = config_wrapper if config_wrapper is not None else meta.get('params', {})

            for data in self.datas:
                df = self._build_dataframe(data)
                # Determine existing position before generating/acting on signals for debug visibility
                position = self.getposition(data)
                pos_size = position.size if position else 0
                price = data.close[0]
                try:
                    signals = logic_obj.generate_signals(df, config_obj)
                    if signals is None or len(signals) == 0:
                        continue
                    latest_sig = signals.iloc[-1]
                    # Optional expanded debug if strategy exposes detailed score attributes
                    if hasattr(logic_obj, 'last_net_score'):
                        extra = (f" net={getattr(logic_obj,'last_net_score',0):.3f}"
                                 f" bull={getattr(logic_obj,'last_bull_score',0):.3f}"
                                 f" bear={getattr(logic_obj,'last_bear_score',0):.3f}")
                    else:
                        extra = ""
                    self.loggers[strat_name].strategy(
                        f"Signal check {data._name} latest_sig={latest_sig} pos_size={pos_size} price={price}{extra}")
                    # record signal for visualization
                    tname = data._name
                    if tname not in self._signal_history:
                        self._signal_history[tname] = {}
                    if strat_name not in self._signal_history[tname]:
                        self._signal_history[tname][strat_name] = []
                    self._signal_history[tname][strat_name].append((self.datetime.datetime(), float(latest_sig),
                                                                     float(getattr(logic_obj,'last_net_score', 0.0)),
                                                                     float(getattr(logic_obj,'last_bull_score', 0.0)),
                                                                     float(getattr(logic_obj,'last_bear_score', 0.0))))
                except Exception as e:
                    self.loggers[strat_name].error(f"Signal generation error {data._name}: {e}")
                    continue

                # Entry (long only for now)
                if latest_sig > 0 and latest_sig != 2 and pos_size <= 0:
                    try:
                        target_value = logic_obj.calculate_position_size(df, alloc_capital, config_obj)
                        # Conviction scaling if strategy exposes last_net_score & dynamic threshold concept
                        if hasattr(logic_obj, 'last_net_score') and target_value > 0:
                            # Use absolute of last_net_score relative to (buy_thr or dynamic) if available
                            # We don't have direct dyn_buy_thr here; approximate with config threshold
                            base_thr = getattr(getattr(config_obj, 'strategy', config_obj), 'buy_score_threshold', 0.5)
                            if base_thr > 0:
                                scale = min(1.5, max(0.25, abs(getattr(logic_obj,'last_net_score',0)) / base_thr))
                                target_value *= scale
                        self.loggers[strat_name].strategy(
                            f"Sizing {data._name} target_value={target_value:.2f} alloc_capital={alloc_capital:.2f}")
                    except Exception as e:
                        self.loggers[strat_name].error(f"Position size error {data._name}: {e}")
                        continue
                    if target_value <= 0 or price <= 0:
                        self.loggers[strat_name].strategy(
                            f"Skip buy {data._name} target_value={target_value} price={price}")
                        continue
                    shares = max(1, int(target_value / price))
                    o = self.buy(data=data, size=shares)
                    self._order_strategy[o.ref] = strat_name
                    self._data_last_strategy[data._name] = strat_name
                    # record extended context
                    self._orders_log.append({
                        'timestamp': self.datetime.datetime(),
                        'ticker': data._name,
                        'strategy': strat_name,
                        'action': 'SIGNAL_BUY',
                        'net_score': float(getattr(logic_obj,'last_net_score',0.0)),
                        'bull_score': float(getattr(logic_obj,'last_bull_score',0.0)),
                        'bear_score': float(getattr(logic_obj,'last_bear_score',0.0)),
                        'confidence': float(getattr(logic_obj, '_diag_rows', [{'confidence': None}])[-1].get('confidence', 0.0)) if getattr(logic_obj, '_diag_rows', None) else None,
                        'peak_confidence': float(getattr(logic_obj, '_diag_rows', [{'peak_confidence': None}])[-1].get('peak_confidence', 0.0)) if getattr(logic_obj, '_diag_rows', None) else None,
                        'shares_planned': shares,
                        'price': price
                    })
                    self.loggers[strat_name].signal(
                        ticker=data._name,
                        signal_type='BUY',
                        value=latest_sig,
                        params={'shares': shares, 'price': price, 'alloc_capital': alloc_capital}
                    )
                # Partial exit (signal==2): trim existing long position if >0
                elif latest_sig == 2 and pos_size > 0:
                    try:
                        strat_cfg = getattr(config_obj, 'strategy', config_obj)
                        reduction = getattr(strat_cfg, 'partial_exit_reduction', 0.5)
                        reduction = max(0.05, min(0.95, float(reduction)))
                        trim_size = int(pos_size * reduction)
                        if trim_size <= 0:
                            trim_size = 1 if pos_size > 1 else 0
                        if trim_size > 0:
                            self.loggers[strat_name].strategy(
                                f"Partial exit {data._name} reduce {trim_size}/{pos_size} (reduction={reduction})")
                            o = self.sell(data=data, size=trim_size)
                            self._order_strategy[o.ref] = strat_name
                            self._data_last_strategy[data._name] = strat_name
                            self._orders_log.append({
                                'timestamp': self.datetime.datetime(),
                                'ticker': data._name,
                                'strategy': strat_name,
                                'action': 'SIGNAL_PARTIAL_EXIT',
                                'reduction_fraction': reduction,
                                'trim_size': trim_size,
                                'remaining_est': pos_size - trim_size,
                                'price': price,
                                'net_score': float(getattr(logic_obj,'last_net_score',0.0))
                                ,'confidence': float(getattr(logic_obj, '_diag_rows', [{'confidence': None}])[-1].get('confidence', 0.0)) if getattr(logic_obj, '_diag_rows', None) else None
                                ,'peak_confidence': float(getattr(logic_obj, '_diag_rows', [{'peak_confidence': None}])[-1].get('peak_confidence', 0.0)) if getattr(logic_obj, '_diag_rows', None) else None
                            })
                            self.loggers[strat_name].signal(
                                ticker=data._name,
                                signal_type='PARTIAL_EXIT',
                                value=latest_sig,
                                params={'trim_size': trim_size, 'pos_size': pos_size, 'price': price}
                            )
                    except Exception as e:
                        self.loggers[strat_name].error(f"Partial exit error {data._name}: {e}")
                # Exit
                elif latest_sig < 0 and pos_size > 0:
                    self.loggers[strat_name].strategy(
                        f"Exit signal {data._name} latest_sig={latest_sig} existing_pos={pos_size}")
                    o = self.close(data=data)
                    self._order_strategy[o.ref] = strat_name
                    self._data_last_strategy[data._name] = strat_name
                    self._orders_log.append({
                        'timestamp': self.datetime.datetime(),
                        'ticker': data._name,
                        'strategy': strat_name,
                        'action': 'SIGNAL_SELL',
                        'net_score': float(getattr(logic_obj,'last_net_score',0.0)),
                        'bull_score': float(getattr(logic_obj,'last_bull_score',0.0)),
                        'bear_score': float(getattr(logic_obj,'last_bear_score',0.0)),
                        'confidence': float(getattr(logic_obj, '_diag_rows', [{'confidence': None}])[-1].get('confidence', 0.0)) if getattr(logic_obj, '_diag_rows', None) else None,
                        'peak_confidence': float(getattr(logic_obj, '_diag_rows', [{'peak_confidence': None}])[-1].get('peak_confidence', 0.0)) if getattr(logic_obj, '_diag_rows', None) else None,
                        'pos_size': pos_size,
                        'price': price
                    })
                    self.loggers[strat_name].signal(
                        ticker=data._name,
                        signal_type='SELL',
                        value=latest_sig,
                        params={'pos_size': pos_size, 'price': price}
                    )

        # 2. Daily (bar) performance attribution (still proportional for valuation layer)
        for name, meta in self.meta.items():
            pct = meta['pct']
            strat_val = total_val * (pct / 100.0)
            strat_cash = cash * (pct / 100.0)
            tracker = self.perf[name]
            tracker.update_daily_stats(dt, strat_val)
            metrics = tracker.get_metrics()
            self.loggers[name].performance({
                'date': dt.strftime('%Y-%m-%d'),
                'portfolio_value': strat_val,
                'cash_est': strat_cash,
                'total_return': metrics['total_return'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'max_drawdown': metrics['max_drawdown'],
                'win_rate': metrics['win_rate'],
                'profit_factor': metrics['profit_factor'],
                'trades': metrics['total_trades']
            })

    # ---- Public accessors ----
    def get_strategy_performance(self, strategy_name: str = None) -> Dict[str, Any]:
        if strategy_name:
            if strategy_name not in self.perf:
                raise ValueError(f'Strategy {strategy_name} not found')
            p = self.perf[strategy_name]
            return {
                'metrics': p.get_metrics(),
                'positions': p.get_position_summary(),
                'trades': p.get_trade_history(),
                'daily_stats': p.get_daily_stats(),
            }
        return {
            name: {
                'metrics': p.get_metrics(),
                'positions': p.get_position_summary(),
                'trades': p.get_trade_history(),
                'daily_stats': p.get_daily_stats(),
            } for name, p in self.perf.items()
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        agg = {
            'total_return': 0.0,
            'total_pnl': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'max_drawdown': 0.0,
        }
        for name, p in self.perf.items():
            m = p.get_metrics()
            agg['total_pnl'] += m['total_pnl']
            agg['total_trades'] += m['total_trades']
            agg['winning_trades'] += m['winning_trades']
            agg['losing_trades'] += m['losing_trades']
            agg['max_drawdown'] = max(agg['max_drawdown'], m['max_drawdown'])
        agg['win_rate'] = (agg['winning_trades'] / agg['total_trades']) if agg['total_trades'] else 0.0
        start_cash = getattr(self.broker, 'startingcash', None)
        if start_cash:
            agg['total_return'] = (self.broker.getvalue() - start_cash) / start_cash
        return agg

    # Visualization data accessors
    def get_signal_history(self) -> Dict[str, Dict[str, list]]:
        return self._signal_history

    def get_orders_log(self) -> list[Dict[str, Any]]:
        return self._orders_log

    def get_equity_curve(self) -> pd.DataFrame:
        """Return equity curve (portfolio value per bar)."""
        # We stored values in _value_history aligned with bars; approximate timestamps from first data feed
        if not self._value_history:
            return pd.DataFrame(columns=['value'])
        try:
            # Use primary data (first feed) index as timeline
            primary = self.datas[0]
            dates = []
            for i in range(len(self._value_history)):
                try:
                    dt = bt.num2date(primary.lines.datetime.array[i])
                except Exception:
                    dt = None
                dates.append(pd.Timestamp(dt) if dt else pd.NaT)
            return pd.DataFrame({'value': self._value_history}, index=dates)
        except Exception:
            return pd.DataFrame({'value': self._value_history})
