#!/usr/bin/env python3
"""
Interactive Chart Viewer - Desktop GUI Application
Displays financial charts with multiple timeframes and technical indicators
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import pandas as pd
import json
import os
import argparse
from datetime import datetime
import numpy as np
from matplotlib.lines import Line2D

class InteractiveChartViewer:
    def __init__(self, root, ticker='AAPL', strategy='day_trading', data_file=None, orders_file=None, equity_file=None):
        self.root = root
        self.root.title(f"Interactive Chart Viewer - {ticker.upper()} ({strategy.upper()})")
        self.root.geometry("1600x1000")

        # Data storage
        self.chart_data = {}
        self.current_timeframe = '1d'
        self.current_ticker = ticker
        self.current_strategy = strategy
        self.data_file = data_file
        self.orders_file = orders_file
        self.equity_file = equity_file

        # Loaded auxiliary dataframes
        self.orders_df = None  # type: pd.DataFrame | None
        self.equity_df = None  # type: pd.DataFrame | None

        # Available timeframes
        self.timeframes = ['1d', '1h', '5m', '1m', '1w']

        # Interactive features
        self.crosshair_enabled = True
        self.drawing_mode = None  # 'trendline', 'horizontal', 'vertical', None
        self.drawing_lines = []  # Store drawn lines
        self.selected_bar = None
        self.zoom_level = 1.0

        # Mouse tracking
        self.mouse_x = None
        self.mouse_y = None

        self.setup_ui()

        # Load primary price data
        if data_file:
            self.load_data_from_file(data_file)
        else:
            self.load_sample_data()

        # Load orders/signals if provided
        if self.orders_file and os.path.isfile(self.orders_file):
            try:
                self.orders_df = pd.read_csv(self.orders_file, parse_dates=['timestamp'])
                self.orders_df.sort_values('timestamp', inplace=True)
            except Exception as e:
                print(f"[WARN] Could not load orders file {self.orders_file}: {e}")
                self.orders_df = None

        # Load equity curve (account balance) if provided or auto-detect
        ecandidate = self.equity_file or 'equity_curve.csv'
        if ecandidate and os.path.isfile(ecandidate):
            try:
                self.equity_df = pd.read_csv(ecandidate, index_col=0, parse_dates=True)
                self.equity_df.sort_index(inplace=True)
            except Exception as e:
                print(f"[WARN] Could not load equity curve file {ecandidate}: {e}")
                self.equity_df = None

        self.setup_interactive_features()

    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Top control panel
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        # File controls
        file_frame = ttk.Frame(control_frame)
        file_frame.pack(side=tk.LEFT)

        ttk.Button(file_frame, text="Load Chart Data", command=self.load_chart_file).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(file_frame, text="Load CSV Data", command=self.load_csv_file).pack(side=tk.LEFT, padx=(0, 10))

        # Interactive controls
        interact_frame = ttk.Frame(control_frame)
        interact_frame.pack(side=tk.LEFT)

        ttk.Button(interact_frame, text="Zoom In", command=self.zoom_in).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Button(interact_frame, text="Zoom Out", command=self.zoom_out).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Button(interact_frame, text="Reset Zoom", command=self.reset_zoom).pack(side=tk.LEFT, padx=(0, 10))

        # Drawing tools
        draw_frame = ttk.Frame(control_frame)
        draw_frame.pack(side=tk.LEFT)

        ttk.Label(draw_frame, text="Drawing:").pack(side=tk.LEFT, padx=(0, 5))
        self.draw_var = tk.StringVar(value="none")
        draw_combo = ttk.Combobox(draw_frame, textvariable=self.draw_var,
                                values=["none", "trendline", "horizontal", "vertical"],
                                state='readonly', width=10)
        draw_combo.pack(side=tk.LEFT, padx=(0, 10))
        draw_combo.bind('<<ComboboxSelected>>', self.on_drawing_mode_change)

        ttk.Button(draw_frame, text="Clear Drawings", command=self.clear_drawings).pack(side=tk.LEFT, padx=(0, 10))

        # Export controls
        export_frame = ttk.Frame(control_frame)
        export_frame.pack(side=tk.LEFT)

        ttk.Button(export_frame, text="Export Chart", command=self.export_chart).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(export_frame, text="Export Data", command=self.export_data).pack(side=tk.LEFT)

        # Timeframe controls
        tf_frame = ttk.Frame(control_frame)
        tf_frame.pack(side=tk.RIGHT)

        ttk.Label(tf_frame, text="Timeframe:").pack(side=tk.LEFT, padx=(0, 5))
        self.tf_var = tk.StringVar(value='1d')
        self.tf_combo = ttk.Combobox(tf_frame, textvariable=self.tf_var,
                                   values=self.timeframes, state='readonly', width=8)
        self.tf_combo.pack(side=tk.LEFT, padx=(0, 10))
        self.tf_combo.bind('<<ComboboxSelected>>', self.on_timeframe_change)

        # Navigation controls
        nav_frame = ttk.Frame(tf_frame)
        nav_frame.pack(side=tk.LEFT)

        ttk.Button(nav_frame, text="◀", width=3, command=self.prev_timeframe).pack(side=tk.LEFT)
        ttk.Button(nav_frame, text="▶", width=3, command=self.next_timeframe).pack(side=tk.LEFT)

        # Chart display area
        chart_frame = ttk.Frame(main_frame)
        chart_frame.pack(fill=tk.BOTH, expand=True)

        # Matplotlib figure with toolbar
        self.figure = plt.Figure(figsize=(16, 10), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=chart_frame)

        # Add navigation toolbar
        toolbar_frame = ttk.Frame(chart_frame)
        toolbar_frame.pack(fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.pack(side=tk.TOP, fill=tk.X)

        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Info panel
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(fill=tk.X, pady=(10, 0))

        self.info_label = ttk.Label(info_frame, text="No data loaded")
        self.info_label.pack(anchor=tk.W)

        # Price info display
        price_frame = ttk.Frame(info_frame)
        price_frame.pack(side=tk.RIGHT)

        ttk.Label(price_frame, text="Mouse Position:").pack(side=tk.LEFT, padx=(0, 5))
        self.price_info_var = tk.StringVar()
        self.price_info_var.set("Move mouse over chart")
        ttk.Label(price_frame, textvariable=self.price_info_var,
                 font=('Arial', 10, 'bold')).pack(side=tk.LEFT)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Click and drag to zoom, right-click for context menu")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var,
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    def setup_interactive_features(self):
        """Setup interactive chart features"""
        # Connect mouse events
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas.mpl_connect('button_press_event', self.on_mouse_click)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.canvas.mpl_connect('scroll_event', self.on_mouse_scroll)

        # Crosshair lines
        self.crosshair_vline = None
        self.crosshair_hline = None

    def on_mouse_move(self, event):
        """Handle mouse movement for crosshair and info display"""
        if not hasattr(self, 'axes') or not self.axes:
            return

        if event.inaxes in self.axes:
            # Update crosshair
            self.update_crosshair(event)

            # Update price info
            self.update_price_info(event)

    def update_crosshair(self, event):
        """Update crosshair lines"""
        if not self.crosshair_enabled:
            return

        # Remove existing crosshair
        if self.crosshair_vline:
            self.crosshair_vline.remove()
        if self.crosshair_hline:
            self.crosshair_hline.remove()

        # Create new crosshair
        self.crosshair_vline = event.inaxes.axvline(x=event.xdata, color='gray',
                                                  linestyle='--', alpha=0.7, linewidth=1)
        self.crosshair_hline = event.inaxes.axhline(y=event.ydata, color='gray',
                                                  linestyle='--', alpha=0.7, linewidth=1)

        self.canvas.draw_idle()

    def update_price_info(self, event):
        """Update price information display"""
        if event.xdata is None or event.ydata is None:
            self.price_info_var.set("Move mouse over chart")
            return

        try:
            # Find closest data point
            df = self.chart_data.get(self.current_timeframe)
            if df is None or df.empty:
                return

            # Convert xdata to datetime
            mouse_date = mdates.num2date(event.xdata)

            # Find closest date in data
            closest_idx = df.index.get_indexer([mouse_date], method='nearest')[0]
            if closest_idx >= 0 and closest_idx < len(df):
                row = df.iloc[closest_idx]
                date_str = df.index[closest_idx].strftime('%Y-%m-%d %H:%M')

                if event.inaxes == self.axes[0]:  # Price chart
                    info = f"Date: {date_str} | O: {row['Open']:.2f} | H: {row['High']:.2f} | L: {row['Low']:.2f} | C: {row['Close']:.2f}"
                elif event.inaxes == self.axes[1]:  # Volume chart
                    info = f"Date: {date_str} | Volume: {row['Volume']:,.0f}"
                elif event.inaxes == self.axes[2]:  # RSI chart
                    rsi_val = self.calculate_rsi(df['Close'], 14).iloc[closest_idx] if len(df) >= 14 else None
                    rsi_str = f"{rsi_val:.1f}" if rsi_val is not None else "N/A"
                    info = f"Date: {date_str} | RSI: {rsi_str}"
                else:
                    info = f"Date: {date_str} | Value: {event.ydata:.2f}"

                self.price_info_var.set(info)
        except Exception as e:
            self.price_info_var.set(f"Error: {str(e)}")

    def on_mouse_click(self, event):
        """Handle mouse clicks for drawing and selection"""
        if event.button == 1:  # Left click
            if self.drawing_mode:
                self.start_drawing(event)
            else:
                self.select_bar(event)
        elif event.button == 3:  # Right click
            self.show_context_menu(event)

    def on_mouse_release(self, event):
        """Handle mouse release for drawing completion"""
        if event.button == 1 and self.drawing_mode and hasattr(self, 'drawing_start'):
            self.finish_drawing(event)

    def on_mouse_scroll(self, event):
        """Handle mouse scroll for zooming"""
        if event.button == 'up':
            self.zoom_in(center_x=event.xdata)
        elif event.button == 'down':
            self.zoom_out(center_x=event.xdata)

    def start_drawing(self, event):
        """Start drawing a line"""
        self.drawing_start = (event.xdata, event.ydata)
        self.status_var.set(f"Drawing {self.drawing_mode} - Click to finish")

    def finish_drawing(self, event):
        """Finish drawing a line"""
        if not hasattr(self, 'drawing_start'):
            return

        start_x, start_y = self.drawing_start
        end_x, end_y = event.xdata, event.ydata

        if self.drawing_mode == 'trendline':
            line = Line2D([start_x, end_x], [start_y, end_y],
                         color='blue', linewidth=2, alpha=0.8)
        elif self.drawing_mode == 'horizontal':
            # Use full x-range for horizontal line
            x_range = self.axes[0].get_xlim()
            line = Line2D(x_range, [start_y, start_y],
                         color='red', linewidth=2, alpha=0.8, linestyle='--')
        elif self.drawing_mode == 'vertical':
            # Use full y-range for vertical line
            y_range = self.axes[0].get_ylim()
            line = Line2D([start_x, start_x], y_range,
                         color='green', linewidth=2, alpha=0.8, linestyle='--')

        event.inaxes.add_line(line)
        self.drawing_lines.append(line)
        self.canvas.draw()

        delattr(self, 'drawing_start')
        self.status_var.set(f"Drew {self.drawing_mode} - Ready")

    def select_bar(self, event):
        """Select a candlestick bar and show details"""
        if event.inaxes != self.axes[0]:  # Only for price chart
            return

        df = self.chart_data.get(self.current_timeframe)
        if df is None or df.empty:
            return

        # Find closest bar
        mouse_date = mdates.num2date(event.xdata)
        closest_idx = df.index.get_indexer([mouse_date], method='nearest')[0]

        if closest_idx >= 0 and closest_idx < len(df):
            row = df.iloc[closest_idx]

            # Show detailed information
            info = f"""Bar Details:
Date: {df.index[closest_idx].strftime('%Y-%m-%d %H:%M:%S')}
Open: {row['Open']:.2f}
High: {row['High']:.2f}
Low: {row['Low']:.2f}
Close: {row['Close']:.2f}
Volume: {row['Volume']:,.0f}
Change: {((row['Close'] - row['Open']) / row['Open'] * 100):+.2f}%"""

            messagebox.showinfo("Bar Details", info)

    def show_context_menu(self, event):
        """Show context menu on right click"""
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(label="Toggle Crosshair", command=self.toggle_crosshair)
        menu.add_command(label="Reset Zoom", command=self.reset_zoom)
        menu.add_separator()
        menu.add_command(label="Clear All Drawings", command=self.clear_drawings)
        menu.add_separator()
        menu.add_command(label="Export Chart", command=self.export_chart)

        try:
            menu.tk_popup(event.guiEvent.x_root, event.guiEvent.y_root)
        finally:
            menu.grab_release()

    def toggle_crosshair(self):
        """Toggle crosshair visibility"""
        self.crosshair_enabled = not self.crosshair_enabled
        if not self.crosshair_enabled:
            if self.crosshair_vline:
                self.crosshair_vline.remove()
            if self.crosshair_hline:
                self.crosshair_hline.remove()
            self.canvas.draw()
        self.status_var.set(f"Crosshair {'enabled' if self.crosshair_enabled else 'disabled'}")

    def zoom_in(self, center_x=None):
        """Zoom in on the chart"""
        if not hasattr(self, 'axes') or not self.axes:
            return

        xlim = self.axes[0].get_xlim()
        ylim = self.axes[0].get_ylim()

        # Calculate zoom center
        if center_x is None:
            center_x = (xlim[0] + xlim[1]) / 2

        # Zoom factor
        zoom_factor = 0.8
        x_range = (xlim[1] - xlim[0]) * zoom_factor
        y_range = (ylim[1] - ylim[0]) * zoom_factor

        # Set new limits
        new_xlim = (center_x - x_range/2, center_x + x_range/2)
        new_ylim = (ylim[0] + (ylim[1]-ylim[0])*(1-zoom_factor)/2,
                   ylim[1] - (ylim[1]-ylim[0])*(1-zoom_factor)/2)

        for ax in self.axes:
            ax.set_xlim(new_xlim)
            if ax == self.axes[0]:  # Only zoom Y for price chart
                ax.set_ylim(new_ylim)

        self.canvas.draw()
        self.zoom_level *= (1 / zoom_factor)

    def zoom_out(self, center_x=None):
        """Zoom out on the chart"""
        if not hasattr(self, 'axes') or not self.axes:
            return

        xlim = self.axes[0].get_xlim()
        ylim = self.axes[0].get_ylim()

        # Calculate zoom center
        if center_x is None:
            center_x = (xlim[0] + xlim[1]) / 2

        # Zoom factor
        zoom_factor = 1.25
        x_range = (xlim[1] - xlim[0]) * zoom_factor
        y_range = (ylim[1] - ylim[0]) * zoom_factor

        # Set new limits
        new_xlim = (center_x - x_range/2, center_x + x_range/2)
        new_ylim = (ylim[0] - (ylim[1]-ylim[0])*(zoom_factor-1)/2,
                   ylim[1] + (ylim[1]-ylim[0])*(zoom_factor-1)/2)

        for ax in self.axes:
            ax.set_xlim(new_xlim)
            if ax == self.axes[0]:  # Only zoom Y for price chart
                ax.set_ylim(new_ylim)

        self.canvas.draw()
        self.zoom_level *= (1 / zoom_factor)

    def reset_zoom(self):
        """Reset zoom to show all data"""
        if hasattr(self, 'axes') and self.axes:
            for ax in self.axes:
                ax.autoscale()
            self.canvas.draw()
        self.zoom_level = 1.0
        self.status_var.set("Zoom reset")

    def on_drawing_mode_change(self, event=None):
        """Handle drawing mode change"""
        mode = self.draw_var.get()
        if mode == "none":
            self.drawing_mode = None
            self.status_var.set("Drawing disabled")
        else:
            self.drawing_mode = mode
            self.status_var.set(f"Drawing mode: {mode} - Click and drag to draw")

    def clear_drawings(self):
        """Clear all drawn lines"""
        if hasattr(self, 'axes') and self.axes:
            for ax in self.axes:
                for line in ax.lines[:]:
                    # Keep only the main plot lines, remove drawn lines
                    if hasattr(line, 'get_alpha') and line.get_alpha() in [0.8, 0.7]:
                        line.remove()
        self.drawing_lines.clear()
        self.canvas.draw()
        self.status_var.set("All drawings cleared")

    def prev_timeframe(self):
        """Switch to previous timeframe"""
        current_idx = self.timeframes.index(self.current_timeframe)
        new_idx = (current_idx - 1) % len(self.timeframes)
        self.tf_var.set(self.timeframes[new_idx])
        self.on_timeframe_change()

    def next_timeframe(self):
        """Switch to next timeframe"""
        current_idx = self.timeframes.index(self.current_timeframe)
        new_idx = (current_idx + 1) % len(self.timeframes)
        self.tf_var.set(self.timeframes[new_idx])
        self.on_timeframe_change()

    def export_chart(self):
        """Export chart as image"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("SVG files", "*.svg")],
            title="Export Chart"
        )

        if file_path:
            try:
                self.figure.savefig(file_path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Export Successful", f"Chart exported to {file_path}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export chart: {str(e)}")

    def export_data(self):
        """Export chart data as CSV"""
        if not self.chart_data or self.current_timeframe not in self.chart_data:
            messagebox.showwarning("No Data", "No data to export")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            title="Export Data"
        )

        if file_path:
            try:
                df = self.chart_data[self.current_timeframe]
                df.to_csv(file_path)
                messagebox.showinfo("Export Successful", f"Data exported to {file_path}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export data: {str(e)}")

    def load_data_from_file(self, file_path):
        """Load chart data from CSV file"""
        try:
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)

            # Ensure proper column names
            rename_map = {}
            for col in df.columns:
                if col.lower() == 'open':
                    rename_map[col] = 'Open'
                elif col.lower() == 'high':
                    rename_map[col] = 'High'
                elif col.lower() == 'low':
                    rename_map[col] = 'Low'
                elif col.lower() == 'close':
                    rename_map[col] = 'Close'
                elif col.lower() == 'volume':
                    rename_map[col] = 'Volume'

            if rename_map:
                df.rename(columns=rename_map, inplace=True)

            # Store data for different timeframes
            self.chart_data = {
                '1d': df,
                '1h': self.resample_data(df, '1h'),
                '5m': self.resample_data(df, '5min'),
                '1m': self.resample_data(df, '1min'),
                '1w': self.resample_data(df, '1W')
            }

            self.update_chart()
            self.status_var.set(f"Loaded data from {os.path.basename(file_path)}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data from file: {str(e)}")
            # Fall back to sample data
            self.load_sample_data()
        """Load sample data for demonstration"""
        try:
            # Create sample OHLCV data
            dates = pd.date_range('2024-09-01', '2024-09-30', freq='D')
            np.random.seed(42)

            # Generate realistic price data
            base_price = 220
            prices = []
            for i in range(len(dates)):
                change = np.random.normal(0, 2)
                base_price += change
                prices.append(max(base_price, 200))  # Floor at 200

            # Create OHLCV data
            data = []
            for i, price in enumerate(prices):
                high = price + abs(np.random.normal(0, 1))
                low = price - abs(np.random.normal(0, 1))
                open_price = prices[i-1] if i > 0 else price + np.random.normal(0, 0.5)
                volume = np.random.randint(30000000, 100000000)

                data.append({
                    'Date': dates[i],
                    'Open': open_price,
                    'High': high,
                    'Low': low,
                    'Close': price,
                    'Volume': volume
                })

            df = pd.DataFrame(data)
            df.set_index('Date', inplace=True)

            # Store data for different timeframes
            self.chart_data = {
                '1d': df,
                '1h': self.resample_data(df, '1H'),
                '5m': self.resample_data(df, '5min'),
                '1m': self.resample_data(df, '1min'),
                '1w': self.resample_data(df, '1W')
            }

            self.update_chart()
            self.status_var.set("Sample data loaded successfully")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load sample data: {str(e)}")

    def resample_data(self, df, freq):
        """Resample data to different timeframes"""
        try:
            # Convert 'H' to 'h' for pandas compatibility
            freq = freq.replace('H', 'h')
            resampled = df.resample(freq).agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
            return resampled
        except:
            return df.copy()

    def load_chart_file(self):
        """Load chart data from HTML file"""
        file_path = filedialog.askopenfilename(
            title="Select Chart HTML File",
            filetypes=[("HTML files", "*.html"), ("All files", "*.*")]
        )

        if not file_path:
            return

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract chart data from HTML
            start_marker = 'const chartData = '
            end_marker = ';\n\n        // Current timeframe'

            start_idx = content.find(start_marker)
            if start_idx == -1:
                raise ValueError("Chart data not found in HTML file")

            start_idx += len(start_marker)
            end_idx = content.find(end_marker, start_idx)
            if end_idx == -1:
                end_idx = content.find('};', start_idx) + 2

            json_str = content[start_idx:end_idx].strip()
            if json_str.endswith(','):
                json_str = json_str[:-1]

            # Parse the data
            data = json.loads(json_str)

            # Convert to DataFrames
            self.chart_data = {}
            for tf, tf_data in data.items():
                df_data = json.loads(tf_data)
                df = pd.DataFrame(df_data)

                # Convert date strings to datetime
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()

                self.chart_data[tf] = df

            # Extract ticker and strategy from filename
            filename = os.path.basename(file_path)
            if '_interactive_chart.html' in filename:
                parts = filename.replace('_interactive_chart.html', '').split('_')
                if len(parts) >= 2:
                    self.current_ticker = parts[0]
                    self.current_strategy = '_'.join(parts[1:])

            self.update_chart()
            self.status_var.set(f"Loaded data from {filename}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load chart file: {str(e)}")

    def load_csv_file(self):
        """Load data from CSV file"""
        file_path = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if not file_path:
            return

        try:
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            df = df.sort_index()

            # Ensure required columns exist
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Store data for different timeframes
            self.chart_data = {
                '1d': df,
                '1h': self.resample_data(df, '1H'),
                '5m': self.resample_data(df, '5min'),
                '1m': self.resample_data(df, '1min'),
                '1w': self.resample_data(df, '1W')
            }

            # Extract ticker from filename
            filename = os.path.basename(file_path)
            self.current_ticker = filename.replace('.csv', '').split('_')[0]

            self.update_chart()
            self.status_var.set(f"Loaded data from {filename}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV file: {str(e)}")

    def on_timeframe_change(self, event=None):
        """Handle timeframe selection change"""
        new_timeframe = self.tf_var.get()
        print(f"Timeframe changed from {self.current_timeframe} to {new_timeframe}")  # Debug
        self.current_timeframe = new_timeframe
        self.update_chart()

    def calculate_sma(self, prices, period):
        """Calculate Simple Moving Average"""
        return prices.rolling(window=period).mean()

    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def update_chart(self):
        """Update the chart display"""
        if not self.chart_data or self.current_timeframe not in self.chart_data:
            return

        # Clear previous plot
        self.figure.clear()

        # Get data for current timeframe
        df = self.chart_data[self.current_timeframe]

        if df.empty:
            self.info_label.config(text="No data available for selected timeframe")
            self.canvas.draw()
            return

        # Determine which optional panels we will show
        have_scores = self.orders_df is not None and {'net_score','bull_score','bear_score'}.issubset(set(self.orders_df.columns))
        have_equity = self.equity_df is not None
        component_path = 'short_term_components.csv'
        have_components = os.path.isfile(component_path)
        comp_df = None
        if have_components:
            try:
                comp_df = pd.read_csv(component_path, index_col=0, parse_dates=True)
                if not comp_df.empty:
                    comp_df = comp_df.loc[comp_df.index.intersection(df.index)]
                else:
                    have_components = False
            except Exception as e:
                print(f"[WARN] Could not load component file: {e}")
                have_components = False

        # Dynamic panel layout
        base_layout = [3, 1, 1]
        if have_scores:
            base_layout.append(1)
        if have_equity:
            base_layout.append(1)
        if have_components:
            base_layout.append(1)
        gs = self.figure.add_gridspec(len(base_layout), 1, height_ratios=base_layout, hspace=0.08)
        ax1 = self.figure.add_subplot(gs[0])
        ax2 = self.figure.add_subplot(gs[1], sharex=ax1)
        ax3 = self.figure.add_subplot(gs[2], sharex=ax1)
        ax_scores = ax_equity = ax_components = None
        next_row = 3
        if have_scores:
            ax_scores = self.figure.add_subplot(gs[next_row], sharex=ax1); next_row += 1
        if have_equity:
            ax_equity = self.figure.add_subplot(gs[next_row], sharex=ax1); next_row += 1
        if have_components:
            ax_components = self.figure.add_subplot(gs[next_row], sharex=ax1)
        self.axes = [ax1, ax2, ax3]
        if ax_scores is not None: self.axes.append(ax_scores)
        if ax_equity is not None: self.axes.append(ax_equity)
        if ax_components is not None: self.axes.append(ax_components)

        # Plot candlestick chart
        for idx, row in df.iterrows():
            color = 'green' if row['Close'] >= row['Open'] else 'red'
            ax1.plot([idx, idx], [row['Low'], row['High']], color=color, linewidth=1)
            ax1.plot([idx, idx], [row['Open'], row['Close']],
                    color=color, linewidth=4, picker=True)  # Add picker for click detection

        # Add moving averages & Bollinger Bands
        legend_items = []
        if len(df) >= 50:
            sma20 = self.calculate_sma(df['Close'], 20)
            sma50 = self.calculate_sma(df['Close'], 50)

            ax1.plot(df.index, sma20, color='orange', linewidth=2, label='SMA 20')
            ax1.plot(df.index, sma50, color='purple', linewidth=2, label='SMA 50')
            legend_items.extend(['SMA 20', 'SMA 50'])

        # VWAP overlay (intraday style). If multiple days present, compute session-wise VWAP reset daily
        if len(df) > 0 and {'High','Low','Close','Volume'}.issubset(df.columns):
            try:
                price_typ = (df['High'] + df['Low'] + df['Close']) / 3.0
                # Reset VWAP each day if timeframe is intraday
                if self.current_timeframe in ['1m','5m','1h']:
                    vwap_vals = []
                    for session_date, session_df in df.groupby(df.index.date):
                        vol_cum = session_df['Volume'].cumsum()
                        tpv_cum = (price_typ.loc[session_df.index] * session_df['Volume']).cumsum()
                        vwap_session = tpv_cum / vol_cum.replace(0, np.nan)
                        vwap_vals.append(vwap_session)
                    vwap_series = pd.concat(vwap_vals).sort_index()
                else:
                    vol_cum = df['Volume'].cumsum()
                    tpv_cum = (price_typ * df['Volume']).cumsum()
                    vwap_series = tpv_cum / vol_cum.replace(0, np.nan)
                ax1.plot(df.index, vwap_series, color='brown', linewidth=1.4, linestyle='-', label='VWAP')
                legend_items.append('VWAP')
            except Exception as e:
                print(f"[WARN] VWAP calc failed: {e}")

        # Bollinger Bands (20, 2) if enough data
        if len(df) >= 20:
            mid = self.calculate_sma(df['Close'], 20)
            std = df['Close'].rolling(20).std()
            upper = mid + 2 * std
            lower = mid - 2 * std
            ax1.plot(df.index, upper, color='grey', linewidth=1, linestyle='--', alpha=0.6, label='BB Upper')
            ax1.plot(df.index, lower, color='grey', linewidth=1, linestyle='--', alpha=0.6, label='BB Lower')
            legend_items.extend(['BB Upper', 'BB Lower'])

        # Volume bars
        colors = ['green' if close >= open_val else 'red'
                 for close, open_val in zip(df['Close'], df['Open'])]
        ax2.bar(df.index, df['Volume'], color=colors, alpha=0.7, picker=True)

        # RSI
        if len(df) >= 14:
            rsi = self.calculate_rsi(df['Close'], 14)
            ax3.plot(df.index, rsi, color='blue', linewidth=2, label='RSI 14')
            ax3.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
            ax3.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
            legend_items.extend(['RSI 14', 'Overbought (70)', 'Oversold (30)'])
            # Stochastic Oscillator (%K and %D)
            try:
                # We'll reuse a manual stochastic computation (14,3,3) using rolling operations
                high14 = df['High'].rolling(14)
                low14 = df['Low'].rolling(14)
                hh = high14.max()
                ll = low14.min()
                k = (df['Close'] - ll) / (hh - ll) * 100
                d = k.rolling(3).mean()
                k_sm = k.rolling(3).mean()  # slow %K equivalent
                # Plot only where sufficient data
                ax3.plot(df.index, k_sm, color='magenta', linewidth=1, label='%K (14,3)')
                ax3.plot(df.index, d, color='orange', linewidth=1, label='%D (3)')
                ax3.axhline(y=80, color='maroon', linestyle=':', alpha=0.6)
                ax3.axhline(y=20, color='darkgreen', linestyle=':', alpha=0.6)
                legend_items.extend(['%K (14,3)','%D (3)'])
            except Exception as e:
                print(f"[WARN] Stochastic calc failed: {e}")

        # Formatting
        ax1.set_title(f'{self.current_ticker.upper()} - {self.current_timeframe.upper()} Chart ({self.current_strategy.upper()} Strategy)')
        ax1.set_ylabel('Price')
        ax1.grid(True, alpha=0.3)

        ax2.set_ylabel('Volume')
        ax2.grid(True, alpha=0.3)

        ax3.set_ylabel('RSI')
        ax3.legend(loc='upper left', fontsize=8)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 100)

        # Plot scores (signals) if available
        if have_scores and ax_scores is not None:
            # Align order timestamps to price index for plotting scores
            scores_df = self.orders_df.dropna(subset=['timestamp']).copy()
            # Use only rows that have net score values (signal rows)
            scores_df = scores_df[~scores_df['net_score'].isna()]
            if not scores_df.empty:
                ax_scores.plot(scores_df['timestamp'], scores_df['net_score'], color='black', label='Net Score', linewidth=1.5)
                ax_scores.plot(scores_df['timestamp'], scores_df['bull_score'], color='green', alpha=0.7, label='Bull Score', linewidth=1)
                ax_scores.plot(scores_df['timestamp'], scores_df['bear_score'], color='red', alpha=0.7, label='Bear Score', linewidth=1)
                ax_scores.axhline(0, color='grey', linewidth=1, alpha=0.5)
                ax_scores.set_ylabel('Scores')
                ax_scores.legend(loc='upper left', fontsize=8)
                ax_scores.grid(True, alpha=0.3)

        # Plot equity curve if available
        if have_equity and ax_equity is not None and self.equity_df is not None:
            ax_equity.plot(self.equity_df.index, self.equity_df['value'], color='teal', label='Equity', linewidth=1.5)
            ax_equity.set_ylabel('Equity')
            ax_equity.legend(loc='upper left', fontsize=8)
            ax_equity.grid(True, alpha=0.3)
            # Show % change annotation
            try:
                start_val = float(self.equity_df['value'].iloc[0])
                end_val = float(self.equity_df['value'].iloc[-1])
                pct = (end_val / start_val - 1) * 100 if start_val else 0
                ax_equity.text(0.01, 0.95, f"Return: {pct:.2f}%", transform=ax_equity.transAxes,
                               fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', fc='white', alpha=0.4))
            except Exception:
                pass

        # Component stacked contributions
        if have_components and ax_components is not None and comp_df is not None and not comp_df.empty:
            cols = [c for c in ['rsi_mean_reversion','boll_lower','boll_upper','short_above_long','short_below_long','momentum_slope','vwap_trend','stoch_osc','mfi_flow','price_momentum'] if c in comp_df.columns]
            if cols:
                stacked = comp_df[cols].fillna(0)
                ax_components.stackplot(stacked.index, [stacked[c] for c in cols], labels=cols, alpha=0.75)
                ax_components.set_ylabel('Components')
                ax_components.legend(loc='upper left', fontsize=6, ncol=3)
                ax_components.grid(True, alpha=0.3)
        bottom_ax = ax_components or ax_equity or ax_scores or ax3
        # Hover tooltip
        if not hasattr(self, 'hover_annotation'):
            self.hover_annotation = self.figure.text(0.01, 0.01, '', fontsize=8, va='bottom', ha='left', alpha=0.0,
                                                     bbox=dict(boxstyle='round', fc='white', ec='grey', alpha=0.85))
        def _format_hover(ts):
            out = []
            if have_scores and 'scores_df' in locals() and not scores_df.empty:
                try:
                    nearest = scores_df['timestamp'].get_indexer([ts], method='nearest')[0]
                    srow = scores_df.iloc[nearest]
                    out.append(f"Net {srow.net_score:.3f} (B {srow.bull_score:.3f}/R {srow.bear_score:.3f})")
                except Exception:
                    pass
            if have_components and comp_df is not None and ts in comp_df.index:
                top = comp_df.loc[ts].sort_values(ascending=False).head(3)
                out.append(', '.join(f"{k}:{v:.2f}" for k, v in top.items()))
            return '\n'.join(out)
        if not hasattr(self, '_installed_component_hover'):
            orig_motion = self.on_mouse_move
            def wrapped_motion(event):
                orig_motion(event)
                if event.xdata is None or event.inaxes not in self.axes:
                    return
                try:
                    dt_cursor = mdates.num2date(event.xdata)
                    nearest = df.index.get_indexer([dt_cursor], method='nearest')[0]
                    ts = df.index[nearest]
                    txt = _format_hover(ts)
                    if txt:
                        self.hover_annotation.set_text(txt)
                        self.hover_annotation.set_alpha(1.0)
                    else:
                        self.hover_annotation.set_alpha(0.0)
                    self.canvas.draw_idle()
                except Exception:
                    pass
            self.canvas.mpl_disconnect(self.canvas.mpl_connect('motion_notify_event', wrapped_motion))
            self._installed_component_hover = True

        # Formatting x-axis dates for the bottom-most axis only
        bottom_ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        bottom_ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(bottom_ax.xaxis.get_majorticklabels(), rotation=45)

        # Update info
        data_points = len(df)
        date_range = f"{df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}"
        self.info_label.config(text=f"Ticker: {self.current_ticker} | Data Points: {data_points} | Range: {date_range}")

        # Redraw canvas
        self.canvas.draw()

        # Update status
        self.status_var.set(f"Chart updated - {self.current_timeframe.upper()} timeframe | Zoom: {self.zoom_level:.1f}x | Drawing: {self.drawing_mode or 'none'}")

def main():
    parser = argparse.ArgumentParser(description='Interactive Chart Viewer')
    parser.add_argument('--ticker', '-t', default='AAPL', help='Stock ticker symbol')
    parser.add_argument('--strategy', '-s', default='day_trading', help='Trading strategy name')
    parser.add_argument('--data', '-d', help='CSV file containing price data')
    parser.add_argument('--orders', '-o', help='CSV file containing orders & signals data (orders_diagnostics.csv)')
    parser.add_argument('--equity', '-e', help='CSV file containing equity curve (equity_curve.csv)')

    args = parser.parse_args()

    root = tk.Tk()
    app = InteractiveChartViewer(
        root,
        ticker=args.ticker,
        strategy=args.strategy,
        data_file=args.data,
        orders_file=args.orders,
        equity_file=args.equity
    )
    root.mainloop()

if __name__ == "__main__":
    main()