import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional

class VolumeVisualizer:
    @staticmethod
    def create_volume_profile(data: pd.DataFrame, bins: int = 12) -> go.Figure:
        """Create volume profile visualization"""
        # Calculate volume profile
        price_bins = pd.qcut(data['Close'], bins)
        volume_profile = data.groupby(price_bins)['Volume'].sum()
        
        # Create horizontal bar chart
        fig = go.Figure(go.Bar(
            y=[str(interval) for interval in volume_profile.index],
            x=volume_profile.values,
            orientation='h',
            name='Volume Profile',
        ))
        
        # Customize layout
        fig.update_layout(
            title='Volume Profile Analysis',
            xaxis_title='Volume',
            yaxis_title='Price Levels',
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def create_combined_indicators_plot(data: pd.DataFrame, 
                                      klinger: pd.Series,
                                      force_idx: pd.Series,
                                      symbol: str) -> go.Figure:
        """Create a combined plot of volume indicators"""
        fig = make_subplots(rows=3, cols=1,
                           shared_xaxes=True,
                           vertical_spacing=0.05,
                           row_heights=[0.5, 0.25, 0.25],
                           subplot_titles=(f'{symbol} Price & Volume',
                                        'Klinger Oscillator',
                                        'Force Index'))
        
        # Price and volume plot
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price'
        ), row=1, col=1)
        
        fig.add_trace(go.Bar(
            x=data.index,
            y=data['Volume'],
            name='Volume',
            opacity=0.5
        ), row=1, col=1)
        
        # Klinger oscillator plot
        fig.add_trace(go.Scatter(
            x=data.index,
            y=klinger,
            name='Klinger Oscillator',
            line=dict(color='blue')
        ), row=2, col=1)
        
        # Force Index plot
        fig.add_trace(go.Scatter(
            x=data.index,
            y=force_idx,
            name='Force Index',
            line=dict(color='orange')
        ), row=3, col=1)
        
        fig.update_layout(
            title=f'{symbol} Volume Analysis',
            height=800,
            showlegend=True,
            xaxis3_title='Date'
        )
        
        return fig
        fig.update_layout(
            title='Volume Profile Analysis',
            yaxis_title='Price Levels',
            xaxis_title='Volume',
            showlegend=True,
            height=800
        )
        
        return fig

    @staticmethod
    def plot_volume_indicators(data: pd.DataFrame, indicators: Dict[str, pd.Series]) -> go.Figure:
        """Create comprehensive volume analysis chart"""
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxis=True,
            vertical_spacing=0.05,
            subplot_titles=('Price & Volume', 'Volume Indicators', 'Money Flow'),
            row_heights=[0.5, 0.25, 0.25]
        )

        # Candlestick chart with volume
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Price'
            ),
            row=1, col=1
        )

        # Volume bars
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume',
                marker_color='rgba(0,128,0,0.3)'
            ),
            row=2, col=1
        )

        # Add volume indicators
        colors = ['blue', 'red', 'green', 'purple', 'orange']
        color_idx = 0
        
        for name, indicator in indicators.items():
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=indicator,
                    name=name,
                    line=dict(color=colors[color_idx % len(colors)])
                ),
                row=3, col=1
            )
            color_idx += 1

        fig.update_layout(
            title='Volume Analysis Dashboard',
            yaxis_title='Price',
            yaxis2_title='Volume',
            yaxis3_title='Indicator Values',
            xaxis3_title='Date',
            height=1000,
            showlegend=True
        )

        return fig

    @staticmethod
    def create_smart_money_flow(data: pd.DataFrame, smi: pd.Series) -> go.Figure:
        """Create Smart Money Flow visualization"""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxis=True,
            vertical_spacing=0.05,
            subplot_titles=('Price', 'Smart Money Index'),
            row_heights=[0.6, 0.4]
        )

        # Price chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Price'
            ),
            row=1, col=1
        )

        # Smart Money Index
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=smi,
                name='Smart Money Index',
                line=dict(color='blue')
            ),
            row=2, col=1
        )

        fig.update_layout(
            title='Smart Money Flow Analysis',
            yaxis_title='Price',
            yaxis2_title='SMI Value',
            xaxis2_title='Date',
            height=800,
            showlegend=True
        )

        return fig

    @staticmethod
    def plot_institutional_activity(data: pd.DataFrame, inst_activity: pd.Series) -> go.Figure:
        """Visualize institutional trading activity"""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxis=True,
            vertical_spacing=0.05,
            subplot_titles=('Price & Institutional Activity', 'Volume'),
            row_heights=[0.7, 0.3]
        )

        # Price chart with institutional activity markers
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Price'
            ),
            row=1, col=1
        )

        # Add markers for institutional activity
        inst_dates = data.index[inst_activity]
        inst_prices = data.loc[inst_activity, 'Close']
        
        fig.add_trace(
            go.Scatter(
                x=inst_dates,
                y=inst_prices,
                mode='markers',
                name='Institutional Activity',
                marker=dict(
                    size=10,
                    symbol='triangle-up',
                    color='green',
                    line=dict(width=2, color='darkgreen')
                )
            ),
            row=1, col=1
        )

        # Volume
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume',
                marker_color='rgba(0,128,0,0.3)'
            ),
            row=2, col=1
        )

        fig.update_layout(
            title='Institutional Trading Activity',
            yaxis_title='Price',
            yaxis2_title='Volume',
            xaxis2_title='Date',
            height=800,
            showlegend=True
        )

        return fig

    @staticmethod
    def save_all_charts(data: pd.DataFrame, output_dir: str) -> None:
        """Generate and save all volume analysis charts"""
        from pathlib import Path
        import plotly.io as pio
        
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Calculate all indicators
        from strategies.volume_analysis import VolumeAnalysis
        
        va = VolumeAnalysis()
        vwap = va.calculate_vwap(data)
        vol_corr = va.volume_price_correlation(data)
        smi = va.smart_money_index(data)
        inst_activity = va.institutional_activity(data)
        
        # Create and save all charts
        indicators = {
            'VWAP': vwap,
            'Volume-Price Correlation': vol_corr,
            'Smart Money Index': smi
        }
        
        # Volume Profile
        vol_profile = VolumeVisualizer.create_volume_profile(data)
        pio.write_html(vol_profile, f'{output_dir}/volume_profile.html')
        
        # Volume Indicators
        vol_indicators = VolumeVisualizer.plot_volume_indicators(data, indicators)
        pio.write_html(vol_indicators, f'{output_dir}/volume_indicators.html')
        
        # Smart Money Flow
        smart_money = VolumeVisualizer.create_smart_money_flow(data, smi)
        pio.write_html(smart_money, f'{output_dir}/smart_money_flow.html')
        
        # Institutional Activity
        inst_chart = VolumeVisualizer.plot_institutional_activity(data, inst_activity)
        pio.write_html(inst_chart, f'{output_dir}/institutional_activity.html')