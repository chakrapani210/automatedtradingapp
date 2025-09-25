from typing import Dict, List, Tuple
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Import our modules
from data.data_acquisition import DataAcquisition
from strategies.volume_analysis import VolumeAnalysis
from strategies.enhanced_volume import EnhancedVolumeAnalysis
from position_sizing.volume_sizer import VolumeBasedPositionSizer
from visualization.volume_visualizer import VolumeVisualizer
from utils.app_config import AppConfig

def analyze_volume_patterns(data: pd.DataFrame) -> Dict[str, str]:
    """Analyze volume patterns and provide trading insights"""
    va = VolumeAnalysis()
    eva = EnhancedVolumeAnalysis()
    
    # Calculate key metrics
    volume_sma = data['Volume'].rolling(window=20).mean()
    price_change = data['Close'].pct_change()
    
    # Analyze recent volume patterns
    recent_volume = data['Volume'].tail(5)
    avg_volume = volume_sma.tail(5).mean()
    volume_trend = 'increasing' if recent_volume.mean() > avg_volume else 'decreasing'
    
    # Analyze price-volume relationship
    vol_price_corr = data['Close'].corr(data['Volume'])
    
    # Get volume-based levels
    levels = eva.volume_based_support_resistance(data)
    current_price = data['Close'].iloc[-1]
    
    # Calculate trading signals
    klinger = eva.calculate_klinger_oscillator(data)
    force_idx = eva.volume_force_index(data)
    
    # Analyze institutional activity
    inst_activity = va.institutional_activity(data)
    recent_inst = inst_activity.tail(5).sum()
    
    # Generate insights
    insights = {}
    
    # Volume trend analysis
    insights['volume_trend'] = (
        f"Volume is {volume_trend}. "
        f"Current volume is {recent_volume.iloc[-1]/avg_volume:.1f}x "
        f"the 20-day average."
    )
    
    # Price-volume relationship
    insights['price_volume'] = (
        f"Price-volume correlation is {vol_price_corr:.2f}, indicating "
        f"{'strong' if abs(vol_price_corr) > 0.7 else 'moderate' if abs(vol_price_corr) > 0.4 else 'weak'} "
        f"{'positive' if vol_price_corr > 0 else 'negative'} relationship."
    )
    
    # Support/Resistance analysis
    insights['key_levels'] = (
        f"Key volume-based support at ${levels['support']:.2f}, "
        f"resistance at ${levels['resistance']:.2f}. "
        f"Current price is ${current_price:.2f}, "
        f"{'near support' if abs(current_price - levels['support'])/current_price < 0.02 else 'near resistance' if abs(current_price - levels['resistance'])/current_price < 0.02 else 'between levels'}."
    )
    
    # Institutional activity
    insights['institutional'] = (
        f"{'Strong' if recent_inst >= 3 else 'Moderate' if recent_inst >= 1 else 'Low'} "
        f"institutional activity detected in the last 5 days "
        f"({recent_inst} days with significant activity)."
    )
    
    # Trading signals
    insights['signals'] = (
        f"Klinger Oscillator: {'Bullish' if klinger.iloc[-1] > 0 else 'Bearish'}, "
        f"Force Index: {'Bullish' if force_idx.iloc[-1] > 0 else 'Bearish'}. "
        f"Overall volume analysis suggests "
        f"{'bullish' if klinger.iloc[-1] > 0 and force_idx.iloc[-1] > 0 else 'bearish' if klinger.iloc[-1] < 0 and force_idx.iloc[-1] < 0 else 'mixed'} "
        f"sentiment."
    )
    
    return insights

import pytest

@pytest.fixture
def setup_test_env():
    """Setup test environment with configuration and directories"""
    import os
    project_root = os.path.dirname(os.path.dirname(__file__))
    config_path = os.path.join(project_root, 'config.yaml')
    config = AppConfig.load_from_yaml(config_path)
    
    # Create output directories
    output_dir = Path(os.path.join(project_root, "analysis_output"))
    output_dir.mkdir(exist_ok=True)
    
    test_dir = os.path.join(project_root, 'test_output')
    os.makedirs(test_dir, exist_ok=True)
    
    return {
        'project_root': project_root,
        'test_dir': test_dir
    }
    
    return project_root
    test_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_dir = output_dir / f"test_run_{test_time}"
    test_dir.mkdir(exist_ok=True)
    
    return {
        'config': config,
        'output_dir': output_dir,
        'test_dir': test_dir
    }

def test_basic_volume_indicators(setup_test_env):
    """Test calculation of basic volume indicators"""
    # Load test data
    project_root = setup_test_env['project_root']
    test_dir = setup_test_env['test_dir']
    data = pd.read_csv(os.path.join(project_root, "test_data/AAPL_daily.csv"))
    data.set_index('Date', inplace=True)
    
    # Initialize analysis objects
    va = VolumeAnalysis()
    eva = EnhancedVolumeAnalysis()
    
    # Test On Balance Volume
    obv = va.on_balance_volume(data)
    assert isinstance(obv, pd.Series), "OBV should return a pandas Series"
    assert len(obv) == len(data), "OBV length should match data length"
    assert not obv.isnull().any(), "OBV should not contain NaN values"
    
    # Test VWAP
    vwap = va.calculate_vwap(data)
    assert isinstance(vwap, pd.Series), "VWAP should return a pandas Series"
    assert len(vwap) == len(data), "VWAP length should match data length"
    assert not vwap.isnull().any(), "VWAP should not contain NaN values"
    assert all(vwap > 0), "VWAP values should be positive"
    
    # Load test data
    data = pd.read_csv(os.path.join(project_root, "test_data/AAPL_daily.csv"))
    data.set_index('Date', inplace=True)
    
    # Initialize analysis objects
    va = VolumeAnalysis()
    eva = EnhancedVolumeAnalysis()
    
    # Save summary report
    report_file = os.path.join(test_dir, "volume_analysis_report.txt")
    with open(report_file, "w") as f:
        f.write("=== Volume Analysis Test Report ===\n\n")
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Symbol: AAPL\n")
        f.write(f"Data Range: {data.index[0]} to {data.index[-1]}\n\n")
        
        print("Testing basic volume indicators...")
        f.write("1. Basic Volume Indicators\n")
        f.write("-----------------------\n")
        try:
            obv = va.on_balance_volume(data)
            vwap = va.calculate_vwap(data)
            volume_trend = va.volume_trend_strength(data)
            f.write("[PASS] On Balance Volume (OBV):\n")
            f.write(f"   Current: {obv.iloc[-1]:.2f}\n")
            f.write(f"   5-day change: {((obv.iloc[-1] - obv.iloc[-6])/obv.iloc[-6]*100):.1f}%\n\n")
            
            f.write("[PASS] Volume Weighted Average Price (VWAP):\n")
            f.write(f"   Current: ${vwap.iloc[-1]:.2f}\n")
            f.write(f"   Price relative to VWAP: {((data['Close'].iloc[-1] - vwap.iloc[-1])/vwap.iloc[-1]*100):.1f}%\n\n")
            
            f.write("[PASS] Volume Trend Analysis:\n")
            f.write(f"   {volume_trend}\n\n")
            print("[PASS] Basic volume indicators calculated successfully")
        except Exception as e:
            error_msg = f"[FAIL] Error calculating basic indicators: {str(e)}"
            print(error_msg)
            f.write(f"{error_msg}\n\n")
    
    # Test multi-stock analysis
    print("\nTesting multi-stock analysis...")
    f.write("\n4. Multi-Stock Volume Analysis\n")
    f.write("--------------------------\n")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # One year of data
    tickers = ['AAPL', 'MSFT', 'GOOGL']  # Test with major liquid stocks
    
    try:
        multi_data = DataAcquisition.get_stock_data(tickers, start_date.strftime('%Y-%m-%d'), 
                                                   end_date.strftime('%Y-%m-%d'))
        
        # Initialize our analysis tools
        va = VolumeAnalysis()
        eva = EnhancedVolumeAnalysis()
        sizer = VolumeBasedPositionSizer(config)
        visualizer = VolumeVisualizer()
        
        results = {}
    
        # Process each ticker
        for ticker in tickers:
            print(f"\nAnalyzing {ticker}...")
            f.write(f"\n[PASS] {ticker} Analysis:\n")
            
            # Extract single stock data
            if isinstance(multi_data.columns, pd.MultiIndex):
                stock_data = multi_data[ticker].copy()
            else:
                stock_data = multi_data.copy()
            
            # 1. Basic Volume Analysis
            print("\nBasic Volume Analysis:")
            f.write("   Basic Volume Metrics:\n")
            vwap = va.calculate_vwap(stock_data)
            vol_corr = va.volume_price_correlation(stock_data)
            smi = va.smart_money_index(stock_data)
            
            f.write(f"   - VWAP: ${vwap.iloc[-1]:.2f}\n")
            f.write(f"   - Volume-Price Correlation: {vol_corr.iloc[-1]:.2f}\n")
            f.write(f"   - Smart Money Index: {smi.iloc[-1]:.2f}\n")
            
            # 2. Enhanced Volume Indicators
            print("\nEnhanced Volume Indicators:")
            f.write("\n   Enhanced Indicators:\n")
            klinger = eva.calculate_klinger_oscillator(stock_data)
            eom = eva.ease_of_movement(stock_data)
            force_idx = eva.volume_force_index(stock_data)
            
            f.write(f"   - Klinger Oscillator: {klinger.iloc[-1]:.2f}\n")
            f.write(f"   - Ease of Movement: {eom.iloc[-1]:.2f}\n")
            f.write(f"   - Force Index: {force_idx.iloc[-1]:.2f}\n")
            
            # 3. Position Sizing Test
            print("\nPosition Sizing Analysis:")
            f.write("\n   Position Sizing Analysis:\n")
            test_capital = 100000  # Test with $100k
            size, metrics = sizer.calculate_position_size(stock_data, test_capital, 1, 0.02)
            
            f.write(f"   - Recommended Size: {size} shares\n")
            f.write(f"   - Liquidity Score: {metrics['liquidity_score']:.2f}\n")
            f.write(f"   - Risk Factor: {metrics['risk_factor']:.2f}\n")
            
            # 4. Support/Resistance Levels
            levels = eva.volume_based_support_resistance(stock_data)
            f.write("\n   Volume-Based Levels:\n")
            f.write(f"   - Support: ${levels['support']:.2f}\n")
            f.write(f"   - Resistance: ${levels['resistance']:.2f}\n")
            f.write(f"   - Current Price: ${stock_data['Close'].iloc[-1]:.2f}\n")
            
            # Add summary to results dictionary
            results[ticker] = {
                'vwap': vwap.iloc[-1],
                'volume_correlation': vol_corr.iloc[-1],
                'smart_money_index': smi.iloc[-1],
                'klinger': klinger.iloc[-1],
                'force_index': force_idx.iloc[-1],
                'position_size': size,
                'liquidity_score': metrics['liquidity_score'],
                'support': levels['support'],
                'resistance': levels['resistance']
            }
        
        # Add comparative analysis
        print("\nGenerating comparative analysis...")
        f.write("\n5. Comparative Analysis\n")
        f.write("--------------------\n")
        
        # Calculate combined metrics for each ticker
        comparative_metrics = {}
        for ticker, metrics in results.items():
            # Volume strength score (0-1 scale)
            strength_score = (
                (0.3 * (1 if metrics['volume_correlation'] > 0.5 else 0)) +
                (0.2 * (1 if metrics['klinger'] > 0 else 0)) +
                (0.2 * (1 if metrics['force_index'] > 0 else 0)) +
                (0.3 * metrics['liquidity_score'])
            )
            
            # Calculate technical status
            tech_status = (
                "Bullish" if metrics['klinger'] > 0 and metrics['force_index'] > 0
                else "Bearish" if metrics['klinger'] < 0 and metrics['force_index'] < 0
                else "Mixed"
            )
            
            # Calculate price position relative to support/resistance
            current_price = multi_data[ticker]['Close'].iloc[-1]
            price_to_support = (current_price - metrics['support']) / metrics['support']
            price_to_resistance = (metrics['resistance'] - current_price) / current_price
            
            comparative_metrics[ticker] = {
                'strength_score': strength_score,
                'technical_status': tech_status,
                'price_to_support': price_to_support,
                'price_to_resistance': price_to_resistance
            }
            
        # Write comparative analysis to report
        f.write("\nVolume Profile Comparison:\n")
        for ticker, metrics in comparative_metrics.items():
            f.write(f"\n{ticker}:\n")
            f.write(f"- Overall Strength Score: {metrics['strength_score']:.2f}\n")
            f.write(f"- Technical Status: {metrics['technical_status']}\n")
            f.write(f"- Distance to Support: {metrics['price_to_support']:.1%}\n")
            f.write(f"- Distance to Resistance: {metrics['price_to_resistance']:.1%}\n")
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        try:
            for ticker in tickers:
                stock_data = multi_data[ticker].copy() if isinstance(multi_data.columns, pd.MultiIndex) else multi_data.copy()
                ticker_dir = test_dir / ticker
                ticker_dir.mkdir(exist_ok=True)
                
                # Generate and save visualizations
                try:
                    # Volume profile plot
                    profile_fig = visualizer.create_volume_profile_plot(stock_data, ticker)
                    profile_fig.write_html(str(ticker_dir / 'volume_profile.html'))
                    
                    # Combined indicators plot
                    indicators_fig = visualizer.create_combined_indicators_plot(
                        stock_data,
                        results[ticker]['volume_correlation'],
                        results[ticker]['klinger'],
                        results[ticker]['force_index'],
                        ticker
                    )
                    indicators_fig.write_html(str(ticker_dir / 'volume_indicators.html'))
                    
                    # Smart money flow
                    flow_fig = visualizer.create_smart_money_flow(
                        stock_data,
                        results[ticker]['smart_money_index'],
                        ticker
                    )
                    flow_fig.write_html(str(ticker_dir / 'smart_money_flow.html'))
                    
                except Exception as viz_error:
                    f.write(f"\n[FAIL] Error generating {ticker} visualizations: {str(viz_error)}\n")
                    continue
            
            print("[PASS] Visualizations generated successfully")
            f.write("\n[PASS] Visualizations saved in test output directory\n")
            
        except Exception as e:
            error_msg = f"[FAIL] Error in visualization process: {str(e)}"
            print(error_msg)
            f.write(f"\n{error_msg}\n")
        
        # Save results to CSV
        try:
            pd.DataFrame(results).round(2).to_csv(test_dir / 'analysis_results.csv')
            f.write("\n[PASS] Analysis results saved to CSV\n")
        except Exception as e:
            f.write(f"\n[FAIL] Error saving results to CSV: {str(e)}\n")
    
            print("\n=== Volume Analysis Test Complete ===")
        print(f"Full report saved to: {report_file}")
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        raise

def test_enhanced_volume_analysis(setup_test_env):
    """Test enhanced volume analysis features"""
    # Load test data
    project_root = setup_test_env['project_root']
    data = pd.read_csv(os.path.join(project_root, "test_data/AAPL_daily.csv"))
    data.set_index('Date', inplace=True)
    
    # Initialize analysis object
    eva = EnhancedVolumeAnalysis()
    
    # Test Klinger oscillator
    klinger = eva.calculate_klinger_oscillator(data)
    assert isinstance(klinger, pd.Series), "Klinger oscillator should return a pandas Series"    
    assert len(klinger) == len(data), "Klinger oscillator length should match data length"       
    # First few values will be NaN due to EMA calculation
    assert not klinger.tail(len(data) - 55).isnull().any(), "Klinger oscillator should not contain NaN values after initial period"
    
    # Test Force Index
    force_idx = eva.volume_force_index(data)
    assert isinstance(force_idx, pd.Series), "Force Index should return a pandas Series"
    assert len(force_idx) == len(data), "Force Index length should match data length"
    # First few values will have NaN due to calculation
    assert not force_idx.tail(len(data) - 13).isnull().any(), "Force Index should not contain NaN values after initial period"
    
    # Test Support/Resistance levels
    levels = eva.volume_based_support_resistance(data)
    assert isinstance(levels, dict), "Support/Resistance should return a dictionary"
    assert 'support' in levels, "Support level should be in results"
    assert 'resistance' in levels, "Resistance level should be in results"
    assert levels['support'] < levels['resistance'], "Support should be lower than resistance"

def test_position_sizing(setup_test_env):
    """Test volume-based position sizing"""
    # Load test data
    project_root = setup_test_env['project_root']
    test_dir = setup_test_env['test_dir']
    data = pd.read_csv(os.path.join(project_root, "test_data/AAPL_daily.csv"))
    data.set_index('Date', inplace=True)
    
    # Create a config object with default settings
    class Strategy:
        max_position_size = 1000000
        min_position_size = 1000
        default_risk_percent = 0.02
    
    class Config:
        strategy = Strategy()
    
    # Initialize position sizer
    sizer = VolumeBasedPositionSizer(Config())
    
    # Test position sizing calculation
    size, metrics = sizer.calculate_position_size(data, 100000, 1, 0.02)
    
    # Validate position size
    assert isinstance(size, int), "Position size should be an integer"
    assert size >= 0, "Position size should be non-negative"
    assert isinstance(metrics, dict), "Metrics should be a dictionary"
    assert 'liquidity_score' in metrics, "Liquidity score should be in metrics"
    assert 'risk_factor' in metrics, "Risk factor should be in metrics"
    assert metrics['liquidity_score'] >= 0, "Liquidity score should be non-negative"
    assert metrics['risk_factor'] >= 0, "Risk factor should be non-negative"

def test_visualization(setup_test_env):
    """Test volume visualization functions"""
    # Load test data
    project_root = setup_test_env['project_root']
    test_dir = setup_test_env['test_dir']
    data = pd.read_csv(os.path.join(project_root, "test_data/AAPL_daily.csv"))
    data.set_index('Date', inplace=True)
    
    # Initialize visualizer
    visualizer = VolumeVisualizer()
    
    # Test volume profile plot
    profile_fig = visualizer.create_volume_profile(data)
    assert profile_fig is not None, "Volume profile plot should be created"
    
    # Test indicators plot
    eva = EnhancedVolumeAnalysis()
    klinger = eva.calculate_klinger_oscillator(data)
    force_idx = eva.volume_force_index(data)
    
    indicators_fig = visualizer.create_combined_indicators_plot(
        data, klinger, force_idx, 'AAPL'
    )
    assert indicators_fig is not None, "Combined indicators plot should be created"
    
    # Save test plots
    test_dir = setup_test_env['test_dir']
    profile_fig.write_html(os.path.join(test_dir, 'test_volume_profile.html'))
    indicators_fig.write_html(os.path.join(test_dir, 'test_indicators.html'))

if __name__ == "__main__":
    pytest.main([__file__, '-v'])