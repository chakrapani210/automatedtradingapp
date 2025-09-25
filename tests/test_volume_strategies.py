import sys
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import pytest
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

from strategies.volume_analysis import VolumeAnalysis
from strategies.enhanced_volume import EnhancedVolumeAnalysis
from position_sizing.volume_sizer import VolumeBasedPositionSizer
from visualization.volume_visualizer import VolumeVisualizer

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    dates = pd.date_range(start='2024-01-01', periods=100)
    data = pd.DataFrame({
        'Open': np.random.randn(100).cumsum() + 100,
        'High': np.random.randn(100).cumsum() + 102,
        'Low': np.random.randn(100).cumsum() + 98,
        'Close': np.random.randn(100).cumsum() + 100,
        'Volume': np.abs(np.random.randn(100) * 1000000).astype(int)
    }, index=dates)
    return data

@pytest.fixture
def config():
    """Create test configuration"""
    return {
        'trading': {
            'max_position_size': 1000000,
            'min_position_size': 1000,
            'default_risk_percent': 0.02,
            'max_risk_percent': 0.05
        },
        'volume_analysis': {
            'support_resistance': {
                'lookback_period': 20,
                'volume_threshold': 1.5,
                'price_buffer': 0.02
            }
        }
    }

def test_volume_analysis_basic(sample_data):
    """Test basic volume analysis features"""
    va = VolumeAnalysis()
    
    # Test VWAP calculation
    vwap = va.calculate_vwap(sample_data)
    assert isinstance(vwap, pd.Series)
    assert len(vwap) == len(sample_data)
    assert not vwap.isnull().any()
    assert all(vwap > 0)

def test_volume_analysis_enhanced(sample_data):
    """Test enhanced volume analysis features"""
    eva = EnhancedVolumeAnalysis()
    
    # Test Klinger oscillator
    klinger = eva.calculate_klinger_oscillator(sample_data)
    assert isinstance(klinger, pd.Series)
    assert len(klinger) == len(sample_data)
    
    # Test Force Index
    force_idx = eva.volume_force_index(sample_data)
    assert isinstance(force_idx, pd.Series)
    assert len(force_idx) == len(sample_data)
    
    # Test volume weighted price ranges
    price_data = sample_data['Close']
    volume_data = sample_data['Volume']
    vw_high = (price_data * volume_data).max() / volume_data.max()
    vw_low = (price_data * volume_data).min() / volume_data.max()
    
    assert vw_high > vw_low
    assert isinstance(vw_high, float)
    assert isinstance(vw_low, float)

def test_position_sizing(sample_data, config):
    """Test position sizing calculations"""
    # Create a config object that matches the expected structure
    class Strategy:
        max_position_size = config['trading']['max_position_size']
        min_position_size = config['trading']['min_position_size']
        default_risk_percent = config['trading']['default_risk_percent']
        
    class Config:
        strategy = Strategy()
    
    sizer = VolumeBasedPositionSizer(Config())
    
    # Test position sizing calculation
    size, metrics = sizer.calculate_position_size(
        sample_data,
        100000,  # capital
        1,  # signal
        0.02  # risk percent
    )
    
    assert isinstance(size, (int, float))
    assert size >= 0
    assert isinstance(metrics, dict)
    assert 'liquidity_score' in metrics
    assert 'risk_factor' in metrics
    assert metrics['liquidity_score'] >= 0
    assert metrics['risk_factor'] >= 0

def test_visualization(sample_data, tmp_path):
    """Test visualization functions"""
    visualizer = VolumeVisualizer()
    
    # Test volume profile
    fig = visualizer.create_volume_profile(sample_data)
    assert fig is not None
    
    # Save test output
    output_file = tmp_path / "test_volume_profile.html"
    fig.write_html(str(output_file))
    assert output_file.exists()
    
    # Test basic price plot with volume
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sample_data.index,
        y=sample_data['Close'],
        name='Price'
    ))
    fig.add_trace(go.Bar(
        x=sample_data.index,
        y=sample_data['Volume'],
        name='Volume'
    ))
    assert fig is not None

if __name__ == "__main__":
    pytest.main([__file__, "-v"])