"""Simplified volume analysis tests (clean replacement)."""

import os
import pandas as pd
import pytest

from strategies.volume_analysis import VolumeAnalysis
from strategies.enhanced_volume import EnhancedVolumeAnalysis
from position_sizing.volume_sizer import VolumeBasedPositionSizer
from visualization.volume_visualizer import VolumeVisualizer


@pytest.fixture
def setup_test_env():
    project_root = os.path.dirname(os.path.dirname(__file__))
    test_dir = os.path.join(project_root, 'test_output')
    os.makedirs(test_dir, exist_ok=True)
    return {'project_root': project_root, 'test_dir': test_dir}


def _load_data(project_root):
    path = os.path.join(project_root, 'test_data', 'AAPL_daily.csv')
    if not os.path.exists(path):
        pytest.skip('AAPL_daily.csv missing; skipping volume tests')
    df = pd.read_csv(path)
    if 'Date' in df.columns:
        df.set_index('Date', inplace=True)
    return df


def test_basic_volume_indicators(setup_test_env):
    data = _load_data(setup_test_env['project_root'])
    va = VolumeAnalysis()
    obv = va.on_balance_volume(data)
    vwap = va.calculate_vwap(data)
    assert len(obv) == len(data)
    assert len(vwap) == len(data)
    assert not obv.isnull().any()
    assert not vwap.isnull().any()
    assert all(vwap > 0)


def test_enhanced_volume_analysis(setup_test_env):
    data = _load_data(setup_test_env['project_root'])
    eva = EnhancedVolumeAnalysis()
    klinger = eva.calculate_klinger_oscillator(data)
    force_idx = eva.volume_force_index(data)
    assert not klinger.tail(max(0, len(klinger) - 60)).isnull().any()
    assert not force_idx.tail(max(0, len(force_idx) - 15)).isnull().any()
    levels = eva.volume_based_support_resistance(data)
    assert levels['support'] < levels['resistance']


def test_position_sizing(setup_test_env):
    data = _load_data(setup_test_env['project_root'])
    class Strategy:
        max_position_size = 1000000
        min_position_size = 1
        default_risk_percent = 0.02
    class Config:
        strategy = Strategy()
    sizer = VolumeBasedPositionSizer(Config())
    size, metrics = sizer.calculate_position_size(data, 100000, 1, 0.02)
    assert size >= 0
    assert 'liquidity_score' in metrics
    assert 'risk_factor' in metrics
    assert isinstance(size, int)
    assert metrics['liquidity_score'] >= 0
    assert metrics['risk_factor'] >= 0

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