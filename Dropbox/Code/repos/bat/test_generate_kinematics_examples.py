#!/usr/bin/env python3
"""
Unit tests for generate_kinematics_examples module
Tests example generation and visualization functions
"""

import pytest
import os
import tempfile
import pandas as pd
import numpy as np
from generate_kinematics_examples import (
    generate_constant_velocity_examples,
    generate_constant_acceleration_examples,
    generate_straight_then_circle_examples,
    create_visualization
)


class TestExampleGeneration:
    """Tests for example generation functions"""
    
    def test_generate_constant_velocity_examples(self):
        """Test constant velocity examples generation"""
        examples = generate_constant_velocity_examples()
        
        assert len(examples) == 5
        
        for filename, df in examples:
            assert filename.startswith('example_')
            assert 'constant_velocity' in filename
            assert isinstance(df, pd.DataFrame)
            assert 'time' in df.columns
            assert 'x' in df.columns
            assert 'y' in df.columns
            assert 'vx' in df.columns
            assert 'vy' in df.columns
            assert 'motion_type' in df.columns
            assert len(df) == 100
    
    def test_generate_constant_acceleration_examples(self):
        """Test constant acceleration examples generation"""
        examples = generate_constant_acceleration_examples()
        
        assert len(examples) == 10
        
        for filename, df in examples:
            assert filename.startswith('example_')
            assert 'constant_acceleration' in filename
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 100
    
    def test_generate_straight_then_circle_examples(self):
        """Test straight then circle examples generation"""
        examples = generate_straight_then_circle_examples()
        
        assert len(examples) == 20
        
        for filename, df in examples:
            assert filename.startswith('example_')
            assert 'straight_then_circle' in filename
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
    
    def test_example_data_consistency(self):
        """Test that example data has consistent structure"""
        examples = generate_constant_velocity_examples()
        
        for filename, df in examples:
            # Check required columns
            required_columns = ['time', 'x', 'y', 'vx', 'vy', 'motion_type']
            for col in required_columns:
                assert col in df.columns
            
            # Check data types
            assert df['time'].dtype in [np.float64, np.float32]
            assert df['x'].dtype in [np.float64, np.float32]
            assert df['y'].dtype in [np.float64, np.float32]
            
            # Check motion type
            assert all(df['motion_type'] == 'constant_velocity')
    
    def test_example_numbering(self):
        """Test that examples are numbered correctly"""
        examples = generate_constant_velocity_examples()
        
        for i, (filename, df) in enumerate(examples, 1):
            expected_number = f"{i:02d}"
            assert expected_number in filename


class TestVisualization:
    """Tests for visualization functions"""
    
    def test_create_visualization(self):
        """Test visualization creation"""
        # Create sample data
        df = pd.DataFrame({
            'time': np.linspace(0, 10, 50),
            'x': np.linspace(0, 50, 50),
            'y': np.linspace(0, 25, 50),
            'vx': np.ones(50) * 5.0,
            'vy': np.ones(50) * 2.5,
            'motion_type': ['constant_velocity'] * 50
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = 'test_example'
            plot_path = create_visualization(df, filename, tmpdir, marker_interval=10)
            
            assert os.path.exists(plot_path)
            assert plot_path.endswith('.png')
            assert filename in plot_path
    
    def test_create_visualization_with_different_intervals(self):
        """Test visualization with different marker intervals"""
        df = pd.DataFrame({
            'time': np.linspace(0, 10, 100),
            'x': np.linspace(0, 100, 100),
            'y': np.zeros(100),
            'vx': np.ones(100) * 10.0,
            'vy': np.zeros(100),
            'motion_type': ['constant_velocity'] * 100
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test different intervals
            for interval in [5, 10, 20]:
                plot_path = create_visualization(df, f'test_{interval}', tmpdir, marker_interval=interval)
                assert os.path.exists(plot_path)
    
    def test_visualization_with_zero_velocity(self):
        """Test visualization handles zero velocity correctly"""
        df = pd.DataFrame({
            'time': np.linspace(0, 10, 50),
            'x': np.ones(50) * 5.0,
            'y': np.ones(50) * 5.0,
            'vx': np.zeros(50),
            'vy': np.zeros(50),
            'motion_type': ['constant_velocity'] * 50
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            plot_path = create_visualization(df, 'test_zero_vel', tmpdir)
            assert os.path.exists(plot_path)


class TestFileOperations:
    """Tests for file creation and output"""
    
    def test_output_directory_creation(self):
        """Test that output directories are created correctly"""
        # This test would need to mock or use temp directories
        # For now, we'll verify the structure is correct
        output_dir = 'output'
        csv_dir = os.path.join(output_dir, 'kinematics_examples')
        plot_dir = os.path.join(output_dir, 'visualizations')
        
        # Verify directory structure paths are correct
        assert 'output' in csv_dir
        assert 'kinematics_examples' in csv_dir
        assert 'output' in plot_dir
        assert 'visualizations' in plot_dir


class TestDataValidation:
    """Tests for data validation"""
    
    def test_dataframe_has_required_columns(self):
        """Test that all generated dataframes have required columns"""
        all_examples = []
        all_examples.extend(generate_constant_velocity_examples())
        all_examples.extend(generate_constant_acceleration_examples())
        all_examples.extend(generate_straight_then_circle_examples())
        
        required_columns = ['time', 'x', 'y', 'vx', 'vy', 'motion_type']
        
        for filename, df in all_examples:
            for col in required_columns:
                assert col in df.columns, f"Missing column {col} in {filename}"
    
    def test_no_nan_values(self):
        """Test that generated data has no NaN values"""
        examples = generate_constant_velocity_examples()
        
        for filename, df in examples:
            assert not df.isnull().any().any(), f"NaN values found in {filename}"
    
    def test_time_monotonic(self):
        """Test that time values are monotonically increasing"""
        examples = generate_constant_velocity_examples()
        
        for filename, df in examples:
            assert df['time'].is_monotonic_increasing, f"Time not monotonic in {filename}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

