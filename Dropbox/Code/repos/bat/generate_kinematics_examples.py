#!/usr/bin/env python3
"""
Kinematics Examples Generator
Generates 35 example trajectories with different motion patterns
"""

import numpy as np
import pandas as pd
import os
from kinematics_formulas import generate_trajectory_points


def generate_constant_velocity_examples():
    """Generate examples of constant velocity motion"""
    examples = []
    
    # Example 1-5: Different velocities in different directions
    velocities = [
        [5.0, 0.0],      # Right
        [0.0, 5.0],      # Up
        [3.0, 4.0],      # Diagonal
        [-4.0, 3.0],     # Left-up
        [2.5, -2.5],     # Right-down
    ]
    
    for i, vel in enumerate(velocities, 1):
        params = {
            'initial_position': [0.0, 0.0],
            'velocity': vel
        }
        positions, velocities_arr = generate_trajectory_points(
            'constant_velocity', params, num_points=100, duration=10.0
        )
        
        df = pd.DataFrame({
            'time': np.linspace(0, 10.0, 100),
            'x': positions[:, 0],
            'y': positions[:, 1],
            'vx': velocities_arr[:, 0],
            'vy': velocities_arr[:, 1],
            'motion_type': 'constant_velocity'
        })
        examples.append((f'example_{i:02d}_constant_velocity', df))
    
    return examples


def generate_constant_acceleration_examples():
    """Generate examples of constant acceleration motion"""
    examples = []
    
    # Example 6-15: Different acceleration patterns
    scenarios = [
        {'initial_position': [0.0, 0.0], 'initial_velocity': [0.0, 0.0], 
         'acceleration': [2.0, 0.0], 'acceleration_duration': 5.0},  # Accelerate right
        {'initial_position': [0.0, 0.0], 'initial_velocity': [5.0, 0.0], 
         'acceleration': [-1.0, 0.0], 'acceleration_duration': 3.0},  # Decelerate right
        {'initial_position': [0.0, 0.0], 'initial_velocity': [0.0, 0.0], 
         'acceleration': [0.0, 2.0], 'acceleration_duration': 4.0},  # Accelerate up
        {'initial_position': [0.0, 0.0], 'initial_velocity': [3.0, 0.0], 
         'acceleration': [1.5, 1.0], 'acceleration_duration': 6.0},  # Accelerate diagonal
        {'initial_position': [0.0, 0.0], 'initial_velocity': [10.0, 0.0], 
         'acceleration': [-2.0, 0.0], 'acceleration_duration': 4.0},  # Brake
        {'initial_position': [0.0, 0.0], 'initial_velocity': [2.0, 2.0], 
         'acceleration': [0.5, 0.5], 'acceleration_duration': 8.0},  # Gentle acceleration
        {'initial_position': [0.0, 0.0], 'initial_velocity': [0.0, 0.0], 
         'acceleration': [3.0, 1.0], 'acceleration_duration': 3.0},  # Quick acceleration
        {'initial_position': [0.0, 0.0], 'initial_velocity': [5.0, 5.0], 
         'acceleration': [-1.0, -1.0], 'acceleration_duration': 5.0},  # Decelerate diagonal
        {'initial_position': [0.0, 0.0], 'initial_velocity': [0.0, 0.0], 
         'acceleration': [1.0, -1.0], 'acceleration_duration': 7.0},  # Curved acceleration
        {'initial_position': [0.0, 0.0], 'initial_velocity': [8.0, 0.0], 
         'acceleration': [-1.5, 0.5], 'acceleration_duration': 4.0},  # Turn while braking
    ]
    
    for i, scenario in enumerate(scenarios, 6):
        positions, velocities_arr = generate_trajectory_points(
            'constant_acceleration', scenario, num_points=100, duration=10.0
        )
        
        df = pd.DataFrame({
            'time': np.linspace(0, 10.0, 100),
            'x': positions[:, 0],
            'y': positions[:, 1],
            'vx': velocities_arr[:, 0],
            'vy': velocities_arr[:, 1],
            'motion_type': 'constant_acceleration'
        })
        examples.append((f'example_{i:02d}_constant_acceleration', df))
    
    return examples


def generate_straight_then_circle_examples():
    """Generate examples of straight line followed by circular motion"""
    examples = []
    
    # Example 16-35: Various combinations of straight + circle
    base_scenarios = [
        # Right turn scenarios (circle_center will be calculated automatically)
        {'initial_position': [0.0, 0.0], 'initial_velocity': [5.0, 0.0],
         'straight_duration': 3.0, 'radius': 2.0,
         'angular_velocity': 0.5, 'degrees_to_turn': 90},
        {'initial_position': [0.0, 0.0], 'initial_velocity': [5.0, 0.0],
         'straight_duration': 2.0, 'radius': 1.5,
         'angular_velocity': 0.8, 'degrees_to_turn': 180},
        {'initial_position': [0.0, 0.0], 'initial_velocity': [5.0, 0.0],
         'straight_duration': 4.0, 'radius': 3.0,
         'angular_velocity': 0.3, 'degrees_to_turn': 45},
        {'initial_position': [0.0, 0.0], 'initial_velocity': [6.0, 0.0],
         'straight_duration': 2.5, 'radius': 2.0,
         'angular_velocity': 1.0, 'degrees_to_turn': 270},
        
        # Up then turn scenarios
        {'initial_position': [0.0, 0.0], 'initial_velocity': [0.0, 5.0],
         'straight_duration': 3.0, 'radius': 2.0,
         'angular_velocity': -0.5, 'degrees_to_turn': 90},
        {'initial_position': [0.0, 0.0], 'initial_velocity': [0.0, 4.0],
         'straight_duration': 3.5, 'radius': 1.5,
         'angular_velocity': -0.6, 'degrees_to_turn': 120},
        
        # Diagonal scenarios
        {'initial_position': [0.0, 0.0], 'initial_velocity': [4.0, 4.0],
         'straight_duration': 2.0, 'radius': 2.5,
         'angular_velocity': 0.4, 'degrees_to_turn': 90},
        {'initial_position': [0.0, 0.0], 'initial_velocity': [3.0, 3.0],
         'straight_duration': 3.0, 'radius': 2.0,
         'angular_velocity': -0.5, 'degrees_to_turn': 135},
        
        # Various speeds and radii
        {'initial_position': [0.0, 0.0], 'initial_velocity': [7.0, 0.0],
         'straight_duration': 2.0, 'radius': 1.0,
         'angular_velocity': 1.2, 'degrees_to_turn': 180},
        {'initial_position': [0.0, 0.0], 'initial_velocity': [4.0, 0.0],
         'straight_duration': 5.0, 'radius': 4.0,
         'angular_velocity': 0.25, 'degrees_to_turn': 90},
        {'initial_position': [0.0, 0.0], 'initial_velocity': [5.0, 0.0],
         'straight_duration': 1.5, 'radius': 1.2,
         'angular_velocity': 0.9, 'degrees_to_turn': 225},
        {'initial_position': [0.0, 0.0], 'initial_velocity': [6.0, 0.0],
         'straight_duration': 3.5, 'radius': 2.8,
         'angular_velocity': 0.35, 'degrees_to_turn': 60},
        {'initial_position': [0.0, 0.0], 'initial_velocity': [3.0, 0.0],
         'straight_duration': 4.0, 'radius': 3.2,
         'angular_velocity': 0.28, 'degrees_to_turn': 150},
        {'initial_position': [0.0, 0.0], 'initial_velocity': [5.5, 0.0],
         'straight_duration': 2.2, 'radius': 1.8,
         'angular_velocity': 0.7, 'degrees_to_turn': 90},
        {'initial_position': [0.0, 0.0], 'initial_velocity': [4.5, 0.0],
         'straight_duration': 3.2, 'radius': 2.5,
         'angular_velocity': 0.45, 'degrees_to_turn': 105},
        {'initial_position': [0.0, 0.0], 'initial_velocity': [5.0, 2.0],
         'straight_duration': 2.5, 'radius': 2.2,
         'angular_velocity': 0.5, 'degrees_to_turn': 75},
        {'initial_position': [0.0, 0.0], 'initial_velocity': [3.5, 0.0],
         'straight_duration': 4.5, 'radius': 3.6,
         'angular_velocity': 0.22, 'degrees_to_turn': 120},
        {'initial_position': [0.0, 0.0], 'initial_velocity': [6.5, 0.0],
         'straight_duration': 1.8, 'radius': 1.4,
         'angular_velocity': 1.1, 'degrees_to_turn': 240},
        {'initial_position': [0.0, 0.0], 'initial_velocity': [4.0, 0.0],
         'straight_duration': 3.8, 'radius': 3.0,
         'angular_velocity': 0.3, 'degrees_to_turn': 90},
        {'initial_position': [0.0, 0.0], 'initial_velocity': [5.0, 0.0],
         'straight_duration': 2.8, 'radius': 2.2,
         'angular_velocity': 0.55, 'degrees_to_turn': 165},
    ]
    
    for i, scenario in enumerate(base_scenarios, 16):
        positions, velocities_arr = generate_trajectory_points(
            'straight_then_circle', scenario, num_points=150, duration=15.0
        )
        
        df = pd.DataFrame({
            'time': np.linspace(0, positions.shape[0] * 0.1, positions.shape[0]),
            'x': positions[:, 0],
            'y': positions[:, 1],
            'vx': velocities_arr[:, 0],
            'vy': velocities_arr[:, 1],
            'motion_type': 'straight_then_circle'
        })
        examples.append((f'example_{i:02d}_straight_then_circle', df))
    
    return examples


def main():
    """Generate all 35 examples and save to CSV files"""
    output_dir = 'kinematics_examples'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating kinematics examples...")
    
    # Generate all examples
    all_examples = []
    all_examples.extend(generate_constant_velocity_examples())
    all_examples.extend(generate_constant_acceleration_examples())
    all_examples.extend(generate_straight_then_circle_examples())
    
    # Save to CSV files
    for filename, df in all_examples:
        filepath = os.path.join(output_dir, f'{filename}.csv')
        df.to_csv(filepath, index=False)
        print(f"✅ Saved {filepath} ({len(df)} points)")
    
    print(f"\n✅ Generated {len(all_examples)} example files in {output_dir}/")
    print(f"   - Constant velocity: 5 examples")
    print(f"   - Constant acceleration: 10 examples")
    print(f"   - Straight then circle: 20 examples")


if __name__ == '__main__':
    main()

