#!/usr/bin/env python3
"""Verify distance formula d = (1/2) * a * t² for constant acceleration"""

from kinematics_formulas import position_constant_acceleration, generate_trajectory_points
import numpy as np

print('=' * 80)
print('DISTANCE FORMULA VERIFICATION: d = (1/2) * a * t²')
print('=' * 80)

# Test parameters: starting from rest with constant acceleration
initial_position = np.array([0.0, 0.0])
initial_velocity = np.array([0.0, 0.0])
acceleration = np.array([2.0, 0.0])  # 2 m/s² in x-direction
acceleration_magnitude = np.linalg.norm(acceleration)

print('\nTest Parameters:')
print(f'  Initial position: {initial_position}')
print(f'  Initial velocity: {initial_velocity}')
print(f'  Acceleration: {acceleration} (magnitude: {acceleration_magnitude:.6f} m/s²)')
print(f'\nFormula: distance = (1/2) * a * t²')
print('\n' + '=' * 80)
print('VERIFICATION AT DISCRETE TIME POINTS:')
print('=' * 80)
print(f'\n{"Time (s)":<12} {"Expected d":<20} {"Actual d":<20} {"Difference":<15} {"Status"}')
print('-' * 80)

# Test at multiple intermediate time points
test_times = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
all_passed = True

for t in test_times:
    # Calculate position using the formula
    position = position_constant_acceleration(initial_position, initial_velocity, acceleration, t)
    
    # Distance traveled from initial point
    distance_traveled = np.linalg.norm(position - initial_position)
    
    # Expected distance: d = (1/2) * a * t²
    expected_distance = 0.5 * acceleration_magnitude * t ** 2
    
    # Difference
    difference = abs(distance_traveled - expected_distance)
    passed = difference < 1e-6
    status = '✓ PASS' if passed else '✗ FAIL'
    
    if not passed:
        all_passed = False
    
    print(f'{t:<12.1f} {expected_distance:<20.10f} {distance_traveled:<20.10f} {difference:<15.2e} {status}')

print('\n' + '=' * 80)
print('VERIFICATION WITH TRAJECTORY GENERATION (Intermediate Points):')
print('=' * 80)

# Generate trajectory and verify intermediate points
params = {
    'initial_position': [0.0, 0.0],
    'initial_velocity': [0.0, 0.0],
    'acceleration': [2.0, 0.0],
    'acceleration_duration': 5.0
}

positions, velocities = generate_trajectory_points(
    'constant_acceleration', params, num_points=100, duration=5.0
)

times = np.linspace(0, 5.0, len(positions))
print(f'\nGenerated {len(positions)} trajectory points')
print(f'\n{"Time (s)":<12} {"Expected d":<20} {"Trajectory d":<20} {"Difference":<15} {"Status"}')
print('-' * 80)

# Check every 10th point (intermediate points)
check_indices = [10, 20, 30, 40, 50, 60, 70, 80, 90]
for i in check_indices:
    if i < len(positions):
        t = times[i]
        position = positions[i]
        distance_traveled = np.linalg.norm(position - initial_position)
        expected_distance = 0.5 * acceleration_magnitude * t ** 2
        difference = abs(distance_traveled - expected_distance)
        passed = difference < 1e-5
        status = '✓ PASS' if passed else '✗ FAIL'
        
        if not passed:
            all_passed = False
        
        print(f'{t:<12.2f} {expected_distance:<20.10f} {distance_traveled:<20.10f} {difference:<15.2e} {status}')

print('\n' + '=' * 80)
if all_passed:
    print('✅ ALL TESTS PASSED: Distance formula d = (1/2) * a * t² verified!')
else:
    print('❌ SOME TESTS FAILED')
print('=' * 80)

