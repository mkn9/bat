#!/usr/bin/env python3
"""
Unit tests for kinematics_formulas module
Tests all kinematic formulas and trajectory generation functions
"""

import pytest
import numpy as np
from kinematics_formulas import (
    position_constant_velocity,
    position_constant_acceleration,
    velocity_constant_acceleration,
    circular_motion_position,
    circular_motion_velocity,
    generate_trajectory_points
)


class TestConstantVelocity:
    """Tests for constant velocity motion"""
    
    def test_position_at_origin(self):
        """Test position calculation starting at origin"""
        initial_position = np.array([0.0, 0.0])
        velocity = np.array([5.0, 0.0])
        time = 2.0
        
        result = position_constant_velocity(initial_position, velocity, time)
        expected = np.array([10.0, 0.0])
        
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_position_with_initial_offset(self):
        """Test position calculation with initial position offset"""
        initial_position = np.array([10.0, 20.0])
        velocity = np.array([3.0, 4.0])
        time = 2.0
        
        result = position_constant_velocity(initial_position, velocity, time)
        expected = np.array([16.0, 28.0])
        
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_position_at_time_zero(self):
        """Test position at time zero returns initial position"""
        initial_position = np.array([5.0, 10.0])
        velocity = np.array([1.0, 2.0])
        time = 0.0
        
        result = position_constant_velocity(initial_position, velocity, time)
        
        np.testing.assert_array_almost_equal(result, initial_position)
    
    def test_position_negative_velocity(self):
        """Test position with negative velocity"""
        initial_position = np.array([10.0, 10.0])
        velocity = np.array([-2.0, -3.0])
        time = 2.0
        
        result = position_constant_velocity(initial_position, velocity, time)
        expected = np.array([6.0, 4.0])
        
        np.testing.assert_array_almost_equal(result, expected)


class TestConstantAcceleration:
    """Tests for constant acceleration motion"""
    
    def test_position_with_zero_initial_velocity(self):
        """Test position with zero initial velocity"""
        initial_position = np.array([0.0, 0.0])
        initial_velocity = np.array([0.0, 0.0])
        acceleration = np.array([2.0, 0.0])
        time = 2.0
        
        result = position_constant_acceleration(
            initial_position, initial_velocity, acceleration, time
        )
        expected = np.array([4.0, 0.0])  # 0.5 * 2 * 2^2 = 4.0
        
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_distance_formula_constant_acceleration(self):
        """Test that distance traveled = (1/2) * a * t² for constant acceleration from rest"""
        initial_position = np.array([0.0, 0.0])
        initial_velocity = np.array([0.0, 0.0])
        acceleration = np.array([3.0, 2.0])  # 2D acceleration
        acceleration_magnitude = np.linalg.norm(acceleration)
        
        # Test at multiple time points
        test_times = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        
        for t in test_times:
            # Calculate position using formula
            position = position_constant_acceleration(
                initial_position, initial_velocity, acceleration, t
            )
            
            # Calculate distance traveled from initial point
            distance_traveled = np.linalg.norm(position - initial_position)
            
            # Expected distance using formula: d = (1/2) * a * t²
            # For 2D: d = (1/2) * |a| * t²
            expected_distance = 0.5 * acceleration_magnitude * t ** 2
            
            # Verify the formula holds
            np.testing.assert_almost_equal(
                distance_traveled, expected_distance, decimal=6,
                err_msg=f"Distance formula failed at t={t}: got {distance_traveled}, expected {expected_distance}"
            )
        
        # Also test with trajectory generation to verify intermediate points
        params = {
            'initial_position': [0.0, 0.0],
            'initial_velocity': [0.0, 0.0],
            'acceleration': [3.0, 2.0],
            'acceleration_duration': 5.0
        }
        
        positions, velocities = generate_trajectory_points(
            'constant_acceleration', params, num_points=50, duration=5.0
        )
        
        # Verify several intermediate points follow the formula
        times = np.linspace(0, 5.0, 50)
        for i in range(5, len(positions), 10):  # Check every 10th point
            t = times[i]
            if t > 0:
                position = positions[i]
                distance_traveled = np.linalg.norm(position - initial_position)
                expected_distance = 0.5 * acceleration_magnitude * t ** 2
                
                np.testing.assert_almost_equal(
                    distance_traveled, expected_distance, decimal=5,
                    err_msg=f"Trajectory point at t={t:.2f}: distance {distance_traveled:.6f} != expected {expected_distance:.6f}"
                )
    
    def test_position_with_initial_velocity(self):
        """Test position with non-zero initial velocity"""
        initial_position = np.array([0.0, 0.0])
        initial_velocity = np.array([5.0, 0.0])
        acceleration = np.array([2.0, 0.0])
        time = 2.0
        
        result = position_constant_acceleration(
            initial_position, initial_velocity, acceleration, time
        )
        expected = np.array([14.0, 0.0])  # 5*2 + 0.5*2*2^2 = 14.0
        
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_velocity_with_acceleration(self):
        """Test velocity calculation with acceleration"""
        initial_velocity = np.array([5.0, 0.0])
        acceleration = np.array([2.0, 1.0])
        time = 3.0
        
        result = velocity_constant_acceleration(initial_velocity, acceleration, time)
        expected = np.array([11.0, 3.0])  # 5 + 2*3, 0 + 1*3
        
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_velocity_at_time_zero(self):
        """Test velocity at time zero returns initial velocity"""
        initial_velocity = np.array([10.0, 5.0])
        acceleration = np.array([2.0, 3.0])
        time = 0.0
        
        result = velocity_constant_acceleration(initial_velocity, acceleration, time)
        
        np.testing.assert_array_almost_equal(result, initial_velocity)


class TestCircularMotion:
    """Tests for circular motion"""
    
    def test_circular_position_at_origin(self):
        """Test circular position with center at origin"""
        center = np.array([0.0, 0.0])
        radius = 5.0
        initial_angle = 0.0
        angular_velocity = np.pi / 2  # 90 degrees per second
        time = 1.0  # 90 degrees
        
        result = circular_motion_position(center, radius, initial_angle, angular_velocity, time)
        expected = np.array([0.0, 5.0])  # At 90 degrees: (0, radius)
        
        np.testing.assert_array_almost_equal(result, expected, decimal=6)
    
    def test_circular_position_centered(self):
        """Test circular position with offset center"""
        center = np.array([10.0, 20.0])
        radius = 3.0
        initial_angle = 0.0
        angular_velocity = np.pi / 2
        time = 0.0
        
        result = circular_motion_position(center, radius, initial_angle, angular_velocity, time)
        expected = np.array([13.0, 20.0])  # Center + radius at angle 0
        
        np.testing.assert_array_almost_equal(result, expected, decimal=6)
    
    def test_circular_velocity_magnitude(self):
        """Test circular velocity magnitude is correct"""
        center = np.array([0.0, 0.0])
        radius = 5.0
        initial_angle = 0.0
        angular_velocity = 1.0
        time = 0.0
        
        result = circular_motion_velocity(center, radius, initial_angle, angular_velocity, time)
        
        # Velocity magnitude should be radius * angular_velocity
        speed = np.sqrt(result[0]**2 + result[1]**2)
        expected_speed = radius * angular_velocity  # 5.0
        
        np.testing.assert_almost_equal(speed, expected_speed, decimal=6)
    
    def test_circular_velocity_perpendicular(self):
        """Test circular velocity is perpendicular to radius"""
        center = np.array([0.0, 0.0])
        radius = 5.0
        initial_angle = 0.0
        angular_velocity = 1.0
        time = 0.0
        
        position = circular_motion_position(center, radius, initial_angle, angular_velocity, time)
        velocity = circular_motion_velocity(center, radius, initial_angle, angular_velocity, time)
        
        # Dot product should be zero (perpendicular)
        dot_product = np.dot(position - center, velocity)
        np.testing.assert_almost_equal(dot_product, 0.0, decimal=6)


class TestTrajectoryGeneration:
    """Tests for trajectory generation function"""
    
    def test_constant_velocity_trajectory(self):
        """Test constant velocity trajectory generation"""
        params = {
            'initial_position': [0.0, 0.0],
            'velocity': [5.0, 0.0]
        }
        
        positions, velocities = generate_trajectory_points(
            'constant_velocity', params, num_points=10, duration=10.0
        )
        
        assert positions.shape == (10, 2)
        assert velocities.shape == (10, 2)
        
        # First position should be initial position
        np.testing.assert_array_almost_equal(positions[0], [0.0, 0.0])
        
        # All velocities should be the same
        for vel in velocities:
            np.testing.assert_array_almost_equal(vel, [5.0, 0.0])
    
    def test_constant_acceleration_trajectory(self):
        """Test constant acceleration trajectory generation"""
        params = {
            'initial_position': [0.0, 0.0],
            'initial_velocity': [0.0, 0.0],
            'acceleration': [2.0, 0.0],
            'acceleration_duration': 5.0
        }
        
        positions, velocities = generate_trajectory_points(
            'constant_acceleration', params, num_points=10, duration=10.0
        )
        
        assert positions.shape == (10, 2)
        assert velocities.shape == (10, 2)
        
        # First position should be initial
        np.testing.assert_array_almost_equal(positions[0], [0.0, 0.0])
        
        # Position should increase over time
        assert positions[-1, 0] > positions[0, 0]
    
    def test_constant_acceleration_distance_intervals(self):
        """Test that constant acceleration has increasing distances between time markers"""
        params = {
            'initial_position': [0.0, 0.0],
            'initial_velocity': [0.0, 0.0],
            'acceleration': [2.0, 0.0],
            'acceleration_duration': 5.0
        }
        
        positions, velocities = generate_trajectory_points(
            'constant_acceleration', params, num_points=100, duration=10.0
        )
        
        # Calculate distances between consecutive points
        distances = np.array([
            np.linalg.norm(positions[i+1] - positions[i]) 
            for i in range(len(positions) - 1)
        ])
        
        # Calculate speeds from velocities
        speeds = np.array([np.linalg.norm(vel) for vel in velocities])
        
        # During acceleration phase, distances should increase over time
        # (object covers more distance in each equal time interval as it speeds up)
        accel_duration = 5.0
        total_duration = 10.0
        dt = total_duration / (len(positions) - 1)
        accel_end_idx = int(accel_duration / total_duration * len(positions))
        
        # Check acceleration phase: distances should be monotonically increasing
        accel_distances = distances[:accel_end_idx]
        if len(accel_distances) > 1:
            # Allow small numerical errors, but overall trend should be increasing
            distance_diffs = np.diff(accel_distances)
            increasing_ratio = np.sum(distance_diffs > 0) / len(distance_diffs)
            assert increasing_ratio > 0.8, f"Distances should be increasing during acceleration, but only {increasing_ratio*100:.1f}% of intervals increase"
        
        # Check that speeds are increasing during acceleration phase
        accel_speeds = speeds[:accel_end_idx]
        if len(accel_speeds) > 1:
            speed_diffs = np.diff(accel_speeds)
            assert np.all(speed_diffs > -0.01), "Speeds should be increasing during acceleration"
        
        # After acceleration, object should maintain constant velocity
        # Distances should be approximately constant (within small tolerance)
        if accel_end_idx < len(distances):
            constant_vel_distances = distances[accel_end_idx:]
            if len(constant_vel_distances) > 1:
                distance_std = np.std(constant_vel_distances)
                distance_mean = np.mean(constant_vel_distances)
                # After acceleration, distances should be relatively constant
                assert distance_std / distance_mean < 0.05, \
                    f"After acceleration, distances should be constant, but CV is {distance_std/distance_mean*100:.2f}%"
    
    def test_straight_then_circle_trajectory(self):
        """Test straight then circle trajectory generation"""
        params = {
            'initial_position': [0.0, 0.0],
            'initial_velocity': [5.0, 0.0],
            'straight_duration': 2.0,
            'radius': 2.0,
            'angular_velocity': 0.5,
            'degrees_to_turn': 90
        }
        
        positions, velocities = generate_trajectory_points(
            'straight_then_circle', params, num_points=50, duration=15.0
        )
        
        assert positions.shape[0] == 50
        assert velocities.shape[0] == 50
        
        # First position should be initial
        np.testing.assert_array_almost_equal(positions[0], [0.0, 0.0])
    
    def test_straight_then_circle_constant_speed(self):
        """Test that straight_then_circle maintains constant speed with equal distances"""
        params = {
            'initial_position': [0.0, 0.0],
            'initial_velocity': [5.0, 0.0],
            'straight_duration': 2.0,
            'radius': 2.0,
            'angular_velocity': 0.5,
            'degrees_to_turn': 90
        }
        
        positions, velocities = generate_trajectory_points(
            'straight_then_circle', params, num_points=100, duration=15.0
        )
        
        # Calculate distances between consecutive points
        distances = np.array([
            np.linalg.norm(positions[i+1] - positions[i]) 
            for i in range(len(positions) - 1)
        ])
        
        # Calculate speeds from velocities
        speeds = np.array([np.linalg.norm(vel) for vel in velocities])
        
        # All speeds should be approximately constant (within 5% tolerance)
        # This ensures constant speed throughout
        speed_std = np.std(speeds)
        speed_mean = np.mean(speeds)
        assert speed_std / speed_mean < 0.05, f"Speed variation too high: std={speed_std}, mean={speed_mean}"
        
        # Distances should be approximately equal (within 10% tolerance)
        # This ensures equal spacing in arc length
        distance_std = np.std(distances)
        distance_mean = np.mean(distances)
        assert distance_std / distance_mean < 0.10, f"Distance variation too high: std={distance_std}, mean={distance_mean}"
    
    def test_invalid_motion_type(self):
        """Test that invalid motion type returns zeros"""
        params = {'initial_position': [0.0, 0.0]}
        
        positions, velocities = generate_trajectory_points('invalid_type', params, num_points=10, duration=10.0)
        
        # Invalid motion type should return zero positions and velocities
        np.testing.assert_array_almost_equal(positions, np.zeros((10, 2)))
        np.testing.assert_array_almost_equal(velocities, np.zeros((10, 2)))


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""
    
    def test_zero_time(self):
        """Test calculations at time zero"""
        initial_position = np.array([5.0, 10.0])
        velocity = np.array([1.0, 2.0])
        
        result = position_constant_velocity(initial_position, velocity, 0.0)
        np.testing.assert_array_almost_equal(result, initial_position)
    
    def test_negative_time(self):
        """Test calculations with negative time"""
        initial_position = np.array([10.0, 10.0])
        velocity = np.array([1.0, 1.0])
        time = -2.0
        
        result = position_constant_velocity(initial_position, velocity, time)
        expected = np.array([8.0, 8.0])
        
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_zero_velocity(self):
        """Test position with zero velocity"""
        initial_position = np.array([5.0, 10.0])
        velocity = np.array([0.0, 0.0])
        time = 5.0
        
        result = position_constant_velocity(initial_position, velocity, time)
        np.testing.assert_array_almost_equal(result, initial_position)
    
    def test_zero_radius_circle(self):
        """Test circular motion with zero radius"""
        center = np.array([5.0, 10.0])
        radius = 0.0
        initial_angle = 0.0
        angular_velocity = 1.0
        time = 2.0
        
        result = circular_motion_position(center, radius, initial_angle, angular_velocity, time)
        np.testing.assert_array_almost_equal(result, center)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

