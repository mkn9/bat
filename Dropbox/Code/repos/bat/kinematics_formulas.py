#!/usr/bin/env python3
"""
Kinematics Formulas Module
Provides formulas for position, velocity, acceleration, and circular motion
"""

import numpy as np
from typing import Tuple, List


def position_constant_velocity(
    initial_position: np.ndarray,
    velocity: np.ndarray,
    time: float
) -> np.ndarray:
    """
    Calculate position with constant velocity.
    
    Formula: p(t) = p₀ + v * t
    
    Args:
        initial_position: Initial position vector [x, y]
        velocity: Constant velocity vector [vx, vy]
        time: Time elapsed
        
    Returns:
        Position vector [x, y] at time t
    """
    return initial_position + velocity * time


def position_constant_acceleration(
    initial_position: np.ndarray,
    initial_velocity: np.ndarray,
    acceleration: np.ndarray,
    time: float
) -> np.ndarray:
    """
    Calculate position with constant acceleration.
    
    Formula: p(t) = p₀ + v₀ * t + (1/2) * a * t²
    
    Args:
        initial_position: Initial position vector [x, y]
        initial_velocity: Initial velocity vector [vx, vy]
        acceleration: Constant acceleration vector [ax, ay]
        time: Time elapsed
        
    Returns:
        Position vector [x, y] at time t
    """
    return initial_position + initial_velocity * time + 0.5 * acceleration * time ** 2


def velocity_constant_acceleration(
    initial_velocity: np.ndarray,
    acceleration: np.ndarray,
    time: float
) -> np.ndarray:
    """
    Calculate velocity with constant acceleration.
    
    Formula: v(t) = v₀ + a * t
    
    Args:
        initial_velocity: Initial velocity vector [vx, vy]
        acceleration: Constant acceleration vector [ax, ay]
        time: Time elapsed
        
    Returns:
        Velocity vector [vx, vy] at time t
    """
    return initial_velocity + acceleration * time


def circular_motion_position(
    center: np.ndarray,
    radius: float,
    initial_angle: float,
    angular_velocity: float,
    time: float
) -> np.ndarray:
    """
    Calculate position in circular motion.
    
    Formula: 
        x(t) = cx + r * cos(θ₀ + ω * t)
        y(t) = cy + r * sin(θ₀ + ω * t)
    
    Args:
        center: Center of circle [cx, cy]
        radius: Radius of circular path
        initial_angle: Initial angle in radians
        angular_velocity: Angular velocity in radians per second
        time: Time elapsed
        
    Returns:
        Position vector [x, y] at time t
    """
    angle = initial_angle + angular_velocity * time
    x = center[0] + radius * np.cos(angle)
    y = center[1] + radius * np.sin(angle)
    return np.array([x, y])


def circular_motion_velocity(
    center: np.ndarray,
    radius: float,
    initial_angle: float,
    angular_velocity: float,
    time: float
) -> np.ndarray:
    """
    Calculate velocity in circular motion.
    
    Formula:
        vx(t) = -r * ω * sin(θ₀ + ω * t)
        vy(t) = r * ω * cos(θ₀ + ω * t)
    
    Args:
        center: Center of circle [cx, cy]
        radius: Radius of circular path
        initial_angle: Initial angle in radians
        angular_velocity: Angular velocity in radians per second
        time: Time elapsed
        
    Returns:
        Velocity vector [vx, vy] at time t
    """
    angle = initial_angle + angular_velocity * time
    vx = -radius * angular_velocity * np.sin(angle)
    vy = radius * angular_velocity * np.cos(angle)
    return np.array([vx, vy])


def generate_trajectory_points(
    motion_type: str,
    params: dict,
    num_points: int = 100,
    duration: float = 10.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate trajectory points for different motion types.
    
    Args:
        motion_type: Type of motion ('constant_velocity', 'constant_acceleration', 
                                   'straight_then_circle', 'circle_then_straight')
        params: Dictionary of parameters for the motion
        num_points: Number of points to generate
        duration: Total duration of motion in seconds
        
    Returns:
        Tuple of (positions array, velocities array)
        positions: Array of shape (num_points, 2) with [x, y] positions
        velocities: Array of shape (num_points, 2) with [vx, vy] velocities
    """
    times = np.linspace(0, duration, num_points)
    positions = np.zeros((num_points, 2))
    velocities = np.zeros((num_points, 2))
    
    if motion_type == 'constant_velocity':
        p0 = np.array(params['initial_position'])
        v = np.array(params['velocity'])
        
        for i, t in enumerate(times):
            positions[i] = position_constant_velocity(p0, v, t)
            velocities[i] = v
    
    elif motion_type == 'constant_acceleration':
        p0 = np.array(params['initial_position'])
        v0 = np.array(params['initial_velocity'])
        a = np.array(params['acceleration'])
        accel_duration = params.get('acceleration_duration', duration)
        
        for i, t in enumerate(times):
            if t <= accel_duration:
                positions[i] = position_constant_acceleration(p0, v0, a, t)
                velocities[i] = velocity_constant_acceleration(v0, a, t)
            else:
                # After acceleration period, continue with constant velocity
                final_accel_pos = position_constant_acceleration(p0, v0, a, accel_duration)
                final_accel_vel = velocity_constant_acceleration(v0, a, accel_duration)
                remaining_time = t - accel_duration
                positions[i] = position_constant_velocity(final_accel_pos, final_accel_vel, remaining_time)
                velocities[i] = final_accel_vel
    
    elif motion_type == 'straight_then_circle':
        # Straight line segment
        p0 = np.array(params['initial_position'])
        v = np.array(params['initial_velocity'])
        straight_duration = params['straight_duration']
        radius = params['radius']
        degrees_to_turn = params.get('degrees_to_turn', 90)
        radians_to_turn = np.deg2rad(degrees_to_turn)
        
        # Calculate constant speed from initial velocity
        speed = np.linalg.norm(v)
        
        # Calculate where straight line ends
        straight_end_pos = position_constant_velocity(p0, v, straight_duration)
        straight_distance = speed * straight_duration
        
        # Calculate circle center such that the circle is tangent to the straight line
        # The center is at a distance 'radius' perpendicular to the velocity direction
        vel_angle = np.arctan2(v[1], v[0])
        
        # Determine turn direction from angular_velocity if provided, otherwise default to right turn
        angular_velocity_param = params.get('angular_velocity', None)
        if angular_velocity_param is not None:
            # Use provided angular velocity but ensure it matches the speed
            angular_velocity = angular_velocity_param
            # Ensure speed consistency: speed = radius * |angular_velocity|
            expected_speed = radius * abs(angular_velocity)
            if abs(expected_speed - speed) > 0.01:
                # Adjust angular velocity to match speed
                angular_velocity = (speed / radius) * np.sign(angular_velocity_param)
        else:
            # Default: right turn (positive angular velocity)
            angular_velocity = speed / radius
        
        # For right turn (positive angular_velocity), center is to the left of velocity
        # For left turn (negative angular_velocity), center is to the right of velocity
        perp_angle = vel_angle + np.pi/2 if angular_velocity > 0 else vel_angle - np.pi/2
        circle_center = straight_end_pos + radius * np.array([np.cos(perp_angle), np.sin(perp_angle)])
        
        # Calculate initial angle for circle (position where straight line ends)
        # Angle from circle center to the transition point
        circle_start_angle = np.arctan2(straight_end_pos[1] - circle_center[1], 
                                       straight_end_pos[0] - circle_center[0])
        
        # Calculate arc length for circular portion
        circle_arc_length = radius * radians_to_turn
        total_distance = straight_distance + circle_arc_length
        
        # Generate points at equal arc length intervals (not time intervals)
        # This ensures constant speed throughout
        arc_lengths = np.linspace(0, total_distance, num_points)
        dt = total_distance / (speed * (num_points - 1)) if num_points > 1 else 0
        
        for i, arc_length in enumerate(arc_lengths):
            if arc_length <= straight_distance:
                # Straight line portion
                t = arc_length / speed
                positions[i] = position_constant_velocity(p0, v, t)
                velocities[i] = v
            else:
                # Circular portion
                circle_arc = arc_length - straight_distance
                circle_angle = circle_arc / radius  # Arc length = radius * angle
                circle_time = circle_angle / abs(angular_velocity)
                
                if circle_angle <= radians_to_turn:
                    positions[i] = circular_motion_position(
                        circle_center, radius, circle_start_angle, angular_velocity, circle_time
                    )
                    velocities[i] = circular_motion_velocity(
                        circle_center, radius, circle_start_angle, angular_velocity, circle_time
                    )
                else:
                    # After circle, continue straight
                    final_circle_pos = circular_motion_position(
                        circle_center, radius, circle_start_angle, angular_velocity, radians_to_turn / abs(angular_velocity)
                    )
                    final_circle_vel = circular_motion_velocity(
                        circle_center, radius, circle_start_angle, angular_velocity, radians_to_turn / abs(angular_velocity)
                    )
                    remaining_arc = arc_length - straight_distance - circle_arc_length
                    remaining_time = remaining_arc / speed
                    positions[i] = position_constant_velocity(final_circle_pos, final_circle_vel, remaining_time)
                    velocities[i] = final_circle_vel
    
    return positions, velocities

