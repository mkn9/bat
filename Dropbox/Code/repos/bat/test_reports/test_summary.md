# Test Cases Summary

Generated: Tue Nov  4 14:29:39 UTC 2025

**Total Test Cases: 32**

## test_kinematics_formulas.py

**Total: 20 test cases**

### TestConstantVelocity

- **test_position_at_origin**
  - Test position calculation starting at origin
- **test_position_with_initial_offset**
  - Test position calculation with initial position offset
- **test_position_at_time_zero**
  - Test position at time zero returns initial position
- **test_position_negative_velocity**
  - Test position with negative velocity

### TestConstantAcceleration

- **test_position_with_zero_initial_velocity**
  - Test position with zero initial velocity
- **test_position_with_initial_velocity**
  - Test position with non-zero initial velocity
- **test_velocity_with_acceleration**
  - Test velocity calculation with acceleration
- **test_velocity_at_time_zero**
  - Test velocity at time zero returns initial velocity

### TestCircularMotion

- **test_circular_position_at_origin**
  - Test circular position with center at origin
- **test_circular_position_centered**
  - Test circular position with offset center
- **test_circular_velocity_magnitude**
  - Test circular velocity magnitude is correct
- **test_circular_velocity_perpendicular**
  - Test circular velocity is perpendicular to radius

### TestTrajectoryGeneration

- **test_constant_velocity_trajectory**
  - Test constant velocity trajectory generation
- **test_constant_acceleration_trajectory**
  - Test constant acceleration trajectory generation
- **test_straight_then_circle_trajectory**
  - Test straight then circle trajectory generation
- **test_invalid_motion_type**
  - Test that invalid motion type returns zeros

### TestEdgeCases

- **test_zero_time**
  - Test calculations at time zero
- **test_negative_time**
  - Test calculations with negative time
- **test_zero_velocity**
  - Test position with zero velocity
- **test_zero_radius_circle**
  - Test circular motion with zero radius

## test_generate_kinematics_examples.py

**Total: 12 test cases**

### TestExampleGeneration

- **test_generate_constant_velocity_examples**
  - Test constant velocity examples generation
- **test_generate_constant_acceleration_examples**
  - Test constant acceleration examples generation
- **test_generate_straight_then_circle_examples**
  - Test straight then circle examples generation
- **test_example_data_consistency**
  - Test that example data has consistent structure
- **test_example_numbering**
  - Test that examples are numbered correctly

### TestVisualization

- **test_create_visualization**
  - Test visualization creation
- **test_create_visualization_with_different_intervals**
  - Test visualization with different marker intervals
- **test_visualization_with_zero_velocity**
  - Test visualization handles zero velocity correctly

### TestFileOperations

- **test_output_directory_creation**
  - Test that output directories are created correctly

### TestDataValidation

- **test_dataframe_has_required_columns**
  - Test that all generated dataframes have required columns
- **test_no_nan_values**
  - Test that generated data has no NaN values
- **test_time_monotonic**
  - Test that time values are monotonically increasing

