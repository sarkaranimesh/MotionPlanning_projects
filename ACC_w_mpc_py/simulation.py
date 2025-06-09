import numpy as np
import matplotlib.pyplot as plt
from vehicle_dynamics import EgoVehicle, LeadVehicle
from mpc_controller import MPCController
from performance_analysis import plot_performance

# Driver settings with predefined calibrations
DRIVER_SETTINGS = {
    'aggressive': {
        'gap': 1,  # Shortest gap (0.8s)
        'desired_speed_factor': 1.2,  # 20% faster than default
        'weights': {
            'q_velocity': 2.0,      # Higher weight for velocity tracking
            'q_distance': 1.0,      # Lower weight for distance
            'q_close': 5.0,         # Moderate penalty for being close
            'q_far': 0.5,           # Lower penalty for being far
            'r_acceleration': 0.05,  # Allow more aggressive acceleration
            'r_jerk': 0.05          # Allow more aggressive jerk
        }
    },
    'balanced': {
        'gap': 2,  # Medium gap (1.5s)
        'desired_speed_factor': 1.0,  # Default speed
        'weights': {
            'q_velocity': 1.0,      # Balanced velocity tracking
            'q_distance': 2.0,      # Higher weight for distance
            'q_close': 10.0,        # Higher penalty for being close
            'q_far': 1.0,           # Balanced penalty for being far
            'r_acceleration': 0.1,   # Moderate acceleration
            'r_jerk': 0.1           # Moderate jerk
        }
    },
    'conservative': {
        'gap': 3,  # Longest gap (2.5s)
        'desired_speed_factor': 0.9,  # 10% slower than default
        'weights': {
            'q_velocity': 0.5,      # Lower weight for velocity tracking
            'q_distance': 3.0,      # Highest weight for distance
            'q_close': 20.0,        # Highest penalty for being close
            'q_far': 2.0,           # Higher penalty for being far
            'r_acceleration': 0.2,   # More conservative acceleration
            'r_jerk': 0.2           # More conservative jerk
        }
    }
}

def apply_driver_settings(mpc_controller, driver_style, base_speed):
    """
    Apply driver settings to the MPC controller
    
    Parameters:
    -----------
    mpc_controller : MPCController
        MPC controller instance
    driver_style : str
        Driver style ('aggressive', 'balanced', or 'conservative')
    base_speed : float
        Base desired speed (m/s)
    """
    if driver_style not in DRIVER_SETTINGS:
        raise ValueError(f"Unknown driver style: {driver_style}")
        
    settings = DRIVER_SETTINGS[driver_style]
    
    # Set gap
    mpc_controller.set_gap(settings['gap'])
    
    # Set desired speed
    mpc_controller.desired_velocity = base_speed * settings['desired_speed_factor']
    
    # Apply weights
    mpc_controller.calibrate_weights(settings['weights'])

def run_simulation(ego_vehicle, lead_vehicle, mpc_controller, dt, simulation_time, driver_style='balanced'):
    """
    Run the ACC simulation
    
    Parameters:
    -----------
    ego_vehicle : EgoVehicle
        Ego vehicle instance
    lead_vehicle : LeadVehicle
        Lead vehicle instance
    mpc_controller : MPCController
        MPC controller instance
    dt : float
        Time step (s)
    simulation_time : float
        Total simulation time (s)
    driver_style : str
        Driver style ('aggressive', 'balanced', or 'conservative')
        
    Returns:
    --------
    tuple
        Time history and simulation results
    """
    print(f"Starting simulation with {driver_style} driver style...")
    
    # Storage for plotting
    time_history = []
    ego_pos_history = []
    ego_vel_history = []
    ego_accel_history = []
    lead_pos_history = []
    lead_vel_history = []
    distance_history = []
    gap_history = []  # Store current gap setting
    weight_history = []  # Store current weights
    speed_history = []  # Store desired speed
    
    # Apply initial driver settings
    base_speed = mpc_controller.desired_velocity
    apply_driver_settings(mpc_controller, driver_style, base_speed)
    
    # Simulation loop
    current_time = 0.0
    while current_time < simulation_time:
        # Store current states
        time_history.append(current_time)
        ego_pos_history.append(ego_vehicle.position)
        ego_vel_history.append(ego_vehicle.velocity)
        ego_accel_history.append(ego_vehicle.acceleration)
        lead_pos_history.append(lead_vehicle.position)
        lead_vel_history.append(lead_vehicle.velocity)
        distance_history.append(lead_vehicle.position - ego_vehicle.position)
        gap_history.append(mpc_controller.gap_settings[mpc_controller.current_gap])
        weight_history.append(mpc_controller.get_current_weights())
        speed_history.append(mpc_controller.desired_velocity)
        
        # Get control input from MPC
        ego_vehicle.acceleration = mpc_controller.calculate_control_input(
            ego_vehicle, lead_vehicle, dt)
        
        # Update vehicle states
        ego_vehicle.update_state(dt)
        lead_vehicle.update_state(dt, current_time)
        
        current_time += dt
        
        # Print progress every 5 seconds
        if int(current_time) % 5 == 0 and current_time > 0:
            print(f"Simulation time: {current_time:.1f}s")
            print(f"Current settings:")
            print(f"  - Gap: {mpc_controller.gap_settings[mpc_controller.current_gap]:.1f}s")
            print(f"  - Desired speed: {mpc_controller.desired_velocity*3.6:.1f} km/h")
            print(f"  - Weights: {mpc_controller.get_current_weights()}")
    
    print("Simulation completed. Preparing plots...")
    return (time_history, ego_pos_history, ego_vel_history, ego_accel_history,
            lead_pos_history, lead_vel_history, distance_history, gap_history,
            weight_history, speed_history)

def main():
    """Main function to run the simulation"""
    print("Initializing simulation parameters...")
    
    # Simulation parameters
    dt = 0.1
    simulation_time = 60.0
    
    # Convert desired speed from km/h to m/s
    base_speed = 55.0 * 1000 / 3600  # 55 km/h to m/s
    
    print("Creating vehicle instances...")
    # Create vehicle instances
    ego_vehicle = EgoVehicle(
        initial_position=0.0,
        initial_velocity=base_speed,  # Start at base speed
        max_acceleration=2.0,
        min_acceleration=-3.0,
        max_velocity=base_speed * 1.2  # Allow some overshoot
    )
    
    # Set initial lead vehicle position with safe distance
    initial_distance = 30.0  # meters
    lead_vehicle = LeadVehicle(
        initial_position=initial_distance,
        initial_velocity=0.0  # Will be updated by the sinusoidal function
    )
    
    print("Creating MPC controller...")
    # Create MPC controller with SLSQP method
    mpc_controller = MPCController(
        prediction_horizon=10,
        control_horizon=10,
        desired_velocity=base_speed,
        desired_distance=20.0,
        min_safe_distance=10.0,
        q_velocity=1.0,
        q_distance=2.0,
        r_acceleration=0.1,
        r_jerk=0.1,
        max_jerk=2.0,
        opt_method='SLSQP'
    )
    
    # Choose driver style
    driver_style = 'balanced'  # Can be 'aggressive', 'balanced', or 'conservative'
    
    print(f"Running simulation with {driver_style} driver style...")
    # Run simulation
    results = run_simulation(ego_vehicle, lead_vehicle, mpc_controller, dt, simulation_time, driver_style)
    
    print("Generating plots...")
    # Plot results and performance metrics
    plot_performance(
        results[0],  # time_history
        results[1],  # ego_pos_history
        results[2],  # ego_vel_history
        results[3],  # ego_accel_history
        results[4],  # lead_pos_history
        results[5],  # lead_vel_history
        results[6],  # distance_history
        mpc_controller.desired_velocity,
        mpc_controller.desired_distance,
        results[7],  # gap_history
        results[8]   # weight_history
    )
    print("Simulation complete!")

if __name__ == "__main__":
    main() 