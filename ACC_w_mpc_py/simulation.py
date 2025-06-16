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

def run_simulation(driver_style='balanced', sim_time=60.0, dt=0.1):
    """
    Run the ACC simulation with specified driver style
    
    Parameters:
    -----------
    driver_style : str
        Driver style ('aggressive', 'balanced', or 'conservative')
    sim_time : float
        Total simulation time (s)
    dt : float
        Time step (s)
    """
    # Initialize simulation parameters
    print("\nInitializing simulation parameters...")
    
    # Create vehicle instances
    print("Creating vehicle instances...")
    ego_vehicle = EgoVehicle(initial_position=0.0, initial_velocity=15.0)  # Start at 15 m/s
    lead_vehicle = LeadVehicle(initial_position=30.0, initial_velocity=15.0)  # Start 30m ahead
    
    # Create MPC controller
    print("Creating MPC controller...")
    mpc = MPCController(horizon=10, dt=dt, min_safe_distance=10.0)
    
    # Apply driver settings
    print(f"Running simulation with {driver_style} driver style...")
    settings = DRIVER_SETTINGS[driver_style]
    apply_driver_settings(mpc, driver_style, 15.0)  # Base speed of 15 m/s
    
    # Initialize data storage
    time_history = []
    ego_vel_history = []
    lead_vel_history = []
    distance_history = []
    acceleration_history = []
    jerk_history = []
    gap_history = []
    weights_history = []
    speed_history = []
    ego_pos_history = []
    lead_pos_history = []
    
    # Run simulation
    print(f"Starting simulation with {driver_style} driver style...")
    t = 0.0
    last_acceleration = 0.0
    
    while t < sim_time:
        # Update lead vehicle
        lead_vehicle.update_state(dt, t)
        
        # Get current states
        ego_state = np.array([ego_vehicle.position, ego_vehicle.velocity])
        lead_state = np.array([lead_vehicle.position, lead_vehicle.velocity])
        
        # Calculate control input
        acceleration = mpc.optimize(ego_state, lead_state, dt)
        
        # Update ego vehicle with the calculated acceleration
        ego_vehicle.acceleration = acceleration
        ego_vehicle.update_state(dt)
        
        # Calculate jerk
        jerk = (acceleration - last_acceleration) / dt
        last_acceleration = acceleration
        
        # Store data
        time_history.append(t)
        ego_vel_history.append(ego_vehicle.velocity)
        lead_vel_history.append(lead_vehicle.velocity)
        distance_history.append(lead_vehicle.position - ego_vehicle.position)
        acceleration_history.append(acceleration)
        jerk_history.append(jerk)
        gap_history.append(settings['gap'])
        weights_history.append(mpc.get_current_weights())
        speed_history.append(settings['desired_speed_factor'])
        ego_pos_history.append(ego_vehicle.position)
        lead_pos_history.append(lead_vehicle.position)
        
        # Print progress
        if t % 0.1 < dt:  # Print every 0.1 seconds
            print(f"Simulation time: {t:.1f}s")
            print("Current settings:")
            print(f"  - Gap: {settings['gap']}s")
            print(f"  - Desired speed: {ego_vehicle.velocity * 3.6:.1f} km/h")
            print(f"  - Weights: {mpc.get_current_weights()}")
        
        t += dt
    
    print("Simulation completed. Preparing plots...")
    
    # Convert lists to numpy arrays
    time_history = np.array(time_history)
    ego_vel_history = np.array(ego_vel_history)
    lead_vel_history = np.array(lead_vel_history)
    distance_history = np.array(distance_history)
    acceleration_history = np.array(acceleration_history)
    jerk_history = np.array(jerk_history)
    gap_history = np.array(gap_history)
    weights_history = np.array(weights_history)
    speed_history = np.array(speed_history)
    ego_pos_history = np.array(ego_pos_history)
    lead_pos_history = np.array(lead_pos_history)
    
    # Plot results
    print("Generating plots...")
    plot_performance(
        time_history,
        ego_pos_history,
        ego_vel_history,
        lead_pos_history,
        lead_vel_history,
        acceleration_history,
        distance_history,
        ego_vel_history[-1] * 3.6,  # Desired velocity
        settings['gap'] * ego_vel_history[-1],  # Desired distance
        gap_history,
        weights_history,
        speed_history
    )
    
    print("Simulation complete!")

if __name__ == "__main__":
    # Run simulation with aggressive driver style
    print("\n=== Running Aggressive Driver Style Simulation ===")
    run_simulation(driver_style='aggressive', sim_time=60.0, dt=0.1)
    
    # Run simulation with conservative driver style
    print("\n=== Running Conservative Driver Style Simulation ===")
    run_simulation(driver_style='conservative', sim_time=60.0, dt=0.1) 